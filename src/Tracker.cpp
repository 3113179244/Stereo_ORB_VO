#include "Tracker.h"
#include "Config.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FrameDrawer.h"
#include <algorithm>
#include <iostream>
#include "ORBmatcher.h"
#include "MotionOnlyBA.h"
#include "LocalMapping.h"
#include "Viewer.h"
Tracker::Tracker(System *pSys, ORBVocabulary *pVoc, std::shared_ptr<Map> pMap, System::eSensor sensor)
    : mpSystem(pSys), mpORBVocabulary(pVoc), mpMap(pMap), mState(NO_IMAGES_YET), mVelocity(Eigen::Matrix4f::Identity()), mpReferenceKF(nullptr), mpLocalMapper(nullptr)
{
    // 从 Config 类中加载 ORB 提取器参数
    int nFeatures = Config::g_nORBnFeatures;
    float fScaleFactor = Config::g_dORBscaleFactor;
    int nLevels = Config::g_nORBnLevels;
    int finiThFAST = Config::g_nORBiniThFAST;
    int fminThFAST = Config::g_nORBminThFAST;

    // 初始化左右图 ORB 提取器
    mpORBextractorLeft = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, finiThFAST, fminThFAST);
    mpORBextractorRight = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, finiThFAST, fminThFAST);
}

Tracker::~Tracker() {}

Eigen::Matrix4f Tracker::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    imRectLeft.copyTo(mImGray);
    // 构建内参矩阵与畸变矩阵
    cv::Mat K = (cv::Mat_<float>(3, 3) << Config::g_dFx, 0, Config::g_dCx,
                 0, Config::g_dFy, Config::g_dCy,
                 0, 0, 1);
    cv::Mat DistCoef = (cv::Mat_<float>(4, 1) << Config::g_dK1, Config::g_dK2, Config::g_dP1, Config::g_dP2);

    // 实例化当前帧 (内部自动触发多线程特征提取 + 双目匹配计算深度)
    mCurrentFrame = Frame(imRectLeft, imRectRight, timestamp,
                          mpORBextractorLeft.get(), mpORBextractorRight.get(),
                          mpORBVocabulary, K, DistCoef, Config::g_dBf, Config::g_dThDepth);
    // 打印左右目图像的特征点，双目匹配成功点数
    // int nLeft = mCurrentFrame.mvKeys.size();
    // int nRight = mCurrentFrame.mvKeysRight.size();
    // int nMatches = std::count_if(mCurrentFrame.mvuRight.begin(), mCurrentFrame.mvuRight.end(),
    //                              [](float d)
    //                              { return d >= 0; }); // 或 mvDepth[i] > 0

    // std::cout << "Frame " << mCurrentFrame.mnId
    //           << " | Left features: " << nLeft
    //           << " | Right features: " << nRight
    //           << " | Stereo matches: " << nMatches << std::endl;
    // 执行跟踪状态机主逻辑
    Track();

    // 返回当前帧姿态
    return mCurrentFrame.mTcw;
}

void Tracker::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // 阶段 A: 未初始化状态 -> 执行双目初始化
    if (mState == NOT_INITIALIZED)
    {
        if (StereoInitialization())
        {
            mState = OK;
            mLastFrame = Frame(mCurrentFrame);
        }
    }
    // 阶段 B: 正常跟踪状态 -> 估计姿态
    else
    {
        bool bOK = false;

        // 优先尝试恒速模型跟踪 (Velocity Model)
        if (!mVelocity.isIdentity() && mLastFrame.mnId == mCurrentFrame.mnId - 1)
        {
            bOK = TrackWithMotionModel();
        }

        // 若运动模型失效，回退到参考关键帧跟踪
        if (!bOK)
        {
            bOK = TrackReferenceKeyFrame();
        }

        // 跟踪局部地图进行位姿精确优化
        if (bOK)
        {
            bOK = TrackLocalMap();
        }

        if (bOK)
        {
            mState = OK;
            mVelocity = mCurrentFrame.mTcw * mLastFrame.mTcw.inverse();
            if (NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }
            if (mpViewer && mState == OK)
            {
                mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);
            }
            mLastFrame = Frame(mCurrentFrame);
        }
        else
        {
            mState = LOST;
            mVelocity.setIdentity();
        }
        mLastFrame = Frame(mCurrentFrame);
    }

    if (mpFrameDrawer)
    {
        mpFrameDrawer->Update(this);
    }
}

bool Tracker::StereoInitialization()
{
    if (mCurrentFrame.N < 500)
        return false;

    // 设置世界坐标系原点
    mCurrentFrame.SetPose(Eigen::Matrix4f::Identity());

    // 创建第一帧对应的 KeyFrame 并加入 Map
    KeyFrame *pKFinit = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKFinit);

    // 地图点筛选与创建
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];

        // 筛选条件：深度必须大于 0 且小于近点阈值 (mThDepth)
        if (z > 0 && z < mCurrentFrame.mThDepth)
        {
            Eigen::Vector3f p3D = mCurrentFrame.UnprojectStereo(i);
            MapPoint *pMP = new MapPoint(p3D, pKFinit, mpMap.get());

            // 建立 KeyFrame 和 MapPoint 的双向绑定
            pMP->AddObservation(pKFinit, i);
            pKFinit->AddMapPoint(pMP, i);

            // 更新点属性（法线、金字塔深度范围、最佳描述子）
            pMP->ComputeDistinctiveDescriptor();
            pMP->UpdateNormalAndDepth();

            // 加入全局地图
            mpMap->AddMapPoint(pMP);
            mCurrentFrame.mvpMapPoints[i] = pMP;
        }
    }

    // 关键帧插入后台：将初始化关键帧推送到 LocalMapping 线程
    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKFinit);
    }

    mpReferenceKF = pKFinit;
    mnLastKeyFrameId = mCurrentFrame.mnId;

    return true;
}

bool Tracker::TrackWithMotionModel()
{
    // 根据恒速模型粗略预测当前位姿: T_cw = V * T_lw
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // 清空当前帧的地图点指针数组
    mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));

    // 利用投影建立上一帧地图点与当前帧特征点的匹配
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 7); // 搜索半径阈值设为15

    if (nmatches < 20)
    {
        // 匹配点太少，放大搜索半径重试一次
        mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * 7);
    }

    if (nmatches < 20)
        return false;

    // 只优化当前帧位姿 (Pose Optimization / Motion-only BA)
    int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 剔除优化时被判定为外点的匹配
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                nmatches--;
            }
        }
    }

    mnMatchesInliers = nInliers;

    // 内点数满足要求则跟踪成功
    return (nInliers >= 10);
}

bool Tracker::TrackReferenceKeyFrame()
{
    // 假设初值继承上一帧位姿
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // 清空当前帧关联的地图点
    mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));

    if (!mpReferenceKF)
        return false;

    // 通过词袋模型 (BoW) 或描述子匹配参考关键帧与当前帧特征点
    ORBmatcher matcher(0.7, true);
    std::vector<MapPoint *> vpMapPointMatches;

    // 搜索参考关键帧 mpReferenceKF 在当前帧中的匹配点
    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
        return false;

    // 将匹配到的 MapPoints 赋值给当前帧
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (vpMapPointMatches[i])
        {
            mCurrentFrame.mvpMapPoints[i] = vpMapPointMatches[i];
        }
    }

    // 位姿优化 (Motion-Only BA)
    int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 剔除外点
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
            mCurrentFrame.mvbOutlier[i] = false;
        }
    }

    mnMatchesInliers = nInliers;

    return nInliers >= 10;
}

bool Tracker::TrackLocalMap()
{
    // 搜集局部地图关键帧 (Local KeyFrames)
    std::vector<KeyFrame *> vpLocalKeyFrames;

    // 将当前匹配点对应的 KeyFrame 以及共视 KeyFrames 放入局部关键帧列表
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (auto mit = observations.begin(); mit != observations.end(); mit++)
                {
                    vpLocalKeyFrames.push_back(mit->first);
                }
            }
        }
    }

    std::sort(vpLocalKeyFrames.begin(), vpLocalKeyFrames.end());
    vpLocalKeyFrames.erase(std::unique(vpLocalKeyFrames.begin(), vpLocalKeyFrames.end()), vpLocalKeyFrames.end());

    if (vpLocalKeyFrames.empty())
        return false;

    // 搜集局部地图点 (Local MapPoints) 并剔除已匹配的点
    std::vector<MapPoint *> vpLocalMapPoints;
    for (KeyFrame *pKF : vpLocalKeyFrames)
    {
        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint *pMP : vpMPs)
        {
            if (!pMP || pMP->isBad())
                continue;
            // 避免重复添加
            if (std::find(vpLocalMapPoints.begin(), vpLocalMapPoints.end(), pMP) == vpLocalMapPoints.end())
            {
                vpLocalMapPoints.push_back(pMP);
            }
        }
    }

    // 将局部地图点投影到当前帧进行二次匹配
    ORBmatcher matcher(0.8, true);
    int nMatches = 0;

    for (MapPoint *pMP : vpLocalMapPoints)
    {
        if (pMP->isBad())
            continue;

        // 检查该地图点是否已在当前帧匹配过
        bool bAlreadyFound = false;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i] == pMP)
            {
                bAlreadyFound = true;
                break;
            }
        }

        if (bAlreadyFound)
            continue;

        // 将地图点投影到当前帧像素平面
        Eigen::Vector3f P_w = pMP->GetWorldPos();
        Eigen::Vector3f P_c = mCurrentFrame.GetRotationInverse().transpose() * P_w + mCurrentFrame.mTcw.block<3, 1>(0, 3);

        if (P_c[2] <= 0)
            continue; // 剔除相机后方的点

        // 计算投影像素坐标
        float u = Frame::fx * P_c[0] / P_c[2] + Frame::cx;
        float v = Frame::fy * P_c[1] / P_c[2] + Frame::cy;

        if (u < Frame::mnMinX || u >= Frame::mnMaxX || v < Frame::mnMinY || v >= Frame::mnMaxY)
            continue;

        // 在投影区域 (半径 5~10 像素) 内查找的最佳描述子点
        std::vector<size_t> vIndices = mCurrentFrame.GetFeaturesInArea(u, v, 5.0f);
        if (vIndices.empty())
            continue;

        int bestDist = 255;
        int bestIdx = -1;

        for (size_t idx : vIndices)
        {
            if (mCurrentFrame.mvpMapPoints[idx])
                continue; // 该像素特征点已有匹配

            // 比对描述子距离
            cv::Mat dMP = pMP->GetDescriptor();
            cv::Mat dFrame = mCurrentFrame.mDescriptors.row(idx);
            int dist = ORBmatcher::DescriptorDistance(dMP, dFrame);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // 若匹配质量达标则建立关联
        if (bestDist < ORBmatcher::TH_HIGH && bestIdx >= 0)
        {
            mCurrentFrame.mvpMapPoints[bestIdx] = pMP;
            nMatches++;
        }
    }

    // 第二次精细位姿优化 (Motion-Only BA)
    int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 更新内点标记并统计
    mnMatchesInliers = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
            }
            else
            {
                mnMatchesInliers++;
            }
        }
    }

    // 局部地图跟踪成功的最低内点门槛
    return mnMatchesInliers >= 30;
}

bool Tracker::NeedNewKeyFrame()
{
    bool bLocalMappingIdle = mpLocalMapper->SetNotStop();

    // 跟踪到的点太少，位姿不可靠，不建帧
    if (mnMatchesInliers < 15)
        return false;

    // 参考关键帧中被跟踪到的点数量
    int nRefMatches = 0;
    if (mpReferenceKF)
    {
        const std::vector<MapPoint *> vpRefMPs = mpReferenceKF->GetMapPointMatches();
        for (size_t i = 0; i < vpRefMPs.size(); i++)
            if (vpRefMPs[i] && !vpRefMPs[i]->isBad())
                nRefMatches++;
    }

    // 近点统计
    int nNonTrackedClose = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mCurrentFrame.mThDepth)
            if (!mCurrentFrame.mvpMapPoints[i] || mCurrentFrame.mvbOutlier[i])
                nNonTrackedClose++;
    }

    // -- ORB-SLAM2 风格的三条条件 --
    // c1a: 距上次关键帧超过 20 帧 → 强制插入
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + 20;
    // c1b: 距上次关键帧超过 2 帧 且 LocalMapping 空闲 → 插入（主要持续来源）
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + 2 && bLocalMappingIdle);
    // c2: 跟踪比跌破参考关键帧的 0.75 → 插入
    const bool c2 = (nRefMatches > 0 &&
                     static_cast<float>(mnMatchesInliers) / static_cast<float>(nRefMatches) < 0.75f);
    // c3: 跟踪点少且近点大量未跟踪 → 插入（提供三角化基线）
    const bool c3 = (mnMatchesInliers < 50 && nNonTrackedClose > 70);

    return (c1a || c1b || c2 || c3);
}

void Tracker::CreateNewKeyFrame()
{
    // 创建新关键帧
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap.get());

    // 遍历当前帧特征点：为新增的未跟踪点创建 MapPoint，为已跟踪点追加 Observation
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

        if (!pMP)
        {
            // 筛选新增点：深度在有效范围 [0, mThDepth] 内
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0 && z < mCurrentFrame.mThDepth)
            {
                Eigen::Vector3f p3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(p3D, pKF, mpMap.get());

                pNewMP->AddObservation(pKF, i);
                pKF->AddMapPoint(pNewMP, i);

                pNewMP->ComputeDistinctiveDescriptor();
                pNewMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }
        else if (!mCurrentFrame.mvbOutlier[i])
        {
            // 如果点已被跟踪，只需更新双向引用关系
            pMP->AddObservation(pKF, i);
            pKF->AddMapPoint(pMP, i);
        }
    }

    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKF);
    }

    mpReferenceKF = pKF;
    mnLastKeyFrameId = mCurrentFrame.mnId;
}

void Tracker::Reset()
{
    mState = NOT_INITIALIZED;
    mVelocity.setIdentity();
    mpReferenceKF = nullptr;
}
