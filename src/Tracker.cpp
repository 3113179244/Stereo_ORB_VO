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
#include "KeyFrameDatabase.h"
#include "Viewer.h"
Tracker::Tracker(System *pSys, ORBVocabulary *pVoc, KeyFrameDatabase *pKFDB, std::shared_ptr<Map> pMap, System::eSensor sensor)
    : mpSystem(pSys), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpMap(pMap), mState(NO_IMAGES_YET), mVelocity(Eigen::Matrix4f::Identity()), mpReferenceKF(nullptr), mpLocalMapper(nullptr)
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

// Tracker.cpp
void Tracker::Track()
{
    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    if (mState == NOT_INITIALIZED)
    {
        if (StereoInitialization())
        {
            mState = OK;
            mLastFrame = Frame(mCurrentFrame);
        }
    }
    else
    {
        bool bOK = false;
        bool bWasOK = false;

        // 1. 系统正常状态 (OK)：尝试运动模型与参考帧跟踪
        if (mState == OK)
        {
            bWasOK = true;

            bool bMM = false, bRF = false, bLM = false;
            if (mLastFrame.mnId == mCurrentFrame.mnId - 1)
                bMM = TrackWithMotionModel();

            bRF = bRF || bMM;

            if (!bRF)
            {
                bRF = TrackReferenceKeyFrame();
            }

            bOK = bRF;
            if (bOK)
            {
                bLM = TrackLocalMap();
                bOK = bLM;
            }
        }
        else // 2. 系统丢失状态 (LOST)：触发重定位
        {
            bOK = Relocalize();
        }

        // 3. 跟踪或重定位结果处理
        if (bOK)
        {
            mState = OK;
            // 恒速模型速度：T_c_current * inv(T_c_last) = T_current_last。
            // 仅在正常连续跟踪时更新；刚重定位成功时保持单位阵，避免旧速度误导。
            if (bWasOK)
                mVelocity = mCurrentFrame.mTcw * mLastFrame.mTcw.inverse();
            else
                mVelocity = Eigen::Matrix4f::Identity();

            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            if (mpViewer)
                mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);
        }
        else
        {
            mState = LOST;
            mVelocity.setIdentity(); // 丢失状态下清空速度
        }

        mLastFrame = Frame(mCurrentFrame);
    }

    if (mpFrameDrawer)
        mpFrameDrawer->Update(this);
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

    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKFinit);

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

    if (!mpReferenceKF || mpReferenceKF->mbBad)
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

bool Tracker::Relocalize()
{
    // 计算当前帧的 BoW 向量（若尚未计算）
    if (mCurrentFrame.mBowVec.empty())
        mCurrentFrame.ComputeBoW();

    if (!mpKeyFrameDB)
        return false;

    // 从关键帧数据库获取用于重定位的候选关键帧
    std::vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (vpCandidateKFs.empty())
    {
        std::cout << "[Reloc]   no reloc candidates (BoW shared words=0)" << std::endl;
        return false;
    }
    std::cout << "[Reloc]   got " << vpCandidateKFs.size() << " candidates" << std::endl;

    ORBmatcher matcher(0.75, true);

    // 依次尝试每个候选关键帧，直到找到一个满足内点门槛的候选
    for (KeyFrame *pKF : vpCandidateKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        // 通过 BoW 在候选关键帧与当前帧之间建立特征匹配
        std::vector<MapPoint *> vpMapPointMatches;
        int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vpMapPointMatches);

        // 匹配点太少，直接跳过该候选
        if (nmatches < 15)
        {
            std::cout << "[Reloc]   candidate kfId=" << pKF->mnId
                      << " BoWmatches=" << nmatches << " (skip<15)" << std::endl;
            continue;
        }

        // 将匹配到的地图点赋给当前帧
        mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (vpMapPointMatches[i])
                mCurrentFrame.mvpMapPoints[i] = vpMapPointMatches[i];
        }

        // 以候选关键帧的位姿作为当前帧位姿初值
        mCurrentFrame.SetPose(pKF->GetPose());

        // Motion-Only BA 优化当前帧位姿
        int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

        // ---- 若第一轮匹配内点偏少，则基于优化后的位姿初值，将候选KF的
        //      未匹配地图点投影到当前帧做局部窗口补充匹配，再重新优化 ----
        if (nInliers >= 10 && nInliers < 20)
        {
            const Eigen::Matrix3f Rcw = mCurrentFrame.mTcw.block<3, 3>(0, 0);
            const Eigen::Vector3f tcw = mCurrentFrame.mTcw.block<3, 1>(0, 3);

            std::vector<MapPoint *> vpMPsKF = pKF->GetMapPointMatches();
            for (MapPoint *pMP : vpMPsKF)
            {
                if (!pMP || pMP->isBad())
                    continue;

                // 跳过已经被匹配的点
                bool bMatched = false;
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] == pMP)
                    {
                        bMatched = true;
                        break;
                    }
                }
                if (bMatched)
                    continue;

                // 将地图点投影到当前帧
                Eigen::Vector3f P_w = pMP->GetWorldPos();
                Eigen::Vector3f P_c = Rcw * P_w + tcw;
                if (P_c[2] <= 0)
                    continue;

                float u = Frame::fx * P_c[0] / P_c[2] + Frame::cx;
                float v = Frame::fy * P_c[1] / P_c[2] + Frame::cy;
                if (u < Frame::mnMinX || u >= Frame::mnMaxX || v < Frame::mnMinY || v >= Frame::mnMaxY)
                    continue;

                std::vector<size_t> vIndices = mCurrentFrame.GetFeaturesInArea(u, v, 7.0f);
                if (vIndices.empty())
                    continue;

                int bestDist = 255;
                int bestIdx = -1;
                for (size_t idx : vIndices)
                {
                    if (mCurrentFrame.mvpMapPoints[idx])
                        continue;

                    cv::Mat dMP = pMP->GetDescriptor();
                    cv::Mat dFrame = mCurrentFrame.mDescriptors.row(idx);
                    int dist = ORBmatcher::DescriptorDistance(dMP, dFrame);
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx = idx;
                    }
                }

                if (bestDist < ORBmatcher::TH_HIGH && bestIdx >= 0)
                    mCurrentFrame.mvpMapPoints[bestIdx] = pMP;
            }

            // 补充匹配后再次优化位姿
            nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);
        }

        std::cout << "[Reloc]   candidate kfId=" << pKF->mnId
                  << " finalInliers=" << nInliers << " (need>=10)" << std::endl;

        if (nInliers >= 10)
        {
            // 剔除优化后仍被判定为外点的匹配
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                    mCurrentFrame.mvbOutlier[i] = false;
                }
            }

            // 设定新的参考关键帧
            mpReferenceKF = pKF;
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mnMatchesInliers = nInliers;
            return true;
        }
    }

    // 所有候选均未满足门槛，重定位失败
    return false;
}

bool Tracker::TrackLocalMap()
{
    // 搜集局部地图关键帧 (Local KeyFrames)
    std::vector<KeyFrame *> vpLocalKeyFrames;

    // 将当前匹配点对应的 KeyFrame 放入局部关键帧列表
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

    // 增加每个局部关键帧的最佳共视邻居，扩大局部地图覆盖范围，
    // 避免局部地图只跟随"当前已匹配点"，从而缓解匹配点萎缩的恶性循环
    std::vector<KeyFrame *> vpLocalKFWithNeighbors = vpLocalKeyFrames;
    for (KeyFrame *pKF : vpLocalKeyFrames)
    {
        if (pKF->mbBad)
            continue;
        std::vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for (KeyFrame *pN : vNeighs)
        {
            if (!pN->mbBad)
                vpLocalKFWithNeighbors.push_back(pN);
        }
    }
    std::sort(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end());
    vpLocalKFWithNeighbors.erase(std::unique(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end()),
                                 vpLocalKFWithNeighbors.end());

    if (vpLocalKFWithNeighbors.empty())
        return false;

    // 搜集局部地图点 (Local MapPoints) 并剔除已匹配的点
    std::vector<MapPoint *> vpLocalMapPoints;
    for (KeyFrame *pKF : vpLocalKFWithNeighbors)
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

    // 局部地图跟踪成功的最低内点门槛（略低于原版 30，降低误判丢失的几率）
    return mnMatchesInliers >= 20;
}

bool Tracker::NeedNewKeyFrame()
{
    // 确保 LocalMapping 不会被外部阻断
    if (mpLocalMapper)
        mpLocalMapper->SetNotStop();

    // 1. 如果局部建图线程被暂停，不插入
    if (mpLocalMapper && mpLocalMapper->isStopped())
        return false;

    // 2. 跟踪到的内点太少，位姿不可靠，不建帧
    if (mnMatchesInliers < 15)
        return false;

    const int nKFs = mpMap->GetKeyFramesInMap();

    // 3. 【核心修复】：参考关键帧中有效的地图点总数
    //    直接统计 mpReferenceKF 里所有非 bad 的地图点，不设 minObs 门槛！
    int nRefMatches = 0;
    if (mpReferenceKF)
    {
        // 原版 TrackedMapPoints(nMinObs) 中，只有 nKFs <= 2 时 minObs才为 2，
        // 但最稳妥、最直观的是直接统计当前参考帧绑定的有效地图点数：
        const std::vector<MapPoint*> vpRefMPs = mpReferenceKF->GetMapPointMatches();
        for (size_t i = 0; i < vpRefMPs.size(); i++)
        {
            if (vpRefMPs[i] && !vpRefMPs[i]->isBad())
                nRefMatches++;
        }
    }

    // 避免除零异常
    if (nRefMatches == 0)
        nRefMatches = 1;

    // 4. 双目近点统计
    int nNonTrackedClose = 0;
    int nTrackedClose = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mCurrentFrame.mThDepth)
        {
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        }
    }

    // 触发近点补充机制
    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // 5. 比例与条件判断
    float thRefRatio = (nKFs < 2) ? 0.40f : 0.75f;
    const int nFramesPassed = mCurrentFrame.mnId - mnLastKeyFrameId;

    // Condition 1a: 距离上一关键帧很长时间 (>= 20 帧)
    const bool c1a = nFramesPassed >= 20;

    // Condition 1b: 至少隔了 2 帧
    const bool c1b = nFramesPassed >= 2;

    // Condition 1c: 跟踪急剧衰减（跌破 25%）或急需补充近点
    const bool c1c = (mnMatchesInliers < nRefMatches * 0.25f) || bNeedToInsertClose;

    // Condition 2: 匹配点降到参考帧 75% 以下，或需要补近点；且内点 > 15
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio) || bNeedToInsertClose) && (mnMatchesInliers > 15);

    // 日志打印（观察修复效果）
    // std::cout << "[KF Check] Inliers: " << mnMatchesInliers 
    //           << " | RefMatches: " << nRefMatches 
    //           << " | Ratio: " << (float)mnMatchesInliers / nRefMatches 
    //           << " | FrameDiff: " << nFramesPassed << std::endl;

    // 6. 决策
    if ((c1a || c1b || c1c) && c2)
    {
        // 只要队列里的阻塞数量少于 3 个，就允许插入
        if (mpLocalMapper)
        {
            if (mpLocalMapper->KeyframesInQueue() < 3)
                return true;
            else
                return false;
        }
        return true;
    }

    return false;
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

    if (mpKeyFrameDB)
    {
        mpKeyFrameDB->add(pKF);
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
