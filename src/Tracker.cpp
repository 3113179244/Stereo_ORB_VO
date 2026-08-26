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
#include "Frame.h"
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
    mImGray = imRectLeft.clone();

    // 构建内参矩阵与畸变矩阵
    cv::Mat K = (cv::Mat_<float>(3, 3) << Config::g_dFx, 0, Config::g_dCx,
                 0, Config::g_dFy, Config::g_dCy,
                 0, 0, 1);
    cv::Mat DistCoef = (cv::Mat_<float>(4, 1) << Config::g_dK1, Config::g_dK2, Config::g_dP1, Config::g_dP2);

    mCurrentFrame = Frame(imRectLeft.clone(), imRectRight.clone(), timestamp,
                          mpORBextractorLeft.get(), mpORBextractorRight.get(),
                          mpORBVocabulary, K, DistCoef, Config::g_dBf, Config::g_dThDepth);

    // 执行跟踪状态机主逻辑
    Track();

    // 返回当前帧姿态
    return mCurrentFrame.mTcw;
}

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
            mVelocity.setIdentity(); // 初始化后速度归零
            std::cout << "[Tracking] 系统初始化成功！" << std::endl;
        }
    }
    else
    {
        UpdateLastFrame(); // 更新上一帧数据
        bool bOK = false;
        bool bFromRelocalization = false; // 标记是否是通过重定位恢复的

        // 1. 系统正常跟踪状态
        if (mState == OK)
        {
            bool bMM = false, bRF = false, bLM = false;

            // 1.1 优先尝试恒速模型
            if (mLastFrame.mnId == mCurrentFrame.mnId - 1 && !mVelocity.isIdentity())
            {
                bMM = TrackWithMotionModel();
            }

            // 1.2 若恒速模型失败，回退到参考关键帧跟踪
            if (!bMM)
            {
                // 清空先前可能留下的不完整匹配，恢复位姿初值
                mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));
                mCurrentFrame.SetPose(mLastFrame.mTcw);

                bRF = TrackReferenceKeyFrame();
            }
            else
            {
                bRF = true;
            }

            // 1.3 粗追踪成功后，再进行局部地图追踪 (TrackLocalMap)
            bOK = bRF;
            if (bOK)
            {
                bLM = TrackLocalMap();
                bOK = bLM;
            }

            if (!bOK)
            {
                std::cout << "[Tracking 警告] 帧 ID " << mCurrentFrame.mnId
                          << " 跟踪丢失！(MM:" << bMM << ", RefKF:" << bRF << ", LocalMap:" << bLM << ")" << std::endl;
            }
        }
        else // 2. 系统丢失状态 (mState == LOST)
        {
            std::cout << "[Tracking] 帧 ID " << mCurrentFrame.mnId << " 处于 LOST 状态，准备执行重定位..." << std::endl;
            bOK = Relocalize(); // 调用重定位
            if (bOK)
            {
                bFromRelocalization = true; // 标记来自重定位
            }
        }

        // 3. 状态更新与位姿预测更新
        if (bOK)
        {
            mState = OK;
            if (bFromRelocalization)
            {
                mVelocity.setIdentity();
            }
            else
            {
                mVelocity = mCurrentFrame.mTcw * mLastFrame.mTcw.inverse();
            }

            // 是否需要插入新的关键帧
            if (NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }

            if (mpViewer)
                mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);
        }
        else
        {
            mState = LOST;
            mVelocity.setIdentity();
        }

        // 保存上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    if (mpFrameDrawer)
        mpFrameDrawer->Update(this);

    if (mState == OK && !mCurrentFrame.mTcw.hasNaN() && !mCurrentFrame.mTcw.isZero())
    {
        Eigen::Matrix4f Tcr = Eigen::Matrix4f::Identity();
        if (mpReferenceKF && !mpReferenceKF->mbBad)
        {
            Tcr = mCurrentFrame.mTcw * mpReferenceKF->GetPoseInverse();
        }
        else
        {
            Tcr = mCurrentFrame.mTcw;
        }

        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(false);
    }
    else
    {
        // 记录丢失帧
        if (!mlRelativeFramePoses.empty())
        {
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        }
        else
        {
            mlRelativeFramePoses.push_back(Eigen::Matrix4f::Identity());
            mlpReferences.push_back(nullptr);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        }
        mlbLost.push_back(true);
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
        mpLocalMapper->RequestStopBA();
    }

    mpReferenceKF = pKFinit;
    mnLastKeyFrameId = mCurrentFrame.mnId;

    return true;
}

bool Tracker::TrackWithMotionModel()
{
    // ORB-SLAM2 标准：基于运动模型投影搜索，使用 0.9 的 NN Ratio，检查旋转一致性
    ORBmatcher matcher(0.9f, true);

    // 1. 基于恒速模型预测当前帧初始位姿
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

    // 阶段 1：使用标准搜索半径 th = 7.0f (单目为 15.0f，双目因有视差约束基准设为 7.0f)
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 7.0f, false);

    // 阶段 2：若匹配数过少，放宽到 15.0f 重新搜索
    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 15.0f, false);
    }

    if (nmatches < 10)
        return false;

    // 第一次位姿优化 (Motion-only BA)
    int num_inliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 阶段 3：如果优化后内点不足 40，使用 2 倍半径 (th = 15.0f) 补搜未匹配点并二次优化
    if (num_inliers < 40)
    {
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i] = nullptr;
        }

        int additional_matches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 15.0f, false);
        if (num_inliers + additional_matches >= 20)
        {
            num_inliers = MotionOnlyBA::Optimize(&mCurrentFrame);
        }
    }

    // 剔除 Outliers
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
            mCurrentFrame.mvpMapPoints[i] = nullptr;
    }

    // ORB-SLAM2 判定恒速模型跟踪成功的标准：内点数 >= 10
    return num_inliers >= 10;
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
    if (mCurrentFrame.mBowVec.empty())
        mCurrentFrame.ComputeBoW();

    if (!mpKeyFrameDB)
    {
        std::cout << "  └─ [Relocalize] KeyFrameDB 为空！" << std::endl;
        return false;
    }

    // 利用 KeyFrameDatabase 寻找候选关键帧
    std::vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (vpCandidateKFs.empty())
    {
        std::cout << "  └─ [Relocalize] 未检索到词袋候选帧" << std::endl;
        return false;
    }

    std::cout << "  └─ [Relocalize] 检索到 " << vpCandidateKFs.size() << " 个候选帧，开始 PnP 匹配..." << std::endl;

    ORBmatcher matcher(0.75, true);

    for (KeyFrame *pKF : vpCandidateKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        std::vector<MapPoint *> vpMapPointMatches;
        int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vpMapPointMatches);

        if (nmatches < 8)
            continue;

        std::vector<cv::Point3f> vPts3D;
        std::vector<cv::Point2f> vPts2D;
        std::vector<int> vMapPointIndices;

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f P3D = pMP->GetWorldPos();
                vPts3D.push_back(cv::Point3f(P3D.x(), P3D.y(), P3D.z()));
                vPts2D.push_back(mCurrentFrame.mvKeysUn[i].pt);
                vMapPointIndices.push_back(i);
            }
        }

        if (vPts3D.size() < 8)
            continue;

        cv::Mat rvec, tvec;
        std::vector<int> inliersPnP;

        bool bPnPSuccess = cv::solvePnPRansac(
            vPts3D, vPts2D,
            mCurrentFrame.mK, cv::Mat(),
            rvec, tvec,
            false, 300, 8.0f, 0.99, inliersPnP, cv::SOLVEPNP_EPNP);

        if (!bPnPSuccess || inliersPnP.size() < 6)
            continue;

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix4f Tcw_pnp = Eigen::Matrix4f::Identity();
        for (int r = 0; r < 3; r++)
        {
            Tcw_pnp(r, 3) = static_cast<float>(tvec.at<double>(r));
            for (int c = 0; c < 3; c++)
            {
                Tcw_pnp(r, c) = static_cast<float>(R_cv.at<double>(r, c));
            }
        }

        mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));
        for (int inlierIdx : inliersPnP)
        {
            int frameIdx = vMapPointIndices[inlierIdx];
            mCurrentFrame.mvpMapPoints[frameIdx] = vpMapPointMatches[frameIdx];
        }

        mCurrentFrame.SetPose(Tcw_pnp);

        int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

        if (nInliers >= 8)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
                    mCurrentFrame.mvbOutlier[i] = false;
                }
            }

            mpReferenceKF = pKF;
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mnMatchesInliers = nInliers;

            std::cout << "  └─ [Relocalize SUCCESS] 重定位成功恢复！参考 KF ID: " << pKF->mnId
                      << " | 匹配点数: " << nInliers << std::endl;
            return true;
        }
    }

    std::cout << "  └─ [Relocalize FAILED] PnP / BA 校验均未通过" << std::endl;
    return false;
}

/**
 * @brief 局部地图跟踪顶层控制与成功门槛
 */
bool Tracker::TrackLocalMap()
{
    // 1. 更新局部关键帧与局部地图点
    UpdateLocalMap();

    // 2. 投影匹配局部地图点
    SearchLocalPoints();

    // 3. 位姿优化 (Motion-Only BA)
    int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 4. 更新内点标记并剔除外点
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

    // 5. 将当前局部地图点同步到 Map 供可视化渲染
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // ORB-SLAM2 官方标准门槛：局部地图跟踪成功判定内点数至少为 30 (保底宽松门槛不低于 15~30)
    return mnMatchesInliers >= 30;
}

bool Tracker::NeedNewKeyFrame()
{
    // 1. 如果 LocalMapping 线程被停止（例如闭环时锁定），禁止插入关键帧
    if (mpLocalMapper && (mpLocalMapper->isStopped() || mpLocalMapper->GetStopRequired()))
        return false;

    const int nKFs = mpMap ? mpMap->GetKeyFramesInMap() : 0;

    // 2. 距离上一次重定位太近且关键帧已有一定积累时，不插入关键帧
    // （若工程中暂无重定位帧记录，此项默认通过）
    const int mMaxFrames = static_cast<int>(Config::g_dFps > 0.0 ? Config::g_dFps : 20.0);
    const int mMinFrames = 0;

    // 3. 统计参考关键帧跟踪到的稳定地图点数量 (nRefMatches)
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;

    int nRefMatches = 0;
    if (mpReferenceKF && !mpReferenceKF->mbBad)
    {
        nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    }
    if (nRefMatches <= 0)
        nRefMatches = 1; // 避免除零

    // 4. 查询 LocalMapping 是否处于空闲状态
    bool bLocalMappingIdle = mpLocalMapper ? (mpLocalMapper->KeyframesInQueue() == 0) : true;

    // 5. 双目专属逻辑：统计近点（Close Points）跟踪状况
    int nNonTrackedClose = 0; // 当前帧中存在有效深度但尚未绑定地图点的近点
    int nTrackedClose = 0;    // 当前帧中已成功跟踪并内点匹配的近点

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        // 深度大于0且小于远近点阈值视为近点
        if (mCurrentFrame.mvDepth[i] > 0.0f && mCurrentFrame.mvDepth[i] < mCurrentFrame.mThDepth)
        {
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        }
    }

    // 双目特有决策：已跟踪近点稀疏(<100)且未跟踪近点充足(>70)，必须立即建关键帧补充地图点
    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // 6. 决策阈值设置
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    const int nFramesPassed = mCurrentFrame.mnId - mnLastKeyFrameId;

    // 条件 1a: 距离上一关键帧已超过最大帧数 (如 1 秒未插帧)
    const bool c1a = nFramesPassed >= mMaxFrames;

    // 条件 1b: 经过了最小间隔帧数且 LocalMapping 空闲
    const bool c1b = (nFramesPassed >= mMinFrames && bLocalMappingIdle);

    // 条件 1c: 跟踪显著变弱 (内点少于参考帧的 25%) 或 满足双目急需近点条件
    const bool c1c = (mnMatchesInliers < nRefMatches * 0.25f) || bNeedToInsertClose;

    // 条件 2: 匹配点相比参考帧明显减少(或需要近点)，且当前内点数满足最低跟踪要求 (>15)
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio) || bNeedToInsertClose) && (mnMatchesInliers > 15);

    // 7. 综合决策与 LocalMapping 队列控制
    if ((c1a || c1b || c1c) && c2)
    {
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            if (mpLocalMapper)
            {
                // 打断局部 BA 优化以尽快响应新关键帧
                mpLocalMapper->RequestStopBA();

                // 双目模式下限制后端缓冲队列长度，避免挤压过多关键帧导致内存暴涨和耗时堆积
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
                else
                    return false;
            }
            return false;
        }
    }

    return false;
}

void Tracker::CreateNewKeyFrame()
{
    // 如果局部建图器处于停止状态且无法阻止其停止，则不插入
    if (mpLocalMapper && !mpLocalMapper->SetNotStop())
        return;

    // Step 1: 创建新关键帧并与当前帧绑定
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap.get());
    mpReferenceKF = pKF;

    // Step 2: 双目/RGB-D 专属建点逻辑（按深度排序，保底 100 个近点）
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0.0f)
        {
            vDepthIdx.push_back(std::make_pair(z, i));
        }
    }

    if (!vDepthIdx.empty())
    {
        // 深度从小到大排序，优先处理近点
        std::sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

            if (!pMP)
            {
                bCreateNew = true;
            }
            else if (pMP->GetObservations().size() < 1)
            {
                // 地图点已无有效观测，重置并重新创建
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }

            if (bCreateNew)
            {
                // 反投影生成世界坐标系下的 3D 点
                Eigen::Vector3f x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap.get());

                pNewMP->AddObservation(pKF, i);
                pKF->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptor();
                pNewMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                nPoints++;
            }
            else
            {
                // 若该点已被成功跟踪且不是外点，只需追加本关键帧的观测
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    pMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pMP, i);
                }
                nPoints++;
            }

            // 停止条件：深度超过阈值且已处理近点数达到 100 个以上
            if (vDepthIdx[j].first > mCurrentFrame.mThDepth && nPoints > 100)
                break;
        }
    }

    // Step 3: 将关键帧送入各个后端处理模块
    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKF);

    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKF);
        mpLocalMapper->Release(); // 允许 LocalMapping 恢复正常状态
    }

    mnLastKeyFrameId = mCurrentFrame.mnId;
}

void Tracker::Reset()
{
    mState = NOT_INITIALIZED;
    mVelocity.setIdentity();
    mpReferenceKF = nullptr;
    // 清空历史轨迹记录
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
}

void Tracker::UpdateLastFrame()
{
    // 如果没有参考关键帧，直接退出
    if (!mpReferenceKF || mpReferenceKF->mbBad)
        return;

    // 获取上一帧相对于参考关键帧的相对位姿: T_lr = T_lw * T_rw^-1
    // 由于在 Track() 结束时保存了 mlRelativeFramePoses 和 mlpReferences，此处更新 mLastFrame：
    if (!mlRelativeFramePoses.empty() && mlpReferences.back() == mpReferenceKF)
    {
        Eigen::Matrix4f Tlr = mlRelativeFramePoses.back();
        // 依据参考关键帧最新可能已被 LocalMapping 优化过的位姿，重新计算上一帧的绝对位姿
        mLastFrame.SetPose(Tlr * mpReferenceKF->GetPose());
    }
}

void Tracker::ResetVelocity()
{
    // 将恒速模型速度矩阵重置为单位矩阵，下一帧将退化为基于上一帧重投影搜索
    mVelocity.setIdentity();
}

/**
 * @brief 调度更新局部地图
 */
void Tracker::UpdateLocalMap()
{
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 搜集局部关键帧 (当前帧观测帧 + 一级共视邻居 + 二级共视邻居)
 */
/**
 * @brief 搜集局部关键帧 (当前帧观测帧 + 一级共视前 K 个邻居 + 二级共视邻居)
 */
void Tracker::UpdateLocalKeyFrames()
{
    mvpLocalKeyFrames.clear();

    // 1. 统计当前帧所有已匹配地图点的所有有效观测关键帧
    std::map<KeyFrame*, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && !pMP->isBad())
        {
            const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (auto mit = observations.begin(); mit != observations.end(); ++mit)
            {
                if (!mit->first->mbBad)
                    keyframeCounter[mit->first]++;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int maxObs = 0;
    KeyFrame *pKFmax = nullptr;
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // 2. 加入当前帧直接观测到的关键帧，并选出共视点最多的作为参考关键帧
    for (auto &mit : keyframeCounter)
    {
        KeyFrame *pKF = mit.first;
        if (!pKF->mbBad)
        {
            mvpLocalKeyFrames.push_back(pKF);
            if (mit.second > maxObs)
            {
                maxObs = mit.second;
                pKFmax = pKF;
            }
        }
    }

    if (pKFmax)
        mpReferenceKF = pKFmax;

    // 3. 扩充一级共视邻居 (ORB-SLAM2 标准：前 10 个最佳共视邻居)、二级共视邻居 (前 5 个)、生成树子/父节点
    std::vector<KeyFrame*> vpLocalKFWithNeighbors = mvpLocalKeyFrames;
    for (KeyFrame *pKF : mvpLocalKeyFrames)
    {
        if (pKF->mbBad)
            continue;

        // (a) 一级最佳共视前 10 个邻居
        const std::vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for (KeyFrame *pN : vNeighs)
        {
            if (pN && !pN->mbBad)
            {
                vpLocalKFWithNeighbors.push_back(pN);
                // (b) 二级最佳共视前 5 个邻居
                const std::vector<KeyFrame *> vSecondNeighs = pN->GetBestCovisibilityKeyFrames(5);
                for (KeyFrame *p2N : vSecondNeighs)
                {
                    if (p2N && !p2N->mbBad)
                        vpLocalKFWithNeighbors.push_back(p2N);
                }
            }
        }

        // (c) 生成树子节点与父节点
        const std::set<KeyFrame *> spChildren = pKF->GetChilds();
        for (KeyFrame *pChild : spChildren)
        {
            if (pChild && !pChild->mbBad)
                vpLocalKFWithNeighbors.push_back(pChild);
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent && !pParent->mbBad)
            vpLocalKFWithNeighbors.push_back(pParent);
    }

    // 4. 排序与去重
    std::sort(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end());
    vpLocalKFWithNeighbors.erase(
        std::unique(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end()),
        vpLocalKFWithNeighbors.end());

    mvpLocalKeyFrames = vpLocalKFWithNeighbors;
}

/**
 * @brief 搜集局部地图点 (从局部关键帧中提取并去重)
 */
void Tracker::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (KeyFrame *pKF : mvpLocalKeyFrames)
    {
        if (!pKF || pKF->mbBad)
            continue;

        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint *pMP : vpMPs)
        {
            if (!pMP || pMP->isBad())
                continue;

            // 避免重复收集
            if (std::find(mvpLocalMapPoints.begin(), mvpLocalMapPoints.end(), pMP) == mvpLocalMapPoints.end())
            {
                mvpLocalMapPoints.push_back(pMP);
            }
        }
    }
}

/**
 * @brief 投影搜索局部地图点
 */
void Tracker::SearchLocalPoints()
{
    if (mvpLocalMapPoints.empty())
        return;

    // 1. 过滤掉当前帧中已经成功匹配且为有效内点（Inlier）的地图点
    std::vector<MapPoint*> vpCandidateMPs;
    vpCandidateMPs.reserve(mvpLocalMapPoints.size());

    for (MapPoint *pMP : mvpLocalMapPoints)
    {
        if (!pMP || pMP->isBad())
            continue;

        bool bAlreadyTracked = false;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            // 核心修复：只有当指针匹配且不是 Outlier 时，才视作已跟踪成功
            if (mCurrentFrame.mvpMapPoints[i] == pMP && !mCurrentFrame.mvbOutlier[i])
            {
                bAlreadyTracked = true;
                break;
            }
        }

        if (!bAlreadyTracked)
        {
            vpCandidateMPs.push_back(pMP);
        }
    }

    if (vpCandidateMPs.empty())
        return;

    // 2. 局部地图匹配设置 nnratio = 0.8f，开启旋转一致性校验，搜索半径 th = 5.0f
    ORBmatcher matcher(0.8f, true);
    matcher.SearchByProjection(mCurrentFrame, vpCandidateMPs, 5.0f);
}