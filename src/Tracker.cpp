#include "Tracker.h"
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
Tracker::Tracker(System *pSys, ORBVocabulary *pVoc, KeyFrameDatabase *pKFDB,
                 std::shared_ptr<Map> pMap, System::eSensor sensor, const std::string &strSettingPath)
    : mpSystem(pSys), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpMap(pMap),
      mState(NO_IMAGES_YET), mVelocity(Eigen::Matrix4f::Identity()),
      mpReferenceKF(nullptr), mpLocalMapper(nullptr), mnLastRelocFrameId(0)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    if (!fSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << strSettingPath << std::endl;
        exit(-1);
    }

    float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;
    mK = cv::Mat::eye(3, 3, CV_32F);
    mDistCoef = cv::Mat::zeros(4, 1, CV_32F);

    std::string datasetType = "KITTI / Common";

    // 1. 读取并识别模式
    if (!fSettings["LEFT.P"].empty() && !fSettings["RIGHT.P"].empty())
    {
        datasetType = "EuRoC (Stereo Rectified)";
        cv::Mat P_l, P_r;
        fSettings["LEFT.P"] >> P_l;
        fSettings["RIGHT.P"] >> P_r;

        P_l.convertTo(P_l, CV_32F);
        P_r.convertTo(P_r, CV_32F);

        fx = P_l.at<float>(0, 0);
        fy = P_l.at<float>(1, 1);
        cx = P_l.at<float>(0, 2);
        cy = P_l.at<float>(1, 2);

        mbf = -P_r.at<float>(0, 3);
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
    }
    else
    {
        fx = fSettings["Camera.fx"];
        fy = fSettings["Camera.fy"];
        cx = fSettings["Camera.cx"];
        cy = fSettings["Camera.cy"];
        mbf = fSettings["Camera.bf"];

        mDistCoef.at<float>(0) = fSettings["Camera.k1"];
        mDistCoef.at<float>(1) = fSettings["Camera.k2"];
        mDistCoef.at<float>(2) = fSettings["Camera.p1"];
        mDistCoef.at<float>(3) = fSettings["Camera.p2"];
    }

    mK.at<float>(0, 0) = fx;
    mK.at<float>(1, 1) = fy;
    mK.at<float>(0, 2) = cx;
    mK.at<float>(1, 2) = cy;

    mFps = fSettings["Camera.fps"];
    if (mFps <= 0.0f)
        mFps = 30.0f;

    mThDepth = fSettings["ThDepth"];
    if (mThDepth <= 0.0f)
        mThDepth = 35.0f;

    // 2. ORB 提取器参数
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);
    mpORBextractorRight = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    // 3. 格式化输出配置参数
    const float b = (fx > 0.0f) ? (mbf / fx) : 0.0f;
    std::cout << "\n================ Loaded Configuration ================" << std::endl;
    std::cout << " Dataset Type     : " << datasetType << std::endl;
    std::cout << " Camera Matrix (K): [ fx: " << fx << ", fy: " << fy
              << ", cx: " << cx << ", cy: " << cy << " ]" << std::endl;
    std::cout << " Distortion Coeff : [ k1: " << mDistCoef.at<float>(0)
              << ", k2: " << mDistCoef.at<float>(1)
              << ", p1: " << mDistCoef.at<float>(2)
              << ", p2: " << mDistCoef.at<float>(3) << " ]" << std::endl;
    std::cout << " Baseline (b)     : " << b << " m (bf = " << mbf << ")" << std::endl;
    std::cout << " FPS / Depth Th   : " << mFps << " / " << mThDepth << " (maxDepth = " << mThDepth * b << " m)" << std::endl;
    std::cout << " ORB Features     : " << nFeatures << " points, " << nLevels << " levels, scale: " << fScaleFactor << std::endl;
    std::cout << " FAST Thresholds  : ini = " << fIniThFAST << ", min = " << fMinThFAST << std::endl;
    std::cout << "======================================================\n"
              << std::endl;
}

Tracker::~Tracker() {}

Eigen::Matrix4f Tracker::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft.clone();

    // 直接使用已解析好的成员变量
    mCurrentFrame = Frame(imRectLeft.clone(), imRectRight.clone(), timestamp,
                          mpORBextractorLeft.get(), mpORBextractorRight.get(),
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();
    return mCurrentFrame.mTcw;
}

void Tracker::Track()
{
    // track包含两部分：估计运动、跟踪局部地图
    // 如果图像复位过、或者第一次运行，则为 NO_IMAGES_YET 状态
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // Get Map Mutex -> Map cannot be changed
    // 地图更新时加锁，保证地图在当前帧处理期间不会被后端线程修改
    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

    // Step 1：初始化
    if (mState == NOT_INITIALIZED)
    {
        StereoInitialization();
        mpFrameDrawer->Update(this);

        if (mState != OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK = false;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 正常 SLAM 模式（定位 + 建图更新）
        if (mState == OK)
        {
            // Local Mapping might have changed some MapPoints tracked in last frame
            // Step 2.1 检查并更新上一帧被替换的 MapPoints
            CheckReplacedInLastFrame();

            // Step 2.2 运动模型为空（刚初始化完成或跟丢刚恢复）或紧跟在重定位帧之后，跟踪参考关键帧；否则恒速模型跟踪
            if (mVelocity.isIdentity(1e-4) || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                bOK = TrackWithMotionModel();
                if (!bOK)
                    bOK = TrackReferenceKeyFrame();
            }
        }
        else
        {
            // 如果跟丢了，进行重定位
            bOK = Relocalize();
        }

        // 将最新的关键帧作为当前帧的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // Step 3：在跟踪得到当前帧初始姿态后，对 local map 进行跟踪得到更多的匹配，并优化当前位姿
        if (bOK)
            bOK = TrackLocalMap();

        if (bOK)
            mState = OK;
        else
            mState = LOST;

        // Step 4：更新显示线程中的图像、特征点、地图点等信息
        if (mpFrameDrawer)
            mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // Update motion model
            // Step 5：跟踪成功，更新恒速运动模型速度 mVelocity = Tcl = Tcw * Twl
            if (!mLastFrame.mTcw.isZero())
            {
                mVelocity = mCurrentFrame.mTcw * mLastFrame.mTcw.inverse();
            }
            else
            {
                mVelocity.setIdentity();
            }

            if (mpViewer)
                mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);

            // Step 6：清除观测不到的地图点（未被任何关键帧观测到）
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (pMP->GetObservations().empty())
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = nullptr;
                    }
                }
            }

            // Delete temporal MapPoints
            // Step 7：清除恒速模型跟踪中 UpdateLastFrame 为上一帧临时添加的地图点
            for (auto lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; ++lit)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            // Step 8：检测并插入关键帧
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            // Step 9：删除那些在 BA 优化中被标记为 Outlier 的地图点（不传递到下一帧）
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
            }
        }

        // Reset if the camera get lost soon after initialization
        // Step 10：如果初始化后不久就跟踪失败，并且重定位失败，重新 Reset 系统
        if (mState == LOST)
        {
            if (mpMap && mpMap->GetKeyFramesInMap() <= 5)
            {
                std::cout << "Track lost soon after initialisation, reseting..." << std::endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存上一帧数据，当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // Step 11：记录位姿信息，用于后续整条轨迹的还原与输出
    if (!mCurrentFrame.mTcw.isZero() && !mCurrentFrame.mTcw.hasNaN())
    {
        Eigen::Matrix4f Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        // 如果跟踪失败，相对位姿复用上一帧记录
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

bool Tracker::StereoInitialization()
{
    // 1. 特征点数量门槛（ORB-SLAM2 官方标准为 500 点）
    if (mCurrentFrame.N <= 500)
        return false;

    // 2. 设定初始位姿为世界坐标系原点 (单位矩阵)
    mCurrentFrame.SetPose(Eigen::Matrix4f::Identity());

    // 3. 创建初始关键帧并插入全局地图与词袋数据库
    KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKFini);

    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKFini);

    // 4. 为每个具有正深度且在测量范围内的特征点创建 MapPoint 并关联
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0.0f) // 官方只要 z > 0 即生成地图点，近远点判定由后续模块处理
        {
            Eigen::Vector3f x3D = mCurrentFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap.get());

            // 建立特征点与关键帧的双向观测
            pNewMP->AddObservation(pKFini, i);
            pKFini->AddMapPoint(pNewMP, i);

            // 计算代表性描述子、平均观测方向与深度范围
            pNewMP->ComputeDistinctiveDescriptor();
            pNewMP->UpdateNormalAndDepth();

            // 注册入全局地图并关联至当前普通帧
            mpMap->AddMapPoint(pNewMP);
            mCurrentFrame.mvpMapPoints[i] = pNewMP;
        }
    }

    std::cout << "[Initialization] New stereo map created with "
              << mpMap->GetMapPointsInMap() << " points" << std::endl;

    // 5. 将初始关键帧推送给局部建图线程
    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKFini);
    }

    // 6. 更新追踪上下文（对齐 ORB-SLAM2 官方核心状态）
    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    // 7. 更新局部关键帧与局部地图点
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();

    // 8. 同步局部参考点给 Map 与 Viewer 可视化
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    if (mpViewer)
    {
        mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);
    }

    // 9. 显式置位状态
    mState = OK;
    return true;
}

bool Tracker::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9f, true);
    UpdateLastFrame();
    // 1. 基于恒速模型预测位姿初值
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

    // 2. 双目基础搜索窗口为 7
    int th = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, false);

    // 3. 若匹配过少，扩大 2 倍窗口重新搜索
    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, false);
    }

    if (nmatches < 20)
        return false;

    // 4. 执行位姿优化 (Motion-only BA，内部已含 4 轮核函数外点剔除)
    int num_inliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 5. 剔除被判定为 Outlier 的地图点
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCurrentFrame.mvbOutlier[i] = false;
        }
    }

    // 6. 原版判定标准：内点数 >= 10 即可交付给 TrackLocalMap
    return num_inliers >= 10;
}

bool Tracker::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // Step 1：将当前帧的描述子转化为 BoW 向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7f, true);
    std::vector<MapPoint *> vpMapPointMatches;

    // Step 2：通过词袋特征匹配当前帧与参考关键帧
    int nmatches = matcher.SearchByBoW(
        mpReferenceKF,
        mCurrentFrame,
        vpMapPointMatches);

    // 匹配数目小于 15 直接判定失败
    if (nmatches < 15)
        return false;

    // Step 3：绑定匹配关系，并将上一帧位姿作为当前帧初值加速收敛
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // Step 4：通过重投影误差优化当前帧位姿 (Pose-Only BA)
    MotionOnlyBA::Optimize(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除优化后的外点（MapPoints），并统计匹配成功的内点数
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mnVisible++; // 保持与官方一致，外点也累计被视野看到的次数
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->GetObservations().size() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    mnMatchesInliers = nmatchesMap;

    // Step 6：成功匹配的内点大于等于 10 个即判定跟踪成功
    return nmatchesMap >= 10;
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

        if (!bPnPSuccess || inliersPnP.size() < 15)
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

        if (nInliers >= 20)
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
            mnLastRelocFrameId = mCurrentFrame.mnId;
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
    const int mMaxFrames = static_cast<int>(mFps > 0.0f ? mFps : 20.0f);
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
    bool bLocalMappingIdle = mpLocalMapper ? mpLocalMapper->AcceptKeyFrames() : true;

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
                mpLocalMapper->RequestStopBA();
            }

            return true;
        }
    }

    return false;
}

void Tracker::CreateNewKeyFrame()
{
    // Step 1: 锁住 LocalMapping，防止其在插入关键帧的过程中响应外部（如闭环）停止
    if (mpLocalMapper && !mpLocalMapper->SetNotStop(true))
        return;

    // Step 2: 将当前帧构造成关键帧，并插入地图
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKF);

    // Step 3: 更新参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // Step 4: 双目专属逻辑 —— 为有有效深度的近点创建新的地图点
    // 4.1 收集所有具有正深度的特征点
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
        // 4.2 按照深度从小到大排序，优先处理高精度的近点
        std::sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

            // 如果当前特征点未绑定地图点，或者绑定的地图点观测数少于1（没有关键帧支持）
            if (!pMP)
            {
                bCreateNew = true;
            }
            else if (pMP->GetObservations().empty())
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }

            if (bCreateNew)
            {
                // 反投影到世界坐标系
                Eigen::Vector3f x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap.get());

                // 建立特征点与关键帧的双向观测关联
                pNewMP->AddObservation(pKF, i);
                pKF->AddMapPoint(pNewMP, i);

                // 计算代表性描述子与法向量
                pNewMP->ComputeDistinctiveDescriptor();
                pNewMP->UpdateNormalAndDepth();

                // 加入全局地图，并关联到当前帧
                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                nPoints++;
            }
            else
            {
                // 该地图点已经被有效跟踪（无需重复生成新点）
                // 若不是外点且尚未观测，为已有地图点追加当前关键帧的观测
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    if (!pMP->IsInKeyFrame(pKF))
                    {
                        pMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pMP, i);
                    }
                }
                nPoints++;
            }

            // 4.3 停止条件：
            // 深度已经超过近点阈值 mThDepth，且已经生成的/处理的近点数超过 100 个
            if (vDepthIdx[j].first > mCurrentFrame.mThDepth && nPoints > 100)
                break;
        }
    }

    // Step 5: 将新关键帧注册入词袋数据库（用于重定位/闭环检测的倒排索引）
    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKF);

    // Step 6: 送入 LocalMapping 线程的处理队列
    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKF);
        // 解除停止阻止，恢复 LocalMapping 的正常控制状态
        mpLocalMapper->SetNotStop(false);
    }

    // Step 7: 更新关键帧记录 ID
    mnLastKeyFrameId = mCurrentFrame.mnId;
}

void Tracker::Reset()
{
    mState = NOT_INITIALIZED;
    mVelocity.setIdentity();
    mpReferenceKF = nullptr;
    mnLastRelocFrameId = 0;
    mnLastKeyFrameId = 0;
    mnMatchesInliers = 0;

    // 清空局部地图缓存（避免持有野指针）
    mvpLocalKeyFrames.clear();
    mvpLocalMapPoints.clear();

    // 清空当前帧与上一帧的关联点
    mCurrentFrame.mvpMapPoints.clear();
    mLastFrame.mvpMapPoints.clear();
    for (auto pMP : mlpTemporalPoints)
        delete pMP;
    mlpTemporalPoints.clear();
    // 清空历史轨迹记录
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
}

void Tracker::UpdateLastFrame()
{
    // 0. 清理残留临时点，防止内存泄漏或野指针悬挂
    for (MapPoint *pMP : mlpTemporalPoints)
        delete pMP;
    mlpTemporalPoints.clear();

    // 1. 检查参考关键帧与位姿记录
    if (mlRelativeFramePoses.empty() || mlpReferences.empty())
        return;

    KeyFrame *pRef = mlpReferences.back();
    if (!pRef)
        return;

    Eigen::Matrix4f Tlr = mlRelativeFramePoses.back();
    KeyFrame *pOrigRef = pRef;

    // 沿生成树回溯已剔除的关键帧
    Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();
    while (pRef->mbBad)
    {
        Trw = Trw * pRef->GetRelativePoseToParent();
        pRef = pRef->GetParent();
        if (!pRef)
            return;
    }

    // 更新上一帧绝对位姿
    mLastFrame.SetPose(Tlr * Trw * pRef->GetPose());

    // 若上一帧本身是关键帧，无需生成临时点
    if (mnLastKeyFrameId == mLastFrame.mnId)
        return;

    // 2. 双目专属逻辑：为上一帧有深度但无地图点的特征点生成临时 VO 点
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);

    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0.0f)
        {
            vDepthIdx.push_back(std::make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    std::sort(vDepthIdx.begin(), vDepthIdx.end());

    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int i = vDepthIdx[j].second;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        // 只有未关联地图点，或者关联的点没有任何关键帧观测时才需要新建
        if (!pMP || pMP->GetObservations().empty())
        {
            Eigen::Vector3f x3D = mLastFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, pOrigRef, mpMap.get());

            mLastFrame.mvpMapPoints[i] = pNewMP;
            mLastFrame.mvbOutlier[i] = false;
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mLastFrame.mThDepth && nPoints > 100)
            break;
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
    std::map<KeyFrame *, int> keyframeCounter;
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
    std::vector<KeyFrame *> vpLocalKFWithNeighbors = mvpLocalKeyFrames;
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
    std::vector<MapPoint *> vpCandidateMPs;
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

void Tracker::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            while (pRep)
            {
                pMP = pRep;
                pRep = pMP->GetReplaced();
            }
            if (pMP->isBad())
            {
                mLastFrame.mvpMapPoints[i] = nullptr;
            }
            else
            {
                mLastFrame.mvpMapPoints[i] = pMP;
            }
        }
    }
}