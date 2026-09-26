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

    mCurrentFrame = Frame(imRectLeft.clone(), imRectRight.clone(), timestamp,
                          mpORBextractorLeft.get(), mpORBextractorRight.get(),
                          mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();
    return mCurrentFrame.mTcw;
}

void Tracker::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

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
        bool bOK = false;

        // 正常 SLAM 模式（定位 + 建图更新）
        if (mState == OK)
        {
            CheckReplacedInLastFrame();

            if (mVelocity.isIdentity(1e-4) || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                bOK = TrackWithMotionModel();
                if (!bOK)
                {
                    bOK = TrackReferenceKeyFrame();
                }
            }
        }
        else
        {
            bOK = Relocalize();
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // Step 3：对 local map 进行跟踪得到更多的匹配，并优化当前位姿
        if (bOK)
        {
            bOK = TrackLocalMap();
        }

        if (bOK)
        {
            mState = OK;
        }
        else
        {
            mState = LOST;
        }

        // Step 4：更新显示线程
        if (mpFrameDrawer)
            mpFrameDrawer->Update(this);

        if (bOK)
        {
            // Step 5：更新恒速运动模型速度
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

            // Step 6：清除未被任何关键帧观测到的地图点
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

            // Step 7：清除上一帧临时添加的地图点
            for (auto lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; ++lit)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Step 8：检测并插入关键帧
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            // Step 9：删除被标记为 Outlier 的地图点
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
            }
        }

        // Step 10：如果初始化后不久就跟踪失败，重置系统
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

        mLastFrame = Frame(mCurrentFrame);
    }

    // Step 11：记录位姿信息
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
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

bool Tracker::StereoInitialization()
{
    if (mCurrentFrame.N <= 500)
        return false;

    mCurrentFrame.SetPose(Eigen::Matrix4f::Identity());

    KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKFini);

    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKFini);

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0.0f)
        {
            Eigen::Vector3f x3D = mCurrentFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap.get());

            pNewMP->AddObservation(pKFini, i);
            pKFini->AddMapPoint(pNewMP, i);

            pNewMP->ComputeDistinctiveDescriptor();
            pNewMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pNewMP);
            mCurrentFrame.mvpMapPoints[i] = pNewMP;
        }
    }

    std::cout << "[Initialization] New stereo map created with "
              << mpMap->GetMapPointsInMap() << " points" << std::endl;

    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKFini);
    }

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    if (mpViewer)
    {
        mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);
    }

    mState = OK;
    return true;
}

bool Tracker::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9f, true);
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

    int th = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, false);

    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, false);
    }

    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 3 * th, false);
    }

    if (nmatches < 15)
    {
        return false;
    }

    int num_inliers = Optimizer::PoseOptimization(&mCurrentFrame);

    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCurrentFrame.mvbOutlier[i] = false;
        }
    }

    return (num_inliers >= 10);
}

bool Tracker::TrackReferenceKeyFrame()
{
    mCurrentFrame.ComputeBoW();

    ORBmatcher matcher(0.7f, true);
    std::vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

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
                pMP->mnVisible++;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->GetObservations().size() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    mnMatchesInliers = nmatchesMap;
    return (nmatchesMap >= 10);
}

bool Tracker::Relocalize()
{
    if (mCurrentFrame.mBowVec.empty())
        mCurrentFrame.ComputeBoW();

    if (!mpKeyFrameDB)
        return false;

    std::vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (vpCandidateKFs.empty())
        return false;

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
        cv::setRNGSeed(0);
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

        int nInliers = Optimizer::PoseOptimization(&mCurrentFrame);

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
            return true;
        }
    }

    return false;
}

bool Tracker::TrackLocalMap()
{
    UpdateLocalMap();
    SearchLocalPoints();
    Optimizer::PoseOptimization(&mCurrentFrame);

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
                if (mCurrentFrame.mvpMapPoints[i]->GetObservations().size() > 0)
                {
                    mnMatchesInliers++;
                }
            }
        }
    }

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    if (mCurrentFrame.mnId < mnLastRelocFrameId + mFps)
    {
        return (mnMatchesInliers > 10);
    }

    if (mnMatchesInliers < 30)
    {
        return (mnMatchesInliers >= 15);
    }

    return true;
}

bool Tracker::NeedNewKeyFrame()
{
    if (mpLocalMapper && (mpLocalMapper->isStopped() || mpLocalMapper->GetStopRequired()))
        return false;

    if (mvpLocalKeyFrames.empty() || mnMatchesInliers < 20)
    {
        return true;
    }

    const int nKFs = mpMap ? mpMap->GetKeyFramesInMap() : 0;

    const int mMaxFrames = static_cast<int>(mFps > 0.0f ? mFps : 20.0f);
    const int mMinFrames = 0;

    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;

    int nRefMatches = 0;
    if (mpReferenceKF && !mpReferenceKF->mbBad)
    {
        nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    }
    if (nRefMatches <= 0)
        nRefMatches = 1;

    bool bLocalMappingIdle = mpLocalMapper ? mpLocalMapper->AcceptKeyFrames() : true;

    int nNonTrackedClose = 0;
    int nTrackedClose = 0;

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvDepth[i] > 0.0f && mCurrentFrame.mvDepth[i] < mCurrentFrame.mThDepth)
        {
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        }
    }

    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    const int nFramesPassed = mCurrentFrame.mnId - mnLastKeyFrameId;

    const bool c1a = nFramesPassed >= mMaxFrames;
    const bool c1b = (nFramesPassed >= mMinFrames && bLocalMappingIdle);
    const bool c1c = (mnMatchesInliers < nRefMatches * 0.25f) || bNeedToInsertClose;
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio) || bNeedToInsertClose) && (mnMatchesInliers > 15);

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
    if (mpLocalMapper && !mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKF);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

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
            else if (pMP->GetObservations().empty())
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }

            if (bCreateNew)
            {
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

            if (vDepthIdx[j].first > mCurrentFrame.mThDepth && nPoints > 100)
                break;
        }
    }

    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKF);

    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKF);
        mpLocalMapper->SetNotStop(false);
    }

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

    mvpLocalKeyFrames.clear();
    mvpLocalMapPoints.clear();

    mCurrentFrame.mvpMapPoints.clear();
    mLastFrame.mvpMapPoints.clear();
    for (auto pMP : mlpTemporalPoints)
        delete pMP;
    mlpTemporalPoints.clear();

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
}

void Tracker::UpdateLastFrame()
{
    for (MapPoint *pMP : mlpTemporalPoints)
        delete pMP;
    mlpTemporalPoints.clear();

    if (mlRelativeFramePoses.empty() || mlpReferences.empty())
        return;

    KeyFrame *pRef = mlpReferences.back();
    if (!pRef)
        return;

    Eigen::Matrix4f Tlr = mlRelativeFramePoses.back();
    KeyFrame *pOrigRef = pRef;

    Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();
    while (pRef->mbBad)
    {
        Trw = Trw * pRef->GetRelativePoseToParent();
        pRef = pRef->GetParent();
        if (!pRef)
            return;
    }

    mLastFrame.SetPose(Tlr * Trw * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId)
        return;

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
    mVelocity.setIdentity();
}

void Tracker::UpdateLocalMap()
{
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracker::UpdateLocalKeyFrames()
{
    mvpLocalKeyFrames.clear();

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
    {
        if (mpReferenceKF && !mpReferenceKF->mbBad)
        {
            mvpLocalKeyFrames.push_back(mpReferenceKF);
            const std::vector<KeyFrame *> vNeighs = mpReferenceKF->GetBestCovisibilityKeyFrames(10);
            for (KeyFrame *pN : vNeighs)
            {
                if (pN && !pN->mbBad)
                    mvpLocalKeyFrames.push_back(pN);
            }
        }
        return;
    }

    int maxObs = 0;
    KeyFrame *pKFmax = nullptr;
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

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

    std::vector<KeyFrame *> vpLocalKFWithNeighbors = mvpLocalKeyFrames;
    for (KeyFrame *pKF : mvpLocalKeyFrames)
    {
        if (pKF->mbBad)
            continue;

        const std::vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for (KeyFrame *pN : vNeighs)
        {
            if (pN && !pN->mbBad)
            {
                vpLocalKFWithNeighbors.push_back(pN);
                const std::vector<KeyFrame *> vSecondNeighs = pN->GetBestCovisibilityKeyFrames(5);
                for (KeyFrame *p2N : vSecondNeighs)
                {
                    if (p2N && !p2N->mbBad)
                        vpLocalKFWithNeighbors.push_back(p2N);
                }
            }
        }

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

    std::sort(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end());
    vpLocalKFWithNeighbors.erase(
        std::unique(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end()),
        vpLocalKFWithNeighbors.end());

    mvpLocalKeyFrames = vpLocalKFWithNeighbors;
}

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

            if (std::find(mvpLocalMapPoints.begin(), mvpLocalMapPoints.end(), pMP) == mvpLocalMapPoints.end())
            {
                mvpLocalMapPoints.push_back(pMP);
            }
        }
    }
}

void Tracker::SearchLocalPoints()
{
    if (mvpLocalMapPoints.empty())
        return;

    std::vector<MapPoint *> vpCandidateMPs;
    vpCandidateMPs.reserve(mvpLocalMapPoints.size());

    for (MapPoint *pMP : mvpLocalMapPoints)
    {
        if (!pMP || pMP->isBad())
            continue;

        bool bAlreadyTracked = false;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
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

    float th = 5.0f;
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
    {
        th = 10.0f;
    }
    else if (mnMatchesInliers < 25)
    {
        th = 8.0f;
    }

    ORBmatcher matcher(0.8f, false);
    matcher.SearchByProjection(mCurrentFrame, vpCandidateMPs, th);
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