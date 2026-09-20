#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "Tracker.h"
#include "Viewer.h"
#include "FrameDrawer.h"
#include "LocalMapping.h"
#include "KeyFrameDatabase.h"
#include <fstream>
#include <algorithm>
#include <iomanip>

System::System(const std::string &strConfigFile, const std::string &strVocFile, const eSensor sensor, const bool bUseViewer)
    : mSensor(sensor), mpViewerThread(nullptr)
{
    std::cout << "Starting ORB-SLAM2 Stereo System..." << std::endl;

    // 1. 验证配置文件能够打开
    cv::FileStorage fsSettings(strConfigFile, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "Failed to open settings file at: " << strConfigFile << std::endl;
        exit(-1);
    }
    fsSettings.release();

    // 2. 加载词典
    mpVocabulary = std::make_shared<ORBVocabulary>();
    std::cout << "Loading Vocabulary file from: " << strVocFile << " ..." << std::endl;
    try
    {
        mpVocabulary->load(strVocFile);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[System] Exception caught while loading vocabulary: " << e.what() << std::endl;
        exit(-1);
    }

    if (mpVocabulary->empty())
    {
        std::cerr << "Failed to load vocabulary at: " << strVocFile << std::endl;
        exit(-1);
    }
    std::cout << "Vocabulary loaded successfully." << std::endl;

    // 3. 初始化全局地图与各个模块
    mpMap = std::make_shared<Map>();
    mpFrameDrawer = std::make_shared<FrameDrawer>(mpMap.get());
    mpKeyFrameDatabase = new KeyFrameDatabase(mpVocabulary.get());

    // 传递 strConfigFile 初始化 Tracker
    mpTracker = std::make_shared<Tracker>(this, mpVocabulary.get(), mpKeyFrameDatabase, mpMap, sensor, strConfigFile);

    mpLocalMapper = std::make_shared<LocalMapping>(this, mpMap);
    mpLoopCloser = std::make_shared<LoopClosing>(mpMap.get(), mpKeyFrameDatabase, mpVocabulary.get(), true);

    mpTracker->SetFrameDrawer(mpFrameDrawer);
    mpTracker->SetLocalMapper(mpLocalMapper.get());
    mpTracker->SetLoopClosing(mpLoopCloser.get());

    mpLocalMapper->SetTracker(mpTracker.get());
    mpLoopCloser->SetTracker(mpTracker.get());
    mpLoopCloser->SetLocalMapper(mpLocalMapper.get());

    if (bUseViewer)
    {
        // 传递 strConfigFile 初始化 Viewer
        mpViewer = std::make_shared<Viewer>(this, mpMap, mpFrameDrawer, strConfigFile);
        mpViewerThread = new std::thread(&Viewer::Run, mpViewer.get());
        mpTracker->SetViewer(mpViewer);
    }
}

System::~System()
{
    Shutdown();
    if (mpKeyFrameDatabase)
    {
        delete mpKeyFrameDatabase;
        mpKeyFrameDatabase = nullptr;
    }
}

cv::Mat System::DrawFrame()
{
    if (mpFrameDrawer)
        return mpFrameDrawer->DrawFrame();
    return cv::Mat();
}

Eigen::Matrix4f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if (mSensor != STEREO)
    {
        std::cerr << "Error: System initialized for non-stereo tracking!" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    // 彩色/灰度检查
    cv::Mat imLeftGray, imRightGray;
    if (imLeft.channels() == 3)
    {
        cv::cvtColor(imLeft, imLeftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imRight, imRightGray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        imLeftGray = imLeft.clone();
        imRightGray = imRight.clone();
    }

    // 调用 Tracker 执行跟踪主流程
    Eigen::Matrix4f Tcw = mpTracker->GrabImageStereo(imLeftGray, imRightGray, timestamp);

    return Tcw;
}

void System::Shutdown()
{
    if (mpLocalMapper)
    {
        mpLocalMapper->RequestStop();
    }
    if (mpLoopCloser)
    {
        mpLoopCloser->RequestStop();
    }
    if (mpViewerThread)
    {
        if (mpViewer)
            mpViewer->RequestFinish();
        mpViewerThread->join();
        delete mpViewerThread;
        mpViewerThread = nullptr;
    }
}

void System::Reset()
{
    std::unique_lock<std::mutex> lock(mMutexMode);
    if (mpTracker)
        mpTracker->Reset();
    if (mpMap)
        mpMap->Clear();
}

void System::SaveTrajectoryKITTI(const std::string &filename)
{
    std::cout << "\nSaving camera trajectory to " << filename << " ..." << std::endl;

    if (mSensor == MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << std::endl;
        return;
    }

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "ERROR: Failed to open file at: " << filename << std::endl;
        return;
    }
    f << std::fixed;

    auto lRit = mpTracker->mlpReferences.begin();
    auto lT   = mpTracker->mlFrameTimes.begin();

    for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend; ++lit, ++lRit, ++lT)
    {
        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        // 1. 初始化级联变换矩阵为单位阵
        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 2. 【核心修复】：严格遵循 ORB-SLAM2 原版右乘级联
        // 每向上回溯一层父节点，变换依次右乘该坏关键帧的 mTcp (即 T_child_parent)
        while (pKF->mbBad)
        {
            Trw = Trw * pKF->GetRelativePoseToParent();
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        // 3. 乘上最终有效父关键帧的最新位姿: Trw = Trw * T_parent_w
        Trw = Trw * pKF->GetPose();

        // 4. 计算当前帧世界位姿: Tcw = Tcr * Trw
        Eigen::Matrix4f Tcw = (*lit) * Trw;

        // 5. 还原相机在世界系下的绝对位姿 Twc = Tcw^-1
        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        // 6. 写入 KITTI 格式
        f << std::setprecision(9)
          << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " "
          << Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " "
          << Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << "\n";
    }

    f.close();
    std::cout << "Trajectory successfully saved to " << filename << std::endl;
}

void System::SaveTrajectoryTUM(const std::string &filename)
{
    std::cout << "\nSaving camera trajectory to " << filename << " ..." << std::endl;

    if (mSensor == MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << std::endl;
        return;
    }

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        return;
    }
    f << std::fixed;

    auto lit  = mpTracker->mlRelativeFramePoses.begin();
    auto lRit = mpTracker->mlpReferences.begin();
    auto lT   = mpTracker->mlFrameTimes.begin();
    auto lbL  = mpTracker->mlbLost.begin();

    for (; lit != mpTracker->mlRelativeFramePoses.end(); ++lit, ++lRit, ++lT, ++lbL)
    {
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 严格遵循 ORB-SLAM2 原版右乘级联
        while (pKF->mbBad)
        {
            Trw = Trw * pKF->GetRelativePoseToParent();
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        Trw = Trw * pKF->GetPose();
        Eigen::Matrix4f Tcw = (*lit) * Trw;

        if (Tcw.hasNaN())
            continue;

        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        Eigen::Quaternionf q(Rwc);
        q.normalize();

        // TUM 格式: timestamp tx ty tz qx qy qz qw
        f << std::setprecision(6) << *lT << " "
          << std::setprecision(9) 
          << twc.x() << " " << twc.y() << " " << twc.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    f.close();
    std::cout << "Trajectory successfully saved to " << filename << std::endl;
}