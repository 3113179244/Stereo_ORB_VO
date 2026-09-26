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
#include <unistd.h>
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
    // 1. 先向所有子线程发送 Finish 请求
    if (mpLocalMapper)
    {
        mpLocalMapper->RequestFinish();
    }
    if (mpLoopCloser)
    {
        mpLoopCloser->RequestFinish();
    }
    if (mpViewer)
    {
        mpViewer->RequestFinish();
    }

    // 2. 等待各子线程真正退出（LocalMapping 析构内部自带 join，或者在此处等待 finished）
    if (mpLocalMapper)
    {
        while (!mpLocalMapper->isFinished())
            usleep(2000);
    }
    if (mpLoopCloser)
    {
        while (!mpLoopCloser->isFinished())
            usleep(2000);
    }

    if (mpViewerThread)
    {
        if (mpViewerThread->joinable())
            mpViewerThread->join();
        delete mpViewerThread;
        mpViewerThread = nullptr;
    }
}

void System::Reset()
{
    std::unique_lock<std::mutex> lock(mMutexMode);

    if (mpLoopCloser)
    {
        mpLoopCloser->RequestStopGBA(); // 中断后台全局 BA
        mpLoopCloser->RequestReset();
    }
    if (mpLocalMapper)
        mpLocalMapper->RequestReset();

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

    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "ERROR: Failed to open file at: " << filename << std::endl;
        return;
    }
    f << std::fixed;

    auto lit = mpTracker->mlRelativeFramePoses.begin();
    auto lRit = mpTracker->mlpReferences.begin();

    // 记录第一帧作为 KITTI 评估的初始基准原点 T_w_0
    Eigen::Matrix4f Two = Eigen::Matrix4f::Identity();
    bool bFirst = true;

    for (; lit != mpTracker->mlRelativeFramePoses.end(); ++lit, ++lRit)
    {
        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        // 声明相对位姿累计矩阵
        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 顺着父节点指针向上回溯第一个有效的关键帧
        while (pKF->mbBad)
        {
            Trw = pKF->GetRelativePoseToParent() * Trw;
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        // 计算当前参考关键帧在世界系下的位姿
        Trw = Trw * pKF->GetPose();

        // 计算当前普通帧在世界系下的位姿
        Eigen::Matrix4f Tcw = (*lit) * Trw;
        Eigen::Matrix4f Twc = Tcw.inverse();

        if (bFirst)
        {
            Two = Twc;
            bFirst = false;
        }

        // KITTI 标准：转换到以首帧为原点的局部坐标系 T_0_c = (T_w_0)^-1 * T_w_c
        Eigen::Matrix4f T0c = Two.inverse() * Twc;

        Eigen::Matrix3f R = T0c.block<3, 3>(0, 0);
        Eigen::Vector3f t = T0c.block<3, 1>(0, 3);

        f << std::setprecision(9)
          << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << t(0) << " "
          << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << t(1) << " "
          << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << t(2) << "\n";
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

    // 1. 加锁保护整个地图位姿
    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filename << std::endl;
        return;
    }
    f << std::fixed;

    auto lit = mpTracker->mlRelativeFramePoses.begin();
    auto lRit = mpTracker->mlpReferences.begin();
    auto lT = mpTracker->mlFrameTimes.begin();
    auto lbL = mpTracker->mlbLost.begin();

    for (; lit != mpTracker->mlRelativeFramePoses.end(); ++lit, ++lRit, ++lT, ++lbL)
    {
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        while (pKF->mbBad)
        {
            Trw = pKF->GetRelativePoseToParent() * Trw;
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        // TUM 标准格式：直接输出世界坐标系位姿，不右乘 Two
        Trw = Trw * pKF->GetPose();

        Eigen::Matrix4f Tcw = (*lit) * Trw;
        if (Tcw.hasNaN())
            continue;

        // Twc = Tcw^-1
        Eigen::Matrix4f Twc = Tcw.inverse();
        Eigen::Matrix3f Rwc = Twc.block<3, 3>(0, 0);
        Eigen::Vector3f twc = Twc.block<3, 1>(0, 3);

        Eigen::Quaternionf q(Rwc);
        q.normalize();

        // 格式: timestamp tx ty tz qx qy qz qw
        f << std::setprecision(6) << *lT << " "
          << std::setprecision(9)
          << twc.x() << " " << twc.y() << " " << twc.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    f.close();
    std::cout << "Trajectory successfully saved to " << filename << std::endl;
}