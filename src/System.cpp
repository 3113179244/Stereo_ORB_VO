#include "System.h"
#include "Config.h"
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

    // 加载参数配置文件
    if (!Config::setParameterFile(strConfigFile))
    {
        std::cerr << "[System] Failed to load config file: " << strConfigFile << std::endl;
        return;
    }
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
    // 初始化全局地图 Map
    mpMap = std::make_shared<Map>();

    mpFrameDrawer = std::make_shared<FrameDrawer>(mpMap.get());

    mpKeyFrameDatabase = new KeyFrameDatabase(mpVocabulary.get());

    mpTracker = std::make_shared<Tracker>(this, mpVocabulary.get(), mpKeyFrameDatabase, mpMap, sensor);

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
        mpViewer = std::make_shared<Viewer>(this, mpMap);
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

    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
    {
        std::cerr << "ERROR: Map has no KeyFrames, cannot save trajectory!" << std::endl;
        return;
    }

    // 1. 严格按照关键帧 ID 从小到大排序，拿到第 0 个关键帧
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b)
              { return a->mnId < b->mnId; });

    // 2. 原版官方定义：Two 即第一帧关键帧的 GetPoseInverse() (T_w0_c0)
    // 变换所有关键帧使其以第一帧为原点
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "ERROR: Failed to open file at: " << filename << std::endl;
        return;
    }
    f << std::fixed;

    auto lRit = mpTracker->mlpReferences.begin();
    auto lT = mpTracker->mlFrameTimes.begin();

    // 3. 遍历每一普通帧的相对位姿队列
    for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend; ++lit, ++lRit, ++lT)
    {
        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 4. 如果参考关键帧在建图过程中被剔除(Bad)，沿生成树向上遍历回溯
        while (pKF->mbBad)
        {
            Trw = Trw * pKF->GetRelativePoseToParent();
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        // 5. 严格按照官方顺序相乘: Trw = Trw * pKF->GetPose() * Two
        Trw = Trw * pKF->GetPose() * Two;

        // 6. 当前普通帧的世界坐标系变换 Tcw = (*lit) * Trw
        Eigen::Matrix4f Tcw = (*lit) * Trw;

        // 7. 计算相机在世界坐标系下的旋转与平移: Rwc = Rcw^T, twc = -Rwc * tcw
        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);

        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        // 8. 严格按照 KITTI 官方格式输出 12 个参数: [Rwc | twc]
        f << std::setprecision(9)
          << Rwc(0, 0) << " " << Rwc(0, 1) << " " << Rwc(0, 2) << " " << twc(0) << " "
          << Rwc(1, 0) << " " << Rwc(1, 1) << " " << Rwc(1, 2) << " " << twc(1) << " "
          << Rwc(2, 0) << " " << Rwc(2, 1) << " " << Rwc(2, 2) << " " << twc(2) << "\n";
    }

    f.close();
    std::cout << "trajectory saved!" << std::endl;
}

void System::SaveTrajectoryTUM(const std::string &filename)
{
    std::cout << std::endl
              << "Saving camera trajectory to " << filename << " ..." << std::endl;
    if (mSensor == MONOCULAR)
    {
        std::cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << std::endl;
        return;
    }

    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
    {
        std::cerr << "ERROR: Map has no KeyFrames!" << std::endl;
        return;
    }

    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b)
              { return a->mnId < b->mnId; });

    // 官方定义：第一帧的逆位姿作为全局原点变换基准
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse();

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
        // 严格丢弃任何被标记为 LOST 的帧
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 沿生成树向上回溯有效父节点
        while (pKF->mbBad)
        {
            Trw = Trw * pKF->GetRelativePoseToParent();
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        Trw = Trw * pKF->GetPose() * Two;
        Eigen::Matrix4f Tcw = (*lit) * Trw;

        // 检查数值是否有效，防止输出 NaN 或 Inf
        if (Tcw.hasNaN())
            continue;

        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        Eigen::Quaternionf q(Rwc);
        q.normalize();

        f << std::setprecision(6) << *lT << " "
          << std::setprecision(9) << twc.x() << " " << twc.y() << " " << twc.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    f.close();
    std::cout << "Trajectory successfully saved to " << filename << std::endl;
}