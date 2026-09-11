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

    // 1. 按关键帧 ID 递增排序，保证第一帧关键帧排在首位
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    // 2. 官方原点校正：以第 0 个关键帧的位姿逆作为全局原点变换矩阵
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "ERROR: Failed to open file at: " << filename << std::endl;
        return;
    }
    f << std::fixed;

    auto lRit = mpTracker->mlpReferences.begin();
    auto lT   = mpTracker->mlFrameTimes.begin();

    // 3. 遍历普通帧队列
    for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend; ++lit, ++lRit, ++lT)
    {
        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 4. 若参考关键帧已被剔除，沿生成树递归向上回溯至有效父节点
        while (pKF->mbBad)
        {
            Trw = Trw * pKF->GetRelativePoseToParent();
            pKF = pKF->GetParent();
            if (!pKF)
                break;
        }

        if (!pKF)
            continue;

        // 5. 按照 ORB-SLAM2 官方顺序级联矩阵：Trw = Trw * T_pKF_w * Two
        Trw = Trw * pKF->GetPose() * Two;

        // 6. 还原当前帧在世界坐标系下的变换 Tcw = Tcr * Trw
        Eigen::Matrix4f Tcw = (*lit) * Trw;

        // 7. 转为相机到世界坐标系（Twc）：Rwc = Rcw^T, twc = -Rwc * tcw
        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        // 8. 写入 12 个参数 [Rwc | twc]
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

    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
    {
        std::cerr << "ERROR: Map has no KeyFrames, cannot save trajectory!" << std::endl;
        return;
    }

    // 1. 按关键帧 ID 排序
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    // 2. 原点校正矩阵
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse();

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
        // 3. TUM 官方标准：跟踪失败/丢失的帧严格跳过不输出
        if (*lbL)
            continue;

        KeyFrame *pKF = *lRit;
        if (!pKF)
            continue;

        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 4. 沿生成树向上回溯有效父节点
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

        if (Tcw.hasNaN())
            continue;

        // 5. 提取 Rwc 与 twc
        Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);
        Eigen::Matrix3f Rwc = Rcw.transpose();
        Eigen::Vector3f twc = -Rwc * tcw;

        // 6. 构造四元数并归一化
        Eigen::Quaternionf q(Rwc);
        q.normalize();

        // 7. TUM 格式输出: timestamp tx ty tz qx qy qz qw
        f << std::setprecision(6) << *lT << " "
          << std::setprecision(9) << twc.x() << " " << twc.y() << " " << twc.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }

    f.close();
    std::cout << "Trajectory successfully saved to " << filename << std::endl;
}