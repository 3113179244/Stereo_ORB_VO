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
    // 初始化前端 Tracker
    mpTracker = std::make_shared<Tracker>(this, mpVocabulary.get(), mpKeyFrameDatabase, mpMap, sensor);
    mpTracker->SetFrameDrawer(mpFrameDrawer);
    mpLocalMapper = std::make_shared<LocalMapping>(this, mpMap);
    mpTracker->SetLocalMapper(mpLocalMapper.get());
    mpLocalMapper->SetTracker(mpTracker.get());
   
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
    cv::Mat imLeftGray = imLeft;
    cv::Mat imRightGray = imRight;

    if (imLeft.channels() == 3)
    {
        cv::cvtColor(imLeft, imLeftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imRight, imRightGray, cv::COLOR_BGR2GRAY);
    }

    // 调用 Tracker 执行跟踪主流程
    Eigen::Matrix4f Tcw = mpTracker->GrabImageStereo(imLeftGray, imRightGray, timestamp);

    return Tcw;
}

void System::Shutdown()
{
    if (mpViewerThread)
    {
        if (mpViewer)
            mpViewer->RequestStop();
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
    if (!mpMap) return;

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty()) {
        std::cerr << "[System] 错误：地图中无关键帧，无法生成轨迹。" << std::endl;
        return;
    }

    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    // 对齐到第一帧坐标系
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f(filename.c_str());
    if (!f.is_open()) {
        std::cerr << "[System] 无法创建轨迹文件: " << filename << std::endl;
        return;
    }
    f << std::fixed << std::setprecision(9);

    auto lRit = mpTracker->mlpReferences.begin();
    auto lbL = mpTracker->mlbLost.begin();
    Eigen::Matrix4f lastValidTwc = Eigen::Matrix4f::Identity();  // 保存最近有效帧的 Twc
    int nValid = 0;

    for (auto lit = mpTracker->mlRelativeFramePoses.begin();
         lit != mpTracker->mlRelativeFramePoses.end();
         ++lit, ++lRit, ++lbL)
    {
        KeyFrame* pKF = *lRit;
        Eigen::Matrix4f Twc;

        // 追溯有效的父关键帧（若当前参考帧变坏）
        while (pKF && pKF->mbBad)
            pKF = pKF->GetParent();

        if (pKF) {
            // 有效帧：计算实际位姿
            Eigen::Matrix4f Trw = pKF->GetPose() * Two;
            Eigen::Matrix4f Tcw = (*lit) * Trw;
            Twc = Tcw.inverse();
            lastValidTwc = Twc;
            nValid++;
        } else {
            // 无效帧：沿用上一有效位姿
            Twc = lastValidTwc;
        }

        // 输出 3x4 矩阵
        f << Twc(0,0) << " " << Twc(0,1) << " " << Twc(0,2) << " " << Twc(0,3) << " "
          << Twc(1,0) << " " << Twc(1,1) << " " << Twc(1,2) << " " << Twc(1,3) << " "
          << Twc(2,0) << " " << Twc(2,1) << " " << Twc(2,2) << " " << Twc(2,3) << std::endl;
    }

    f.close();
    std::cout << "[System] KITTI 轨迹保存完成！有效帧 " << nValid
              << "，总行数 " << mpTracker->mlRelativeFramePoses.size() << std::endl;
}