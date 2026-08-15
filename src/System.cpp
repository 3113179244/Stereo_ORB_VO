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
    std::cout << "[System] 正在保存 KITTI 轨迹至: " << filename << " ..." << std::endl;

    if (mSensor == MONOCULAR)
    {
        std::cerr << "[System 警告] 单目模式下 KITTI 轨迹尺度未定，可能需要尺度对齐。" << std::endl;
    }

    // 获取并检查关键帧
    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
    {
        std::cerr << "[System 错误] 地图中无关键帧，无法生成轨迹！" << std::endl;
        return;
    }

    // 按创建顺序 (mnId) 升序排序
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    // 计算第一帧的世界坐标系逆位姿 T_w0^{-1} (用于将第一帧归一化为世界原点)
    // 注意：GetPoseInverse() 返回的是相机的世界位姿 Twc0
    // 其逆矩阵就是第一帧的 T_cw0 (即 GetPose())
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse().inverse(); // 等价于 vpKFs[0]->GetPose()

    // 打开输出文件
    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "[System 错误] 无法创建轨迹文件: " << filename << std::endl;
        return;
    }
    f << std::fixed << std::setprecision(9);

    // 遍历 Tracking 线程记录的所有普通帧 (共 1101 帧)
    auto lRit = mpTracker->mlpReferences.begin();
    auto lT = mpTracker->mlFrameTimes.begin();
    auto lbL = mpTracker->mlbLost.begin();

    Eigen::Matrix4f lastValidTwc = Eigen::Matrix4f::Identity();
    int nValid = 0;
    int nTotal = 0;

    for (auto lit = mpTracker->mlRelativeFramePoses.begin();
         lit != mpTracker->mlRelativeFramePoses.end();
         ++lit, ++lRit, ++lT, ++lbL)
    {
        nTotal++;
        KeyFrame* pKF = *lRit;
        Eigen::Matrix4f Tcr = *lit; // 当前普通帧相对于参考关键帧的位姿
        bool bLost = *lbL;

        Eigen::Matrix4f Twc;

        if (!bLost && pKF)
        {
            // 若该参考关键帧在 LocalMapping 剔除中变为了 bad，向上找非 bad 的父关键帧
            while (pKF && pKF->mbBad)
            {
                pKF = pKF->GetParent();
            }

            if (pKF)
            {
                // 获取当前参考关键帧最新的世界位姿 T_rw (世界系 -> 参考帧系)
                Eigen::Matrix4f Trw = pKF->GetPose();

                // 还原当前普通帧在优化后的世界坐标系中的变换 T_cw = T_cr * T_rw
                Eigen::Matrix4f Tcw = Tcr * Trw;

                // 得到当前帧相机在世界系下的绝对位姿 T_wc (相机系 -> 世界系)
                Twc = Tcw.inverse();
                lastValidTwc = Twc;
                nValid++;
            }
            else
            {
                // 如果父节点全部失效，沿用上一有效帧
                Twc = lastValidTwc;
            }
        }
        else
        {
            // 丢失帧沿用上一帧位姿，保证 1101 行严格对齐
            Twc = lastValidTwc;
        }

        // 将位姿统一变换到以第一帧为世界原点的参考系下: T = T_w0^{-1} * T_wc
        Eigen::Matrix4f T = Two * Twc;

        // 按 KITTI 格式输出 3x4 矩阵（12 个浮点数）
        f << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " "
          << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " "
          << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << std::endl;
    }

    f.close();
    std::cout << "[System] KITTI 轨迹保存成功！有效帧: " << nValid 
              << " / 总记录帧数: " << nTotal << std::endl;
}