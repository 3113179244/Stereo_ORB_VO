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

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
    {
        std::cerr << "[System 错误] 地图中无关键帧，无法生成轨迹！" << std::endl;
        return;
    }

    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse().inverse();

    std::ofstream f(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "[System 错误] 无法创建轨迹文件: " << filename << std::endl;
        return;
    }
    f << std::fixed << std::setprecision(9);

    auto lRit = mpTracker->mlpReferences.begin();
    auto lT   = mpTracker->mlFrameTimes.begin();
    auto lbL  = mpTracker->mlbLost.begin();

    Eigen::Matrix4f lastValidTwc = Eigen::Matrix4f::Identity();
    int nValid = 0;
    int nTotal = 0;

    for (auto lit = mpTracker->mlRelativeFramePoses.begin();
         lit != mpTracker->mlRelativeFramePoses.end();
         ++lit, ++lRit, ++lT, ++lbL)
    {
        nTotal++;
        KeyFrame* pKF = *lRit;
        Eigen::Matrix4f Tcr = *lit;
        bool bLost = *lbL;

        Eigen::Matrix4f Twc = lastValidTwc;
        bool bCurrentValid = false;

        if (!bLost && pKF)
        {
            // 严格的坐标链变换：记录从当前帧向上传递的相对位姿
            Eigen::Matrix4f T_c_curr = Tcr;
            KeyFrame* pCurrKF = pKF;
            int maxDepth = 0;

            // 沿着生成树向上回溯，使用被剔除时固化的相对位姿 mTcp 进行变换传递
            while (pCurrKF && pCurrKF->mbBad && pCurrKF->GetParent() && maxDepth < 10)
            {
                KeyFrame* pParent = pCurrKF->GetParent();
                
                // 【核心修改】：直接使用剔除时保存的固定相对位姿，绝不能用两帧当前不一致的绝对位姿实时计算！
                Eigen::Matrix4f T_child_parent = pCurrKF->GetRelativePoseToParent();

                // 累乘相对位姿: T_c_parent = T_c_child * T_child_parent
                T_c_curr = T_c_curr * T_child_parent;

                pCurrKF = pParent;
                maxDepth++;
            }

            if (pCurrKF && !pCurrKF->mbBad)
            {
                // 用累乘变换后的 T_c_curr 乘上最终有效父帧的位姿
                Eigen::Matrix4f Trw = pCurrKF->GetPose();
                Eigen::Matrix4f Tcw = T_c_curr * Trw;
                Twc = Tcw.inverse();
                lastValidTwc = Twc;
                bCurrentValid = true;
            }
            else
            {
                // 保底：直接使用原参考关键帧被标记为 bad 时的自身位姿
                Eigen::Matrix4f Trw = pKF->GetPose();
                Eigen::Matrix4f Tcw = Tcr * Trw;
                Twc = Tcw.inverse();
                lastValidTwc = Twc;
                bCurrentValid = true;
            }
        }

        if (bCurrentValid)
            nValid++;

        Eigen::Matrix4f T = Two * Twc;

        f << T(0,0) << " " << T(0,1) << " " << T(0,2) << " " << T(0,3) << " "
          << T(1,0) << " " << T(1,1) << " " << T(1,2) << " " << T(1,3) << " "
          << T(2,0) << " " << T(2,1) << " " << T(2,2) << " " << T(2,3) << "\n";
    }

    f.flush();
    f.close();
    std::cout << "[System] KITTI 轨迹保存成功！有效帧: " << nValid 
              << " / 总记录帧数: " << nTotal << std::endl;
}