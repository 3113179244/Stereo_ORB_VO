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

/**
 * @brief 保存关键帧轨迹，KITTI 格式（供 evo 评估）
 * @details 参考 ORB-SLAM2 的 SaveKeyFrameTrajectoryKITTI：
 *          - 遍历地图中所有关键帧，按 mnId 排序；
 *          - 每个关键帧取其相机位姿 T_wc（GetPoseInverse() 返回的 [R_wc | t_wc]），
 *            与 KITTI ground truth（T_w_cam0）坐标系约定一致；
 *          - 将左上 3x4 矩阵行优先展平成 12 个浮点数，每行写一个关键帧，无时间戳。
 */
void System::SaveKeyFrameTrajectoryKITTI(const std::string &filename)
{
    if (!mpMap)
        return;

    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
        return;

    // 按关键帧 ID 排序，确保输出严格按时间（插入）顺序
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    std::ofstream f;
    f.open(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "[System] 无法打开轨迹文件: " << filename << std::endl;
        return;
    }

    std::cout << "[System] 保存关键帧轨迹 (KITTI) 到 " << filename << " ..." << std::endl;
    f << std::fixed << std::setprecision(9);

    int nValid = 0;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        // T_wc = [R_wc | t_wc]
        Eigen::Matrix4f Twc = pKF->GetPoseInverse();

        f << Twc(0, 0) << " " << Twc(0, 1) << " " << Twc(0, 2) << " " << Twc(0, 3) << " "
          << Twc(1, 0) << " " << Twc(1, 1) << " " << Twc(1, 2) << " " << Twc(1, 3) << " "
          << Twc(2, 0) << " " << Twc(2, 1) << " " << Twc(2, 2) << " " << Twc(2, 3) << std::endl;
        ++nValid;
    }

    f.close();
    std::cout << "[System] 轨迹保存完成，共 " << nValid << " 个关键帧。" << std::endl;
}

/**
 * @brief 保存关键帧轨迹，TUM 格式（供 evo 评估）
 * @details 参考 ORB-SLAM2 的 SaveKeyFrameTrajectoryTUM：
 *          - 遍历地图中所有关键帧，按 mnId 排序；
 *          - 每个关键帧取其相机位姿 T_wc（GetPoseInverse()），
 *            输出格式：timestamp tx ty tz qx qy qz qw；
 *          - 带时间戳，evo 按时间自动对齐，不要求与真值行数相同。
 */
void System::SaveKeyFrameTrajectoryTUM(const std::string &filename)
{
    if (!mpMap)
        return;

    std::vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
        return;

    // 按关键帧 ID 排序，确保输出严格按时间（插入）顺序
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    std::ofstream f;
    f.open(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "[System] 无法打开轨迹文件: " << filename << std::endl;
        return;
    }

    std::cout << "[System] 保存关键帧轨迹 (TUM) 到 " << filename << " ..." << std::endl;
    f << std::fixed << std::setprecision(9);

    int nValid = 0;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        // T_wc = [R_wc | t_wc]
        Eigen::Matrix4f Twc = pKF->GetPoseInverse();
        Eigen::Vector3f t = Twc.block<3, 1>(0, 3);                       // 平移（世界系下相机位置）
        Eigen::Matrix3f R = Twc.block<3, 3>(0, 0);                        // 旋转（相机到世界）
        Eigen::Quaternionf q(R);                                          // 旋转矩阵 -> 四元数

        // TUM: timestamp tx ty tz qx qy qz qw
        f << pKF->mTimeStamp << " "
          << t.x() << " " << t.y() << " " << t.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        ++nValid;
    }

    f.close();
    std::cout << "[System] 轨迹保存完成，共 " << nValid << " 个关键帧。" << std::endl;
}

void System::SaveTrajectoryKITTI(const std::string &filename)
{
    if (!mpMap)
        return;

    std::cout << "\n[System] 正在按照 ORB-SLAM2 思路导出轨迹 (KITTI) 到 " << filename << " ..." << std::endl;

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty())
    {
        std::cerr << "[System] 错误：地图中无关键帧，无法生成轨迹。" << std::endl;
        return;
    }

    // 按 KeyFrame ID 排序
    std::sort(vpKFs.begin(), vpKFs.end(),
              [](KeyFrame *a, KeyFrame *b) { return a->mnId < b->mnId; });

    // 第一帧关键帧位姿的逆（用于校准世界坐标系原点）
    Eigen::Matrix4f Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "[System] 无法创建轨迹文件: " << filename << std::endl;
        return;
    }

    f << std::fixed << std::setprecision(9);

    auto lRit = mpTracker->mlpReferences.begin();
    auto lbL = mpTracker->mlbLost.begin();
    int nValid = 0;

    // 结合后端优化后的最新参考帧位姿，动态重算每一帧的位姿
    for (auto lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end();
         lit != lend; lit++, lRit++, lbL++)
    {
        if (*lbL) continue; // 跳过追踪丢失的帧

        KeyFrame* pKF = *lRit;
        Eigen::Matrix4f Trw = Eigen::Matrix4f::Identity();

        // 若参考帧坏掉，向上追溯生成树上的有效父关键帧
        while (pKF && pKF->mbBad)
        {
            pKF = pKF->GetParent();
        }

        if (!pKF) continue;

        // 获取后端 BA 优化后的最新参考帧位姿 T_rw
        Trw = pKF->GetPose() * Two;

        // 重算当前帧优化后的 T_cw = T_cr * T_rw
        Eigen::Matrix4f Tcw = (*lit) * Trw;

        // 相机在世界坐标系下的变换矩阵 T_wc = (T_cw)^-1
        Eigen::Matrix4f Twc = Tcw.inverse();

        // KITTI 格式：展开输出 3x4 变换矩阵的 12 个浮点数
        f << Twc(0, 0) << " " << Twc(0, 1) << " " << Twc(0, 2) << " " << Twc(0, 3) << " "
          << Twc(1, 0) << " " << Twc(1, 1) << " " << Twc(1, 2) << " " << Twc(1, 3) << " "
          << Twc(2, 0) << " " << Twc(2, 1) << " " << Twc(2, 2) << " " << Twc(2, 3) << std::endl;

        nValid++;
    }

    f.close();
    std::cout << "[System] KITTI 轨迹保存完成！共导出 " << nValid << " 帧位姿。" << std::endl;
}