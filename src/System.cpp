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

    // 记录当前帧位姿（方案C：逐帧轨迹）
    // T_wc = (T_cw)^-1，与 KITTI ground truth 坐标系约定一致（相机在世界坐标系位姿）
    {
        std::lock_guard<std::mutex> lock(mMutexMode);
        mvFrameTrajectory.push_back(Tcw.inverse());
    }

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
    // 清空逐帧轨迹，避免与下次运行的数据混叠
    ClearFrameTrajectory();
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
 * @brief 保存每一帧轨迹，KITTI 格式（供 evo 评估）
 * @details 方案C：遍历记录在 mvFrameTrajectory 中的每一帧位姿（T_wc），
 *          输出 KITTI 格式（每行 12 个数，无时间戳）。与关键帧轨迹不同，
 *          该文件行数与序列图像帧数一致，可与 KITTI ground truth 逐行对齐，
 *          避免 evo_ape 因行数不匹配而报 "data matrices must have the same shape"。
 */
void System::SaveFrameTrajectoryKITTI(const std::string &filename)
{
    if (mvFrameTrajectory.empty())
    {
        std::cerr << "[System] 未记录到任何逐帧轨迹，未生成文件: " << filename << std::endl;
        return;
    }

    std::ofstream f;
    f.open(filename.c_str());
    if (!f.is_open())
    {
        std::cerr << "[System] 无法打开轨迹文件: " << filename << std::endl;
        return;
    }

    std::cout << "[System] 保存逐帧轨迹 (KITTI) 到 " << filename << " ..." << std::endl;
    f << std::fixed << std::setprecision(9);

    int nValid = 0;
    for (const Eigen::Matrix4f &Twc : mvFrameTrajectory)
    {
        // T_wc = [R_wc | t_wc]
        f << Twc(0, 0) << " " << Twc(0, 1) << " " << Twc(0, 2) << " " << Twc(0, 3) << " "
          << Twc(1, 0) << " " << Twc(1, 1) << " " << Twc(1, 2) << " " << Twc(1, 3) << " "
          << Twc(2, 0) << " " << Twc(2, 1) << " " << Twc(2, 2) << " " << Twc(2, 3) << std::endl;
        ++nValid;
    }

    f.close();
    std::cout << "[System] 逐帧轨迹保存完成，共 " << nValid << " 行。" << std::endl;
}

void System::ClearFrameTrajectory()
{
    std::lock_guard<std::mutex> lock(mMutexMode);
    mvFrameTrajectory.clear();
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