#ifndef SYSTEM_H
#define SYSTEM_H

#include <iostream>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <DBoW3/DBoW3.h>
// 前向声明
class Map;
class Tracker;
class Viewer;
class FrameDrawer;
class LocalMapping;
typedef DBoW3::Vocabulary ORBVocabulary;
class KeyFrameDatabase;

class System
{
public:
    // 传感器模式枚举
    enum eSensor
    {
        MONOCULAR = 0,
        STEREO = 1,
        RGBD = 2
    };

public:
    System(const std::string &strConfigFile, const std::string &strVocFile, const eSensor sensor = STEREO, const bool bUseViewer = true);
    ~System();
    cv::Mat DrawFrame();
    std::shared_ptr<FrameDrawer> GetFrameDrawer() const { return mpFrameDrawer; }
    // 核心输入接口：传入左右目图像和时间戳，返回世界到相机的变换 Tcw
    Eigen::Matrix4f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);
    ORBVocabulary *GetVocabulary() const { return mpVocabulary.get(); }
    // 控制接口
    void Reset();
    void Shutdown();

    // 数据获取接口
    std::shared_ptr<Map> GetMap() const { return mpMap; }

    // 保存关键帧轨迹（KITTI 格式，每行 12 个浮点数：T_wc = [R_wc | t_wc] 行优先展平）
    // 供 evo 等工具评估，可直接与 KITTI ground truth 对比。
    void SaveKeyFrameTrajectoryKITTI(const std::string &filename);

    // 保存每一帧轨迹（KITTI 格式，每行 12 个浮点数：T_wc = [R_wc | t_wc] 行优先展平）
    // 注意：这是「逐帧」轨迹，行数与序列图像帧数一致，可与 KITTI ground truth 逐行对齐，
    //       避免「关键帧轨迹」因行数 != 真值行数而导致的 evo 对齐报错。
    void SaveFrameTrajectoryKITTI(const std::string &filename);
    // 清空已记录的逐帧轨迹（Reset 时调用以保证不残留旧序列数据）
    void ClearFrameTrajectory();

    // 保存关键帧轨迹（TUM 格式，每行: timestamp tx ty tz qx qy qz qw，带时间戳）
    // evo 按时间戳自动对齐，不要求与真值行数相同，评估更稳健。
    void SaveKeyFrameTrajectoryTUM(const std::string &filename);

private:
    eSensor mSensor;
    std::shared_ptr<ORBVocabulary> mpVocabulary;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracker> mpTracker;
    std::shared_ptr<Viewer> mpViewer;
    std::shared_ptr<LocalMapping> mpLocalMapper;
    // 可视化与后台线程句柄
    std::thread *mpViewerThread;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    // 线程安全互斥锁
    std::mutex mMutexMode;
    KeyFrameDatabase *mpKeyFrameDatabase;
    std::vector<Eigen::Matrix4f> mvFrameTrajectory;
};

#endif // SYSTEM_H