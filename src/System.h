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
#include "LoopClosing.h"
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
    std::shared_ptr<Viewer> GetViewer() const { return mpViewer; }
    // 核心输入接口：传入左右目图像和时间戳，返回世界到相机的变换 Tcw
    Eigen::Matrix4f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);
    ORBVocabulary *GetVocabulary() const { return mpVocabulary.get(); }
    std::shared_ptr<LoopClosing> GetLoopCloser() const { return mpLoopCloser; }
    // 控制接口
    void Reset();
    void Shutdown();

    // 数据获取接口
    std::shared_ptr<Map> GetMap() const { return mpMap; }
    void SaveTrajectoryKITTI(const std::string &filename);
    void SaveTrajectoryTUM(const std::string &filename);
    
private:
    eSensor mSensor;
    std::shared_ptr<ORBVocabulary> mpVocabulary;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracker> mpTracker;
    std::shared_ptr<Viewer> mpViewer;
    std::shared_ptr<LocalMapping> mpLocalMapper;
    std::shared_ptr<LoopClosing> mpLoopCloser;
    // 可视化与后台线程句柄
    std::thread *mpViewerThread;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    // 线程安全互斥锁
    std::mutex mMutexMode;
    KeyFrameDatabase *mpKeyFrameDatabase;
};

#endif // SYSTEM_H