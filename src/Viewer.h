#ifndef VIEWER_H
#define VIEWER_H

#include <memory>
#include <mutex>
#include <thread>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

class System;
class Map;
class Tracker;
class FrameDrawer;
class KeyFrame;
class MapPoint;

class Viewer
{
public:
    Viewer(System *pSystem, std::shared_ptr<Map> pMap, std::shared_ptr<FrameDrawer> pFrameDrawer = nullptr);
    ~Viewer();

    void SetTracker(Tracker *pTracker) { mpTracker = pTracker; }
    void SetFrameDrawer(std::shared_ptr<FrameDrawer> pFrameDrawer) { mpFrameDrawer = pFrameDrawer; }

    // 主渲染循环
    void Run();

    // 线程安全的相机位姿更新接口（供 Tracker 调用）
    void UpdateCurrentCameraPose(const Eigen::Matrix4f &Tcw);

    // 线程控制
    void RequestStop();
    bool isStopped();
    bool Stop();
    void Release();

    void RequestFinish();
    bool isFinished();
    bool CheckFinish();
    void SetFinish();

private:
    void DrawMapPoints();
    void DrawKeyFrames(bool bDrawKF, bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &M);

    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    void SetCurrentCameraPose(const Eigen::Matrix4f &Tcw);

    System *mpSystem;
    std::shared_ptr<Map> mpMap;
    Tracker *mpTracker;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;

    // 绘制尺寸与视角参数
    float mCameraSize;
    float mCameraLineWidth;
    float mPointSize;
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;

    float mViewpointX;
    float mViewpointY;
    float mViewpointZ;
    float mViewpointF;

    Eigen::Matrix4f mCameraPose;
    std::mutex mMutexCamera;

    double mFPS;
    double mT;

    bool mbStopRequested;
    bool mbStopped;
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexStop;
    std::mutex mMutexFinish;
};

#endif // VIEWER_H