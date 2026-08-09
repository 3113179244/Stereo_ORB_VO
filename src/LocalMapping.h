#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include <list>
#include <mutex>
#include <thread>
#include <memory>

class System;
class Map;
class KeyFrame;
class MapPoint;
class Tracker;

class LocalMapping
{
public:
    LocalMapping(System* pSys, std::shared_ptr<Map> pMap);
    ~LocalMapping();

    void SetTracker(Tracker* pTracker) { mpTracker = pTracker; }

    // 主线程循环函数
    void Run();

    // 由 Tracking 线程调用：插入新的待处理关键帧
    void InsertKeyFrame(KeyFrame* pKF);

    // 状态请求与交互接口
    void RequestStop();
    bool isStopped();
    bool Stop();
    void Release();
    bool SetNotStop();
    bool AcceptKeyFrames();
    bool GetStopRequired();
    // 查询是否有待处理的关键帧（Tracking 线程据此判断 LocalMapping 是否空闲）
    bool KeyframesInQueue();

private:
    // 地图点与关键帧处理主流程函数
    void ProcessNewKeyFrame();
    void MapPointCulling();
    void CreateNewMapPoints();
    void SearchInNeighbors();
    void KeyFrameCulling();

    // 线程控制与标记
    bool CheckNewKeyFrames();

    System* mpSystem;
    std::shared_ptr<Map> mpMap;
    Tracker* mpTracker;

    // 待处理的关键帧队列及互斥锁
    std::list<KeyFrame*> mlNewKeyFrames;
    std::mutex mMutexNewKeyBase;
    
    // 当前正在处理的关键帧与最近新增的地图点列表
    KeyFrame* mpCurrentKeyFrame;
    std::list<MapPoint*> mlpRecentAddedMapPoints;

    // 线程与控制变量
    std::thread* mpThread;
    std::mutex mMutexStop;
    bool mbStopRequested;
    bool mbStopped;
    bool mbNotStop;
    bool mbAcceptKeyFrames;
};

#endif // LOCALMAPPING_H