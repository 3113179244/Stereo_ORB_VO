#pragma once

#include <thread>
#include <mutex>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "DBoW3/DBoW3.h"

class KeyFrame;
class MapPoint;
class Map;
class Tracker;
class LocalMapping;
class KeyFrameDatabase;
class Optimizer;
class LoopClosing
{
public:
    // 连续性检验结构体：记录候选组与其连续命中的次数
    typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;

public:
    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, DBoW3::Vocabulary* pVoc, const bool bFixScale = true);
    ~LoopClosing();

    void SetTracker(Tracker* pTracker) { mpTracker = pTracker; }
    void SetLocalMapper(LocalMapping* pLocalMapper) { mpLocalMapper = pLocalMapper; }

    // 主线程循环
    void Run();

    // 插入待检测的关键帧
    void InsertKeyFrame(KeyFrame* pKF);

    // 请求与状态查询
    void RequestStop();
    bool isStopped();
    void RequestReset();

private:
    bool CheckNewKeyFrames();

    // 1. 闭环候选帧检测
    bool DetectLoop();

    // 2. 几何位姿求解与验证 (双目使用 SE3 / PnP)
    bool ComputeSE3();

    // 3. 闭环校正与融合
    void CorrectLoop();

    // 局部地图点投影融合辅助函数
    void SearchAndFuse(const std::vector<KeyFrame*>& vpLoopConnectedKFs);

private:
    Map* mpMap;
    KeyFrameDatabase* mpKeyFrameDB;
    DBoW3::Vocabulary* mpORBVocabulary;
    Tracker* mpTracker;
    LocalMapping* mpLocalMapper;

    std::thread* mpThread;
    bool mbFixScale; // 双目 SLAM 固定尺度为 true (SE3)

    // 关键帧队列
    std::list<KeyFrame*> mlpLoopKeyFrameQueue;
    std::mutex mMutexLoopQueue;

    // 当前处理的关键帧与匹配到的闭环候选帧
    KeyFrame* mpCurrentKF;
    KeyFrame* mpMatchedKF;
    Eigen::Matrix4f mTcw_loop; // 闭环计算出的当前帧位姿

    // 匹配关系缓存
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    std::vector<MapPoint*> mvpLoopMatchedPoints;

    // 连续性检验 (Temporal Consistency) 历史记录
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame*> mvpEnoughConsistentCandidates;

    // 线程控制标志
    bool mbStopRequested;
    bool mbStopped;
    std::mutex mMutexStop;
    bool mbResetRequested;
    std::mutex mMutexReset;
};