#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <vector>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// 前向声明，避免循环包含
class KeyFrame;
class Frame;
class Map;

class MapPoint
{
public:
    // 构造函数：已知空间位置、参考关键帧及所属地图
    MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap);
    
    // 设置和获取地图点在世界坐标系下的三维坐标
    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();
    
    // 获取地图点的平均观测法线方向
    Eigen::Vector3f GetNormal(); 
    
    // 添加和删除对该地图点的观测（记录哪个关键帧的哪个特征点观测到了它）
    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);
    
    // 获取所有观测到该地图点的关键帧及对应的特征点索引
    std::map<KeyFrame*, size_t> GetObservations();
    
    // 获取指定关键帧观测到该地图点的特征点索引，若未观测到返回-1
    int GetIndexInKeyFrame(KeyFrame* pKF);
    // 判断该地图点是否被某个指定关键帧观测到了
    bool IsInKeyFrame(KeyFrame* pKF);
    
    // 计算该地图点的最具代表性描述子（与其他描述子汉明距离中位数最小的那个）
    void ComputeDistinctiveDescriptor(); 
    // 获取该地图点的代表性描述子
    cv::Mat GetDescriptor();
    
    // 更新地图点的平均法线方向与可见的深度范围（金字塔层级对应的距离上下限）
    void UpdateNormalAndDepth();         
    
    // 将该地图点标记为坏点（例如在优化中被剔除或观测数过少）
    void SetBadFlag();
    // 判断该地图点是否为坏点
    bool isBad();
    // 替换地图点（通常在闭环检测或局部建图融合重复点时调用），用传入的pMP替换当前点
    void Replace(MapPoint* pMP);    
    
    // 增加可见次数（通常在局部地图跟踪时使用）
    void IncreaseVisible(int n=1);
    // 增加被成功匹配到的次数
    void IncreaseFound(int n=1);
    // 获取匹配率（Found / Visible），用于评估该点质量
    float GetFoundRatio();
    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    // 静态成员：用于生成唯一的地图点ID
    static long unsigned int nNextId;
    // 当前地图点的唯一ID
    long unsigned int mnId;
    // 全局互斥锁
    static std::mutex mGlobalMutex;
    long unsigned int mnFirstKFid; // 记录创建该点的关键帧 ID
    // 跟踪统计：在视野内的次数和实际被匹配到的次数
    int mnVisible;
    int mnFound;
    
    // 标记与状态变量
    bool mbBad;               // 是否是坏点
    MapPoint* mpReplaced;     // 如果被替换，指向替换它的新地图点
    float mfMinDistance;      // 能够观测到该点的最小距离
    float mfMaxDistance;      // 能够观测到该点的最大距离

private:
    // ---- 线程安全数据，防止多线程（如跟踪、局部建图、闭环）发生数据竞争 ----
    std::mutex mMutexPos;      // 保护位置数据的互斥锁
    std::mutex mMutexFeatures; // 保护特征观测数据的互斥锁

    Eigen::Vector3f mWorldPos;                 // 3D世界坐标
    std::map<KeyFrame*, size_t> mObservations; // 观测到该点的关键帧字典：<关键帧指针, 特征点在关键帧中的索引>
    Eigen::Vector3f mNormalVector;             // 平均观测法向量
    cv::Mat mDescriptor;                       // 最具代表性的描述子

    // 引用关键帧（通常是第一次创建该地图点的关键帧）与所在的地图指针
    KeyFrame* mpRefKF;
    Map* mpMap;
};

#endif // MAPPOINT_H