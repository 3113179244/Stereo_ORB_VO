#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include <map>
#include <set>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

// 前向声明
class Frame;
class Camera;
class Map;
class MapPoint;
class KeyFrame;
class ORBextractor;

class Optimizer
{
public:
    /**
     * @brief 仅优化当前帧位姿 (Pose-Only BA / Motion-Only BA)
     * @param pFrame 当前帧指针
     * @return 优化后的内点 (Inlier) 数量
     */
    static int PoseOptimization(Frame *pFrame);

    /**
     * @brief 局部 Bundle Adjustment (优化局部关键帧位姿与局部地图点)
     */
    static void LocalBundleAdjustment(KeyFrame *pCurKF, bool *pbStopFlag, std::shared_ptr<Map> pMap);

    /**
     * @brief 全局 Bundle Adjustment (优化所有关键帧与地图点)
     */
    static void GlobalBundleAdjustment(Map *pMap, int nIterations = 20, bool *pbStopFlag = nullptr, 
                                       const unsigned long nLoopKF = 0, const bool bRunGBA = false);

    /**
     * @brief 闭环位姿图优化 (Essential Graph)
     */
    static void OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                       const std::map<KeyFrame*, Eigen::Matrix4f> &NonCorrectedPoses,
                                       const std::map<KeyFrame*, Eigen::Matrix4f> &CorrectedPoses,
                                       const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections);
};

#endif // OPTIMIZER_H