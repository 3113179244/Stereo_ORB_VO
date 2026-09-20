#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

// 前向声明
class Frame;
class Camera;
class Map;
class MapPoint;
class KeyFrame;

class Optimizer
{
public:
    static void LocalBundleAdjustment(KeyFrame *pCurKF, bool *pbStopFlag, std::shared_ptr<Map> pMap);
    static void GlobalBundleAdjustment(Map *pMap, int nIterations = 20, bool *pbStopFlag = nullptr, const unsigned long nLoopKF = 0, const bool bRunGBA = false);
    static void OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                   const std::map<KeyFrame*, Eigen::Matrix4f> &NonCorrectedPoses,
                                   const std::map<KeyFrame*, Eigen::Matrix4f> &CorrectedPoses,
                                   const std::map<KeyFrame*, std::set<KeyFrame*>> &LoopConnections);
};

#endif // OPTIMIZER_H