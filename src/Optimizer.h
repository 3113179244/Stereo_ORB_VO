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

class Optimizer {
public:
    static int PoseOptimization(Frame *pFrame);
    static void LocalBundleAdjustment(KeyFrame *pCurKF, bool *pbStopFlag, std::shared_ptr<Map> pMap);
    static void GlobalBundleAdjustment(std::shared_ptr<Map> pMap);
};

#endif // OPTIMIZER_H