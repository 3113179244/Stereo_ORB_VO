#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Frame.h"
#include "System.h"

class Map;
class Viewer;
class ORBextractor;
class KeyFrame;
class ORBmatcher;
class MotionOnlyBA;
class LocalMapping;
class KeyFrameDatabase;

class Tracker
{
public:
    enum eTrackingState
    {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };
    Tracker(System *pSys, ORBVocabulary* pVoc, KeyFrameDatabase* pKFDB, std::shared_ptr<Map> pMap, System::eSensor sensor);
    ~Tracker();
    void SetFrameDrawer(std::shared_ptr<FrameDrawer> pFrameDrawer) { mpFrameDrawer = pFrameDrawer; }
    // 图像数据Grab接口
    Eigen::Matrix4f GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp);
    void SetLocalMapper(LocalMapping *pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void SetViewer(std::shared_ptr<Viewer> pViewer) { mpViewer = pViewer; }
    void Reset();
    bool Relocalize();
    eTrackingState mState;
    // 当前帧与上一帧
    Frame mCurrentFrame;
    Frame mLastFrame;
    cv::Mat mImGray;
private:
    void Track();
    bool StereoInitialization();
    bool TrackWithMotionModel();
    bool TrackReferenceKeyFrame();
    bool TrackLocalMap();
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    System *mpSystem;
    KeyFrameDatabase* mpKeyFrameDB;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Viewer> mpViewer;
    ORBVocabulary* mpORBVocabulary;
    LocalMapping *mpLocalMapper;
    // 特征提取器指针 (双目需要左右各一个 extractor)
    std::unique_ptr<ORBextractor> mpORBextractorLeft;
    std::unique_ptr<ORBextractor> mpORBextractorRight;

    // 恒速模型：相对位姿速度 Velocity (T_current_last = T_c_w * T_w_last)
    Eigen::Matrix4f mVelocity;

    // 计数器与关键帧参考
    KeyFrame *mpReferenceKF;
    int mnLastKeyFrameId;
    int mnMatchesInliers;
};

#endif // TRACKER_H