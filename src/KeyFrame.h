#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <vector>
#include <set>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <DBoW3/DBoW3.h>
// 前向声明
class Frame;
class MapPoint;
class Map;
typedef DBoW3::Vocabulary ORBVocabulary;
class KeyFrame
{
public:
    // 构造函数：根据当前普通帧 (Frame) 和所属地图 (Map) 构造关键帧
    KeyFrame(Frame &F, Map *pMap);

    // 位姿 (Pose) 相关函数
    // 设置相机到世界坐标系的变换矩阵 Tcw (Thread-safe)
    void SetPose(const Eigen::Matrix4f &Tcw);
    // 获取相机位姿 Tcw (Thread-safe)
    Eigen::Matrix4f GetPose();
    // 获取位姿的逆 Twc，即世界坐标系到相机坐标系的变换 (Thread-safe)
    Eigen::Matrix4f GetPoseInverse();
    // 获取相机光心在世界坐标系下的三维坐标 (Thread-safe)
    Eigen::Vector3f GetCameraCenter();
    // 获取旋转矩阵 Rcw (Thread-safe)
    Eigen::Matrix3f GetRotation();
    // 获取平移向量 tcw (Thread-safe)
    Eigen::Vector3f GetTranslation();

    // 共视图 (Covisibility Graph) 相关函数
    // 添加或更新与其他关键帧的连接关系及权重（权重通常是共视地图点的数量）
    void AddConnection(KeyFrame *pKF, const int &weight);
    // 删除与其他关键帧的连接关系
    void EraseConnection(KeyFrame *pKF);
    // 重新计算并更新当前关键帧与所有其他关键帧的共视连接关系
    void UpdateConnections();
    // 更新共视权重排序，将共视程度最高的关键帧排在前面
    void UpdateBestCovisibles();
    // 获取权重（共视点数）大于指定值 w 的所有相连关键帧
    std::vector<KeyFrame *> GetCovisibleByWeight(const int &w);
    // 获取共视程度最高的前 N 个关键帧
    std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N);
    // 获取所有建立连接的关键帧
    std::vector<KeyFrame *> GetConnectedKeyFrames();
    // 获取当前关键帧与指定关键帧之间的共视权重
    int GetWeight(KeyFrame *pKF);

    // 地图点 (MapPoint) 相关函数
    // 为当前关键帧的第 idx 个特征点关联一个 3D 地图点
    void AddMapPoint(MapPoint *pMP, const size_t &idx);
    // 解除第 idx 个特征点与地图点的绑定
    void EraseMapPointMatch(const size_t &idx);
    // 解除特定地图点与当前关键帧的绑定
    void EraseMapPointMatch(MapPoint *pMP);
    // 替换第 idx 个特征点关联的地图点（常用于闭环或局部建图的重复点融合）
    void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);
    // 获取当前关键帧中所有特征点对应的地图点列表
    std::vector<MapPoint *> GetMapPointMatches();
    // 获取第 idx 个特征点对应的地图点
    MapPoint *GetMapPoint(const size_t &idx);
    void SetParent(KeyFrame *pKF);
    KeyFrame *GetParent();
    void AddChild(KeyFrame *pKF);
    void EraseChild(KeyFrame *pKF);
    std::set<KeyFrame *> GetChilds();
    // 词袋模型 (Bag of Words)
    // 计算当前关键帧的词袋向量，用于重定位和闭环检测
    void ComputeBoW();
    void SetBadFlag();
    int TrackedMapPoints(const int &minObs);
    // 获取当自身被标记为 bad 时相对于父节点的相对位姿
    Eigen::Matrix4f GetRelativePoseToParent();
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r,
                                          const int minLevel = -1, const int maxLevel = -1) const;
    // 网格划分相关参数（每个关键帧独立，用于加速局部特征投影匹配）
    static const int mnGridCols = 64; // 或复用 FRAME_GRID_COLS
    static const int mnGridRows = 48;
    float mfGridElementWidthInv;
    float mfGridElementHeightInv;
    float mnMinX, mnMaxX, mnMinY, mnMaxY;
    std::vector<std::size_t> mGrid[64][48]; // 二维网格，存特征点索引

    // 基础标识与时间戳
    static long unsigned int nNextId;  // 静态全局变量，用于生成下一个关键帧的唯一ID
    long unsigned int mnId;            // 当前关键帧的唯一ID
    const long unsigned int mnFrameId; // 提取出该关键帧的原始普通帧(Frame)的ID
    const double mTimeStamp;           // 对应图像的时间戳

    // 相机内参及双目/深度参数
    const float fx, fy, cx, cy, invfx, invfy;
    const float mbf, mb, mThDepth;
    const cv::Mat mK; // 相机内参矩阵

    // 特征点与描述子
    const int N;                              // 特征点总数
    const std::vector<cv::KeyPoint> mvKeys;   // 原始提取的二维特征点
    const std::vector<cv::KeyPoint> mvKeysUn; // 去畸变后的二维特征点
    const std::vector<float> mvuRight;        // 双目右图对应的横坐标 (若是单目则为负)
    const std::vector<float> mvDepth;         // 对应的深度值
    const cv::Mat mDescriptors;               // 特征点对应的描述子矩阵

    // 状态与图像金字塔参数
    bool mbBad; // 标记该关键帧是否已被剔除（如因冗余被 Local Mapping 线程删除）

    int mnScaleLevels;                   // 图像金字塔的层数
    float mfScaleFactor;                 // 金字塔缩放因子
    std::vector<float> mvScaleFactors;   // 各层级的缩放因子
    std::vector<float> mvLevelSigma2;    // 各层级缩放因子的平方
    std::vector<float> mvInvLevelSigma2; // 各层级缩放因子平方的倒数

    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;

    long unsigned int mnRelocQuery = 0;
    int mnRelocWords = 0;
    float mRelocScore = 0.0f;

private:
    KeyFrame *mpParent = nullptr;     // 父节点指针
    std::set<KeyFrame *> mspChildren; // 子节点集合
    // 将特征点分配到网格
    void AssignFeaturesToGrid();
    // 判断特征点是否在网格内并返回网格坐标
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
    // 线程安全控制锁
    std::mutex mMutexPose;        // 保护相机位姿读写的互斥锁
    std::mutex mMutexConnections; // 保护共视图连接关系读写的互斥锁
    std::mutex mMutexFeatures;    // 保护特征点与地图点匹配关系读写的互斥锁

    // 相机位姿数据
    Eigen::Matrix4f Tcw; // 世界坐标系到相机坐标系的变换矩阵
    Eigen::Vector3f Ow;  // 相机光心在世界坐标系下的坐标
    Eigen::Matrix3f Rcw; // 旋转矩阵 (世界 -> 相机)
    Eigen::Vector3f tcw; // 平移向量 (世界 -> 相机)
    Eigen::Matrix3f Rwc; // 旋转矩阵的逆 (相机 -> 世界)
                         // 保存被剔除时相对于父节点的位姿: T_child_parent
    Eigen::Matrix4f mTcp;
    // 记录特征点关联的 3D 地图点（按特征点索引排列，空则为 nullptr）
    std::vector<MapPoint *> mvpMapPoints;

    // 共视图 (Covisibility Graph) 数据结构
    std::map<KeyFrame *, int> mConnectedKeyFrameWeights;  // 记录相连的关键帧及其权重（共享的地图点数量）
    std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames; // 按权重降序排列的相连关键帧列表
    std::vector<int> mvOrderedWeights;                    // 与上述列表对应的权重列表

    // 关联的地图指针与词典指针
    Map *mpMap;
    ORBVocabulary *mpORBvocabulary;
};

#endif // KEYFRAME_H