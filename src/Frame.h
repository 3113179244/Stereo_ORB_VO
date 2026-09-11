#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <DBoW3/DBoW3.h>
class MapPoint;
class ORBextractor;
class KeyFrame;
typedef DBoW3::Vocabulary ORBVocabulary;
// 定义图像网格的行数和列数，用于将特征点分配到网格中，以加速局部特征匹配
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame
{
public:
    // 默认构造函数
    Frame();

    // 双目帧构造函数：传入左右目图像、时间戳、ORB特征提取器、词袋模型、相机内参及双目基线参数
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
          ORBextractor *extractorLeft, ORBextractor *extractorRight,
          ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // 提取 ORB 特征，flag=0 代表左图，flag=1 代表右图
    void ExtractORB(int flag, const cv::Mat &im);

    // 设置相机到世界坐标系的变换矩阵 $T_{cw}$ (使用 Eigen)
    void SetPose(const Eigen::Matrix4f &Tcw);

    // 从位姿矩阵 $T_{cw}$ 更新内部的旋转矩阵 $R_{cw}$、平移向量 $t_{cw}$ 以及光心位置 $O_w$
    void UpdatePoseMatrices();

    // 获取相机光心在世界坐标系下的 3D 坐标 $O_w$
    inline Eigen::Vector3f GetCameraCenter() { return mOw; }

    // 获取相机到世界的旋转矩阵 $R_{wc}$ ($R_{cw}$ 的逆)
    inline Eigen::Matrix3f GetRotationInverse() { return mRwc; }

    // 将带有深度信息的第 i 个双目/RGB-D特征点反投影为 3D 世界坐标
    Eigen::Vector3f UnprojectStereo(const int &i);

    // 获取图像中以 (x,y) 为中心、r 为半径的圆形区域内的所有特征点索引
    // minLevel 和 maxLevel 用于限制提取特征点的金字塔层级
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r,
                                          const int minLevel = -1, const int maxLevel = -1) const;
    /**
     * @brief 判断第 i 个特征点在当前帧视野下是否为近点 (Depth < mThDepth)
     */
    bool isNear(int i) const;

    /**
     * @brief 判断第 i 个特征点在当前帧视野下是否为远点 (Depth >= mThDepth)
     */
    bool isFar(int i) const;

    /**
     * @brief 根据地图点世界坐标，判断其相对于当前帧相机光心是否为近点
     */
    bool isMapPointNear(MapPoint* pMP) const;

    /**
     * @brief 根据地图点世界坐标，判断其相对于当前帧相机光心是否为远点
     */
    bool isMapPointFar(MapPoint* pMP) const;
    // 计算 BoW 向量的函数
    void ComputeBoW();
    // 全局静态变量，用于分配唯一的帧 ID
    static long unsigned int nNextId;
    long unsigned int mnId; // 当前帧的 ID
    double mTimeStamp;      // 时间戳
    cv::Mat mImGrayLeft;
    cv::Mat mImGrayRight;
    // 相机内参相关
    cv::Mat mK;
    static float fx, fy, cx, cy, invfx, invfy; // 焦距及其倒数、主点坐标
    cv::Mat mDistCoef;                         // 畸变系数
    float mbf;                                 // 基线长度乘以焦距 (baseline * fx)
    float mb;                                  // 物理基线长度
    float mThDepth;                            // 区分远近点的深度阈值

    // 特征点数据
    int N;                                                   // 提取到的特征点总数
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight, mvKeysUn; // 左图特征点、右图特征点、去畸变后的左图特征点
    cv::Mat mDescriptors, mDescriptorsRight;                 // 左图和右图的特征描述子

    std::vector<float> mvuRight; // 左图特征点在右图中的匹配横坐标 (用于双目)
    std::vector<float> mvDepth;  // 左图特征点对应的深度值
    DBoW3::BowVector mBowVec;
    DBoW3::FeatureVector mFeatVec;
    // 地图点及外点标记
    std::vector<MapPoint *> mvpMapPoints; // 每个特征点关联的 3D 地图点 (若无关联则为空指针)
    std::vector<bool> mvbOutlier;         // 标记每个特征点对应的地图点是否被判断为外点 (Outlier)

    // 相机位姿矩阵 $T_{cw}$
    Eigen::Matrix4f mTcw;

    // 网格划分相关参数 (静态变量，所有帧共享)，用于加速区域内的特征点查找
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    static float mnMinX, mnMaxX, mnMinY, mnMaxY;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS]; // 二维网格，存储落入各个网格的特征点索引

    // 工具类指针
    ORBextractor *mpORBextractorLeft;
    ORBextractor *mpORBextractorRight;
    ORBVocabulary *mpORBvocabulary;

    // 标志位：是否是第一次进行相机的内参和网格参数计算
    static bool mbInitialComputations;
    KeyFrame* mpReferenceKF = nullptr; // 当前帧绑定的参考关键帧

private:
    // 计算双目匹配，得到视差和深度
    void ComputeStereoMatches();
    // 计算图像的有效边界 (去畸变后)
    void ComputeImageBounds(const cv::Mat &imLeft);
    // 将提取到的特征点分配到图像网格中
    void AssignFeaturesToGrid();
    // 判断给定的特征点是否在网格内，并返回其网格坐标 posX 和 posY
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    // 位姿数据的内部缓存
    Eigen::Matrix3f mRcw; // 旋转矩阵 (世界到相机)
    Eigen::Vector3f mtcw; // 平移向量 (世界到相机)
    Eigen::Matrix3f mRwc; // 旋转矩阵的逆 (相机到世界)
    Eigen::Vector3f mOw;  // 相机光心在世界坐标系下的位置
};

#endif // FRAME_H