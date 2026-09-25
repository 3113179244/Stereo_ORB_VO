#pragma once

#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

class KeyFrame;
class MapPoint;

class Sim3Solver
{
public:
    // bFixScale: 双目/RGB-D 设为 true (SE3 模式，尺度 s=1.0)；单目设为 false (Sim3 模式)
    Sim3Solver(KeyFrame* pKF1, KeyFrame* pKF2, 
               const std::vector<MapPoint*>& vpMatched12, 
               const bool bFixScale = true);

    // 设置 RANSAC 参数
    void SetRansacParameters(double probability = 0.99, int minInliers = 20, int maxIterations = 300);

    // 迭代求解，返回是否求解成功
    bool iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

    // 获取估计的旋转、平移和尺度
    Eigen::Matrix3f GetEstimatedRotation() const { return mBestRotation; }
    Eigen::Vector3f GetEstimatedTranslation() const { return mBestTranslation; }
    float GetEstimatedScale() const { return mBestScale; }
    
    // 获取相机 2 到相机 1 的变换矩阵 T12 (Pc1 = s * R12 * Pc2 + t12)
    Eigen::Matrix4f GetEstimatedTransformation() const;

protected:
    // Horn 绝对定向闭式求解函数
    void ComputeSim3(const std::vector<Eigen::Vector3f>& P1, 
                     const std::vector<Eigen::Vector3f>& P2,
                     Eigen::Matrix3f& R12, Eigen::Vector3f& t12, float& s12);

    // 双向重投影误差卡方检验
    int CheckInliers(const Eigen::Matrix3f& R12, const Eigen::Vector3f& t12, const float s12,
                     std::vector<bool>& vbInliers);

private:
    KeyFrame* mpKF1;
    KeyFrame* mpKF2;

    // 存储有效 3D 点及其特征点像素坐标
    std::vector<Eigen::Vector3f> mvX3Dc1;
    std::vector<Eigen::Vector3f> mvX3Dc2;
    std::vector<cv::Point2f> mvKeys1;
    std::vector<cv::Point2f> mvKeys2;
    std::vector<float> mvSigma2_1;
    std::vector<float> mvSigma2_2;
    std::vector<size_t> mvAllIndices;

    int mN1;
    bool mbFixScale;

    // RANSAC 参数与状态
    int mMaxIterations;
    int mIterations;
    int mMinInliers;
    int mBestInliers;
    
    Eigen::Matrix3f mBestRotation;
    Eigen::Vector3f mBestTranslation;
    float mBestScale;
    std::vector<bool> mvbBestInliers;

    std::mt19937 mRng; // 修复：随机数生成器成员变量，避免每次调用重复固定种子
};