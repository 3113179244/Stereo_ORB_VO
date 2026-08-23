#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;

    // 构造函数
    Camera();
    Camera(double fx, double fy, double cx, double cy,
           double k1 = 0, double k2 = 0, double p1 = 0, double p2 = 0,
           double width = 0, double height = 0, double fps = 30.0,
           double bf = 0.0, double thDepth = 40.0);

    ~Camera() = default;

    inline double fx() const { return fx_; }
    inline double fy() const { return fy_; }
    inline double cx() const { return cx_; }
    inline double cy() const { return cy_; }
    inline double bf() const { return bf_; }                 // baseline * fx
    inline double b() const { return b_; }                   // baseline (m)
    inline double thDepth() const { return thDepth_; }       // 远近点判定阈值 (以基线为单位)
    inline double maxDepth() const { return thDepth_ * b_; } // 最大有效深度(m)

    // 相机内参矩阵 K 与 畸变矩阵 D
    Eigen::Matrix3d K() const;
    cv::Mat K_cv() const;
    cv::Mat D_cv() const;

    // 1. 像素坐标 (u, v) -> 相机归一化平面坐标 (x, y) = ((u-cx)/fx, (v-cy)/fy)
    Eigen::Vector2d pixel2norm(const Eigen::Vector2d &p_p) const;
    cv::Point2f pixel2norm(const cv::Point2f &p_p) const;

    // 2. 相机归一化坐标 (x, y) -> 像素坐标 (u, v)
    Eigen::Vector2d norm2pixel(const Eigen::Vector2d &p_n) const;
    cv::Point2f norm2pixel(const cv::Point2f &p_n) const;

    // 3. 相机坐标系 3D 点 (X, Y, Z) -> 像素坐标 (u, v)
    Eigen::Vector2d camera2pixel(const Eigen::Vector3d &p_c) const;

    // 4. 像素坐标 (u, v) + 深度 Z -> 相机坐标系 3D 点 (X, Y, Z)
    Eigen::Vector3d pixel2camera(const Eigen::Vector2d &p_p, double depth) const;

    // 5. 根据双目视差 (disparity = u_L - u_R) 计算深度 Z
    // Z = (fx * b) / disparity = bf / disparity
    inline double disparity2depth(double disparity) const
    {
        if (disparity <= 0.0)
            return -1.0;
        return bf_ / disparity;
    }

    // 6. 根据深度 Z 计算视差 disparity
    inline double depth2disparity(double depth) const
    {
        if (depth <= 0.0)
            return -1.0;
        return bf_ / depth;
    }

    // 7. 判断 3D 点在当前相机坐标系下是否在视场内 (In Front & In Bound)
    bool isInFrustum(const Eigen::Vector3d &p_c, double margin = 0.0) const;

    // 8. 畸变矫正 (支持单点去畸变)
    Eigen::Vector2d undistortPoint(const Eigen::Vector2d &p_p) const;

private:
    // 相机基础内参
    double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};

    // 畸变参数 (k1, k2 为径向畸变, p1, p2 为切向畸变)
    double k1_{0.0}, k2_{0.0}, p1_{0.0}, p2_{0.0};

    // 图像尺寸与帧率
    double width_{0.0}, height_{0.0};
    double fps_{30.0};

    // 双目专属参数
    double bf_{0.0};       // baseline * fx
    double b_{0.0};        // baseline (基线长度, 单位: 米)
    double thDepth_{40.0}; // ORB-SLAM2 近点/远点阈值: 深度小于 thDepth * b 为近点
};

#endif // CAMERA_H