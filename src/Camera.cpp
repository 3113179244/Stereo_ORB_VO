#include "Camera.h"

Camera::Camera() {}

Camera::Camera(double fx, double fy, double cx, double cy,
               double k1, double k2, double p1, double p2,
               double width, double height, double fps,
               double bf, double thDepth)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
      k1_(k1), k2_(k2), p1_(p1), p2_(p2),
      width_(width), height_(height), fps_(fps),
      bf_(bf), thDepth_(thDepth)
{
    // 根据 bf 和 fx 自动计算基线长度 baseline b (单位: m)
    if (fx_ > 0.0)
    {
        b_ = bf_ / fx_;
    }
}

Eigen::Matrix3d Camera::K() const
{
    Eigen::Matrix3d K;
    K << fx_, 0.0, cx_,
        0.0, fy_, cy_,
        0.0, 0.0, 1.0;
    return K;
}

cv::Mat Camera::K_cv() const
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
    return K.clone();
}

cv::Mat Camera::D_cv() const {
    cv::Mat D = (cv::Mat_<double>(4, 1) << k1_, k2_, p1_, p2_);
    return D.clone();
}

Eigen::Vector2d Camera::pixel2norm(const Eigen::Vector2d &p_p) const
{
    return Eigen::Vector2d(
        (p_p.x() - cx_) / fx_,
        (p_p.y() - cy_) / fy_);
}

cv::Point2f Camera::pixel2norm(const cv::Point2f &p_p) const
{
    return cv::Point2f(
        static_cast<float>((p_p.x - cx_) / fx_),
        static_cast<float>((p_p.y - cy_) / fy_));
}

Eigen::Vector2d Camera::norm2pixel(const Eigen::Vector2d &p_n) const
{
    return Eigen::Vector2d(
        p_n.x() * fx_ + cx_,
        p_n.y() * fy_ + cy_);
}

cv::Point2f Camera::norm2pixel(const cv::Point2f &p_n) const
{
    return cv::Point2f(
        static_cast<float>(p_n.x * fx_ + cx_),
        static_cast<float>(p_n.y * fy_ + cy_));
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d &p_c) const
{
    return Eigen::Vector2d(
        fx_ * p_c.x() / p_c.z() + cx_,
        fy_ * p_c.y() / p_c.z() + cy_);
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth) const
{
    return Eigen::Vector3d(
        (p_p.x() - cx_) * depth / fx_,
        (p_p.y() - cy_) * depth / fy_,
        depth);
}

bool Camera::isInFrustum(const Eigen::Vector3d &p_c, double margin) const
{
    // 深度必须大于 0
    if (p_c.z() <= 0.0)
        return false;

    // 投影到像素坐标
    Eigen::Vector2d p_p = camera2pixel(p_c);

    // 检查是否在图像边界内（考虑边界 margin）
    return (p_p.x() >= margin && p_p.x() < (width_ - margin) &&
            p_p.y() >= margin && p_p.y() < (height_ - margin));
}

Eigen::Vector2d Camera::undistortPoint(const Eigen::Vector2d &p_p) const
{
    // 如果没有畸变参数，直接返回
    if (k1_ == 0.0 && k2_ == 0.0 && p1_ == 0.0 && p2_ == 0.0)
    {
        return p_p;
    }

    cv::Mat mat(1, 2, CV_32F);
    mat.at<float>(0, 0) = static_cast<float>(p_p.x());
    mat.at<float>(0, 1) = static_cast<float>(p_p.y());

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K_cv(), D_cv(), cv::Mat(), K_cv());
    mat = mat.reshape(1);

    return Eigen::Vector2d(mat.at<float>(0, 0), mat.at<float>(0, 1));
}