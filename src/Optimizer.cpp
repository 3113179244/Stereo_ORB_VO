#include "Optimizer.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBextractor.h"

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/rotation.h>
#include <sophus/se3.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <mutex>

/**
 * @brief Ceres 迭代回调函数：用于在每次迭代结束时响应外部中断请求
 */
class AbortCallback : public ceres::IterationCallback
{
public:
    explicit AbortCallback(bool *pbStopFlag) : pbStopFlag_(pbStopFlag) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &) override
    {
        if (pbStopFlag_ && *pbStopFlag_)
        {
            return ceres::SOLVER_ABORT;
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    bool *pbStopFlag_;
};

/**
 * @brief Sophus SE(3) 自定义 Ceres Manifold (环境: Ceres >= 2.1)
 * 参数块布局 (Ambient Size = 7):
 *   - x[0..3]: Quaternion (x, y, z, w)
 *   - x[4..6]: Translation (tx, ty, tz)
 * 切空间布局 (Tangent Size = 6):
 *   - delta[0..2]: 旋转李代数 phi (omega)
 *   - delta[3..5]: 平移李代数 rho (v)
 * 更新模型 (左乘扰动): T_plus = exp([rho, phi]) * T
 */
class SophusSE3Manifold : public ceres::Manifold
{
public:
    ~SophusSE3Manifold() override = default;

    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    bool Plus(const double *x, const double *delta, double *x_plus) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q(x);
        Eigen::Map<const Eigen::Vector3d> t(x + 4);
        Sophus::SE3d T(q, t);

        // delta: [omega, v] -> Sophus xi: [v, omega]
        Eigen::Matrix<double, 6, 1> xi;
        xi << delta[3], delta[4], delta[5], delta[0], delta[1], delta[2];

        Sophus::SE3d T_plus = Sophus::SE3d::exp(xi) * T;

        Eigen::Map<Eigen::Quaterniond> q_plus(x_plus);
        Eigen::Map<Eigen::Vector3d> t_plus(x_plus + 4);

        q_plus = T_plus.unit_quaternion();
        t_plus = T_plus.translation();

        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        const double qx = x[0], qy = x[1], qz = x[2], qw = x[3];
        const double tx = x[4], ty = x[5], tz = x[6];

        // 1. 四元数关于旋转扰动 phi (4x3): q(+) = [0.5*phi, 1] * q
        J(0, 0) = 0.5 * qw;
        J(0, 1) = 0.5 * qz;
        J(0, 2) = -0.5 * qy;
        J(1, 0) = -0.5 * qz;
        J(1, 1) = 0.5 * qw;
        J(1, 2) = 0.5 * qx;
        J(2, 0) = 0.5 * qy;
        J(2, 1) = -0.5 * qx;
        J(2, 2) = 0.5 * qw;
        J(3, 0) = -0.5 * qx;
        J(3, 1) = -0.5 * qy;
        J(3, 2) = -0.5 * qz;

        // 2. 平移关于旋转扰动 phi (3x3): -[t]_x
        J(4, 0) = 0.0;
        J(4, 1) = tz;
        J(4, 2) = -ty;
        J(5, 0) = -tz;
        J(5, 1) = 0.0;
        J(5, 2) = tx;
        J(6, 0) = ty;
        J(6, 1) = -tx;
        J(6, 2) = 0.0;

        // 3. 平移关于平移扰动 rho (3x3): I
        J(4, 3) = 1.0;
        J(5, 4) = 1.0;
        J(6, 5) = 1.0;

        return true;
    }

    bool Minus(const double *y, const double *x, double *delta) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q_x(x), q_y(y);
        Eigen::Map<const Eigen::Vector3d> t_x(x + 4), t_y(y + 4);

        Sophus::SE3d T_x(q_x, t_x);
        Sophus::SE3d T_y(q_y, t_y);

        Sophus::SE3d T_delta = T_y * T_x.inverse();
        Eigen::Matrix<double, 6, 1> xi = T_delta.log();

        delta[0] = xi[3];
        delta[1] = xi[4];
        delta[2] = xi[5];
        delta[3] = xi[0];
        delta[4] = xi[1];
        delta[5] = xi[2];

        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override
    {
        double J_plus_arr[42];
        PlusJacobian(x, J_plus_arr);
        Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J_plus(J_plus_arr);

        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_minus(jacobian);
        J_minus = (J_plus.transpose() * J_plus).inverse() * J_plus.transpose();

        return true;
    }
};

/**
 * @brief 仅位姿优化 - 单目残差块 (Pose-Only Monocular)
 */
class PoseOnlyRepoErrorMono : public ceres::SizedCostFunction<2, 7>
{
public:
    PoseOnlyRepoErrorMono(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                          const Eigen::Matrix3d &K, double inv_sigma)
        : observed_(observed), point_3d_(point_3d),
          fx_(K(0, 0)), fy_(K(1, 1)), cx_(K(0, 2)), cy_(K(1, 2)),
          inv_sigma_(inv_sigma) {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
        Sophus::SE3d T_cw(q, t);

        const Eigen::Vector3d P_c = T_cw * point_3d_;
        const double X = P_c[0];
        const double Y = P_c[1];
        const double Z = P_c[2];

        if (Z <= 1e-4)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            if (jacobians && jacobians[0])
                std::fill(jacobians[0], jacobians[0] + 14, 0.0);
            return true;
        }

        const double inv_z = 1.0 / Z;
        const double inv_z2 = inv_z * inv_z;

        const double u = fx_ * X * inv_z + cx_;
        const double v = fy_ * Y * inv_z + cy_;

        residuals[0] = (u - observed_[0]) * inv_sigma_;
        residuals[1] = (v - observed_[1]) * inv_sigma_;

        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();

            Eigen::Matrix<double, 2, 3> J_proj;
            J_proj(0, 0) = fx_ * inv_z * inv_sigma_;
            J_proj(0, 1) = 0.0;
            J_proj(0, 2) = -fx_ * X * inv_z2 * inv_sigma_;

            J_proj(1, 0) = 0.0;
            J_proj(1, 1) = fy_ * inv_z * inv_sigma_;
            J_proj(1, 2) = -fy_ * Y * inv_z2 * inv_sigma_;

            const double Xw = point_3d_[0], Yw = point_3d_[1], Zw = point_3d_[2];
            const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

            Eigen::Matrix<double, 3, 4> dPc_dq;
            dPc_dq(0, 0) = 2.0 * (qy * Yw + qz * Zw);
            dPc_dq(0, 1) = 2.0 * (-2.0 * qy * Xw + qx * Yw + qw * Zw);
            dPc_dq(0, 2) = 2.0 * (-2.0 * qz * Xw - qw * Yw + qx * Zw);
            dPc_dq(0, 3) = 2.0 * (-qz * Yw + qy * Zw);

            dPc_dq(1, 0) = 2.0 * (qy * Xw - 2.0 * qx * Yw - qw * Zw);
            dPc_dq(1, 1) = 2.0 * (qx * Xw + qz * Zw);
            dPc_dq(1, 2) = 2.0 * (qw * Xw - 2.0 * qz * Yw + qy * Zw);
            dPc_dq(1, 3) = 2.0 * (qz * Xw - qx * Zw);

            dPc_dq(2, 0) = 2.0 * (qz * Xw + qw * Yw - 2.0 * qx * Zw);
            dPc_dq(2, 1) = 2.0 * (-qw * Xw + qz * Yw - 2.0 * qy * Zw);
            dPc_dq(2, 2) = 2.0 * (qx * Xw + qy * Yw);
            dPc_dq(2, 3) = 2.0 * (-qy * Xw + qx * Yw);

            J.block<2, 4>(0, 0) = J_proj * dPc_dq;
            J.block<2, 3>(0, 4) = J_proj;
        }

        return true;
    }

private:
    const Eigen::Vector2d observed_;
    const Eigen::Vector3d point_3d_;
    const double fx_, fy_, cx_, cy_, inv_sigma_;
};

/**
 * @brief 仅位姿优化 - 双目残差块 (Pose-Only Stereo)
 */
class PoseOnlyRepoErrorStereo : public ceres::SizedCostFunction<3, 7>
{
public:
    PoseOnlyRepoErrorStereo(const Eigen::Vector3d &observed, const Eigen::Vector3d &point_3d,
                            const Eigen::Matrix3d &K, double bf, double inv_sigma)
        : observed_(observed), point_3d_(point_3d),
          fx_(K(0, 0)), fy_(K(1, 1)), cx_(K(0, 2)), cy_(K(1, 2)),
          bf_(bf), inv_sigma_(inv_sigma) {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
        Sophus::SE3d T_cw(q, t);

        const Eigen::Vector3d P_c = T_cw * point_3d_;
        const double X = P_c[0];
        const double Y = P_c[1];
        const double Z = P_c[2];

        if (Z <= 1e-4)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            residuals[2] = 0.0;
            if (jacobians && jacobians[0])
                std::fill(jacobians[0], jacobians[0] + 21, 0.0);
            return true;
        }

        const double inv_z = 1.0 / Z;
        const double inv_z2 = inv_z * inv_z;

        const double u = fx_ * X * inv_z + cx_;
        const double v = fy_ * Y * inv_z + cy_;
        const double u_r = u - bf_ * inv_z;

        residuals[0] = (u - observed_[0]) * inv_sigma_;
        residuals[1] = (v - observed_[1]) * inv_sigma_;
        residuals[2] = (u_r - observed_[2]) * inv_sigma_;

        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J(jacobians[0]);
            J.setZero();

            Eigen::Matrix<double, 3, 3> J_proj;
            J_proj(0, 0) = fx_ * inv_z * inv_sigma_;
            J_proj(0, 1) = 0.0;
            J_proj(0, 2) = -fx_ * X * inv_z2 * inv_sigma_;

            J_proj(1, 0) = 0.0;
            J_proj(1, 1) = fy_ * inv_z * inv_sigma_;
            J_proj(1, 2) = -fy_ * Y * inv_z2 * inv_sigma_;

            J_proj(2, 0) = fx_ * inv_z * inv_sigma_;
            J_proj(2, 1) = 0.0;
            J_proj(2, 2) = -(fx_ * X - bf_) * inv_z2 * inv_sigma_;

            const double Xw = point_3d_[0], Yw = point_3d_[1], Zw = point_3d_[2];
            const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

            Eigen::Matrix<double, 3, 4> dPc_dq;
            dPc_dq(0, 0) = 2.0 * (qy * Yw + qz * Zw);
            dPc_dq(0, 1) = 2.0 * (-2.0 * qy * Xw + qx * Yw + qw * Zw);
            dPc_dq(0, 2) = 2.0 * (-2.0 * qz * Xw - qw * Yw + qx * Zw);
            dPc_dq(0, 3) = 2.0 * (-qz * Yw + qy * Zw);

            dPc_dq(1, 0) = 2.0 * (qy * Xw - 2.0 * qx * Yw - qw * Zw);
            dPc_dq(1, 1) = 2.0 * (qx * Xw + qz * Zw);
            dPc_dq(1, 2) = 2.0 * (qw * Xw - 2.0 * qz * Yw + qy * Zw);
            dPc_dq(1, 3) = 2.0 * (qz * Xw - qx * Zw);

            dPc_dq(2, 0) = 2.0 * (qz * Xw + qw * Yw - 2.0 * qx * Zw);
            dPc_dq(2, 1) = 2.0 * (-qw * Xw + qz * Yw - 2.0 * qy * Zw);
            dPc_dq(2, 2) = 2.0 * (qx * Xw + qy * Yw);
            dPc_dq(2, 3) = 2.0 * (-qy * Xw + qx * Yw);

            J.block<3, 4>(0, 0) = J_proj * dPc_dq;
            J.block<3, 3>(0, 4) = J_proj;
        }

        return true;
    }

private:
    const Eigen::Vector3d observed_;
    const Eigen::Vector3d point_3d_;
    const double fx_, fy_, cx_, cy_, bf_, inv_sigma_;
};

/**
 * @brief 联合 BA - 单目重投影误差 (位姿 7 维 + 地图点 3 维)
 */
class LocalRepoErrorAnalytic : public ceres::SizedCostFunction<2, 7, 3>
{
public:
    LocalRepoErrorAnalytic(double fx, double fy, double cx, double cy,
                           double u, double v, double sqrtInvSigma2)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          observed_(u, v), inv_sigma_(sqrtInvSigma2) {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
        Sophus::SE3d T_cw(q, t);
        Eigen::Map<const Eigen::Vector3d> point_3d(parameters[1]);

        const Eigen::Vector3d P_c = T_cw * point_3d;
        const double X = P_c[0];
        const double Y = P_c[1];
        const double Z = P_c[2];

        if (Z <= 1e-4)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            if (jacobians)
            {
                if (jacobians[0])
                    std::fill(jacobians[0], jacobians[0] + 14, 0.0);
                if (jacobians[1])
                    std::fill(jacobians[1], jacobians[1] + 6, 0.0);
            }
            return true;
        }

        const double inv_z = 1.0 / Z;
        const double inv_z2 = inv_z * inv_z;

        const double u = fx_ * X * inv_z + cx_;
        const double v = fy_ * Y * inv_z + cy_;

        residuals[0] = (u - observed_[0]) * inv_sigma_;
        residuals[1] = (v - observed_[1]) * inv_sigma_;

        Eigen::Matrix<double, 2, 3> J_proj;
        J_proj(0, 0) = fx_ * inv_z * inv_sigma_;
        J_proj(0, 1) = 0.0;
        J_proj(0, 2) = -fx_ * X * inv_z2 * inv_sigma_;

        J_proj(1, 0) = 0.0;
        J_proj(1, 1) = fy_ * inv_z * inv_sigma_;
        J_proj(1, 2) = -fy_ * Y * inv_z2 * inv_sigma_;

        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose.setZero();

            const double Xw = point_3d[0], Yw = point_3d[1], Zw = point_3d[2];
            const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

            Eigen::Matrix<double, 3, 4> dPc_dq;
            dPc_dq(0, 0) = 2.0 * (qy * Yw + qz * Zw);
            dPc_dq(0, 1) = 2.0 * (-2.0 * qy * Xw + qx * Yw + qw * Zw);
            dPc_dq(0, 2) = 2.0 * (-2.0 * qz * Xw - qw * Yw + qx * Zw);
            dPc_dq(0, 3) = 2.0 * (-qz * Yw + qy * Zw);

            dPc_dq(1, 0) = 2.0 * (qy * Xw - 2.0 * qx * Yw - qw * Zw);
            dPc_dq(1, 1) = 2.0 * (qx * Xw + qz * Zw);
            dPc_dq(1, 2) = 2.0 * (qw * Xw - 2.0 * qz * Yw + qy * Zw);
            dPc_dq(1, 3) = 2.0 * (qz * Xw - qx * Zw);

            dPc_dq(2, 0) = 2.0 * (qz * Xw + qw * Yw - 2.0 * qx * Zw);
            dPc_dq(2, 1) = 2.0 * (-qw * Xw + qz * Yw - 2.0 * qy * Zw);
            dPc_dq(2, 2) = 2.0 * (qx * Xw + qy * Yw);
            dPc_dq(2, 3) = 2.0 * (-qy * Xw + qx * Yw);

            J_pose.block<2, 4>(0, 0) = J_proj * dPc_dq;
            J_pose.block<2, 3>(0, 4) = J_proj;
        }

        if (jacobians && jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_point(jacobians[1]);
            J_point = J_proj * T_cw.rotationMatrix();
        }

        return true;
    }

private:
    const double fx_, fy_, cx_, cy_;
    const Eigen::Vector2d observed_;
    const double inv_sigma_;
};

/**
 * @brief 联合 BA - 双目重投影误差 (位姿 7 维 + 地图点 3 维)
 */
class LocalRepoErrorStereoAnalytic : public ceres::SizedCostFunction<3, 7, 3>
{
public:
    LocalRepoErrorStereoAnalytic(double fx, double fy, double cx, double cy, double mbf,
                                 double u, double v, double u_r, double sqrtInvSigma2)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), bf_(mbf),
          observed_(u, v, u_r), inv_sigma_(sqrtInvSigma2) {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
        Sophus::SE3d T_cw(q, t);
        Eigen::Map<const Eigen::Vector3d> point_3d(parameters[1]);

        const Eigen::Vector3d P_c = T_cw * point_3d;
        const double X = P_c[0];
        const double Y = P_c[1];
        const double Z = P_c[2];

        if (Z <= 1e-4)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            residuals[2] = 0.0;
            if (jacobians)
            {
                if (jacobians[0])
                    std::fill(jacobians[0], jacobians[0] + 21, 0.0);
                if (jacobians[1])
                    std::fill(jacobians[1], jacobians[1] + 9, 0.0);
            }
            return true;
        }

        const double inv_z = 1.0 / Z;
        const double inv_z2 = inv_z * inv_z;

        const double u = fx_ * X * inv_z + cx_;
        const double v = fy_ * Y * inv_z + cy_;
        const double u_r = u - bf_ * inv_z;

        residuals[0] = (u - observed_[0]) * inv_sigma_;
        residuals[1] = (v - observed_[1]) * inv_sigma_;
        residuals[2] = (u_r - observed_[2]) * inv_sigma_;

        Eigen::Matrix<double, 3, 3> J_proj;
        J_proj(0, 0) = fx_ * inv_z * inv_sigma_;
        J_proj(0, 1) = 0.0;
        J_proj(0, 2) = -fx_ * X * inv_z2 * inv_sigma_;

        J_proj(1, 0) = 0.0;
        J_proj(1, 1) = fy_ * inv_z * inv_sigma_;
        J_proj(1, 2) = -fy_ * Y * inv_z2 * inv_sigma_;

        J_proj(2, 0) = fx_ * inv_z * inv_sigma_;
        J_proj(2, 1) = 0.0;
        J_proj(2, 2) = -(fx_ * X - bf_) * inv_z2 * inv_sigma_;

        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose.setZero();

            const double Xw = point_3d[0], Yw = point_3d[1], Zw = point_3d[2];
            const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

            Eigen::Matrix<double, 3, 4> dPc_dq;
            dPc_dq(0, 0) = 2.0 * (qy * Yw + qz * Zw);
            dPc_dq(0, 1) = 2.0 * (-2.0 * qy * Xw + qx * Yw + qw * Zw);
            dPc_dq(0, 2) = 2.0 * (-2.0 * qz * Xw - qw * Yw + qx * Zw);
            dPc_dq(0, 3) = 2.0 * (-qz * Yw + qy * Zw);

            dPc_dq(1, 0) = 2.0 * (qy * Xw - 2.0 * qx * Yw - qw * Zw);
            dPc_dq(1, 1) = 2.0 * (qx * Xw + qz * Zw);
            dPc_dq(1, 2) = 2.0 * (qw * Xw - 2.0 * qz * Yw + qy * Zw);
            dPc_dq(1, 3) = 2.0 * (qz * Xw - qx * Zw);

            dPc_dq(2, 0) = 2.0 * (qz * Xw + qw * Yw - 2.0 * qx * Zw);
            dPc_dq(2, 1) = 2.0 * (-qw * Xw + qz * Yw - 2.0 * qy * Zw);
            dPc_dq(2, 2) = 2.0 * (qx * Xw + qy * Yw);
            dPc_dq(2, 3) = 2.0 * (-qy * Xw + qx * Yw);

            J_pose.block<3, 4>(0, 0) = J_proj * dPc_dq;
            J_pose.block<3, 3>(0, 4) = J_proj;
        }

        if (jacobians && jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_point(jacobians[1]);
            J_point = J_proj * T_cw.rotationMatrix();
        }

        return true;
    }

private:
    const double fx_, fy_, cx_, cy_, bf_;
    const Eigen::Vector3d observed_;
    const double inv_sigma_;
};

/**
 * @brief 位姿图 SE3 解析求导残差块 (7 维位姿参数块版)
 */
class PoseGraphSE3Analytic : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    explicit PoseGraphSE3Analytic(const Sophus::SE3d &T_ij_meas)
        : T_ij_meas_(T_ij_meas) {}

    static Eigen::Matrix<double, 6, 6> curlyHat(const Eigen::Matrix<double, 6, 1> &xi)
    {
        Eigen::Vector3d w = xi.head<3>();
        Eigen::Vector3d v = xi.tail<3>();

        Eigen::Matrix<double, 6, 6> ad = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix3d w_hat = Sophus::SO3d::hat(w);
        Eigen::Matrix3d v_hat = Sophus::SO3d::hat(v);

        ad.block<3, 3>(0, 0) = w_hat;
        ad.block<3, 3>(3, 0) = v_hat;
        ad.block<3, 3>(3, 3) = w_hat;
        return ad;
    }

    static Eigen::Matrix<double, 6, 6> JrInvSE3(const Eigen::Matrix<double, 6, 1> &xi)
    {
        Eigen::Matrix<double, 6, 6> ad = curlyHat(xi);
        return Eigen::Matrix<double, 6, 6>::Identity() + 0.5 * ad + (1.0 / 12.0) * (ad * ad);
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q_i(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t_i(parameters[0] + 4);
        Sophus::SE3d T_iw(q_i, t_i);

        Eigen::Map<const Eigen::Quaterniond> q_j(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> t_j(parameters[1] + 4);
        Sophus::SE3d T_jw(q_j, t_j);

        Sophus::SE3d T_ij_est = T_iw * T_jw.inverse();
        Sophus::SE3d error_SE3 = T_ij_meas_ * T_ij_est.inverse();

        Eigen::Matrix<double, 6, 1> sophus_xi = error_SE3.log();

        Eigen::Matrix<double, 6, 1> error_vec;
        error_vec.head<3>() = sophus_xi.tail<3>(); // omega
        error_vec.tail<3>() = sophus_xi.head<3>(); // v

        for (int k = 0; k < 6; ++k)
            residuals[k] = error_vec[k];

        if (jacobians)
        {
            auto ComputePinvJPlus = [](const double *param) -> Eigen::Matrix<double, 6, 7>
            {
                const double qx = param[0], qy = param[1], qz = param[2], qw = param[3];
                const double tx = param[4], ty = param[5], tz = param[6];

                Eigen::Matrix<double, 6, 7> pinv;
                pinv.setZero();

                pinv(0, 0) = 2.0 * qw;
                pinv(0, 1) = -2.0 * qz;
                pinv(0, 2) = 2.0 * qy;
                pinv(0, 3) = -2.0 * qx;
                pinv(1, 0) = 2.0 * qz;
                pinv(1, 1) = 2.0 * qw;
                pinv(1, 2) = -2.0 * qx;
                pinv(1, 3) = -2.0 * qy;
                pinv(2, 0) = -2.0 * qy;
                pinv(2, 1) = 2.0 * qx;
                pinv(2, 2) = 2.0 * qw;
                pinv(2, 3) = -2.0 * qz;

                pinv.block<3, 3>(3, 4) = Eigen::Matrix3d::Identity();
                pinv.block<3, 4>(3, 0) = Sophus::SO3d::hat(Eigen::Vector3d(tx, ty, tz)) * pinv.block<3, 4>(0, 0);

                return pinv;
            };

            Eigen::Matrix<double, 6, 6> J_r_inv = JrInvSE3(error_vec);

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_i(jacobians[0]);
                J_i = -J_r_inv * ComputePinvJPlus(parameters[0]);
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_j(jacobians[1]);
                const Eigen::Matrix3d R_est = T_ij_est.rotationMatrix();
                const Eigen::Vector3d t_est = T_ij_est.translation();

                Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Zero();
                Adj.block<3, 3>(0, 0) = R_est;
                Adj.block<3, 3>(3, 0) = Sophus::SO3d::hat(t_est) * R_est;
                Adj.block<3, 3>(3, 3) = R_est;

                J_j = (J_r_inv * Adj) * ComputePinvJPlus(parameters[1]);
            }
        }
        return true;
    }

private:
    Sophus::SE3d T_ij_meas_;
};

int Optimizer::PoseOptimization(Frame *pFrame)
{
    if (!pFrame)
        return 0;

    const int N = pFrame->mvpMapPoints.size();
    pFrame->mvbOutlier.assign(N, false);

    int nInitialCorrespondences = 0;
    for (int i = 0; i < N; ++i)
    {
        if (pFrame->mvpMapPoints[i])
            nInitialCorrespondences++;
    }

    if (nInitialCorrespondences < 3)
        return 0;

    Eigen::Matrix3d K;
    K << pFrame->mK.at<float>(0, 0), 0.0, pFrame->mK.at<float>(0, 2),
        0.0, pFrame->mK.at<float>(1, 1), pFrame->mK.at<float>(1, 2),
        0.0, 0.0, 1.0;

    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);
    const double mbf = pFrame->mbf;

    Eigen::Matrix3d R_cw = pFrame->mTcw.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d t_cw = pFrame->mTcw.block<3, 1>(0, 3).cast<double>();
    Eigen::Quaterniond q_cw(R_cw);
    q_cw.normalize();

    // 状态量: [qx, qy, qz, qw, tx, ty, tz]
    double pose_param[7] = {q_cw.x(), q_cw.y(), q_cw.z(), q_cw.w(), t_cw.x(), t_cw.y(), t_cw.z()};

    const int its[4] = {10, 10, 10, 10};
    const double chi2_mono = 5.991;
    const double chi2_stereo = 7.815;

    int num_inliers = 0;
    std::vector<Eigen::Vector3d> vPoints3D(N, Eigen::Vector3d::Zero());
    std::vector<bool> vbValidMP(N, false);

    for (int i = 0; i < N; ++i)
    {
        MapPoint *pMP = pFrame->mvpMapPoints[i];
        if (pMP && !pMP->isBad())
        {
            vPoints3D[i] = pMP->GetWorldPos().cast<double>();
            vbValidMP[i] = true;
        }
    }

    for (int it = 0; it < 4; ++it)
    {
        ceres::Problem problem;
        problem.AddParameterBlock(pose_param, 7, new SophusSE3Manifold());

        for (int i = 0; i < N; ++i)
        {
            if (!vbValidMP[i] || pFrame->mvbOutlier[i])
                continue;

            const Eigen::Vector3d &P_w = vPoints3D[i];
            const int level = pFrame->mvKeysUn[i].octave;
            const double inv_sigma = 1.0 / std::sqrt(pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level]);
            const float u_r = pFrame->mvuRight[i];
            const float depth = pFrame->mvDepth[i];

            if (u_r < 0.0f || depth >= pFrame->mThDepth || depth <= 0.0f)
            {
                Eigen::Vector2d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y);
                ceres::CostFunction *cost_function = new PoseOnlyRepoErrorMono(obs, P_w, K, inv_sigma);
                ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_mono)) : nullptr;
                problem.AddResidualBlock(cost_function, loss_function, pose_param);
            }
            else
            {
                Eigen::Vector3d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y, u_r);
                ceres::CostFunction *cost_function = new PoseOnlyRepoErrorStereo(obs, P_w, K, mbf, inv_sigma);
                ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_stereo)) : nullptr;
                problem.AddResidualBlock(cost_function, loss_function, pose_param);
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = its[it];
        options.num_threads = 1;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond q_curr(pose_param[3], pose_param[0], pose_param[1], pose_param[2]);
        q_curr.normalize();
        Sophus::SE3d T_curr(q_curr, Eigen::Vector3d(pose_param[4], pose_param[5], pose_param[6]));

        num_inliers = 0;
        for (int i = 0; i < N; ++i)
        {
            if (!vbValidMP[i])
                continue;

            const Eigen::Vector3d &P_w = vPoints3D[i];
            Eigen::Vector3d P_c = T_curr * P_w;
            const double depth = P_c[2];

            if (depth <= 0.0)
            {
                pFrame->mvbOutlier[i] = true;
                continue;
            }

            const double inv_z = 1.0 / depth;
            const double u = fx * P_c[0] * inv_z + cx;
            const double v = fy * P_c[1] * inv_z + cy;
            const int level = pFrame->mvKeysUn[i].octave;
            const double inv_sigma2 = 1.0 / pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level];
            const float u_r = pFrame->mvuRight[i];

            const double du = u - pFrame->mvKeysUn[i].pt.x;
            const double dv = v - pFrame->mvKeysUn[i].pt.y;

            if (u_r < 0.0f || depth >= pFrame->mThDepth)
            {
                const double chi2 = (du * du + dv * dv) * inv_sigma2;
                if (chi2 > chi2_mono)
                {
                    pFrame->mvbOutlier[i] = true;
                }
                else
                {
                    pFrame->mvbOutlier[i] = false;
                    num_inliers++;
                }
            }
            else
            {
                const double u_r_proj = u - mbf * inv_z;
                const double du_r = u_r_proj - u_r;
                const double chi2 = (du * du + dv * dv + du_r * du_r) * inv_sigma2;

                if (chi2 > chi2_stereo)
                {
                    pFrame->mvbOutlier[i] = true;
                }
                else
                {
                    pFrame->mvbOutlier[i] = false;
                    num_inliers++;
                }
            }
        }
    }

    Eigen::Quaterniond q_res(pose_param[3], pose_param[0], pose_param[1], pose_param[2]);
    q_res.normalize();
    Sophus::SE3d T_cw_normalized(q_res, Eigen::Vector3d(pose_param[4], pose_param[5], pose_param[6]));
    pFrame->SetPose(T_cw_normalized.matrix().cast<float>());

    return num_inliers;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pCurKF, bool *pbStopFlag, std::shared_ptr<Map> pMap)
{
    if (!pCurKF || !pMap || pCurKF->mbBad)
        return;

    std::vector<KeyFrame *> vpLocalKFs;
    vpLocalKFs.push_back(pCurKF);
    {
        std::vector<KeyFrame *> vNeigh = pCurKF->GetBestCovisibilityKeyFrames(10);
        for (size_t i = 0; i < vNeigh.size(); ++i)
        {
            if (vNeigh[i] && !vNeigh[i]->mbBad)
                vpLocalKFs.push_back(vNeigh[i]);
        }
    }

    std::vector<KeyFrame *> vpFixedKFs;
    for (size_t i = 0; i < vpLocalKFs.size(); ++i)
    {
        std::vector<KeyFrame *> vNeigh = vpLocalKFs[i]->GetBestCovisibilityKeyFrames(10);
        for (size_t j = 0; j < vNeigh.size(); ++j)
        {
            KeyFrame *pKF = vNeigh[j];
            if (!pKF || pKF->mbBad)
                continue;
            if (std::find(vpLocalKFs.begin(), vpLocalKFs.end(), pKF) != vpLocalKFs.end())
                continue;
            if (std::find(vpFixedKFs.begin(), vpFixedKFs.end(), pKF) != vpFixedKFs.end())
                continue;
            vpFixedKFs.push_back(pKF);
        }
    }

    std::vector<MapPoint *> vpLocalMPs;
    std::set<MapPoint *> sLocalMPs;
    for (size_t i = 0; i < vpLocalKFs.size(); ++i)
    {
        std::vector<MapPoint *> vpMPs = vpLocalKFs[i]->GetMapPointMatches();
        for (size_t j = 0; j < vpMPs.size(); ++j)
        {
            MapPoint *pMP = vpMPs[j];
            if (pMP && !pMP->isBad() && pMP->GetObservations().size() >= 2 && !sLocalMPs.count(pMP))
            {
                sLocalMPs.insert(pMP);
                vpLocalMPs.push_back(pMP);
            }
        }
    }

    if (vpLocalKFs.size() < 2 || vpLocalMPs.size() < 5)
        return;

    std::map<KeyFrame *, std::vector<double>> mapKF_Params;
    auto InitPoseParam = [&](KeyFrame *pKF)
    {
        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();
        mapKF_Params[pKF] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};
    };

    for (KeyFrame *pKF : vpLocalKFs)
        InitPoseParam(pKF);
    for (KeyFrame *pKF : vpFixedKFs)
        InitPoseParam(pKF);

    std::map<MapPoint *, Eigen::Vector3d> mapMP_Point;
    for (MapPoint *pMP : vpLocalMPs)
    {
        mapMP_Point[pMP] = pMP->GetWorldPos().cast<double>();
    }

    ceres::Problem problem;

    for (KeyFrame *pKF : vpLocalKFs)
    {
        problem.AddParameterBlock(mapKF_Params[pKF].data(), 7, new SophusSE3Manifold());
        if (pKF->mnId == 0)
            problem.SetParameterBlockConstant(mapKF_Params[pKF].data());
    }

    for (KeyFrame *pKF : vpFixedKFs)
    {
        problem.AddParameterBlock(mapKF_Params[pKF].data(), 7, new SophusSE3Manifold());
        problem.SetParameterBlockConstant(mapKF_Params[pKF].data());
    }

    for (MapPoint *pMP : vpLocalMPs)
    {
        problem.AddParameterBlock(mapMP_Point[pMP].data(), 3);
    }

    struct ObservationInfo
    {
        KeyFrame *pKF;
        MapPoint *pMP;
        size_t featIdx;
        ceres::ResidualBlockId resId;
        bool isStereo;
    };
    std::vector<ObservationInfo> vObsInfo;

    auto AddObservations = [&](const std::vector<KeyFrame *> &vpKFs)
    {
        for (KeyFrame *pKF : vpKFs)
        {
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
            double *pose_param_ptr = mapKF_Params[pKF].data();
            for (size_t j = 0; j < vpMPs.size(); ++j)
            {
                MapPoint *pMP = vpMPs[j];
                if (!pMP || pMP->isBad() || !mapMP_Point.count(pMP))
                    continue;

                const cv::KeyPoint &kp = pKF->mvKeysUn[j];
                const int level = kp.octave;
                if (level < 0 || level >= pKF->mnScaleLevels)
                    continue;

                const double invSigma2 = pKF->mvInvLevelSigma2[level];
                const double sqrtInvSigma2 = std::sqrt(invSigma2);
                const float u_r = pKF->mvuRight[j];
                const float depth = pKF->mvDepth[j];

                if (u_r >= 0.0f && depth > 0.0f && depth < pKF->mThDepth)
                {
                    ceres::CostFunction *cost = new LocalRepoErrorStereoAnalytic(
                        fx, fy, cx, cy, pKF->mbf, kp.pt.x, kp.pt.y, u_r, sqrtInvSigma2);

                    ceres::ResidualBlockId id = problem.AddResidualBlock(
                        cost, new ceres::HuberLoss(std::sqrt(7.815)), pose_param_ptr, mapMP_Point[pMP].data());

                    vObsInfo.push_back({pKF, pMP, j, id, true});
                }
                else
                {
                    ceres::CostFunction *cost = new LocalRepoErrorAnalytic(
                        fx, fy, cx, cy, kp.pt.x, kp.pt.y, sqrtInvSigma2);

                    ceres::ResidualBlockId id = problem.AddResidualBlock(
                        cost, new ceres::HuberLoss(std::sqrt(5.991)), pose_param_ptr, mapMP_Point[pMP].data());

                    vObsInfo.push_back({pKF, pMP, j, id, false});
                }
            }
        }
    };

    AddObservations(vpLocalKFs);
    AddObservations(vpFixedKFs);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-4;

    AbortCallback callback(pbStopFlag);
    if (pbStopFlag)
        options.callbacks.push_back(&callback);

    options.max_num_iterations = 5;
    ceres::Solver::Summary summary1;
    ceres::Solve(options, &problem, &summary1);

    if (pbStopFlag && *pbStopFlag)
        return;

    const double chi2_mono = 5.991;
    const double chi2_stereo = 7.815;
    std::vector<std::pair<KeyFrame *, MapPoint *>> vToEraseObservations;

    for (auto &info : vObsInfo)
    {
        if (!info.resId)
            continue;

        KeyFrame *pKF = info.pKF;
        MapPoint *pMP = info.pMP;
        const size_t idx = info.featIdx;

        const auto &param = mapKF_Params[pKF];
        Eigen::Quaterniond q_curr(param[3], param[0], param[1], param[2]);
        Eigen::Vector3d t_curr(param[4], param[5], param[6]);
        Sophus::SE3d T_cw_curr(q_curr.normalized(), t_curr);

        Eigen::Vector3d Pc = T_cw_curr * mapMP_Point[pMP];
        if (Pc.z() <= 0.0)
        {
            vToEraseObservations.push_back({pKF, pMP});
            problem.RemoveResidualBlock(info.resId);
            info.resId = nullptr;
            continue;
        }

        const double invz = 1.0 / Pc.z();
        const double u = pKF->fx * Pc.x() * invz + pKF->cx;
        const double v = pKF->fy * Pc.y() * invz + pKF->cy;
        const double du = u - pKF->mvKeysUn[idx].pt.x;
        const double dv = v - pKF->mvKeysUn[idx].pt.y;

        const int level = pKF->mvKeysUn[idx].octave;
        const double invSigma2 = (level >= 0 && level < pKF->mnScaleLevels) ? pKF->mvInvLevelSigma2[level] : 1.0;

        bool isOutlier = false;
        if (info.isStereo)
        {
            const double u_r_proj = u - pKF->mbf * invz;
            const double du_r = u_r_proj - pKF->mvuRight[idx];
            if ((du * du + dv * dv + du_r * du_r) * invSigma2 > chi2_stereo)
                isOutlier = true;
        }
        else
        {
            if ((du * du + dv * dv) * invSigma2 > chi2_mono)
                isOutlier = true;
        }

        if (isOutlier)
        {
            vToEraseObservations.push_back({pKF, pMP});
            problem.RemoveResidualBlock(info.resId);
            info.resId = nullptr;
        }
    }

    options.max_num_iterations = 10;
    ceres::Solver::Summary summary2;
    ceres::Solve(options, &problem, &summary2);

    if (pbStopFlag && *pbStopFlag)
        return;

    {
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        for (size_t i = 0; i < vToEraseObservations.size(); ++i)
        {
            KeyFrame *pKF = vToEraseObservations[i].first;
            MapPoint *pMP = vToEraseObservations[i].second;
            if (pKF && pMP)
            {
                pKF->EraseMapPointMatch(pMP);
                pMP->EraseObservation(pKF);
            }
        }

        for (KeyFrame *pKF : vpLocalKFs)
        {
            const auto &param = mapKF_Params[pKF];
            Eigen::Quaterniond q(param[3], param[0], param[1], param[2]);
            q.normalize();
            Eigen::Vector3d t(param[4], param[5], param[6]);
            Sophus::SE3d T_opt(q, t);
            pKF->SetPose(T_opt.matrix().cast<float>());
        }

        for (MapPoint *pMP : vpLocalMPs)
        {
            if (pMP->isBad())
                continue;
            pMP->SetWorldPos(mapMP_Point[pMP].cast<float>());
            pMP->UpdateNormalAndDepth();
        }
    }
}

void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                       const std::map<KeyFrame *, Eigen::Matrix4f> &NonCorrectedPoses,
                                       const std::map<KeyFrame *, Eigen::Matrix4f> &CorrectedPoses,
                                       const std::map<KeyFrame *, std::set<KeyFrame *>> &LoopConnections)
{
    if (!pMap || !pLoopKF || !pCurKF)
        return;

    std::vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const int N = vpKFs.size();
    if (N < 2)
        return;

    ceres::Problem problem;

    std::map<KeyFrame *, std::vector<double>> mapKF_Params;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        Eigen::Matrix4f Tcw;
        auto itCorrected = CorrectedPoses.find(pKF);
        if (itCorrected != CorrectedPoses.end())
            Tcw = itCorrected->second;
        else
            Tcw = pKF->GetPose();

        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();

        mapKF_Params[pKF] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};
        problem.AddParameterBlock(mapKF_Params[pKF].data(), 7, new SophusSE3Manifold());

        // 固定闭环帧 (参考帧)
        if (pKF == pLoopKF)
            problem.SetParameterBlockConstant(mapKF_Params[pKF].data());
    }

    std::map<KeyFrame *, Sophus::SE3d> mapOriginalPoses;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        Eigen::Matrix4f Tcw;
        auto itNonCorrected = NonCorrectedPoses.find(pKF);
        if (itNonCorrected != NonCorrectedPoses.end())
            Tcw = itNonCorrected->second;
        else
            Tcw = pKF->GetPose();

        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        mapOriginalPoses[pKF] = Sophus::SE3d(Eigen::Quaterniond(R).normalized(), t);
    }

    auto GetCorrectedPose = [&](KeyFrame *pKF) -> Sophus::SE3d {
        auto it = CorrectedPoses.find(pKF);
        if (it != CorrectedPoses.end())
        {
            const Eigen::Matrix4f &T = it->second;
            return Sophus::SE3d(Eigen::Quaterniond(T.block<3, 3>(0, 0).cast<double>()).normalized(),
                                T.block<3, 1>(0, 3).cast<double>());
        }
        return mapOriginalPoses[pKF];
    };

    std::set<std::pair<KeyFrame *, KeyFrame *>> sInsertedEdges;

    // ==================== 1. 添加当前核心回环边 (优先级最高) ====================
    if (pCurKF != pLoopKF && mapKF_Params.count(pCurKF) && mapKF_Params.count(pLoopKF))
    {
        Sophus::SE3d T_cur_corr = GetCorrectedPose(pCurKF);
        Sophus::SE3d T_loop_corr = GetCorrectedPose(pLoopKF);
        Sophus::SE3d T_cur_loop_meas = T_cur_corr * T_loop_corr.inverse();

        ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_cur_loop_meas);
        problem.AddResidualBlock(cost, nullptr, mapKF_Params[pCurKF].data(), mapKF_Params[pLoopKF].data());
        sInsertedEdges.insert(std::make_pair(std::min(pCurKF, pLoopKF), std::max(pCurKF, pLoopKF)));
    }

    // ==================== 2. 添加回环新增的关联边 LoopConnections ====================
    for (const auto &mit : LoopConnections)
    {
        KeyFrame *pKF = mit.first;
        if (!pKF || pKF->mbBad || !mapKF_Params.count(pKF))
            continue;

        const std::set<KeyFrame *> &sLoopNeighbors = mit.second;
        for (KeyFrame *pLKF : sLoopNeighbors)
        {
            if (!pLKF || pLKF->mbBad || !mapKF_Params.count(pLKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pLKF), std::max(pKF, pLKF));
            if (sInsertedEdges.count(edgePair))
                continue;

            // 闭环新增连接必须使用校正后的姿态作为测量值！
            Sophus::SE3d T_i_corr = GetCorrectedPose(pKF);
            Sophus::SE3d T_j_corr = GetCorrectedPose(pLKF);
            Sophus::SE3d T_ij_meas = T_i_corr * T_j_corr.inverse();

            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_ij_meas);
            problem.AddResidualBlock(cost, nullptr, mapKF_Params[pKF].data(), mapKF_Params[pLKF].data());
            sInsertedEdges.insert(edgePair);
        }
    }

    // ==================== 3. 添加生成树边 (Spanning Tree) ====================
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        KeyFrame *pParent = pKF->GetParent();
        if (pParent && !pParent->mbBad && mapKF_Params.count(pParent) && mapKF_Params.count(pKF))
        {
            auto edgePair = std::make_pair(std::min(pKF, pParent), std::max(pKF, pParent));
            if (!sInsertedEdges.count(edgePair))
            {
                Sophus::SE3d T_child_parent_meas = mapOriginalPoses[pKF] * mapOriginalPoses[pParent].inverse();
                ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_child_parent_meas);
                problem.AddResidualBlock(cost, nullptr, mapKF_Params[pKF].data(), mapKF_Params[pParent].data());
                sInsertedEdges.insert(edgePair);
            }
        }

        // 历史回环边 (如果有)
        const std::set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (KeyFrame *pLKF : sLoopEdges)
        {
            if (!pLKF || pLKF->mbBad || !mapKF_Params.count(pLKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pLKF), std::max(pKF, pLKF));
            if (sInsertedEdges.count(edgePair))
                continue;

            // 历史回环边统一使用校正后相对位姿
            Sophus::SE3d T_ij_meas = GetCorrectedPose(pKF) * GetCorrectedPose(pLKF).inverse();
            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_ij_meas);
            problem.AddResidualBlock(cost, nullptr, mapKF_Params[pKF].data(), mapKF_Params[pLKF].data());
            sInsertedEdges.insert(edgePair);
        }
    }

    // ==================== 4. 添加高共视边 (权重 >= 100，排斥回环连接) ====================
    const int minWeight = 100;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad || !mapKF_Params.count(pKF))
            continue;

        const std::vector<KeyFrame *> vpCovKFs = pKF->GetCovisibleByWeight(minWeight);
        for (KeyFrame *pCovKF : vpCovKFs)
        {
            if (!pCovKF || pCovKF->mbBad || !mapKF_Params.count(pCovKF))
                continue;

            // 显式检查：如果该边属于回环连接，绝对不能用未校正位姿！
            if (LoopConnections.count(pKF) && LoopConnections.at(pKF).count(pCovKF))
                continue;
            if (LoopConnections.count(pCovKF) && LoopConnections.at(pCovKF).count(pKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pCovKF), std::max(pKF, pCovKF));
            if (sInsertedEdges.count(edgePair))
                continue;

            // 纯历史内部共视边，可以使用原始相对位姿约束刚度
            Sophus::SE3d T_ij_meas = mapOriginalPoses[pKF] * mapOriginalPoses[pCovKF].inverse();
            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_ij_meas);
            problem.AddResidualBlock(cost, nullptr, mapKF_Params[pKF].data(), mapKF_Params[pCovKF].data());
            sInsertedEdges.insert(edgePair);
        }
    }

    // ==================== 5. 求解与地图点位姿更新 ====================
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    {
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);
        std::map<KeyFrame *, Sophus::SE3d> mapNewPoses;
        for (auto &kv : mapKF_Params)
        {
            KeyFrame *pKF = kv.first;
            const auto &param = kv.second;
            Eigen::Quaterniond q(param[3], param[0], param[1], param[2]);
            q.normalize();
            Sophus::SE3d T_new(q, Eigen::Vector3d(param[4], param[5], param[6]));

            pKF->SetPose(T_new.matrix().cast<float>());
            mapNewPoses[pKF] = T_new;
        }

        std::vector<MapPoint *> vpAllMPs = pMap->GetAllMapPoints();
        for (MapPoint *pMP : vpAllMPs)
        {
            if (!pMP || pMP->isBad())
                continue;

            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            if (!pRefKF || !mapNewPoses.count(pRefKF))
                continue;

            // 对于已经经过粗校正的点，从初始校正位姿换算至优化后位姿；其余点从旧位姿换算
            Sophus::SE3d T_ref_before_opt = (pMP->mnCorrectedByKF == pCurKF->mnId)
                                                ? GetCorrectedPose(pRefKF)
                                                : mapOriginalPoses[pRefKF];
            Sophus::SE3d T_new_cw = mapNewPoses[pRefKF];

            Eigen::Vector3d Pw_old = pMP->GetWorldPos().cast<double>();
            Eigen::Vector3d Pc = T_ref_before_opt * Pw_old;
            Eigen::Vector3d Pw_new = T_new_cw.inverse() * Pc;

            pMP->SetWorldPos(Pw_new.cast<float>());
            pMP->UpdateNormalAndDepth();
        }
    }
}

void Optimizer::GlobalBundleAdjustment(Map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRunGBA)
{
    if (!pMap)
        return;

    std::vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    std::vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();
    if (vpKFs.size() < 2 || vpMPs.empty())
        return;

    std::vector<KeyFrame *> vpKFsToOptimize;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;
        if (bRunGBA && pKF->mnId > nLoopKF)
            continue;
        vpKFsToOptimize.push_back(pKF);
    }

    std::map<KeyFrame *, std::vector<double>> mapKF_Params;
    std::map<KeyFrame *, Sophus::SE3d> mapPosesBeforeGBA;
    ceres::Problem problem;

    for (KeyFrame *pKF : vpKFsToOptimize)
    {
        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();

        mapPosesBeforeGBA[pKF] = Sophus::SE3d(q, t);
        mapKF_Params[pKF] = {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()};

        problem.AddParameterBlock(mapKF_Params[pKF].data(), 7, new SophusSE3Manifold());
        if (pKF->mnId == 0)
            problem.SetParameterBlockConstant(mapKF_Params[pKF].data());
    }

    std::map<MapPoint *, Eigen::Vector3d> mapMP_Point;
    for (MapPoint *pMP : vpMPs)
    {
        if (!pMP || pMP->isBad() || pMP->GetObservations().size() < 3)
            continue;
        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
        if (bRunGBA && pRefKF && pRefKF->mnId > nLoopKF)
            continue;

        mapMP_Point[pMP] = pMP->GetWorldPos().cast<double>();
        problem.AddParameterBlock(mapMP_Point[pMP].data(), 3);
    }

    for (KeyFrame *pKF : vpKFsToOptimize)
    {
        const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
        const std::vector<MapPoint *> vpMatches = pKF->GetMapPointMatches();
        double *pose_param = mapKF_Params[pKF].data();

        for (size_t i = 0; i < vpMatches.size(); ++i)
        {
            MapPoint *pMP = vpMatches[i];
            if (!pMP || pMP->isBad() || !mapMP_Point.count(pMP))
                continue;

            const cv::KeyPoint &kp = pKF->mvKeysUn[i];
            const int level = kp.octave;
            if (level < 0 || level >= pKF->mnScaleLevels)
                continue;

            const double invSigma2 = pKF->mvInvLevelSigma2[level];
            const double sqrtInvSigma2 = std::sqrt(invSigma2);
            const float u_r = pKF->mvuRight[i];
            const float depth = pKF->mvDepth[i];

            ceres::CostFunction *cost = nullptr;
            if (u_r >= 0.0f && depth > 0.0f && depth < pKF->mThDepth)
            {
                cost = new LocalRepoErrorStereoAnalytic(fx, fy, cx, cy, pKF->mbf,
                                                        kp.pt.x, kp.pt.y, u_r, sqrtInvSigma2);
                problem.AddResidualBlock(cost, new ceres::HuberLoss(std::sqrt(7.815)), pose_param, mapMP_Point[pMP].data());
            }
            else
            {
                cost = new LocalRepoErrorAnalytic(fx, fy, cx, cy,
                                                  kp.pt.x, kp.pt.y, sqrtInvSigma2);
                problem.AddResidualBlock(cost, new ceres::HuberLoss(std::sqrt(5.991)), pose_param, mapMP_Point[pMP].data());
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = nIterations;
    options.num_threads = 1;

    AbortCallback callback(pbStopFlag);
    if (pbStopFlag)
        options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (pbStopFlag && *pbStopFlag)
        return;

    {
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        std::map<KeyFrame *, Sophus::SE3d> mapNewPoses;
        for (auto &kv : mapKF_Params)
        {
            KeyFrame *pKF = kv.first;
            const auto &param = kv.second;
            Eigen::Quaterniond q(param[3], param[0], param[1], param[2]);
            q.normalize();
            Sophus::SE3d T_new(q, Eigen::Vector3d(param[4], param[5], param[6]));

            pKF->SetPose(T_new.matrix().cast<float>());
            mapNewPoses[pKF] = T_new;
        }

        for (auto &kv : mapMP_Point)
        {
            MapPoint *pMP = kv.first;
            if (pMP->isBad())
                continue;
            pMP->SetWorldPos(kv.second.cast<float>());
            pMP->UpdateNormalAndDepth();
        }

        if (bRunGBA)
        {
            std::vector<KeyFrame *> vpAllKFs = pMap->GetAllKeyFrames();
            std::sort(vpAllKFs.begin(), vpAllKFs.end(), [](KeyFrame *a, KeyFrame *b)
                      { return a->mnId < b->mnId; });

            for (KeyFrame *pKFi : vpAllKFs)
            {
                if (!pKFi || pKFi->mnId <= nLoopKF || pKFi->mbBad)
                    continue;

                KeyFrame *pParent = pKFi->GetParent();
                if (!pParent)
                    continue;

                if (mapPosesBeforeGBA.count(pParent) && mapNewPoses.count(pParent))
                {
                    Sophus::SE3d T_parent_old = mapPosesBeforeGBA[pParent];
                    Sophus::SE3d T_parent_new = mapNewPoses[pParent];

                    Eigen::Matrix4f T_child_old_mat = pKFi->GetPose();
                    Sophus::SE3d T_child_old(Eigen::Quaterniond(T_child_old_mat.block<3, 3>(0, 0).cast<double>()).normalized(),
                                             T_child_old_mat.block<3, 1>(0, 3).cast<double>());

                    Sophus::SE3d T_rel = T_child_old * T_parent_old.inverse();
                    Sophus::SE3d T_child_new = T_rel * T_parent_new;

                    pKFi->SetPose(T_child_new.matrix().cast<float>());

                    mapPosesBeforeGBA[pKFi] = T_child_old;
                    mapNewPoses[pKFi] = T_child_new;

                    std::vector<MapPoint *> vpKFMPs = pKFi->GetMapPointMatches();
                    for (MapPoint *pMPi : vpKFMPs)
                    {
                        if (!pMPi || pMPi->isBad() || pMPi->GetReferenceKeyFrame() != pKFi)
                            continue;

                        Eigen::Vector3d Pw_old = pMPi->GetWorldPos().cast<double>();
                        Eigen::Vector3d Pc = T_child_old * Pw_old;
                        Eigen::Vector3d Pw_new = T_child_new.inverse() * Pc;

                        pMPi->SetWorldPos(Pw_new.cast<float>());
                        pMPi->UpdateNormalAndDepth();
                    }
                }
            }
        }
    }
}