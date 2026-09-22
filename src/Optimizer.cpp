#include "Optimizer.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "MotionOnlyBA.h"
#include <ceres/rotation.h>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

/**
 * @brief Ceres 迭代回调函数：用于在每次迭代结束时响应外部中断请求 (如插入新关键帧)
 */
class AbortCallback : public ceres::IterationCallback
{
public:
    explicit AbortCallback(bool *pbStopFlag) : pbStopFlag_(pbStopFlag) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &) override
    {
        if (pbStopFlag_ && *pbStopFlag_)
        {
            return ceres::SOLVER_ABORT; // 提前终止 Ceres 优化
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

        // 利用流形理论性质：在 y = x 处，MinusJacobian 即为 PlusJacobian 的伪逆
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_minus(jacobian);
        J_minus = (J_plus.transpose() * J_plus).inverse() * J_plus.transpose();

        return true;
    }
};

/**
 * @brief 单目重投影误差 (Analytic / 解析求导)
 * 残差维度: 2
 * 参数块 0: 7 维位姿 (qx, qy, qz, qw, tx, ty, tz)，与 MotionOnlyBA 一致
 * 参数块 1: 3 维世界坐标系地图点 (Xw, Yw, Zw)
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
        // 1. 解析参数
        Eigen::Map<const Sophus::SE3d> T_cw(parameters[0]);
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

        // 投影关于相机系三维点导数: d(res) / d(P_c) (2x3)
        Eigen::Matrix<double, 2, 3> J_proj;
        J_proj(0, 0) = fx_ * inv_z * inv_sigma_;
        J_proj(0, 1) = 0.0;
        J_proj(0, 2) = -fx_ * X * inv_z2 * inv_sigma_;

        J_proj(1, 0) = 0.0;
        J_proj(1, 1) = fy_ * inv_z * inv_sigma_;
        J_proj(1, 2) = -fy_ * Y * inv_z2 * inv_sigma_;

        // 2. 对位姿参数块的偏导 (2x7)
        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose.setZero();

            const double Xw = point_3d[0], Yw = point_3d[1], Zw = point_3d[2];
            const Eigen::Quaterniond q = T_cw.unit_quaternion();
            const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

            // d(P_c) / d(q) (3x4)
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

            // 回填: 前4列对四元数，后3列对平移 (dPc/dt = I_3)
            J_pose.block<2, 4>(0, 0) = J_proj * dPc_dq;
            J_pose.block<2, 3>(0, 4) = J_proj;
        }

        // 3. 对地图点的偏导 (2x3): d(res) / d(P_w) = J_proj * R_cw
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
 * @brief 双目重投影误差 (Analytic / 解析求导)
 * 残差维度: 3
 * 参数块 0: 7 维位姿 (qx, qy, qz, qw, tx, ty, tz)
 * 参数块 1: 3 维世界坐标系地图点 (Xw, Yw, Zw)
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
        Eigen::Map<const Sophus::SE3d> T_cw(parameters[0]);
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
                {
                    // 3 个残差 x 7 维位姿 = 21 个 double
                    std::fill(jacobians[0], jacobians[0] + 21, 0.0);
                }
                if (jacobians[1])
                {
                    // 3 个残差 x 3 维地图点 = 9 个 double
                    std::fill(jacobians[1], jacobians[1] + 9, 0.0);
                }
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

        // 投影关于相机系三维点导数: d(res) / d(P_c) (3x3)
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

        // 对位姿参数块的偏导 (3x7)
        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose.setZero();

            const double Xw = point_3d[0], Yw = point_3d[1], Zw = point_3d[2];
            const Eigen::Quaterniond q = T_cw.unit_quaternion();
            const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();

            // d(P_c) / d(q) (3x4)
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

        // 对地图点的偏导 (3x3): d(res) / d(P_w) = J_proj * R_cw
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
 * 残差: 6 维
 * 参数块 0: 帧 i 位姿 (7 维: qx, qy, qz, qw, tx, ty, tz)
 * 参数块 1: 帧 j 位姿 (7 维: qx, qy, qz, qw, tx, ty, tz)
 */
class PoseGraphSE3Analytic : public ceres::SizedCostFunction<6, 7, 7>
{
public:
    explicit PoseGraphSE3Analytic(const Sophus::SE3d &T_ij_meas)
        : T_ij_meas_(T_ij_meas) {}

    // 李代数小伴随矩阵 ad_xi, 排列为 [omega, v]
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

    // 严密二阶展开的 SE(3) Jr^-1 矩阵，完全消除截断误差
    static Eigen::Matrix<double, 6, 6> JrInvSE3(const Eigen::Matrix<double, 6, 1> &xi)
    {
        Eigen::Matrix<double, 6, 6> ad = curlyHat(xi);
        Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Identity() + 0.5 * ad + (1.0 / 12.0) * (ad * ad);
        return J;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        Eigen::Map<const Sophus::SE3d> T_iw(parameters[0]);
        Eigen::Map<const Sophus::SE3d> T_jw(parameters[1]);

        // T_ij_est = T_iw * T_jw^-1
        Sophus::SE3d T_ij_est = T_iw * T_jw.inverse();
        // error_SE3 = T_ij_meas * (T_ij_est)^-1 = T_ij_meas * T_jw * T_iw^-1
        Sophus::SE3d error_SE3 = T_ij_meas_ * T_ij_est.inverse();

        Eigen::Matrix<double, 6, 1> sophus_xi = error_SE3.log();

        // 排列为 [omega, v]
        Eigen::Matrix<double, 6, 1> error_vec;
        error_vec.head<3>() = sophus_xi.tail<3>(); // omega
        error_vec.tail<3>() = sophus_xi.head<3>(); // v

        residuals[0] = error_vec[0];
        residuals[1] = error_vec[1];
        residuals[2] = error_vec[2];
        residuals[3] = error_vec[3];
        residuals[4] = error_vec[4];
        residuals[5] = error_vec[5];

        if (jacobians)
        {
            // 辅助 Lambda：直接闭式构造 7 维环境空间到 6 维切空间的逆映射矩阵 (6x7)
            // 避免运行时进行 (J+^T * J+)^-1 矩阵求逆与连乘
            auto ComputePinvJPlus = [](const double *param) -> Eigen::Matrix<double, 6, 7>
            {
                const double qx = param[0], qy = param[1], qz = param[2], qw = param[3];
                const double tx = param[4], ty = param[5], tz = param[6];

                Eigen::Matrix<double, 6, 7> pinv;
                pinv.setZero();

                // 上半部分 3x4: 旋转四元数切空间投影的解析伪逆 (2.0 * W(q)^T)
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

                // 下半部分 3x3: 平移切空间
                pinv.block<3, 3>(3, 4) = Eigen::Matrix3d::Identity();
                // 下半部分 3x4: 消除左乘扰动对平移耦合项 (-[t]_x) 的影响
                pinv.block<3, 4>(3, 0) = Sophus::SO3d::hat(Eigen::Vector3d(tx, ty, tz)) * pinv.block<3, 4>(0, 0);

                return pinv;
            };

            Eigen::Matrix<double, 6, 6> J_r_inv = JrInvSE3(error_vec);

            // 1. 对 i 帧雅可比: J_tangent_i * pinv_i
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_i(jacobians[0]);
                Eigen::Matrix<double, 6, 6> J_tangent_i = -J_r_inv;
                J_i = J_tangent_i * ComputePinvJPlus(parameters[0]);
            }

            // 2. 对 j 帧雅可比: J_tangent_j * pinv_j
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_j(jacobians[1]);

                const Eigen::Matrix3d R_est = T_ij_est.rotationMatrix();
                const Eigen::Vector3d t_est = T_ij_est.translation();

                // SE(3) 大伴随矩阵 Adj(T)，按 [omega, v] 排序
                Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Zero();
                Adj.block<3, 3>(0, 0) = R_est;
                Adj.block<3, 3>(3, 0) = Sophus::SO3d::hat(t_est) * R_est;
                Adj.block<3, 3>(3, 3) = R_est;

                Eigen::Matrix<double, 6, 6> J_tangent_j = J_r_inv * Adj;
                J_j = J_tangent_j * ComputePinvJPlus(parameters[1]);
            }
        }
        return true;
    }

private:
    Sophus::SE3d T_ij_meas_;
};

static void PoseToArray(KeyFrame *pKF, double out[6])
{
    const Eigen::Matrix4f Tcw = pKF->GetPose();
    const Eigen::Matrix3f R_cw = Tcw.block<3, 3>(0, 0);
    const Eigen::Vector3f t_cw = Tcw.block<3, 1>(0, 3);

    const Eigen::AngleAxisd aa(R_cw.cast<double>());
    const Eigen::Vector3d r_vec = aa.angle() * aa.axis();

    out[0] = r_vec[0];
    out[1] = r_vec[1];
    out[2] = r_vec[2];
    out[3] = t_cw[0];
    out[4] = t_cw[1];
    out[5] = t_cw[2];
}

static void ArrayToPose(KeyFrame *pKF, const double in[6])
{
    const Eigen::Vector3d r_vec(in[0], in[1], in[2]);
    const Eigen::Vector3d t_cw(in[3], in[4], in[5]);

    Eigen::Matrix3d R_cw = Eigen::Matrix3d::Identity();
    const double angle = r_vec.norm();
    if (angle > 1e-12)
        R_cw = Eigen::AngleAxisd(angle, r_vec.normalized()).toRotationMatrix();

    Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
    Tcw.block<3, 3>(0, 0) = R_cw.cast<float>();
    Tcw.block<3, 1>(0, 3) = t_cw.cast<float>();
    pKF->SetPose(Tcw);
}

/**
 * @brief 局部 Bundle Adjustment (7 维位姿 + 3 维地图点 解析求导版)
 */
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

    std::map<KeyFrame *, Sophus::SE3d> mapKF_SE3;
    for (KeyFrame *pKF : vpLocalKFs)
    {
        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();
        mapKF_SE3[pKF] = Sophus::SE3d(q, t);
    }
    for (KeyFrame *pKF : vpFixedKFs)
    {
        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();
        mapKF_SE3[pKF] = Sophus::SE3d(q, t);
    }

    std::map<MapPoint *, Eigen::Vector3d> mapMP_Point;
    for (MapPoint *pMP : vpLocalMPs)
    {
        mapMP_Point[pMP] = pMP->GetWorldPos().cast<double>();
    }

    ceres::Problem problem;

    for (KeyFrame *pKF : vpLocalKFs)
    {
        problem.AddParameterBlock(mapKF_SE3[pKF].data(), 7, new SophusSE3Manifold());
        if (pKF->mnId == 0)
            problem.SetParameterBlockConstant(mapKF_SE3[pKF].data());
    }

    for (KeyFrame *pKF : vpFixedKFs)
    {
        problem.AddParameterBlock(mapKF_SE3[pKF].data(), 7, new SophusSE3Manifold());
        problem.SetParameterBlockConstant(mapKF_SE3[pKF].data());
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
            double *pose_param = mapKF_SE3[pKF].data();

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
                    // 1. 双目：使用双目重投影误差 7.815 卡方核函数
                    ceres::CostFunction *cost = new LocalRepoErrorStereoAnalytic(
                        fx, fy, cx, cy, pKF->mbf, kp.pt.x, kp.pt.y, u_r, sqrtInvSigma2);

                    ceres::ResidualBlockId id = problem.AddResidualBlock(
                        cost, new ceres::HuberLoss(std::sqrt(7.815)), pose_param, mapMP_Point[pMP].data());

                    vObsInfo.push_back({pKF, pMP, j, id, true});
                }
                else
                {
                    // 2. 单目/远点：使用单目重投影误差 5.991 卡方核函数
                    ceres::CostFunction *cost = new LocalRepoErrorAnalytic(
                        fx, fy, cx, cy, kp.pt.x, kp.pt.y, sqrtInvSigma2);

                    ceres::ResidualBlockId id = problem.AddResidualBlock(
                        cost, new ceres::HuberLoss(std::sqrt(5.991)), pose_param, mapMP_Point[pMP].data());

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

        const Sophus::SE3d &T_cw_curr = mapKF_SE3[pKF];

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
            Eigen::Quaterniond q = mapKF_SE3[pKF].unit_quaternion();
            q.normalize();
            Sophus::SE3d T_opt(q, mapKF_SE3[pKF].translation());
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

    std::map<KeyFrame *, Sophus::SE3d> mapPoses;
    std::cout << "  --> [Opt-DEBUG] 1. 开始提取位姿初值与设置常数顶点..." << std::endl;
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
        mapPoses[pKF] = Sophus::SE3d(q, t);

        problem.AddParameterBlock(mapPoses[pKF].data(), 7, new SophusSE3Manifold());

        if (pKF == pLoopKF)
        {
            problem.SetParameterBlockConstant(mapPoses[pKF].data());
        }
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

    std::set<std::pair<KeyFrame *, KeyFrame *>> sInsertedEdges;
    std::cout << "  --> [Opt-DEBUG] 2. 开始添加生成树约束边..." << std::endl;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad || pKF->mnId == 0)
            continue;

        // 3.1 父节点边
        KeyFrame *pParent = pKF->GetParent();
        if (pParent && !pParent->mbBad && mapPoses.count(pParent) && mapPoses.count(pKF))
        {
            auto edgePair = std::make_pair(std::min(pKF, pParent), std::max(pKF, pParent));
            if (!sInsertedEdges.count(edgePair))
            {
                Sophus::SE3d T_child_parent_meas = mapOriginalPoses[pKF] * mapOriginalPoses[pParent].inverse();
                ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_child_parent_meas);
                problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                         mapPoses[pKF].data(), mapPoses[pParent].data());
                sInsertedEdges.insert(edgePair);
            }
        }

        // 3.2 子节点边
        const std::set<KeyFrame *> sChildren = pKF->GetChilds();
        for (KeyFrame *pChild : sChildren)
        {
            if (!pChild || pChild->mbBad || !mapPoses.count(pChild) || !mapPoses.count(pKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pChild), std::max(pKF, pChild));
            if (sInsertedEdges.count(edgePair))
                continue;

            Sophus::SE3d T_parent_child_meas = mapOriginalPoses[pKF] * mapOriginalPoses[pChild].inverse();
            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_parent_child_meas);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                     mapPoses[pKF].data(), mapPoses[pChild].data());
            sInsertedEdges.insert(edgePair);
        }

        // 3.3 历史累积的回环边
        const std::set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (KeyFrame *pLKF : sLoopEdges)
        {
            if (!pLKF || pLKF->mbBad || !mapPoses.count(pLKF))
                continue;

            // 关键：跳过本次刚刚触发的闭环帧对，由第 6 步专属负责
            if ((pKF == pCurKF && pLKF == pLoopKF) || (pKF == pLoopKF && pLKF == pCurKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pLKF), std::max(pKF, pLKF));
            if (sInsertedEdges.count(edgePair))
                continue;

            Sophus::SE3d T_ij_meas = mapOriginalPoses[pKF] * mapOriginalPoses[pLKF].inverse();
            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_ij_meas);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                     mapPoses[pKF].data(), mapPoses[pLKF].data());
            sInsertedEdges.insert(edgePair);
        }
    }

    std::cout << "  --> [Opt-DEBUG] 3. 开始添加高共视边与回环边..." << std::endl;
    // 4. 高权重共视边 (weight >= 100)
    const int minWeight = 100;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad || !mapPoses.count(pKF))
            continue;

        const std::vector<KeyFrame *> vpCovKFs = pKF->GetCovisibleByWeight(minWeight);
        for (KeyFrame *pCovKF : vpCovKFs)
        {
            if (!pCovKF || pCovKF->mbBad || !mapPoses.count(pCovKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pCovKF), std::max(pKF, pCovKF));
            if (sInsertedEdges.count(edgePair))
                continue;

            Sophus::SE3d T_ij_meas = mapOriginalPoses[pKF] * mapOriginalPoses[pCovKF].inverse();

            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_ij_meas);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                     mapPoses[pKF].data(), mapPoses[pCovKF].data());

            sInsertedEdges.insert(edgePair);
        }
    }

    // 5. 闭环融合引入的新共视边 (LoopConnections)
    for (auto &mit : LoopConnections)
    {
        KeyFrame *pKF = mit.first;
        if (!pKF || pKF->mbBad || !mapPoses.count(pKF))
            continue;

        const std::set<KeyFrame *> &sLoopNeighbors = mit.second;
        for (KeyFrame *pLKF : sLoopNeighbors)
        {
            if (!pLKF || pLKF->mbBad || !mapPoses.count(pLKF))
                continue;

            auto edgePair = std::make_pair(std::min(pKF, pLKF), std::max(pKF, pLKF));
            if (sInsertedEdges.count(edgePair))
                continue;

            Sophus::SE3d T_i_corr = mapPoses[pKF];
            Sophus::SE3d T_j_corr = mapPoses[pLKF];
            Sophus::SE3d T_ij_meas = T_i_corr * T_j_corr.inverse();

            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_ij_meas);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                     mapPoses[pKF].data(), mapPoses[pLKF].data());

            sInsertedEdges.insert(edgePair);
        }
    }

    // 6. 添加本次闭环的直接约束边 (当前帧与目标闭环帧)
    if (pCurKF != pLoopKF && mapPoses.count(pCurKF) && mapPoses.count(pLoopKF))
    {
        auto edgePair = std::make_pair(std::min(pCurKF, pLoopKF), std::max(pCurKF, pLoopKF));
        if (!sInsertedEdges.count(edgePair))
        {
            // 获取闭环前端求解出来的当前帧矫正位姿 T_cur_corrected (T_cw_loop)
            Sophus::SE3d T_cur_corrected = mapPoses[pCurKF];
            auto itCur = CorrectedPoses.find(pCurKF);
            if (itCur != CorrectedPoses.end())
            {
                Eigen::Matrix4f T_mat = itCur->second;
                Eigen::Matrix3d R = T_mat.block<3, 3>(0, 0).cast<double>();
                Eigen::Vector3d t = T_mat.block<3, 1>(0, 3).cast<double>();
                T_cur_corrected = Sophus::SE3d(Eigen::Quaterniond(R).normalized(), t);
            }

            // pLoopKF 作为固定参考锚点，使用其原位姿 T_loop_w
            Sophus::SE3d T_loop_w = mapOriginalPoses[pLoopKF];

            // 测量值 T_cur_loop_meas = T_cur_w * (T_loop_w)^-1
            Sophus::SE3d T_cur_loop_meas = T_cur_corrected * T_loop_w.inverse();

            ceres::CostFunction *cost = new PoseGraphSE3Analytic(T_cur_loop_meas);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                     mapPoses[pCurKF].data(), mapPoses[pLoopKF].data());
            sInsertedEdges.insert(edgePair);
        }
    }

    // 7. 配置求解器并优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 40;
    options.num_threads = 2;
    options.minimizer_progress_to_stdout = false;
    std::cout << "  --> [Opt-DEBUG] 4. 配置完成，开始 Ceres Solve..." << std::endl;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "  --> [Opt-DEBUG] 5. Ceres Solve 求解完毕！" << std::endl;
    std::cout << "  --> [Opt-DEBUG] 6. 准备回写关键帧与地图点..." << std::endl;

    // 8 & 9. 回写优化后的关键帧与地图点
    {
        std::map<KeyFrame *, Sophus::SE3d> mapNewPoses;
        for (auto &kv : mapPoses)
        {
            KeyFrame *pKF = kv.first;
            Eigen::Quaterniond q = kv.second.unit_quaternion();
            q.normalize();
            Sophus::SE3d T_new(q, kv.second.translation());

            pKF->SetPose(T_new.matrix().cast<float>());
            mapNewPoses[pKF] = T_new;
        }

        std::vector<MapPoint *> vpAllMPs = pMap->GetAllMapPoints();
        for (MapPoint *pMP : vpAllMPs)
        {
            if (!pMP || pMP->isBad())
                continue;

            if (pMP->mnCorrectedByKF == pCurKF->mnId)
                continue;

            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            if (!pRefKF || !mapOriginalPoses.count(pRefKF) || !mapNewPoses.count(pRefKF))
                continue;

            // 提取参考关键帧优化前后的相机位姿 T_cw
            Sophus::SE3d T_old_cw = mapOriginalPoses[pRefKF];
            Sophus::SE3d T_new_cw = mapNewPoses[pRefKF];

            // 1. 将旧世界坐标系下的点转至参考帧相机系: P_c = T_old_cw * P_w_old
            Eigen::Vector3d Pw_old = pMP->GetWorldPos().cast<double>();
            Eigen::Vector3d Pc = T_old_cw * Pw_old;

            // 2. 将相机系下的点转至新世界坐标系: P_w_new = (T_new_cw)^(-1) * P_c
            Eigen::Vector3d Pw_new = T_new_cw.inverse() * Pc;

            pMP->SetWorldPos(Pw_new.cast<float>());
            pMP->UpdateNormalAndDepth();
        }
    }
    std::cout << "  --> [Opt-DEBUG] 7. 回写完毕，退出 OptimizeEssentialGraph！" << std::endl;
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

    std::map<KeyFrame *, Sophus::SE3d> mapPoses;
    ceres::Problem problem;

    for (KeyFrame *pKF : vpKFsToOptimize)
    {
        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Sophus::SE3d T_cw(Eigen::Quaterniond(R).normalized(), t);

        mapPoses[pKF] = T_cw;

        problem.AddParameterBlock(mapPoses[pKF].data(), 7, new SophusSE3Manifold());
        if (pKF->mnId == 0)
            problem.SetParameterBlockConstant(mapPoses[pKF].data());
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
        double *pose_param = mapPoses[pKF].data();

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
    options.num_threads = 4;

    AbortCallback callback(pbStopFlag);
    if (pbStopFlag)
        options.callbacks.push_back(&callback);

    std::map<KeyFrame *, Sophus::SE3d> mapPosesBeforeGBA = mapPoses;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "\033[36m[GBA-Verify] Ceres 终止原因: " << summary.termination_type
              << " (" << summary.message << ")\033[0m" << std::endl;
    std::cout << "\033[36m[GBA-Verify] 优化帧数: " << vpKFsToOptimize.size()
              << " | 地图点数: " << mapMP_Point.size()
              << " | 迭代次数: " << summary.iterations.size() << "\033[0m" << std::endl;
    std::cout << "\033[36m[GBA-Verify] 初始 Cost: " << summary.initial_cost
              << " -> 最终 Cost: " << summary.final_cost << "\033[0m" << std::endl;

    if (pbStopFlag && *pbStopFlag)
        return;

    {
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        std::map<KeyFrame *, Sophus::SE3d> mapNewPoses;
        for (auto &kv : mapPoses)
        {
            KeyFrame *pKF = kv.first;
            Eigen::Quaterniond q = kv.second.unit_quaternion();
            q.normalize();
            Sophus::SE3d T_new(q, kv.second.translation());

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
                    // 父节点 GBA 优化前后的位姿 (都是世界到相机 T_cw)
                    Sophus::SE3d T_parent_old = mapPosesBeforeGBA[pParent];
                    Sophus::SE3d T_parent_new = mapNewPoses[pParent];

                    // 子节点自身当前的位姿
                    Eigen::Matrix4f T_child_old_mat = pKFi->GetPose();
                    Sophus::SE3d T_child_old(Eigen::Quaterniond(T_child_old_mat.block<3, 3>(0, 0).cast<double>()).normalized(),
                                             T_child_old_mat.block<3, 1>(0, 3).cast<double>());

                    // 子节点相对于父节点的相对变换保持不变: T_child_parent = T_child_old * (T_parent_old)^-1
                    Sophus::SE3d T_rel = T_child_old * T_parent_old.inverse();

                    // 更新子节点在新世界坐标系下的绝对位姿: T_child_new = T_child_parent * T_parent_new
                    Sophus::SE3d T_child_new = T_rel * T_parent_new;

                    pKFi->SetPose(T_child_new.matrix().cast<float>());

                    // 更新缓存映射表，以便后续层级的子孙节点能够继续向上回溯
                    mapPosesBeforeGBA[pKFi] = T_child_old;
                    mapNewPoses[pKFi] = T_child_new;

                    // 同步更新以该子帧作为参考关键帧的地图点世界坐标
                    std::vector<MapPoint *> vpKFMPs = pKFi->GetMapPointMatches();
                    for (MapPoint *pMPi : vpKFMPs)
                    {
                        if (!pMPi || pMPi->isBad() || pMPi->GetReferenceKeyFrame() != pKFi)
                            continue;

                        // 1. 旧世界坐标转到子相机系: P_c = T_child_old * P_w_old
                        Eigen::Vector3d Pw_old = pMPi->GetWorldPos().cast<double>();
                        Eigen::Vector3d Pc = T_child_old * Pw_old;

                        // 2. 子相机系转到新世界坐标: P_w_new = (T_child_new)^-1 * P_c
                        Eigen::Vector3d Pw_new = T_child_new.inverse() * Pc;

                        pMPi->SetWorldPos(Pw_new.cast<float>());
                        pMPi->UpdateNormalAndDepth();
                    }
                }
            }
        }
    }
}