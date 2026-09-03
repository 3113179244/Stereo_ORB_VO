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

    /**
     * @brief 流形加法更新: x_plus = delta (+) x
     */
    bool Plus(const double *x, const double *delta, double *x_plus) const override
    {
        // 1. 读取当前估计的位姿
        Eigen::Map<const Eigen::Quaterniond> q(x);
        Eigen::Map<const Eigen::Vector3d> t(x + 4);
        Sophus::SE3d T(q, t);

        // 2. 读取 6 维李代数增量
        // Sophus 的向量排布为 [v, omega]，这里 delta 为 [omega, v]
        Eigen::Matrix<double, 6, 1> xi;
        xi << delta[3], delta[4], delta[5], delta[0], delta[1], delta[2];

        // 3. 施加左乘扰动
        Sophus::SE3d T_plus = Sophus::SE3d::exp(xi) * T;

        // 4. 写回 7 维参数
        Eigen::Map<Eigen::Quaterniond> q_plus(x_plus);
        Eigen::Map<Eigen::Vector3d> t_plus(x_plus + 4);

        q_plus = T_plus.unit_quaternion();
        t_plus = T_plus.translation();

        return true;
    }

    /**
     * @brief 解析求导计算 Plus 雅可比: J = d(x (+) delta) / d(delta) |_{delta=0} (7x6 矩阵)
     * 行优先存储: 7 行 6 列
     */
    bool PlusJacobian(const double *x, double *jacobian) const override
    {
        // 初始化全部为 0
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        const double qx = x[0];
        const double qy = x[1];
        const double qz = x[2];
        const double qw = x[3];

        const double tx = x[4];
        const double ty = x[5];
        const double tz = x[6];

        // 1. 四元数关于旋转扰动 delta_phi 的导数 (4x3):
        // q(+) = exp(phi/2) * q ≈ (1 + 0.5 * [phi]_x) * q
        // d(q_plus)/d(phi) |_{phi=0} = 0.5 * [ qw*I + [q_vec]_x ; -q_vec^T ]
        J(0, 0) = 0.5 * qw;
        J(0, 1) = -0.5 * qz;
        J(0, 2) = 0.5 * qy;
        J(1, 0) = 0.5 * qz;
        J(1, 1) = 0.5 * qw;
        J(1, 2) = -0.5 * qx;
        J(2, 0) = -0.5 * qy;
        J(2, 1) = 0.5 * qx;
        J(2, 2) = 0.5 * qw;
        J(3, 0) = -0.5 * qx;
        J(3, 1) = -0.5 * qy;
        J(3, 2) = -0.5 * qz;

        // 四元数关于平移扰动 delta_rho 的导数全为 0:
        // J.block<4, 3>(0, 3).setZero(); 已初始化

        // 2. 平移关于旋转与平移扰动的导数 (3x6):
        // T_plus.t = exp(phi) * t + J_l * rho ≈ (I + [phi]_x) * t + rho = t - [t]_x * phi + rho
        // d(t_plus)/d(phi) |_{phi=0} = -[t]_x
        J(4, 0) = 0.0;
        J(4, 1) = tz;
        J(4, 2) = -ty;
        J(5, 0) = -tz;
        J(5, 1) = 0.0;
        J(5, 2) = tx;
        J(6, 0) = ty;
        J(6, 1) = -tx;
        J(6, 2) = 0.0;

        // d(t_plus)/d(rho) |_{rho=0} = I (3x3 单位阵)
        J(4, 3) = 1.0;
        J(5, 4) = 1.0;
        J(6, 5) = 1.0;

        return true;
    }

    /**
     * @brief 减法运算: delta = x_plus (-) x = log(T_plus * T^-1)
     */
    bool Minus(const double *y, const double *x, double *delta) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q_x(x), q_y(y);
        Eigen::Map<const Eigen::Vector3d> t_x(x + 4), t_y(y + 4);

        Sophus::SE3d T_x(q_x, t_x);
        Sophus::SE3d T_y(q_y, t_y);

        Sophus::SE3d T_delta = T_y * T_x.inverse();
        Eigen::Matrix<double, 6, 1> xi = T_delta.log(); // [v, omega]

        delta[0] = xi[3];
        delta[1] = xi[4];
        delta[2] = xi[5];
        delta[3] = xi[0];
        delta[4] = xi[1];
        delta[5] = xi[2];

        return true;
    }

    /**
     * @brief 减法雅可比: J = d(y (-) x) / d(y) |_{y=x} (6x7 矩阵)
     */
    bool MinusJacobian(const double *x, double *jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        const double qx = x[0];
        const double qy = x[1];
        const double qz = x[2];
        const double qw = x[3];

        const double tx = x[4];
        const double ty = x[5];
        const double tz = x[6];

        // 旋转部分关于四元数的求导
        J(0, 0) = 2.0 * qw;
        J(0, 1) = 2.0 * qz;
        J(0, 2) = -2.0 * qy;
        J(0, 3) = -2.0 * qx;
        J(1, 0) = -2.0 * qz;
        J(1, 1) = 2.0 * qw;
        J(1, 2) = 2.0 * qx;
        J(1, 3) = -2.0 * qy;
        J(2, 0) = 2.0 * qy;
        J(2, 1) = -2.0 * qx;
        J(2, 2) = 2.0 * qw;
        J(2, 3) = -2.0 * qz;

        // 平移部分
        J(3, 4) = 1.0;
        J(4, 5) = 1.0;
        J(5, 6) = 1.0;

        // 平移关于四元数求导 (在 y=x 处为 [t]_x * 2 * q_rel)
        Eigen::Matrix3d tx_skew;
        tx_skew << 0, -tz, ty,
            tz, 0, -tx,
            -ty, tx, 0;
        J.block<3, 4>(3, 0) = tx_skew * J.block<3, 4>(0, 0);

        return true;
    }
};

/**
 * @brief 单目解析求导残差块（残差 2，参数块 0: 6维位姿增量, 参数块 1: 3维地图点）
 */
class LocalRepoErrorAnalytic : public ceres::SizedCostFunction<2, 6, 3>
{
public:
    LocalRepoErrorAnalytic(double fx, double fy, double cx, double cy,
                           double u, double v, const Sophus::SE3d &T_cw_init, double sqrtInvSigma2)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          u_(u), v_(v), T_cw_init_(T_cw_init), sqrtInvSigma2_(sqrtInvSigma2) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        const double *xi_raw = parameters[0];
        const double *point = parameters[1];

        // 1. 计算当前估计的位姿与空间点
        Eigen::Matrix<double, 6, 1> xi;
        xi << xi_raw[3], xi_raw[4], xi_raw[5], xi_raw[0], xi_raw[1], xi_raw[2];
        Sophus::SE3d T_cw = Sophus::SE3d::exp(xi) * T_cw_init_;

        Eigen::Vector3d P_w(point[0], point[1], point[2]);
        Eigen::Vector3d P_c = T_cw * P_w;

        const double x = P_c[0];
        const double y = P_c[1];
        const double z = P_c[2];

        if (z <= 1e-4)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            if (jacobians)
            {
                if (jacobians[0])
                    std::fill(jacobians[0], jacobians[0] + 12, 0.0); // 2x6
                if (jacobians[1])
                    std::fill(jacobians[1], jacobians[1] + 6, 0.0); // 2x3
            }
            return true; // 返回 true，避免 Ceres 报错
        }

        const double inv_z = 1.0 / z;
        const double inv_z2 = inv_z * inv_z;

        const double pred_u = fx_ * x * inv_z + cx_;
        const double pred_v = fy_ * y * inv_z + cy_;

        // 2. 残差
        residuals[0] = (pred_u - u_) * sqrtInvSigma2_;
        residuals[1] = (pred_v - v_) * sqrtInvSigma2_;

        const double dedx_0 = fx_ * inv_z;
        const double dedz_0 = -fx_ * x * inv_z2;
        const double dedy_1 = fy_ * inv_z;
        const double dedz_1 = -fy_ * y * inv_z2;

        // 3. 对位姿求导 (2x6)
        if (jacobians && jacobians[0])
        {
            double *j_pose = jacobians[0];

            // du 对旋转与平移
            j_pose[0] = sqrtInvSigma2_ * (dedz_0 * y);
            j_pose[1] = sqrtInvSigma2_ * (dedx_0 * z - dedz_0 * x);
            j_pose[2] = sqrtInvSigma2_ * (-dedx_0 * y);
            j_pose[3] = sqrtInvSigma2_ * dedx_0;
            j_pose[4] = 0.0;
            j_pose[5] = sqrtInvSigma2_ * dedz_0;

            // dv 对旋转与平移
            j_pose[6] = sqrtInvSigma2_ * (-dedy_1 * z + dedz_1 * y);
            j_pose[7] = sqrtInvSigma2_ * (-dedz_1 * x);
            j_pose[8] = sqrtInvSigma2_ * (dedy_1 * x);
            j_pose[9] = 0.0;
            j_pose[10] = sqrtInvSigma2_ * dedy_1;
            j_pose[11] = sqrtInvSigma2_ * dedz_1;
        }

        // 4. 对地图点求导 (2x3): J_point = J_Pc * R_cw
        if (jacobians && jacobians[1])
        {
            double *j_pt = jacobians[1];
            const Eigen::Matrix3d &R = T_cw.rotationMatrix();

            // Row 0 (du): [dedx_0, 0, dedz_0] * R
            j_pt[0] = sqrtInvSigma2_ * (dedx_0 * R(0, 0) + dedz_0 * R(2, 0));
            j_pt[1] = sqrtInvSigma2_ * (dedx_0 * R(0, 1) + dedz_0 * R(2, 1));
            j_pt[2] = sqrtInvSigma2_ * (dedx_0 * R(0, 2) + dedz_0 * R(2, 2));

            // Row 1 (dv): [0, dedy_1, dedz_1] * R
            j_pt[3] = sqrtInvSigma2_ * (dedy_1 * R(1, 0) + dedz_1 * R(2, 0));
            j_pt[4] = sqrtInvSigma2_ * (dedy_1 * R(1, 1) + dedz_1 * R(2, 1));
            j_pt[5] = sqrtInvSigma2_ * (dedy_1 * R(1, 2) + dedz_1 * R(2, 2));
        }

        return true;
    }

private:
    double fx_, fy_, cx_, cy_;
    double u_, v_;
    Sophus::SE3d T_cw_init_;
    double sqrtInvSigma2_;
};

/**
 * @brief 双目解析求导残差块（残差 3，参数块 0: 6维位姿增量, 参数块 1: 3维地图点）
 */
class LocalRepoErrorStereoAnalytic : public ceres::SizedCostFunction<3, 6, 3>
{
public:
    LocalRepoErrorStereoAnalytic(double fx, double fy, double cx, double cy, double mbf,
                                 double u, double v, double u_r, const Sophus::SE3d &T_cw_init, double sqrtInvSigma2)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), mbf_(mbf),
          u_(u), v_(v), u_r_(u_r), T_cw_init_(T_cw_init), sqrtInvSigma2_(sqrtInvSigma2) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        const double *xi_raw = parameters[0];
        const double *point = parameters[1];

        Eigen::Matrix<double, 6, 1> xi;
        xi << xi_raw[3], xi_raw[4], xi_raw[5], xi_raw[0], xi_raw[1], xi_raw[2];
        Sophus::SE3d T_cw = Sophus::SE3d::exp(xi) * T_cw_init_;

        Eigen::Vector3d P_w(point[0], point[1], point[2]);
        Eigen::Vector3d P_c = T_cw * P_w;

        const double x = P_c[0];
        const double y = P_c[1];
        const double z = P_c[2];

        if (z <= 1e-4)
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            residuals[2] = 0.0;
            if (jacobians)
            {
                if (jacobians[0])
                    std::fill(jacobians[0], jacobians[0] + 18, 0.0); // 3x6
                if (jacobians[1])
                    std::fill(jacobians[1], jacobians[1] + 9, 0.0); // 3x3
            }
            return true; // 返回 true，避免 Ceres 报错
        }

        const double inv_z = 1.0 / z;
        const double inv_z2 = inv_z * inv_z;

        const double pred_u = fx_ * x * inv_z + cx_;
        const double pred_v = fy_ * y * inv_z + cy_;
        const double pred_ur = pred_u - mbf_ * inv_z;

        // 残差
        residuals[0] = (pred_u - u_) * sqrtInvSigma2_;
        residuals[1] = (pred_v - v_) * sqrtInvSigma2_;
        residuals[2] = (pred_ur - u_r_) * sqrtInvSigma2_;

        const double dedx_0 = fx_ * inv_z;
        const double dedz_0 = -fx_ * x * inv_z2;
        const double dedy_1 = fy_ * inv_z;
        const double dedz_1 = -fy_ * y * inv_z2;
        const double dedx_2 = dedx_0;
        const double dedz_2 = -(fx_ * x - mbf_) * inv_z2;

        // 对位姿求导 (3x6)
        if (jacobians && jacobians[0])
        {
            double *j_pose = jacobians[0];

            // duL
            j_pose[0] = sqrtInvSigma2_ * (dedz_0 * y);
            j_pose[1] = sqrtInvSigma2_ * (dedx_0 * z - dedz_0 * x);
            j_pose[2] = sqrtInvSigma2_ * (-dedx_0 * y);
            j_pose[3] = sqrtInvSigma2_ * dedx_0;
            j_pose[4] = 0.0;
            j_pose[5] = sqrtInvSigma2_ * dedz_0;

            // dvL
            j_pose[6] = sqrtInvSigma2_ * (-dedy_1 * z + dedz_1 * y);
            j_pose[7] = sqrtInvSigma2_ * (-dedz_1 * x);
            j_pose[8] = sqrtInvSigma2_ * (dedy_1 * x);
            j_pose[9] = 0.0;
            j_pose[10] = sqrtInvSigma2_ * dedy_1;
            j_pose[11] = sqrtInvSigma2_ * dedz_1;

            // duR
            j_pose[12] = sqrtInvSigma2_ * (dedz_2 * y);
            j_pose[13] = sqrtInvSigma2_ * (dedx_2 * z - dedz_2 * x);
            j_pose[14] = sqrtInvSigma2_ * (-dedx_2 * y);
            j_pose[15] = sqrtInvSigma2_ * dedx_2;
            j_pose[16] = 0.0;
            j_pose[17] = sqrtInvSigma2_ * dedz_2;
        }

        // 对地图点求导 (3x3)
        if (jacobians && jacobians[1])
        {
            double *j_pt = jacobians[1];
            const Eigen::Matrix3d &R = T_cw.rotationMatrix();

            // Row 0 (duL)
            j_pt[0] = sqrtInvSigma2_ * (dedx_0 * R(0, 0) + dedz_0 * R(2, 0));
            j_pt[1] = sqrtInvSigma2_ * (dedx_0 * R(0, 1) + dedz_0 * R(2, 1));
            j_pt[2] = sqrtInvSigma2_ * (dedx_0 * R(0, 2) + dedz_0 * R(2, 2));

            // Row 1 (dvL)
            j_pt[3] = sqrtInvSigma2_ * (dedy_1 * R(1, 0) + dedz_1 * R(2, 0));
            j_pt[4] = sqrtInvSigma2_ * (dedy_1 * R(1, 1) + dedz_1 * R(2, 1));
            j_pt[5] = sqrtInvSigma2_ * (dedy_1 * R(1, 2) + dedz_1 * R(2, 2));

            // Row 2 (duR)
            j_pt[6] = sqrtInvSigma2_ * (dedx_2 * R(0, 0) + dedz_2 * R(2, 0));
            j_pt[7] = sqrtInvSigma2_ * (dedx_2 * R(0, 1) + dedz_2 * R(2, 1));
            j_pt[8] = sqrtInvSigma2_ * (dedx_2 * R(0, 2) + dedz_2 * R(2, 2));
        }

        return true;
    }

private:
    double fx_, fy_, cx_, cy_, mbf_;
    double u_, v_, u_r_;
    Sophus::SE3d T_cw_init_;
    double sqrtInvSigma2_;
};

/**
 * @brief 位姿图 SE3 解析求导残差块（残差 6，参数块 0: 帧 i 扰动, 参数块 1: 帧 j 扰动）
 */
class PoseGraphSE3Analytic : public ceres::SizedCostFunction<6, 6, 6>
{
public:
    PoseGraphSE3Analytic(const Sophus::SE3d &T_ij_meas, const Sophus::SE3d &T_iw_init, const Sophus::SE3d &T_jw_init)
        : T_ij_meas_(T_ij_meas), T_iw_init_(T_iw_init), T_jw_init_(T_jw_init) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        const double *xi_i_raw = parameters[0];
        const double *xi_j_raw = parameters[1];

        Eigen::Matrix<double, 6, 1> xi_i, xi_j;
        xi_i << xi_i_raw[3], xi_i_raw[4], xi_i_raw[5], xi_i_raw[0], xi_i_raw[1], xi_i_raw[2];
        xi_j << xi_j_raw[3], xi_j_raw[4], xi_j_raw[5], xi_j_raw[0], xi_j_raw[1], xi_j_raw[2];

        Sophus::SE3d T_iw = Sophus::SE3d::exp(xi_i) * T_iw_init_;
        Sophus::SE3d T_jw = Sophus::SE3d::exp(xi_j) * T_jw_init_;

        // 误差 e = log( T_ij_meas * T_jw * T_iw^-1 )
        Sophus::SE3d T_ij_est = T_iw * T_jw.inverse();
        Sophus::SE3d error_SE3 = T_ij_meas_ * T_ij_est.inverse();

        // 提取 6 维李代数残差 [omega, v]
        Eigen::Matrix<double, 6, 1> error_vec = error_SE3.log(); // Sophus::log 返回 [v, w]
        residuals[0] = error_vec[3];                             // wx
        residuals[1] = error_vec[4];                             // wy
        residuals[2] = error_vec[5];                             // wz
        residuals[3] = error_vec[0];                             // vx
        residuals[4] = error_vec[1];                             // vy
        residuals[5] = error_vec[2];                             // vz

        if (jacobians)
        {
            // 对帧 i 的偏导: J_i = -I (6x6)
            if (jacobians[0])
            {
                std::fill(jacobians[0], jacobians[0] + 36, 0.0);
                for (int k = 0; k < 6; ++k)
                    jacobians[0][k * 6 + k] = -1.0;
            }

            // 对帧 j 的偏导: J_j = Ad(T_ij_meas) (在 [w, v] 排布下)
            if (jacobians[1])
            {
                std::fill(jacobians[1], jacobians[1] + 36, 0.0);
                const Eigen::Matrix3d R = T_ij_meas_.rotationMatrix();
                const Eigen::Vector3d t = T_ij_meas_.translation();
                const Eigen::Matrix3d tx = Sophus::SO3d::hat(t); // [t]x

                Eigen::Matrix<double, 6, 6> J_j = Eigen::Matrix<double, 6, 6>::Zero();
                // [w] 对 [w] -> R
                J_j.block<3, 3>(0, 0) = R;
                // [w] 对 [v] -> 0
                J_j.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
                // [v] 对 [w] -> [t]x * R
                J_j.block<3, 3>(3, 0) = tx * R;
                // [v] 对 [v] -> R
                J_j.block<3, 3>(3, 3) = R;

                for (int r = 0; r < 6; ++r)
                    for (int c = 0; c < 6; ++c)
                        jacobians[1][r * 6 + c] = J_j(r, c);
            }
        }

        return true;
    }

private:
    Sophus::SE3d T_ij_meas_;
    Sophus::SE3d T_iw_init_;
    Sophus::SE3d T_jw_init_;
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

int Optimizer::PoseOptimization(Frame *pFrame)
{
    return MotionOnlyBA::Optimize(pFrame);
}

/**
 * @brief 局部 Bundle Adjustment (解析求导版)
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

    std::map<KeyFrame *, std::vector<double>> mapKF_delta_xi;
    for (KeyFrame *pKF : vpLocalKFs)
    {
        mapKF_delta_xi[pKF] = std::vector<double>(6, 0.0);
        problem.AddParameterBlock(mapKF_delta_xi[pKF].data(), 6);
        if (pKF->mnId == 0)
            problem.SetParameterBlockConstant(mapKF_delta_xi[pKF].data());
    }

    std::map<KeyFrame *, std::vector<double>> mapFixed_delta_xi;
    for (KeyFrame *pKF : vpFixedKFs)
    {
        mapFixed_delta_xi[pKF] = std::vector<double>(6, 0.0);
        problem.AddParameterBlock(mapFixed_delta_xi[pKF].data(), 6);
        problem.SetParameterBlockConstant(mapFixed_delta_xi[pKF].data());
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

    for (KeyFrame *pKF : vpLocalKFs)
    {
        const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        double *pose_param = mapKF_delta_xi[pKF].data();
        const Sophus::SE3d &T_init = mapKF_SE3[pKF];

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

            ceres::CostFunction *cost = nullptr;
            bool isStereo = false;
            if (u_r >= 0.0f && depth > 0.0f && depth < pKF->mThDepth)
            {
                cost = new LocalRepoErrorStereoAnalytic(fx, fy, cx, cy, pKF->mbf,
                                                        kp.pt.x, kp.pt.y, u_r, T_init, sqrtInvSigma2);
                isStereo = true;
            }
            else
            {
                cost = new LocalRepoErrorAnalytic(fx, fy, cx, cy,
                                                  kp.pt.x, kp.pt.y, T_init, sqrtInvSigma2);
            }

            ceres::LossFunction *loss = new ceres::HuberLoss(std::sqrt(5.991));
            ceres::ResidualBlockId id = problem.AddResidualBlock(cost, loss, pose_param, mapMP_Point[pMP].data());
            vObsInfo.push_back({pKF, pMP, j, id, isStereo});
        }
    }

    for (KeyFrame *pKF : vpFixedKFs)
    {
        const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        double *pose_param = mapFixed_delta_xi[pKF].data();
        const Sophus::SE3d &T_init = mapKF_SE3[pKF];

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

            ceres::CostFunction *cost = new LocalRepoErrorAnalytic(fx, fy, cx, cy,
                                                                   kp.pt.x, kp.pt.y, T_init, sqrtInvSigma2);
            ceres::LossFunction *loss = new ceres::HuberLoss(std::sqrt(5.991));
            ceres::ResidualBlockId id = problem.AddResidualBlock(cost, loss, pose_param, mapMP_Point[pMP].data());
            vObsInfo.push_back({pKF, pMP, j, id, false});
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_threads = 4;
    options.minimizer_progress_to_stdout = false;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-4;

    AbortCallback callback(pbStopFlag);
    if (pbStopFlag)
        options.callbacks.push_back(&callback);

    // 第一阶段：粗优化
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary1;
    ceres::Solve(options, &problem, &summary1);

    if (pbStopFlag && *pbStopFlag)
        return;

    // 阶段间外点剔除
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

        const auto &d_xi = mapKF_delta_xi.count(pKF) ? mapKF_delta_xi[pKF] : mapFixed_delta_xi[pKF];
        Eigen::Matrix<double, 6, 1> xi;
        xi << d_xi[3], d_xi[4], d_xi[5], d_xi[0], d_xi[1], d_xi[2];
        Sophus::SE3d T_cw_curr = Sophus::SE3d::exp(xi) * mapKF_SE3[pKF];

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

    // 第二阶段：精优化
    options.max_num_iterations = 10;
    ceres::Solver::Summary summary2;
    ceres::Solve(options, &problem, &summary2);

    if (pbStopFlag && *pbStopFlag)
        return;

    // 回写优化结果
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
            const auto &d_xi = mapKF_delta_xi[pKF];
            Eigen::Matrix<double, 6, 1> xi;
            xi << d_xi[3], d_xi[4], d_xi[5], d_xi[0], d_xi[1], d_xi[2];
            Sophus::SE3d T_opt = Sophus::SE3d::exp(xi) * mapKF_SE3[pKF];
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

void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF, const Eigen::Matrix4f &Tcw_loop)
{
    std::vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const int N = vpKFs.size();
    if (N < 2)
        return;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.5);

    std::map<KeyFrame *, Sophus::SE3d> mapOldPoses;
    std::map<KeyFrame *, std::vector<double>> mapKF_delta;

    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();
        Sophus::SE3d T_cw(q, t);

        mapOldPoses[pKF] = T_cw;
        mapKF_delta[pKF] = std::vector<double>(6, 0.0);

        problem.AddParameterBlock(mapKF_delta[pKF].data(), 6);

        // 固定初始第 0 帧与闭环目标帧，作为绝对基准
        if (pKF->mnId == 0 || pKF == pLoopKF)
        {
            problem.SetParameterBlockConstant(mapKF_delta[pKF].data());
        }
    }

    // 1. 生成树（Spanning Tree）相对边约束
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad || pKF->mnId == 0)
            continue;
        KeyFrame *pParent = pKF->GetParent();
        if (!pParent || pParent == pKF || !mapOldPoses.count(pParent) || !mapOldPoses.count(pKF))
            continue;

        Sophus::SE3d T_child_parent_meas = mapOldPoses[pKF] * mapOldPoses[pParent].inverse();

        ceres::CostFunction *cost = new PoseGraphSE3Analytic(
            T_child_parent_meas, mapOldPoses[pKF], mapOldPoses[pParent]);
        problem.AddResidualBlock(cost, loss_function,
                                 mapKF_delta[pKF].data(), mapKF_delta[pParent].data());
    }

    // 2. 添加闭环强约束边
    if (pCurKF != pLoopKF && mapOldPoses.count(pCurKF) && mapOldPoses.count(pLoopKF))
    {
        Eigen::Matrix3d R_loop = Tcw_loop.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t_loop = Tcw_loop.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q_loop(R_loop);
        q_loop.normalize();
        Sophus::SE3d Tcw_loop_se3(q_loop, t_loop);

        Sophus::SE3d T_cur_loop_meas = Tcw_loop_se3 * mapOldPoses[pLoopKF].inverse();

        ceres::CostFunction *cost = new PoseGraphSE3Analytic(
            T_cur_loop_meas, mapOldPoses[pCurKF], mapOldPoses[pLoopKF]);
        problem.AddResidualBlock(cost, nullptr,
                                 mapKF_delta[pCurKF].data(), mapKF_delta[pLoopKF].data());
    }

    // 3. 求解位姿图
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 4. 回写优化后的位姿
    std::map<KeyFrame *, Sophus::SE3d> mapNewPoses;
    for (auto &kv : mapOldPoses)
    {
        KeyFrame *pKF = kv.first;
        const auto &d = mapKF_delta[pKF];
        Eigen::Matrix<double, 6, 1> xi;
        xi << d[3], d[4], d[5], d[0], d[1], d[2];
        Sophus::SE3d T_new = Sophus::SE3d::exp(xi) * kv.second;

        pKF->SetPose(T_new.matrix().cast<float>());
        mapNewPoses[pKF] = T_new;
    }

    // 5. 地图点根据参考帧位姿变化更新
    std::vector<MapPoint *> vpAllMPs = pMap->GetAllMapPoints();
    for (MapPoint *pMP : vpAllMPs)
    {
        if (!pMP || pMP->isBad())
            continue;
        KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
        if (!pRefKF || !mapOldPoses.count(pRefKF) || !mapNewPoses.count(pRefKF))
            continue;

        Sophus::SE3d T_old_cw = mapOldPoses[pRefKF];
        Sophus::SE3d T_new_wc = mapNewPoses[pRefKF].inverse();

        Eigen::Vector3d Pw_old = pMP->GetWorldPos().cast<double>();
        Eigen::Vector3d Pw_new = T_new_wc * (T_old_cw * Pw_old);

        pMP->SetWorldPos(Pw_new.cast<float>());
        pMP->UpdateNormalAndDepth();
    }
}

void Optimizer::GlobalBundleAdjustment(Map *pMap, int nIterations, bool *pbStopFlag)
{
    if (!pMap)
        return;

    std::vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    std::vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();
    if (vpKFs.size() < 2 || vpMPs.empty())
        return;

    std::map<KeyFrame *, Sophus::SE3d> mapKF_SE3;
    for (KeyFrame *pKF : vpKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;
        Eigen::Matrix4f Tcw = pKF->GetPose();
        Eigen::Matrix3d R = Tcw.block<3, 3>(0, 0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(R);
        q.normalize();
        mapKF_SE3[pKF] = Sophus::SE3d(q, t);
    }

    std::map<MapPoint *, Eigen::Vector3d> mapMP_Point;
    for (MapPoint *pMP : vpMPs)
    {
        if (!pMP || pMP->isBad() || pMP->GetObservations().size() < 2)
            continue;
        mapMP_Point[pMP] = pMP->GetWorldPos().cast<double>();
    }

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(5.991));

    std::map<KeyFrame *, std::vector<double>> mapKF_delta;
    for (auto &kv : mapKF_SE3)
    {
        KeyFrame *pKF = kv.first;
        mapKF_delta[pKF] = std::vector<double>(6, 0.0);
        problem.AddParameterBlock(mapKF_delta[pKF].data(), 6);
        if (pKF->mnId == 0)
            problem.SetParameterBlockConstant(mapKF_delta[pKF].data());
    }

    for (auto &kv : mapMP_Point)
    {
        problem.AddParameterBlock(kv.second.data(), 3);
    }

    for (auto &kv : mapKF_SE3)
    {
        KeyFrame *pKF = kv.first;
        const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
        const std::vector<MapPoint *> vpMatches = pKF->GetMapPointMatches();
        double *pose_param = mapKF_delta[pKF].data();
        const Sophus::SE3d &T_init = kv.second;

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
                                                        kp.pt.x, kp.pt.y, u_r, T_init, sqrtInvSigma2);
            }
            else
            {
                cost = new LocalRepoErrorAnalytic(fx, fy, cx, cy,
                                                  kp.pt.x, kp.pt.y, T_init, sqrtInvSigma2);
            }
            problem.AddResidualBlock(cost, loss_function, pose_param, mapMP_Point[pMP].data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = nIterations;
    options.num_threads = 4; // 开启多线程并行舒尔补分解
    options.minimizer_progress_to_stdout = false;

    AbortCallback callback(pbStopFlag);
    if (pbStopFlag)
        options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (pbStopFlag && *pbStopFlag)
        return;

    {
        std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

        for (auto &kv : mapKF_SE3)
        {
            KeyFrame *pKF = kv.first;
            const auto &d = mapKF_delta[pKF];
            Eigen::Matrix<double, 6, 1> xi;
            xi << d[3], d[4], d[5], d[0], d[1], d[2];
            Sophus::SE3d T_opt = Sophus::SE3d::exp(xi) * kv.second;
            pKF->SetPose(T_opt.matrix().cast<float>());
        }

        for (auto &kv : mapMP_Point)
        {
            MapPoint *pMP = kv.first;
            if (pMP->isBad())
                continue;
            pMP->SetWorldPos(kv.second.cast<float>());
            pMP->UpdateNormalAndDepth();
        }
    }
}