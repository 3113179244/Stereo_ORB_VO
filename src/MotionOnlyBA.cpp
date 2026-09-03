#include "MotionOnlyBA.h"
#include "Frame.h"
#include "MapPoint.h"
#include "ORBextractor.h"

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <mutex>

/**
 * @brief Sophus::SE3d 的 Ceres Manifold 实现
 * 状态量环境空间维度 AmbientSize = 7 (quaternion [x, y, z, w], translation [x, y, z])
 * 切空间局部扰动维度 TangentSize = 6 (se(3) Lie algebra [v, w] 或 [w, v])
 */
class SophusSE3Manifold : public ceres::Manifold
{
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    // x_plus_delta = exp(delta) * x (左乘扰动更新)
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override
    {
        Eigen::Map<const Sophus::SE3d> T(x);
        // delta 映射为 6 维切空间向量: [vx, vy, vz, wx, wy, wz]
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> xi(delta);

        Sophus::SE3d T_plus = Sophus::SE3d::exp(xi) * T;

        Eigen::Map<Sophus::SE3d> result(x_plus_delta);
        result = T_plus;
        return true;
    }

    // Plus 的雅可比矩阵: d(Plus(x, delta)) / d(delta)|_{delta=0}
    bool PlusJacobian(const double *x, double *jacobian) const override
    {
        ceres::MatrixRef J(jacobian, 7, 6);
        J.setZero();

        // 也可以使用 ceres::AutoDiffManifold<SophusSE3ManifoldFunctor, 7, 6> 自动求导
        // 若手动推导，数值有限差分最为稳健通用：
        const double eps = 1e-8;
        double x_plus[7];
        double delta[6] = {0};

        for (int i = 0; i < 6; ++i)
        {
            delta[i] = eps;
            Plus(x, delta, x_plus);
            for (int r = 0; r < 7; ++r)
            {
                J(r, i) = (x_plus[r] - x[r]) / eps;
            }
            delta[i] = 0.0;
        }
        return true;
    }

    // 计算切空间差异: y ⊖ x
    bool Minus(const double *y, const double *x, double *y_minus_x) const override
    {
        Eigen::Map<const Sophus::SE3d> T_y(y);
        Eigen::Map<const Sophus::SE3d> T_x(x);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> xi(y_minus_x);
        xi = (T_y * T_x.inverse()).log();
        return true;
    }

    bool MinusJacobian(const double *x, double *jacobian) const override
    {
        ceres::MatrixRef J(jacobian, 6, 7);
        J.setZero();
        return true;
    }
};

/**
 * @brief 单目重投影误差 (AutoDiff)
 */
struct ReprojectionErrorMono
{
    ReprojectionErrorMono(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                          const Eigen::Matrix3d &K, double inv_sigma)
        : observed_(observed), point_3d_(point_3d), fx_(K(0, 0)), fy_(K(1, 1)),
          cx_(K(0, 2)), cy_(K(1, 2)), inv_sigma_(inv_sigma) {}

    template <typename T>
    bool operator()(const T *const se3_raw, T *residuals) const
    {
        // 映射为 Sophus 对象
        Eigen::Map<const Sophus::SE3<T>> T_cw(se3_raw);
        Eigen::Matrix<T, 3, 1> p_w = point_3d_.cast<T>();
        Eigen::Matrix<T, 3, 1> p_c = T_cw * p_w;

        T inv_z = T(1.0) / p_c[2];
        T u = T(fx_) * p_c[0] * inv_z + T(cx_);
        T v = T(fy_) * p_c[1] * inv_z + T(cy_);

        residuals[0] = (u - T(observed_[0])) * T(inv_sigma_);
        residuals[1] = (v - T(observed_[1])) * T(inv_sigma_);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                                       const Eigen::Matrix3d &K, double inv_sigma)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorMono, 2, 7>(
            new ReprojectionErrorMono(observed, point_3d, K, inv_sigma));
    }

    Eigen::Vector2d observed_;
    Eigen::Vector3d point_3d_;
    double fx_, fy_, cx_, cy_, inv_sigma_;
};

/**
 * @brief 双目重投影误差 (AutoDiff)
 */
struct ReprojectionErrorStereo
{
    ReprojectionErrorStereo(const Eigen::Vector3d &observed, const Eigen::Vector3d &point_3d,
                            const Eigen::Matrix3d &K, double bf, double inv_sigma)
        : observed_(observed), point_3d_(point_3d), fx_(K(0, 0)), fy_(K(1, 1)),
          cx_(K(0, 2)), cy_(K(1, 2)), bf_(bf), inv_sigma_(inv_sigma) {}

    template <typename T>
    bool operator()(const T *const se3_raw, T *residuals) const
    {
        Eigen::Map<const Sophus::SE3<T>> T_cw(se3_raw);
        Eigen::Matrix<T, 3, 1> p_w = point_3d_.cast<T>();
        Eigen::Matrix<T, 3, 1> p_c = T_cw * p_w;

        T inv_z = T(1.0) / p_c[2];
        T u = T(fx_) * p_c[0] * inv_z + T(cx_);
        T v = T(fy_) * p_c[1] * inv_z + T(cy_);
        T u_r = u - T(bf_) * inv_z;

        residuals[0] = (u - T(observed_[0])) * T(inv_sigma_);
        residuals[1] = (v - T(observed_[1])) * T(inv_sigma_);
        residuals[2] = (u_r - T(observed_[2])) * T(inv_sigma_);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &observed, const Eigen::Vector3d &point_3d,
                                       const Eigen::Matrix3d &K, double bf, double inv_sigma)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorStereo, 3, 7>(
            new ReprojectionErrorStereo(observed, point_3d, K, bf, inv_sigma));
    }

    Eigen::Vector3d observed_;
    Eigen::Vector3d point_3d_;
    double fx_, fy_, cx_, cy_, bf_, inv_sigma_;
};

int MotionOnlyBA::Optimize(Frame *pFrame)
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

    // 内参提取
    Eigen::Matrix3d K;
    K << pFrame->mK.at<float>(0, 0), 0.0, pFrame->mK.at<float>(0, 2),
        0.0, pFrame->mK.at<float>(1, 1), pFrame->mK.at<float>(1, 2),
        0.0, 0.0, 1.0;

    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);
    const double mbf = pFrame->mbf;

    // 位姿初始化
    Eigen::Matrix3d R_cw = pFrame->mTcw.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d t_cw = pFrame->mTcw.block<3, 1>(0, 3).cast<double>();

    Eigen::Quaterniond q_cw(R_cw);
    q_cw.normalize(); // 消除数值漂移，保证正交性

    Sophus::SE3d T_cw(q_cw, t_cw);

    const int its[4] = {10, 10, 10, 10};
    const double chi2_mono = 5.991;
    const double chi2_stereo = 7.815;

    int num_inliers = 0;

    // 4 轮迭代优化（含外点剔除）
    for (int it = 0; it < 4; ++it)
    {
        ceres::Problem problem;

        // 绑定 Sophus 自定义 Manifold
        SophusSE3Manifold *se3_manifold = new SophusSE3Manifold();
        problem.AddParameterBlock(T_cw.data(), 7, se3_manifold);

        {
            std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; ++i)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (!pMP || pMP->isBad() || pFrame->mvbOutlier[i])
                    continue;

                Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
                const int level = pFrame->mvKeysUn[i].octave;
                const double inv_sigma = 1.0 / std::sqrt(pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level]);
                const float u_r = pFrame->mvuRight[i];

                // 前两轮引入 Huber 核函数抑制粗差点
                ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_mono)) : nullptr;

                if (u_r < 0.0f) // 单目残差
                {
                    Eigen::Vector2d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y);
                    ceres::CostFunction *cost_function =
                        ReprojectionErrorMono::Create(obs, P_w, K, inv_sigma);
                    problem.AddResidualBlock(cost_function, loss_function, T_cw.data());
                }
                else // 双目残差
                {
                    Eigen::Vector3d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y, u_r);
                    if (loss_function)
                        loss_function = new ceres::HuberLoss(std::sqrt(chi2_stereo));

                    ceres::CostFunction *cost_function =
                        ReprojectionErrorStereo::Create(obs, P_w, K, mbf, inv_sigma);
                    problem.AddResidualBlock(cost_function, loss_function, T_cw.data());
                }
            }
        }

        // Ceres 配置与求解
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = its[it];
        options.num_threads = 1;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 重新投影检验内点并标记 Outlier
        num_inliers = 0;
        {
            std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
            for (int i = 0; i < N; ++i)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (!pMP || pMP->isBad())
                    continue;

                Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
                Eigen::Vector3d P_c = T_cw * P_w;
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

                if (u_r < 0.0f)
                {
                    const double chi2 = (du * du + dv * dv) * inv_sigma2;
                    if (chi2 > chi2_mono)
                        pFrame->mvbOutlier[i] = true;
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
                        pFrame->mvbOutlier[i] = true;
                    else
                    {
                        pFrame->mvbOutlier[i] = false;
                        num_inliers++;
                    }
                }
            }
        }
    }

    // 回写优化后的位姿
    Eigen::Quaterniond q_res = T_cw.unit_quaternion();
    q_res.normalize();
    Sophus::SE3d T_cw_normalized(q_res, T_cw.translation());
    pFrame->SetPose(T_cw_normalized.matrix().cast<float>());
    
    return num_inliers;
}