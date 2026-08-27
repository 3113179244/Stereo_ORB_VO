#include "MotionOnlyBA.h"
#include "Frame.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <mutex>
#include <iostream>

/**
 * @brief 单目解析求导（优化变量为切空间 6 维扰动量 xi = [w, v]，初值为 0）
 */
class ReprojectionErrorMonoAnalytic : public ceres::SizedCostFunction<2, 6>
{
public:
    ReprojectionErrorMonoAnalytic(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                                  const Sophus::SE3d &T_cw_initial, const Eigen::Matrix3d &K, double inv_sigma)
        : observed_(observed), point_3d_(point_3d), T_cw_initial_(T_cw_initial), K_(K), inv_sigma_(inv_sigma) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        const double *xi_raw = parameters[0];
        
        // 增量扰动 xi: [wx, wy, wz, vx, vy, vz]
        Eigen::Matrix<double, 6, 1> xi;
        xi << xi_raw[3], xi_raw[4], xi_raw[5], xi_raw[0], xi_raw[1], xi_raw[2]; // Sophus::exp 顺序: [v, w]
        
        Sophus::SE3d T_cw = Sophus::SE3d::exp(xi) * T_cw_initial_;
        Eigen::Vector3d P_c = T_cw * point_3d_;

        const double x = P_c[0];
        const double y = P_c[1];
        const double z = P_c[2];

        if (z <= 1e-4)
            return false;

        const double inv_z = 1.0 / z;
        const double inv_z2 = inv_z * inv_z;
        const double fx = K_(0, 0);
        const double fy = K_(1, 1);
        const double cx = K_(0, 2);
        const double cy = K_(1, 2);

        const double u = fx * x * inv_z + cx;
        const double v = fy * y * inv_z + cy;

        // 残差
        residuals[0] = (u - observed_[0]) * inv_sigma_;
        residuals[1] = (v - observed_[1]) * inv_sigma_;

        // 雅可比 (2x6)
        if (jacobians && jacobians[0])
        {
            double *jacobian = jacobians[0];

            const double dedx_0 = fx * inv_z;
            const double dedz_0 = -fx * x * inv_z2;
            const double dedy_1 = fy * inv_z;
            const double dedz_1 = -fy * y * inv_z2;

            // Row 0 (du)
            jacobian[0] = inv_sigma_ * (dedz_0 * y);
            jacobian[1] = inv_sigma_ * (dedx_0 * z - dedz_0 * x);
            jacobian[2] = inv_sigma_ * (-dedx_0 * y);
            jacobian[3] = inv_sigma_ * dedx_0;
            jacobian[4] = 0.0;
            jacobian[5] = inv_sigma_ * dedz_0;

            // Row 1 (dv)
            jacobian[6]  = inv_sigma_ * (-dedy_1 * z + dedz_1 * y);
            jacobian[7]  = inv_sigma_ * (-dedz_1 * x);
            jacobian[8]  = inv_sigma_ * (dedy_1 * x);
            jacobian[9]  = 0.0;
            jacobian[10] = inv_sigma_ * dedy_1;
            jacobian[11] = inv_sigma_ * dedz_1;
        }

        return true;
    }

private:
    Eigen::Vector2d observed_;
    Eigen::Vector3d point_3d_;
    Sophus::SE3d T_cw_initial_;
    Eigen::Matrix3d K_;
    double inv_sigma_;
};

/**
 * @brief 双目解析求导（优化变量为切空间 6 维扰动量 xi = [w, v]，初值为 0）
 */
class ReprojectionErrorStereoAnalytic : public ceres::SizedCostFunction<3, 6>
{
public:
    ReprojectionErrorStereoAnalytic(const Eigen::Vector3d &observed, const Eigen::Vector3d &point_3d,
                                    const Sophus::SE3d &T_cw_initial, const Eigen::Matrix3d &K, double bf, double inv_sigma)
        : observed_(observed), point_3d_(point_3d), T_cw_initial_(T_cw_initial), K_(K), bf_(bf), inv_sigma_(inv_sigma) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override
    {
        const double *xi_raw = parameters[0];

        Eigen::Matrix<double, 6, 1> xi;
        xi << xi_raw[3], xi_raw[4], xi_raw[5], xi_raw[0], xi_raw[1], xi_raw[2];

        Sophus::SE3d T_cw = Sophus::SE3d::exp(xi) * T_cw_initial_;
        Eigen::Vector3d P_c = T_cw * point_3d_;

        const double x = P_c[0];
        const double y = P_c[1];
        const double z = P_c[2];

        if (z <= 1e-4)
            return false;

        const double inv_z = 1.0 / z;
        const double inv_z2 = inv_z * inv_z;
        const double fx = K_(0, 0);
        const double fy = K_(1, 1);
        const double cx = K_(0, 2);
        const double cy = K_(1, 2);

        const double u = fx * x * inv_z + cx;
        const double v = fy * y * inv_z + cy;
        const double u_r = u - bf_ * inv_z;

        // 残差
        residuals[0] = (u - observed_[0]) * inv_sigma_;
        residuals[1] = (v - observed_[1]) * inv_sigma_;
        residuals[2] = (u_r - observed_[2]) * inv_sigma_;

        // 雅可比 (3x6)
        if (jacobians && jacobians[0])
        {
            double *jacobian = jacobians[0];

            const double dedx_0 = fx * inv_z;
            const double dedz_0 = -fx * x * inv_z2;
            const double dedy_1 = fy * inv_z;
            const double dedz_1 = -fy * y * inv_z2;
            const double dedx_2 = dedx_0;
            const double dedz_2 = -(fx * x - bf_) * inv_z2;

            // Row 0 (duL)
            jacobian[0] = inv_sigma_ * (dedz_0 * y);
            jacobian[1] = inv_sigma_ * (dedx_0 * z - dedz_0 * x);
            jacobian[2] = inv_sigma_ * (-dedx_0 * y);
            jacobian[3] = inv_sigma_ * dedx_0;
            jacobian[4] = 0.0;
            jacobian[5] = inv_sigma_ * dedz_0;

            // Row 1 (dvL)
            jacobian[6]  = inv_sigma_ * (-dedy_1 * z + dedz_1 * y);
            jacobian[7]  = inv_sigma_ * (-dedz_1 * x);
            jacobian[8]  = inv_sigma_ * (dedy_1 * x);
            jacobian[9]  = 0.0;
            jacobian[10] = inv_sigma_ * dedy_1;
            jacobian[11] = inv_sigma_ * dedz_1;

            // Row 2 (duR)
            jacobian[12] = inv_sigma_ * (dedz_2 * y);
            jacobian[13] = inv_sigma_ * (dedx_2 * z - dedz_2 * x);
            jacobian[14] = inv_sigma_ * (-dedx_2 * y);
            jacobian[15] = inv_sigma_ * dedx_2;
            jacobian[16] = 0.0;
            jacobian[17] = inv_sigma_ * dedz_2;
        }

        return true;
    }

private:
    Eigen::Vector3d observed_;
    Eigen::Vector3d point_3d_;
    Sophus::SE3d T_cw_initial_;
    Eigen::Matrix3d K_;
    double bf_;
    double inv_sigma_;
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

    Eigen::Matrix3d K_eigen;
    K_eigen << pFrame->mK.at<float>(0, 0), 0, pFrame->mK.at<float>(0, 2),
               0, pFrame->mK.at<float>(1, 1), pFrame->mK.at<float>(1, 2),
               0, 0, 1;

    Eigen::Matrix3d R_cw = pFrame->mTcw.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d t_cw = pFrame->mTcw.block<3, 1>(0, 3).cast<double>();

    // 保证初始 R_cw 严格正交
    Eigen::Quaterniond q_init(R_cw);
    q_init.normalize();
    Sophus::SE3d T_cw(q_init, t_cw);

    const int its[4] = {10, 10, 10, 10};
    const double chi2_mono = 5.991;
    const double chi2_stereo = 7.815;

    int num_inliers = 0;

    for (int it = 0; it < 4; ++it)
    {
        ceres::Problem problem;
        int num_edges = 0;

        // 本轮迭代优化的 6 维增量（初值严格为 0）
        double delta_xi[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        {
            std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; ++i)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (!pMP || pMP->isBad())
                    continue;

                if (pFrame->mvbOutlier[i])
                    continue;

                Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
                Eigen::Vector3d P_c = T_cw * P_w;
                double depth = P_c[2];

                if (depth <= 0.0)
                {
                    pFrame->mvbOutlier[i] = true;
                    continue;
                }

                const int level = pFrame->mvKeysUn[i].octave;
                const double inv_sigma = 1.0 / std::sqrt(pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level]);
                const float u_r = pFrame->mvuRight[i];

                if (u_r < 0.0f)
                {
                    ceres::CostFunction *cost_function = new ReprojectionErrorMonoAnalytic(
                        Eigen::Vector2d(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y),
                        P_w, T_cw, K_eigen, inv_sigma);
                    ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_mono)) : nullptr;
                    problem.AddResidualBlock(cost_function, loss_function, delta_xi);
                }
                else
                {
                    ceres::CostFunction *cost_function = new ReprojectionErrorStereoAnalytic(
                        Eigen::Vector3d(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y, u_r),
                        P_w, T_cw, K_eigen, pFrame->mbf, inv_sigma);
                    ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_stereo)) : nullptr;
                    problem.AddResidualBlock(cost_function, loss_function, delta_xi);
                }
                num_edges++;
            }
        }

        if (num_edges < 10)
            break;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = its[it];
        options.function_tolerance = 1e-4;
        options.gradient_tolerance = 1e-4;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << std::endl;
        // 优化结束后将增量更新到 T_cw
        Eigen::Matrix<double, 6, 1> xi_opt;
        xi_opt << delta_xi[3], delta_xi[4], delta_xi[5], delta_xi[0], delta_xi[1], delta_xi[2];
        T_cw = Sophus::SE3d::exp(xi_opt) * T_cw;

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
                double depth = P_c[2];

                if (depth <= 0.0)
                {
                    pFrame->mvbOutlier[i] = true;
                    continue;
                }

                double inv_z = 1.0 / depth;
                double u = K_eigen(0, 0) * P_c[0] * inv_z + K_eigen(0, 2);
                double v = K_eigen(1, 1) * P_c[1] * inv_z + K_eigen(1, 2);

                const int level = pFrame->mvKeysUn[i].octave;
                const double inv_sigma2 = 1.0 / pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level];
                const float u_r = pFrame->mvuRight[i];

                double du = u - pFrame->mvKeysUn[i].pt.x;
                double dv = v - pFrame->mvKeysUn[i].pt.y;

                if (u_r < 0.0f)
                {
                    double chi2 = (du * du + dv * dv) * inv_sigma2;
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
                    double u_r_proj = u - pFrame->mbf * inv_z;
                    double du_r = u_r_proj - u_r;
                    double chi2 = (du * du + dv * dv + du_r * du_r) * inv_sigma2;

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
    }

    // 回写优化位姿
    Eigen::Matrix4f optimizedTcw = T_cw.matrix().cast<float>();
    pFrame->SetPose(optimizedTcw);

    return num_inliers;
}