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
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            if (jacobians && jacobians[0])
            {
                std::fill(jacobians[0], jacobians[0] + 12, 0.0);
            }
            return true;
        }

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
            jacobian[6] = inv_sigma_ * (-dedy_1 * z + dedz_1 * y);
            jacobian[7] = inv_sigma_ * (-dedz_1 * x);
            jacobian[8] = inv_sigma_ * (dedy_1 * x);
            jacobian[9] = 0.0;
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
        {
            residuals[0] = 0.0;
            residuals[1] = 0.0;
            residuals[2] = 0.0;
            if (jacobians && jacobians[0])
            {
                std::fill(jacobians[0], jacobians[0] + 18, 0.0);
            }
            return true;
        }

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
            jacobian[6] = inv_sigma_ * (-dedy_1 * z + dedz_1 * y);
            jacobian[7] = inv_sigma_ * (-dedz_1 * x);
            jacobian[8] = inv_sigma_ * (dedy_1 * x);
            jacobian[9] = 0.0;
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

    const float fx = pFrame->mK.at<float>(0, 0);
    const float fy = pFrame->mK.at<float>(1, 1);
    const float cx = pFrame->mK.at<float>(0, 2);
    const float cy = pFrame->mK.at<float>(1, 2);
    const float mbf = pFrame->mbf;

    Eigen::Matrix3d R_cw = pFrame->mTcw.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d t_cw = pFrame->mTcw.block<3, 1>(0, 3).cast<double>();
    Eigen::Quaterniond q_init(R_cw);
    q_init.normalize();
    Sophus::SE3d T_cw(q_init, t_cw);

    const int its[4] = {10, 10, 10, 10};
    const double chi2_mono = 5.991;
    const double chi2_stereo = 7.815;
    const double delta_mono = std::sqrt(chi2_mono);
    const double delta_stereo = std::sqrt(chi2_stereo);

    int num_inliers = 0;

    for (int it = 0; it < 4; ++it)
    {
        for (int iter = 0; iter < its[it]; ++iter)
        {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

            {
                std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

                for (int i = 0; i < N; ++i)
                {
                    MapPoint *pMP = pFrame->mvpMapPoints[i];
                    if (!pMP || pMP->isBad() || pFrame->mvbOutlier[i])
                        continue;

                    Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
                    Eigen::Vector3d P_c = T_cw * P_w;
                    const double x = P_c[0];
                    const double y = P_c[1];
                    const double z = P_c[2];

                    if (z <= 1e-4)
                        continue;

                    const double inv_z = 1.0 / z;
                    const double inv_z2 = inv_z * inv_z;
                    const int level = pFrame->mvKeysUn[i].octave;
                    const double inv_sigma2 = 1.0 / pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level];
                    const float u_r = pFrame->mvuRight[i];

                    const double u = fx * x * inv_z + cx;
                    const double v = fy * y * inv_z + cy;

                    if (u_r < 0.0f) // 单目残差
                    {
                        const double e_u = (u - pFrame->mvKeysUn[i].pt.x);
                        const double e_v = (v - pFrame->mvKeysUn[i].pt.y);
                        const double chi2 = (e_u * e_u + e_v * e_v) * inv_sigma2;

                        double w = 1.0;
                        if (it < 2)
                        {
                            const double r = std::sqrt(chi2);
                            if (r > delta_mono)
                                w = delta_mono / r;
                        }

                        Eigen::Matrix<double, 2, 6> J;
                        const double dedx_0 = fx * inv_z;
                        const double dedz_0 = -fx * x * inv_z2;
                        const double dedy_1 = fy * inv_z;
                        const double dedz_1 = -fy * y * inv_z2;

                        J(0, 0) = dedz_0 * y;
                        J(0, 1) = dedx_0 * z - dedz_0 * x;
                        J(0, 2) = -dedx_0 * y;
                        J(0, 3) = dedx_0;
                        J(0, 4) = 0.0;
                        J(0, 5) = dedz_0;

                        J(1, 0) = -dedy_1 * z + dedz_1 * y;
                        J(1, 1) = -dedz_1 * x;
                        J(1, 2) = dedy_1 * x;
                        J(1, 3) = 0.0;
                        J(1, 4) = dedy_1;
                        J(1, 5) = dedz_1;

                        Eigen::Vector2d e(e_u, e_v);
                        const double weight = inv_sigma2 * w;
                        H += weight * J.transpose() * J;
                        g += -weight * J.transpose() * e;
                    }
                    else // 双目残差
                    {
                        const double u_r_proj = u - mbf * inv_z;
                        const double e_u = (u - pFrame->mvKeysUn[i].pt.x);
                        const double e_v = (v - pFrame->mvKeysUn[i].pt.y);
                        const double e_ur = (u_r_proj - u_r);
                        const double chi2 = (e_u * e_u + e_v * e_v + e_ur * e_ur) * inv_sigma2;

                        double w = 1.0;
                        if (it < 2)
                        {
                            const double r = std::sqrt(chi2);
                            if (r > delta_stereo)
                                w = delta_stereo / r;
                        }

                        Eigen::Matrix<double, 3, 6> J;
                        const double dedx_0 = fx * inv_z;
                        const double dedz_0 = -fx * x * inv_z2;
                        const double dedy_1 = fy * inv_z;
                        const double dedz_1 = -fy * y * inv_z2;
                        const double dedx_2 = dedx_0;
                        const double dedz_2 = -(fx * x - mbf) * inv_z2;

                        J(0, 0) = dedz_0 * y;
                        J(0, 1) = dedx_0 * z - dedz_0 * x;
                        J(0, 2) = -dedx_0 * y;
                        J(0, 3) = dedx_0;
                        J(0, 4) = 0.0;
                        J(0, 5) = dedz_0;

                        J(1, 0) = -dedy_1 * z + dedz_1 * y;
                        J(1, 1) = -dedz_1 * x;
                        J(1, 2) = dedy_1 * x;
                        J(1, 3) = 0.0;
                        J(1, 4) = dedy_1;
                        J(1, 5) = dedz_1;

                        J(2, 0) = dedz_2 * y;
                        J(2, 1) = dedx_2 * z - dedz_2 * x;
                        J(2, 2) = -dedx_2 * y;
                        J(2, 3) = dedx_2;
                        J(2, 4) = 0.0;
                        J(2, 5) = dedz_2;

                        Eigen::Vector3d e(e_u, e_v, e_ur);
                        const double weight = inv_sigma2 * w;
                        H += weight * J.transpose() * J;
                        g += -weight * J.transpose() * e;
                    }
                }
            }

            Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(g);
            if (std::isnan(dx[0]))
                break;

            Eigen::Matrix<double, 6, 1> xi;
            xi << dx[3], dx[4], dx[5], dx[0], dx[1], dx[2];
            T_cw = Sophus::SE3d::exp(xi) * T_cw;

            if (dx.dot(dx) < 1e-12)
                break;
        }

        // 统计并标记 Outliers
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

    pFrame->SetPose(T_cw.matrix().cast<float>());
    return num_inliers;
}