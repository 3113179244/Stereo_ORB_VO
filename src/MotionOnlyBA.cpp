#include "MotionOnlyBA.h"
#include "Frame.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

/**
 * @brief 单目重投影误差残差块（2D-2D 残差，带信息权重）
 */
struct ReprojectionErrorMonocular
{
    ReprojectionErrorMonocular(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                               const Eigen::Matrix3d &K, double inv_sigma)
        : observed_(observed), point_3d_(point_3d), K_(K), inv_sigma_(inv_sigma) {}

    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        T p_w[3] = {T(point_3d_[0]), T(point_3d_[1]), T(point_3d_[2])};
        T p_c[3];

        ceres::AngleAxisRotatePoint(se3, p_w, p_c);

        p_c[0] += se3[3];
        p_c[1] += se3[4];
        p_c[2] += se3[5];

        T inv_z = T(1.0) / p_c[2];
        T u = T(K_(0, 0)) * p_c[0] * inv_z + T(K_(0, 2));
        T v = T(K_(1, 1)) * p_c[1] * inv_z + T(K_(1, 2));

        // 加权残差: (预测 - 观测) * 权重
        residuals[0] = (u - T(observed_[0])) * T(inv_sigma_);
        residuals[1] = (v - T(observed_[1])) * T(inv_sigma_);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                                       const Eigen::Matrix3d &K, double inv_sigma)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorMonocular, 2, 6>(
            new ReprojectionErrorMonocular(observed, point_3d, K, inv_sigma));
    }

    Eigen::Vector2d observed_;
    Eigen::Vector3d point_3d_;
    Eigen::Matrix3d K_;
    double inv_sigma_;
};

/**
 * @brief 双目重投影误差残差块（3D-2D 残差，包含左目 (u, v) 与右目 uR，带信息权重）
 */
struct ReprojectionErrorStereo
{
    ReprojectionErrorStereo(const Eigen::Vector3d &observed, const Eigen::Vector3d &point_3d,
                            const Eigen::Matrix3d &K, double bf, double inv_sigma)
        : observed_(observed), point_3d_(point_3d), K_(K), bf_(bf), inv_sigma_(inv_sigma) {}

    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        T p_w[3] = {T(point_3d_[0]), T(point_3d_[1]), T(point_3d_[2])};
        T p_c[3];

        ceres::AngleAxisRotatePoint(se3, p_w, p_c);

        p_c[0] += se3[3];
        p_c[1] += se3[4];
        p_c[2] += se3[5];

        T inv_z = T(1.0) / p_c[2];
        T u = T(K_(0, 0)) * p_c[0] * inv_z + T(K_(0, 2));
        T v = T(K_(1, 1)) * p_c[1] * inv_z + T(K_(1, 2));
        T u_r = u - T(bf_) * inv_z;

        // 3维残差加权: [uL, vL, uR]
        residuals[0] = (u - T(observed_[0])) * T(inv_sigma_);
        residuals[1] = (v - T(observed_[1])) * T(inv_sigma_);
        residuals[2] = (u_r - T(observed_[2])) * T(inv_sigma_);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &observed, const Eigen::Vector3d &point_3d,
                                       const Eigen::Matrix3d &K, double bf, double inv_sigma)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorStereo, 3, 6>(
            new ReprojectionErrorStereo(observed, point_3d, K, bf, inv_sigma));
    }

    Eigen::Vector3d observed_; // [u, v, uR]
    Eigen::Vector3d point_3d_;
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

    if (N < 5)
        return 0;

    Eigen::Matrix3d K_eigen;
    K_eigen << pFrame->mK.at<float>(0, 0), 0, pFrame->mK.at<float>(0, 2),
               0, pFrame->mK.at<float>(1, 1), pFrame->mK.at<float>(1, 2),
               0, 0, 1;

    Eigen::Matrix3d R_cw = pFrame->mTcw.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d t_cw = pFrame->mTcw.block<3, 1>(0, 3).cast<double>();

    Eigen::AngleAxisd angle_axis_cw(R_cw);
    Eigen::Vector3d r_vec = angle_axis_cw.angle() * angle_axis_cw.axis();

    double camera_pose[6] = {
        r_vec[0], r_vec[1], r_vec[2],
        t_cw[0], t_cw[1], t_cw[2]
    };

    const int its[4] = {10, 10, 10, 10};
    const double chi2_mono = 5.991;   // 2自由度 Chi-Square 阈值
    const double chi2_stereo = 7.815; // 3自由度 Chi-Square 阈值
    int num_inliers = 0;

    for (int it = 0; it < 4; ++it)
    {
        ceres::Problem problem;
        int num_edges = 0;

        Eigen::Vector3d cur_r(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Vector3d cur_t(camera_pose[3], camera_pose[4], camera_pose[5]);
        double cur_angle = cur_r.norm();
        Eigen::Matrix3d cur_R = Eigen::Matrix3d::Identity();
        if (cur_angle > 1e-12)
        {
            cur_R = Eigen::AngleAxisd(cur_angle, cur_r.normalized()).toRotationMatrix();
        }

        for (int i = 0; i < N; ++i)
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (!pMP || pMP->isBad())
                continue;

            if (pFrame->mvbOutlier[i])
                continue;

            Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
            Eigen::Vector3d P_c = cur_R * P_w + cur_t;
            double depth = P_c[2];

            if (depth <= 0)
            {
                pFrame->mvbOutlier[i] = true;
                continue;
            }

            const int level = pFrame->mvKeysUn[i].octave;
            const double inv_sigma = 1.0 / std::sqrt(pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level]);
            const float u_r = pFrame->mvuRight[i];

            ceres::CostFunction *cost_function = nullptr;
            if (u_r >= 0.0f)
            {
                // 双目/近点
                Eigen::Vector3d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y, u_r);
                cost_function = ReprojectionErrorStereo::Create(obs, P_w, K_eigen, pFrame->mbf, inv_sigma);
            }
            else
            {
                // 单目/远点
                Eigen::Vector2d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y);
                cost_function = ReprojectionErrorMonocular::Create(obs, P_w, K_eigen, inv_sigma);
            }

            ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(1.0) : nullptr;
            problem.AddResidualBlock(cost_function, loss_function, camera_pose);
            num_edges++;
        }

        if (num_edges < 5)
            break;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = its[it];
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 提取位姿并根据加权卡方统计量更新 Outlier
        Eigen::Vector3d opt_r(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Vector3d opt_t(camera_pose[3], camera_pose[4], camera_pose[5]);
        double angle = opt_r.norm();
        Eigen::Matrix3d opt_R = Eigen::Matrix3d::Identity();
        if (angle > 1e-12)
        {
            opt_R = Eigen::AngleAxisd(angle, opt_r.normalized()).toRotationMatrix();
        }

        num_inliers = 0;
        for (int i = 0; i < N; ++i)
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (!pMP || pMP->isBad())
                continue;

            Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
            Eigen::Vector3d P_c = opt_R * P_w + opt_t;

            if (P_c[2] <= 0)
            {
                pFrame->mvbOutlier[i] = true;
                continue;
            }

            double inv_z = 1.0 / P_c[2];
            double u = K_eigen(0, 0) * P_c[0] * inv_z + K_eigen(0, 2);
            double v = K_eigen(1, 1) * P_c[1] * inv_z + K_eigen(1, 2);

            const int level = pFrame->mvKeysUn[i].octave;
            const double inv_sigma2 = 1.0 / pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level];

            double du = u - pFrame->mvKeysUn[i].pt.x;
            double dv = v - pFrame->mvKeysUn[i].pt.y;
            const float u_r = pFrame->mvuRight[i];

            if (u_r >= 0.0f)
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
            else
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
        }
    }

    // 写回优化后的位姿
    Eigen::Vector3d final_r(camera_pose[0], camera_pose[1], camera_pose[2]);
    Eigen::Vector3d final_t(camera_pose[3], camera_pose[4], camera_pose[5]);
    double angle = final_r.norm();
    Eigen::Matrix3d final_R = Eigen::Matrix3d::Identity();
    if (angle > 1e-12)
    {
        final_R = Eigen::AngleAxisd(angle, final_r.normalized()).toRotationMatrix();
    }

    Eigen::Matrix4f optimizedTcw = Eigen::Matrix4f::Identity();
    optimizedTcw.block<3, 3>(0, 0) = final_R.cast<float>();
    optimizedTcw.block<3, 1>(0, 3) = final_t.cast<float>();
    pFrame->SetPose(optimizedTcw);

    return num_inliers;
}