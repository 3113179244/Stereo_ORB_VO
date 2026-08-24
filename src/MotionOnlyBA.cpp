#include "MotionOnlyBA.h"
#include "Frame.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <mutex>

/**
 * @brief 单目重投影误差残差块（2D 残差，约束左目 u, v）
 */
struct ReprojectionErrorMono
{
    ReprojectionErrorMono(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
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

        // 加权残差: [uL, vL]
        residuals[0] = (u - T(observed_[0])) * T(inv_sigma_);
        residuals[1] = (v - T(observed_[1])) * T(inv_sigma_);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d,
                                       const Eigen::Matrix3d &K, double inv_sigma)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErrorMono, 2, 6>(
            new ReprojectionErrorMono(observed, point_3d, K, inv_sigma));
    }

    Eigen::Vector2d observed_; // [u, v]
    Eigen::Vector3d point_3d_;
    Eigen::Matrix3d K_;
    double inv_sigma_;
};

/**
 * @brief 双目重投影误差残差块（3D 残差，约束左目 u, v 以及右目 uR）
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

        // 加权残差: [uL, vL, uR]
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

    // 对应 ORB-SLAM2: 检查初始有效匹配点数
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

    Eigen::AngleAxisd angle_axis_cw(R_cw);
    Eigen::Vector3d r_vec = angle_axis_cw.angle() * angle_axis_cw.axis();

    double camera_pose[6] = {
        r_vec[0], r_vec[1], r_vec[2],
        t_cw[0], t_cw[1], t_cw[2]
    };

    const int its[4] = {10, 10, 10, 10};
    const double chi2_mono = 5.991;    // 2自由度 Chi-Square 阈值
    const double chi2_stereo = 7.815;  // 3自由度 Chi-Square 阈值

    int num_inliers = 0;

    // 4 轮迭代优化
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

        // 修改点 3：添加 MapPoint 全局互斥锁，防止读取时被其他线程并发修改
        {
            std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; ++i)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (!pMP || pMP->isBad())
                    continue;

                // 排除当前被标记为 Outlier 的点（不参与本轮优化）
                if (pFrame->mvbOutlier[i])
                    continue;

                Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
                Eigen::Vector3d P_c = cur_R * P_w + cur_t;
                double depth = P_c[2];

                if (depth <= 0.0)
                {
                    pFrame->mvbOutlier[i] = true;
                    continue;
                }

                const int level = pFrame->mvKeysUn[i].octave;
                const double inv_sigma = 1.0 / std::sqrt(pFrame->mpORBextractorLeft->GetScaleSigmaSquares()[level]);
                const float u_r = pFrame->mvuRight[i];

                // 修改点 1：单目观测与双目观测自适应支持（兼容远点/右目未匹配点）
                if (u_r < 0.0f)
                {
                    // 单目 2DoF 观测
                    Eigen::Vector2d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y);
                    ceres::CostFunction *cost_function = ReprojectionErrorMono::Create(obs, P_w, K_eigen, inv_sigma);
                    ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_mono)) : nullptr;
                    problem.AddResidualBlock(cost_function, loss_function, camera_pose);
                }
                else
                {
                    // 双目 3DoF 观测
                    Eigen::Vector3d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y, u_r);
                    ceres::CostFunction *cost_function = ReprojectionErrorStereo::Create(obs, P_w, K_eigen, pFrame->mbf, inv_sigma);
                    ceres::LossFunction *loss_function = (it < 2) ? new ceres::HuberLoss(std::sqrt(chi2_stereo)) : nullptr;
                    problem.AddResidualBlock(cost_function, loss_function, camera_pose);
                }
                num_edges++;
            }
        } // 离开临界区

        // 边数过少时提前退出
        if (num_edges < 10)
            break;

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = its[it];
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 提取优化后的位姿用于卡方检验
        Eigen::Vector3d opt_r(camera_pose[0], camera_pose[1], camera_pose[2]);
        Eigen::Vector3d opt_t(camera_pose[3], camera_pose[4], camera_pose[5]);
        double angle = opt_r.norm();
        Eigen::Matrix3d opt_R = Eigen::Matrix3d::Identity();
        if (angle > 1e-12)
        {
            opt_R = Eigen::AngleAxisd(angle, opt_r.normalized()).toRotationMatrix();
        }

        // 修改点 2：修复外点复活逻辑，每一轮优化后均对全部有效地图点进行卡方重判
        num_inliers = 0;
        {
            std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; ++i)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (!pMP || pMP->isBad())
                    continue;

                Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
                Eigen::Vector3d P_c = opt_R * P_w + opt_t;
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
                    // 单目卡方检验 (2DoF)
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
                    // 双目卡方检验 (3DoF)
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
        } // 离开临界区
    }

    // 回写优化位姿
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