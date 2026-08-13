#include "MotionOnlyBA.h"
#include "Frame.h"
#include "MapPoint.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

/**
 * @brief 包含平移和旋转的标准重投影误差残差块（3D-2D）
 */
struct ReprojectionError
{
    ReprojectionError(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K)
        : observed_(observed), point_3d_(point_3d), K_(K) {}

    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        T p_w[3] = {T(point_3d_[0]), T(point_3d_[1]), T(point_3d_[2])};
        T p_c[3];

        ceres::AngleAxisRotatePoint(se3, p_w, p_c);

        p_c[0] += se3[3];
        p_c[1] += se3[4];
        p_c[2] += se3[5];

        T x = p_c[0] / p_c[2];
        T y = p_c[1] / p_c[2];

        T fx = T(K_(0, 0)), fy = T(K_(1, 1));
        T cx = T(K_(0, 2)), cy = T(K_(1, 2));

        T u = fx * x + cx;
        T v = fy * y + cy;

        residuals[0] = u - T(observed_[0]);
        residuals[1] = v - T(observed_[1]);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
            new ReprojectionError(observed, point_3d, K));
    }

    Eigen::Vector2d observed_;
    Eigen::Vector3d point_3d_;
    Eigen::Matrix3d K_;
};

int MotionOnlyBA::Optimize(Frame *pFrame)
{
    if (!pFrame)
        return 0;

    const int N = pFrame->mvpMapPoints.size();
    pFrame->mvbOutlier.resize(N, false);

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
        t_cw[0], t_cw[1], t_cw[2]};

    ceres::Problem problem;

    int num_edges = 0;

    for (int i = 0; i < N; ++i)
    {
        MapPoint *pMP = pFrame->mvpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
        Eigen::Vector2d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y);

        Eigen::Vector3d P_c = R_cw * P_w + t_cw;
        double depth = P_c[2];

        // 剔除异常深度点
        if (depth <= 0 || depth >= pFrame->mThDepth)
        {
            pFrame->mvbOutlier[i] = true;
            continue;
        }

        ceres::CostFunction *cost_function = ReprojectionError::Create(obs, P_w, K_eigen);

        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), camera_pose);
        num_edges++;
    }

    if (num_edges < 5)
        return 0;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 更新优化后的 Pose
    Eigen::Vector3d opt_r(camera_pose[0], camera_pose[1], camera_pose[2]);
    Eigen::Vector3d opt_t(camera_pose[3], camera_pose[4], camera_pose[5]);

    double angle = opt_r.norm();
    Eigen::Matrix3d opt_R = Eigen::Matrix3d::Identity();
    if (angle > 1e-12)
    {
        opt_R = Eigen::AngleAxisd(angle, opt_r.normalized()).toRotationMatrix();
    }

    Eigen::Matrix4f optimizedTcw = Eigen::Matrix4f::Identity();
    optimizedTcw.block<3, 3>(0, 0) = opt_R.cast<float>();
    optimizedTcw.block<3, 1>(0, 3) = opt_t.cast<float>();
    pFrame->SetPose(optimizedTcw);

    // 统计优化后的内点/外点
    int num_inliers = 0;
    const double chi2_threshold = 7.815; // 2 DOF, 95% confidence

    for (int i = 0; i < N; ++i)
    {
        MapPoint *pMP = pFrame->mvpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
        Eigen::Vector3d P_c = opt_R * P_w + opt_t;

        if (P_c[2] <= 0 || P_c[2] >= pFrame->mThDepth)
        {
            pFrame->mvbOutlier[i] = true;
            continue;
        }

        double inv_z = 1.0 / P_c[2];
        double u = K_eigen(0, 0) * P_c[0] * inv_z + K_eigen(0, 2);
        double v = K_eigen(1, 1) * P_c[1] * inv_z + K_eigen(1, 2);

        double du = u - pFrame->mvKeysUn[i].pt.x;
        double dv = v - pFrame->mvKeysUn[i].pt.y;
        double err_sq = du * du + dv * dv;

        if (err_sq > chi2_threshold)
        {
            pFrame->mvbOutlier[i] = true;
        }
        else
        {
            pFrame->mvbOutlier[i] = false;
            num_inliers++;
        }
    }

    return num_inliers;
}