// Source File: MotionOnlyBA.cpp
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
    // 构造函数：传入观测到的2D像素坐标、3D世界坐标点和相机内参矩阵 K
    ReprojectionError(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K)
        : observed_(observed), point_3d_(point_3d), K_(K) {}

    /**
     * @brief 计算重投影残差的运算符函数
     * @param se3 待优化的相机位姿参数数组，前3位为旋转向量(Angle-Axis)，后3位为平移向量(Translation)
     * @param residuals 输出的残差数组，包含 [u残差, v残差]
     */
    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        // 将 3D 点坐标转为 Ceres 的 Jet/双精度类型
        T p_w[3] = {T(point_3d_[0]), T(point_3d_[1]), T(point_3d_[2])};
        T p_c[3];

        // 使用旋转向量对世界坐标系下的点进行旋转: p_c = R * p_w
        ceres::AngleAxisRotatePoint(se3, p_w, p_c);

        // 加上平移向量: p_c = R * p_w + t
        p_c[0] += se3[3];
        p_c[1] += se3[4];
        p_c[2] += se3[5];

        // 归一化平面坐标转换
        T x = p_c[0] / p_c[2];
        T y = p_c[1] / p_c[2];

        // 获取内参参数 (fx, fy, cx, cy)
        T fx = T(K_(0, 0)), fy = T(K_(1, 1));
        T cx = T(K_(0, 2)), cy = T(K_(1, 2));

        // 计算投影后的像素坐标
        T u = fx * x + cx;
        T v = fy * y + cy;

        // 计算观测值与预测投影值之间的重投影残差
        residuals[0] = u - T(observed_[0]);
        residuals[1] = v - T(observed_[1]);

        return true;
    }

    /**
     * @brief 工厂函数，创建 Ceres 的 CostFunction 实例（使用自动求导机制）
     */
    static ceres::CostFunction *Create(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K)
    {
        // 维度说明：<残差维度=2, 待优化参数维度=6>
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6>(
            new ReprojectionError(observed, point_3d, K));
    }

    Eigen::Vector2d observed_; // 2D 观察像素坐标
    Eigen::Vector3d point_3d_; // 3D 世界坐标点
    Eigen::Matrix3d K_;        // 相机内参矩阵
};

/**
 * @brief 仅包含旋转约束的残差块（忽略平移量，常用于远点优化）
 */
struct RotationOnlyError
{
    // 构造函数：传入观测到的2D像素坐标、3D世界坐标点和相机内参矩阵 K
    RotationOnlyError(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K)
        : observed_(observed), point_3d_(point_3d), K_(K) {}

    /**
     * @brief 计算仅旋转的重投影残差
     */
    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        T p_w[3] = {T(point_3d_[0]), T(point_3d_[1]), T(point_3d_[2])};
        T p_c[3];

        // 仅对点进行旋转运算，忽略平移量 se3[3..5]
        ceres::AngleAxisRotatePoint(se3, p_w, p_c);

        // 归一化平面坐标
        T x = p_c[0] / p_c[2];
        T y = p_c[1] / p_c[2];

        // 获取内参
        T fx = T(K_(0, 0)), fy = T(K_(1, 1));
        T cx = T(K_(0, 2)), cy = T(K_(1, 2));

        // 计算投影像素残差
        residuals[0] = (fx * x + cx) - T(observed_[0]);
        residuals[1] = (fy * y + cy) - T(observed_[1]);

        return true;
    }

    /**
     * @brief 工厂函数，创建仅旋转误差的 CostFunction
     */
    static ceres::CostFunction *Create(const Eigen::Vector2d &observed, const Eigen::Vector3d &point_3d, const Eigen::Matrix3d &K)
    {
        return new ceres::AutoDiffCostFunction<RotationOnlyError, 2, 6>(
            new RotationOnlyError(observed, point_3d, K));
    }

    Eigen::Vector2d observed_; // 2D 观察像素坐标
    Eigen::Vector3d point_3d_; // 3D 世界坐标点
    Eigen::Matrix3d K_;        // 相机内参矩阵
};

/**
 * @brief 执行相机位姿(Pose)的优化计算
 */
int MotionOnlyBA::Optimize(Frame *pFrame)
{
    // 检查输入指针合法性
    if (!pFrame)
        return 0;

    // 获取地图点数量并判断约束数量是否满足要求
    const int N = pFrame->mvpMapPoints.size();
    if (N < 5)
        return 0;

    // 将 cv::Mat 格式的相机内参解析为 Eigen::Matrix3d 格式
    Eigen::Matrix3d K_eigen;
    K_eigen << pFrame->mK.at<float>(0, 0), 0, pFrame->mK.at<float>(0, 2),
        0, pFrame->mK.at<float>(1, 1), pFrame->mK.at<float>(1, 2),
        0, 0, 1;

    // 提取当前帧变换矩阵中的旋转矩阵 R 和平移向量 t，显式转换为 double 类型
    Eigen::Matrix3d R_cw = pFrame->mTcw.block<3, 3>(0, 0).cast<double>();
    Eigen::Vector3d t_cw = pFrame->mTcw.block<3, 1>(0, 3).cast<double>();

    // 将旋转矩阵转换为旋转向量（轴角表示法）
    Eigen::AngleAxisd angle_axis_cw(R_cw);
    Eigen::Vector3d r_vec = angle_axis_cw.angle() * angle_axis_cw.axis();

    // 组装待优化的 6 维相机位姿参数数组 [r_x, r_y, r_z, t_x, t_y, t_z]
    double camera_pose[6] = {
        r_vec[0], r_vec[1], r_vec[2],
        t_cw[0], t_cw[1], t_cw[2]};

    // 构建 Ceres 优化问题，设置 Huber 鲁棒核函数以抑制外点噪声
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    int num_edges = 0; // 记录添加到问题中的有效边/残差块数量

    // 遍历当前帧的所有特征点及其对应的地图点，建立优化约束
    for (int i = 0; i < N; ++i)
    {
        MapPoint *pMP = pFrame->mvpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        // 获取地图点的 3D 世界坐标和当前帧上的 2D 观测点坐标
        Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
        Eigen::Vector2d obs(pFrame->mvKeysUn[i].pt.x, pFrame->mvKeysUn[i].pt.y);

        // 计算地图点在当前相机坐标系下的位置与深度
        Eigen::Vector3d P_c = R_cw * P_w + t_cw;
        double depth = P_c[2];

        // 剔除相机后方的负深度无效点
        if (depth <= 0)
            continue;

        // 双目/立体视觉下所有地图点均有绝对深度，统一使用带平移的标准重投影误差，
        // 为位姿的平移分量提供充分的约束，避免因远点被当作“纯旋转观测”而导致平移漂移。
        ceres::CostFunction *cost_function = ReprojectionError::Create(obs, P_w, K_eigen);

        // 向求解器问题中添加残差块
        problem.AddResidualBlock(cost_function, loss_function, camera_pose);
        num_edges++;
    }

    // 若构建的有效约束数量过少，则取消优化
    if (num_edges < 5)
        return 0;

    // 配置 Ceres 求解器参数
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // 使用密集的 QR 分解求解线性系统
    options.max_num_iterations = 10;              // 限制最大迭代次数为 10 次
    options.minimizer_progress_to_stdout = false; // 不在标准输出流中打印详细迭代进度

    // 运行 Ceres 求解器开始优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 从优化后的参数数组中提取旋转向量与平移向量
    Eigen::Vector3d opt_r(camera_pose[0], camera_pose[1], camera_pose[2]);
    Eigen::Vector3d opt_t(camera_pose[3], camera_pose[4], camera_pose[5]);

    // 将优化后的旋转向量转换回 3x3 旋转矩阵
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

    // 统计内点数量，并利用卡方检验(Chi-Square Test)筛选并标记外点(Outliers)
    int num_inliers = 0;
    const double chi2_threshold = 7.815; // 自由度为 2 时，95% 置信度下的卡方检验阈值

    for (int i = 0; i < N; ++i)
    {
        MapPoint *pMP = pFrame->mvpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        // 计算优化后的相机坐标系下 3D 点坐标
        Eigen::Vector3d P_w = pMP->GetWorldPos().cast<double>();
        Eigen::Vector3d P_c = opt_R * P_w + opt_t;

        // 剔除优化后深度非正的点
        if (P_c[2] <= 0)
        {
            pFrame->mvbOutlier[i] = true;
            continue;
        }

        // 计算重投影像素坐标
        double inv_z = 1.0 / P_c[2];
        double u = K_eigen(0, 0) * P_c[0] * inv_z + K_eigen(0, 2);
        double v = K_eigen(1, 1) * P_c[1] * inv_z + K_eigen(1, 2);

        // 计算像素偏差（投影残差）
        double du = u - pFrame->mvKeysUn[i].pt.x;
        double dv = v - pFrame->mvKeysUn[i].pt.y;
        double err_sq = du * du + dv * dv;

        // 判断重投影误差平方是否超出卡方分布阈值
        if (err_sq > chi2_threshold)
        {
            pFrame->mvbOutlier[i] = true; // 标记为外点
        }
        else
        {
            pFrame->mvbOutlier[i] = false; // 标记为内点
            num_inliers++;
        }
    }

    // 返回成功优化的内点数量
    return num_inliers;
}