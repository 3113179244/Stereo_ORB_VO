// Source File: Optimizer.cpp
#include "Optimizer.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "MotionOnlyBA.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <algorithm>
#include <cmath>

namespace
{

    /**
     * @brief 局部 BA 用的重投影误差（3D-2D）
     * @details 同时优化相机位姿(6DOF)和地图点(3DOF)两个参数块。
     *          位姿参数使用 Tcw（世界->相机），旋转采用轴角(Angle-Axis)表示。
     *            p_c = R_cw * p_w + t_cw
     *          残差已按特征金字塔尺度信息矩阵(1/sigma)加权，维度为 2。
     */
    struct LocalRepoError
    {
        LocalRepoError(double fx, double fy, double cx, double cy,
                       double u, double v, double sqrtInvSigma2)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
              u_(u), v_(v), sqrtInvSigma2_(sqrtInvSigma2) {}

        template <typename T>
        bool operator()(const T *const pose, const T *const point, T *residuals) const
        {
            // pose: [rx, ry, rz, tx, ty, tz], Tcw (世界 -> 相机)
            // point: [x, y, z] 世界坐标
            T p_w[3] = {point[0], point[1], point[2]};
            T p_c[3];

            // p_c = R_cw * p_w
            ceres::AngleAxisRotatePoint(pose, p_w, p_c);
            // p_c = R_cw * p_w + t_cw
            p_c[0] += pose[3];
            p_c[1] += pose[4];
            p_c[2] += pose[5];
            // 安全检查：如果点在相机后面，赋予固定大残差或避免除以接近0的数
            if (p_c[2] <= T(1e-4))
            {
                residuals[0] = T(0.0);
                residuals[1] = T(0.0);
                return false; // 返回 false 告诉 Ceres 该步不可行
            }
            // 归一化 + 内参投影
            const T invz = T(1.0) / p_c[2];
            const T pred_u = fx_ * p_c[0] * invz + cx_;
            const T pred_v = fy_ * p_c[1] * invz + cy_;

            // 按信息矩阵(1/sigma)加权残差
            residuals[0] = (pred_u - T(u_)) * T(sqrtInvSigma2_);
            residuals[1] = (pred_v - T(v_)) * T(sqrtInvSigma2_);
            return true;
        }

        static ceres::CostFunction *Create(double fx, double fy, double cx, double cy,
                                           double u, double v, double sqrtInvSigma2)
        {
            // <残差维度=2, 位姿=6, 地图点=3>
            return new ceres::AutoDiffCostFunction<LocalRepoError, 2, 6, 3>(
                new LocalRepoError(fx, fy, cx, cy, u, v, sqrtInvSigma2));
        }

        double fx_, fy_, cx_, cy_;
        double u_, v_;
        double sqrtInvSigma2_;
    };

    /**
     * @brief 将关键帧位姿 Tcw 转成 Ceres 的 6 维参数数组 [轴角, 平移]
     */
    void PoseToArray(KeyFrame *pKF, double out[6])
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

    /**
     * @brief 把优化后的 6 维参数数组写回关键帧位姿
     */
    void ArrayToPose(KeyFrame *pKF, const double in[6])
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

} // namespace

/**
 * @brief 仅优化帧位姿（Pose-Only BA），直接复用 MotionOnlyBA
 */
int Optimizer::PoseOptimization(Frame *pFrame)
{
    return MotionOnlyBA::Optimize(pFrame);
}

/**
 * @brief 局部 Bundle Adjustment
 * @details 基于 Ceres 实现：
 *           1. 取地图中最近的一个关键帧作为当前关键帧；
 *           2. 收集一级共视邻居作为"局部关键帧"(待优化)，二级邻居作为"固定关键帧"(只提供约束)；
 *           3. 收集所有被局部关键帧观测到的局部地图点(待优化)；
 *           4. 构建 Ceres 问题：同时优化局部关键帧位姿与局部地图点，固定关键帧位姿置为常量；
 *           5. 分两轮优化：第一轮加入全部残差，用卡方检验剔除外点(标记坏点)，第二轮只用内点再优化一次。
 */
void Optimizer::LocalBundleAdjustment(std::shared_ptr<Map> pMap)
{
    if (!pMap)
        return;

    std::vector<KeyFrame *> vpAllKFs = pMap->GetAllKeyFrames();
    if (vpAllKFs.empty())
        return;

    // ------------------------------------------------------------------
    // Step 1&2&3 : 选取当前关键帧、收集局部/固定关键帧与局部地图点
    // ------------------------------------------------------------------
    KeyFrame *pCurKF = vpAllKFs.back(); // 最近插入的关键帧

    // 收集局部关键帧（当前关键帧 + 高共视邻居）
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

    // 收取固定关键帧（局部关键帧的一级邻居，且不在局部集合内）
    std::vector<KeyFrame *> vpFixedKFs;
    for (size_t i = 0; i < vpLocalKFs.size(); ++i)
    {
        std::vector<KeyFrame *> vNeigh = vpLocalKFs[i]->GetBestCovisibilityKeyFrames(5);
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

    // 收集所有被局部关键帧观测的局部地图点（去重）
    std::vector<MapPoint *> vpLocalMPs;
    std::set<MapPoint *> sLocalMPs;
    for (size_t i = 0; i < vpLocalKFs.size(); ++i)
    {
        std::vector<MapPoint *> vpMPs = vpLocalKFs[i]->GetMapPointMatches();
        for (size_t j = 0; j < vpMPs.size(); ++j)
        {
            MapPoint *pMP = vpMPs[j];
            // 增加 pMP->GetObservations().size() >= 2 判断
            if (pMP && !pMP->isBad() && pMP->GetObservations().size() >= 2 && !sLocalMPs.count(pMP))
            {
                sLocalMPs.insert(pMP);
                vpLocalMPs.push_back(pMP);
            }
        }
    }

    const int nLocalKFs = static_cast<int>(vpLocalKFs.size());
    const int nLocalMPs = static_cast<int>(vpLocalMPs.size());

    if (nLocalKFs < 2 || nLocalMPs < 5)
        return; // 约束太少跳过优化

    // ------------------------------------------------------------------
    // Step 4 : 构建 Ceres 问题
    // ------------------------------------------------------------------
    // 为每个局部关键帧分配位姿参数数组，并记录索引
    std::map<KeyFrame *, double *> mapKFPose;
    std::vector<double *> vPoseArrays;
    for (int i = 0; i < nLocalKFs; ++i)
    {
        double *pose = new double[6];
        PoseToArray(vpLocalKFs[i], pose);
        mapKFPose[vpLocalKFs[i]] = pose;
        vPoseArrays.push_back(pose);
    }
    // 固定关键帧
    std::map<KeyFrame *, double *> mapFixedPose;
    for (size_t i = 0; i < vpFixedKFs.size(); ++i)
    {
        double *pose = new double[6];
        PoseToArray(vpFixedKFs[i], pose);
        mapFixedPose[vpFixedKFs[i]] = pose;
    }
    // 为每个局部地图点分配三维坐标参数数组
    std::map<MapPoint *, double *> mapMPPoint;
    std::vector<double *> vPointArrays;
    for (int i = 0; i < nLocalMPs; ++i)
    {
        double *p = new double[3];
        const Eigen::Vector3f pos = vpLocalMPs[i]->GetWorldPos();
        p[0] = pos[0];
        p[1] = pos[1];
        p[2] = pos[2];
        mapMPPoint[vpLocalMPs[i]] = p;
        vPointArrays.push_back(p);
    }

    // 两轮优化：第一轮用于剔除外点，第二轮只用内点
    // badge 记录第一次优化后判定为坏点的地图点
    std::set<MapPoint *> sBadMPs;

    for (int iter = 0; iter < 2; ++iter)
    {
        ceres::Problem problem;

        // 注册局部关键帧位姿参数块
        for (int i = 0; i < nLocalKFs; ++i)
        {
            KeyFrame *pKF = vpLocalKFs[i];
            problem.AddParameterBlock(mapKFPose[pKF], 6);
        }
        // 注册局部地图点参数块
        for (int i = 0; i < nLocalMPs; ++i)
        {
            MapPoint *pMP = vpLocalMPs[i];
            if (sBadMPs.count(pMP))
                continue; // 第二轮时跳过坏点
            problem.AddParameterBlock(mapMPPoint[pMP], 3);
        }

        // 添加局部关键帧 -> 地图点的残差
        for (int i = 0; i < nLocalKFs; ++i)
        {
            KeyFrame *pKF = vpLocalKFs[i];
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
            double *pose = mapKFPose[pKF];

            for (size_t j = 0; j < vpMPs.size(); ++j)
            {
                MapPoint *pMP = vpMPs[j];
                if (!pMP || pMP->isBad())
                    continue;
                if (!mapMPPoint.count(pMP))
                    continue; // 不属于局部地图点
                if (sBadMPs.count(pMP))
                    continue; // 第二轮剔除坏点

                const cv::KeyPoint &kp = pKF->mvKeysUn[j];
                const int level = kp.octave;
                if (level < 0 || level >= pKF->mnScaleLevels)
                    continue;
                const double invSigma2 = pKF->mvInvLevelSigma2[level];
                const double sqrtInvSigma2 = std::sqrt(invSigma2);

                ceres::CostFunction *cost =
                    LocalRepoError::Create(fx, fy, cx, cy, kp.pt.x, kp.pt.y, sqrtInvSigma2);
                ceres::LossFunction *loss = new ceres::HuberLoss(std::sqrt(5.991));
                problem.AddResidualBlock(cost, loss, pose, mapMPPoint[pMP]);
            }
        }

        // 添加固定关键帧 -> 地图点的残差（固定位姿提供约束，不优化）
        for (size_t i = 0; i < vpFixedKFs.size(); ++i)
        {
            KeyFrame *pKF = vpFixedKFs[i];
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
            double *pose = mapFixedPose[pKF];
            problem.AddParameterBlock(pose, 6);
            problem.SetParameterBlockConstant(pose); // 固定位姿不优化

            for (size_t j = 0; j < vpMPs.size(); ++j)
            {
                MapPoint *pMP = vpMPs[j];
                if (!pMP || pMP->isBad())
                    continue;
                if (!mapMPPoint.count(pMP))
                    continue; // 该点不在局部地图点集合中，忽略
                if (sBadMPs.count(pMP))
                    continue;

                const cv::KeyPoint &kp = pKF->mvKeysUn[j];
                const int level = kp.octave;
                if (level < 0 || level >= pKF->mnScaleLevels)
                    continue;
                const double invSigma2 = pKF->mvInvLevelSigma2[level];
                const double sqrtInvSigma2 = std::sqrt(invSigma2);

                ceres::CostFunction *cost =
                    LocalRepoError::Create(fx, fy, cx, cy, kp.pt.x, kp.pt.y, sqrtInvSigma2);
                ceres::LossFunction *loss = new ceres::HuberLoss(std::sqrt(5.991));
                problem.AddResidualBlock(cost, loss, pose, mapMPPoint[pMP]);
            }
        }

        // 求解器配置
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = 10;
        options.num_threads = 1;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 第一轮结束后，用卡方检验剔除外点并标记坏点
        if (iter == 0)
        {
            const double chi2_th = 5.991; // 2 自由度 95% 置信度
            for (int i = 0; i < nLocalMPs; ++i)
            {
                MapPoint *pMP = vpLocalMPs[i];
                if (sBadMPs.count(pMP))
                    continue;

                const double *p = mapMPPoint[pMP];
                const Eigen::Vector3d Pw(p[0], p[1], p[2]);
                double maxErr2 = 0.0;

                // 统计该点在所有观测上的最大加权残差
                std::map<KeyFrame *, size_t> obs = pMP->GetObservations();
                for (auto mit = obs.begin(); mit != obs.end(); ++mit)
                {
                    KeyFrame *pKF = mit->first;
                    const size_t idx = mit->second;
                    if (idx >= pKF->mvKeysUn.size())
                        continue;

                    // 位姿（可能是局部优化的也可能是固定不变的）
                    double *pose = nullptr;
                    auto itL = mapKFPose.find(pKF);
                    auto itF = mapFixedPose.find(pKF);
                    if (itL != mapKFPose.end())
                        pose = itL->second;
                    else if (itF != mapFixedPose.end())
                        pose = itF->second;
                    else
                        continue;

                    // 计算残差（不加权，仅用像素误差）
                    const cv::Point2f &kp = pKF->mvKeysUn[idx].pt;
                    Eigen::Vector3d Pc =
                        Eigen::AngleAxisd(
                            Eigen::Vector3d(pose[0], pose[1], pose[2]).norm(),
                            Eigen::Vector3d(pose[0], pose[1], pose[2]).normalized())
                            .toRotationMatrix() *
                        Pw;
                    Pc += Eigen::Vector3d(pose[3], pose[4], pose[5]);
                    if (Pc.z() <= 0.0)
                    {
                        maxErr2 = 1e9;
                        break;
                    }
                    const double du = pKF->fx * Pc.x() / Pc.z() + pKF->cx - kp.x;
                    const double dv = pKF->fy * Pc.y() / Pc.z() + pKF->cy - kp.y;
                    double err2 = du * du + dv * dv;

                    // 缩放至信息矩阵加权空间，与优化时的 Huber 权重和卡方阈值保持一致
                    const int level = pKF->mvKeysUn[idx].octave;
                    if (level >= 0 && level < pKF->mnScaleLevels)
                        err2 /= pKF->mvLevelSigma2[level];

                    maxErr2 = std::max(maxErr2, err2);
                }

                if (maxErr2 > chi2_th)
                {
                    sBadMPs.insert(pMP);
                    pMP->SetBadFlag();
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 保存优化结果到位姿/地图点
    // ------------------------------------------------------------------
    for (int i = 0; i < nLocalKFs; ++i)
        ArrayToPose(vpLocalKFs[i], mapKFPose[vpLocalKFs[i]]);
    for (int i = 0; i < nLocalMPs; ++i)
    {
        MapPoint *pMP = vpLocalMPs[i];
        if (sBadMPs.count(pMP))
            continue; // 已成坏点，不需回写
        double *p = mapMPPoint[pMP];
        pMP->SetWorldPos(Eigen::Vector3f((float)p[0], (float)p[1], (float)p[2]));
    }

    // 清理动态分配的内存
    for (size_t i = 0; i < vPoseArrays.size(); ++i)
        delete[] vPoseArrays[i];
    for (auto &kv : mapFixedPose)
        delete[] kv.second;
    for (size_t i = 0; i < vPointArrays.size(); ++i)
        delete[] vPointArrays[i];
}

/**
 * @brief 全局 Bundle Adjustment
 * @details 优化地图中所有关键帧位姿与所有地图点。
 *          这里作为骨架提供，解析失败时不改动地图（可替换为完整实现）。
 */
void Optimizer::GlobalBundleAdjustment(std::shared_ptr<Map> pMap)
{
    (void)pMap;
}
