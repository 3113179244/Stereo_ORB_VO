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
 * @brief 局部 BA 用的重投影误差（3D-2D）
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
        T p_w[3] = {point[0], point[1], point[2]};
        T p_c[3];

        ceres::AngleAxisRotatePoint(pose, p_w, p_c);
        p_c[0] += pose[3];
        p_c[1] += pose[4];
        p_c[2] += pose[5];

        const T invz = T(1.0) / p_c[2];
        const T pred_u = fx_ * p_c[0] * invz + cx_;
        const T pred_v = fy_ * p_c[1] * invz + cy_;

        residuals[0] = (pred_u - T(u_)) * T(sqrtInvSigma2_);
        residuals[1] = (pred_v - T(v_)) * T(sqrtInvSigma2_);
        return true;
    }

    static ceres::CostFunction *Create(double fx, double fy, double cx, double cy,
                                       double u, double v, double sqrtInvSigma2)
    {
        return new ceres::AutoDiffCostFunction<LocalRepoError, 2, 6, 3>(
            new LocalRepoError(fx, fy, cx, cy, u, v, sqrtInvSigma2));
    }

    double fx_, fy_, cx_, cy_;
    double u_, v_;
    double sqrtInvSigma2_;
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
 * @brief 局部 Bundle Adjustment (支持中断)
 * @param pbStopFlag 外部中断标志位指针（例如由 LocalMapping 线程传入）
 */
void Optimizer::LocalBundleAdjustment(KeyFrame *pCurKF, bool *pbStopFlag, std::shared_ptr<Map> pMap)
{
    if (!pCurKF || !pMap || pCurKF->mbBad)
        return;

    // ------------------------------------------------------------------
    // Step 1~3: 收集局部关键帧、固定关键帧与局部地图点
    // ------------------------------------------------------------------
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

    const int nLocalKFs = static_cast<int>(vpLocalKFs.size());
    const int nLocalMPs = static_cast<int>(vpLocalMPs.size());

    if (nLocalKFs < 2 || nLocalMPs < 5)
        return;

    // ------------------------------------------------------------------
    // Step 4: 构建 Ceres 变量内存
    // ------------------------------------------------------------------
    std::map<KeyFrame *, double *> mapKFPose;
    std::vector<double *> vPoseArrays;
    for (int i = 0; i < nLocalKFs; ++i)
    {
        double *pose = new double[6];
        PoseToArray(vpLocalKFs[i], pose);
        mapKFPose[vpLocalKFs[i]] = pose;
        vPoseArrays.push_back(pose);
    }

    std::map<KeyFrame *, double *> mapFixedPose;
    for (size_t i = 0; i < vpFixedKFs.size(); ++i)
    {
        double *pose = new double[6];
        PoseToArray(vpFixedKFs[i], pose);
        mapFixedPose[vpFixedKFs[i]] = pose;
    }

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

    auto CleanupMemory = [&]() {
        for (size_t i = 0; i < vPoseArrays.size(); ++i)
            delete[] vPoseArrays[i];
        for (auto &kv : mapFixedPose)
            delete[] kv.second;
        for (size_t i = 0; i < vPointArrays.size(); ++i)
            delete[] vPointArrays[i];
    };

    // 配置 Ceres 求解器参数
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = true; // 确保每一步迭代的优化结果就地写入数组

    AbortCallback callback(pbStopFlag);
    if (pbStopFlag)
    {
        options.callbacks.push_back(&callback);
    }

    // ==================================================================
    // 第一阶段：粗优化（迭代 5 次）
    // ==================================================================
    options.max_num_iterations = 5;
    ceres::Problem problem1;

    // 1. 注册局部关键帧
    for (int i = 0; i < nLocalKFs; ++i)
    {
        KeyFrame *pKF = vpLocalKFs[i];
        problem1.AddParameterBlock(mapKFPose[pKF], 6);
        if ((vpFixedKFs.empty() && i == 0) || pKF->mnId == 0)
            problem1.SetParameterBlockConstant(mapKFPose[pKF]);
    }

    // 2. 注册局部地图点
    for (int i = 0; i < nLocalMPs; ++i)
    {
        problem1.AddParameterBlock(mapMPPoint[vpLocalMPs[i]], 3);
    }

    // 3. 添加局部关键帧观测残差
    for (int i = 0; i < nLocalKFs; ++i)
    {
        KeyFrame *pKF = vpLocalKFs[i];
        const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        double *pose = mapKFPose[pKF];

        for (size_t j = 0; j < vpMPs.size(); ++j)
        {
            MapPoint *pMP = vpMPs[j];
            if (!pMP || pMP->isBad() || !mapMPPoint.count(pMP))
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
            problem1.AddResidualBlock(cost, loss, pose, mapMPPoint[pMP]);
        }
    }

    // 4. 添加固定关键帧约束残差
    for (size_t i = 0; i < vpFixedKFs.size(); ++i)
    {
        KeyFrame *pKF = vpFixedKFs[i];
        const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        double *pose = mapFixedPose[pKF];
        problem1.AddParameterBlock(pose, 6);
        problem1.SetParameterBlockConstant(pose);

        for (size_t j = 0; j < vpMPs.size(); ++j)
        {
            MapPoint *pMP = vpMPs[j];
            if (!pMP || pMP->isBad() || !mapMPPoint.count(pMP))
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
            problem1.AddResidualBlock(cost, loss, pose, mapMPPoint[pMP]);
        }
    }

    ceres::Solver::Summary summary1;
    ceres::Solve(options, &problem1, &summary1);

    // ==================================================================
    // 检查中断标志：若被中断，则跳过外点剔除与第二阶段精优化
    // ==================================================================
    bool bDoMore = true;
    if (pbStopFlag && *pbStopFlag)
    {
        bDoMore = false;
    }

    std::set<MapPoint *> sBadMPs;

    if (bDoMore)
    {
        // --------------------------------------------------------------
        // 卡方检验：剔除外点
        // --------------------------------------------------------------
        const double chi2_th = 5.991;
        for (int i = 0; i < nLocalMPs; ++i)
        {
            MapPoint *pMP = vpLocalMPs[i];
            const double *p = mapMPPoint[pMP];
            const Eigen::Vector3d Pw(p[0], p[1], p[2]);
            double maxErr2 = 0.0;

            std::map<KeyFrame *, size_t> obs = pMP->GetObservations();
            for (auto mit = obs.begin(); mit != obs.end(); ++mit)
            {
                KeyFrame *pKF = mit->first;
                const size_t idx = mit->second;
                if (idx >= pKF->mvKeysUn.size())
                    continue;

                double *pose = nullptr;
                auto itL = mapKFPose.find(pKF);
                auto itF = mapFixedPose.find(pKF);
                if (itL != mapKFPose.end())
                    pose = itL->second;
                else if (itF != mapFixedPose.end())
                    pose = itF->second;
                else
                    continue;

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

                const int level = pKF->mvKeysUn[idx].octave;
                if (level >= 0 && level < pKF->mnScaleLevels)
                    err2 /= pKF->mvLevelSigma2[level];

                maxErr2 = std::max(maxErr2, err2);
            }

            if (maxErr2 > chi2_th)
            {
                sBadMPs.insert(pMP);
            }
        }

        // ==============================================================
        // 第二阶段：纯内点精优化（迭代 10 次）
        // ==============================================================
        options.max_num_iterations = 10;
        ceres::Problem problem2;

        // 注册局部关键帧
        for (int i = 0; i < nLocalKFs; ++i)
        {
            KeyFrame *pKF = vpLocalKFs[i];
            problem2.AddParameterBlock(mapKFPose[pKF], 6);
            if ((vpFixedKFs.empty() && i == 0) || pKF->mnId == 0)
                problem2.SetParameterBlockConstant(mapKFPose[pKF]);
        }

        // 注册内点地图点
        for (int i = 0; i < nLocalMPs; ++i)
        {
            MapPoint *pMP = vpLocalMPs[i];
            if (sBadMPs.count(pMP))
                continue;
            problem2.AddParameterBlock(mapMPPoint[pMP], 3);
        }

        // 局部帧内点残差
        for (int i = 0; i < nLocalKFs; ++i)
        {
            KeyFrame *pKF = vpLocalKFs[i];
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
            double *pose = mapKFPose[pKF];

            for (size_t j = 0; j < vpMPs.size(); ++j)
            {
                MapPoint *pMP = vpMPs[j];
                if (!pMP || pMP->isBad() || !mapMPPoint.count(pMP) || sBadMPs.count(pMP))
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
                problem2.AddResidualBlock(cost, loss, pose, mapMPPoint[pMP]);
            }
        }

        // 固定帧内点残差
        for (size_t i = 0; i < vpFixedKFs.size(); ++i)
        {
            KeyFrame *pKF = vpFixedKFs[i];
            const double fx = pKF->fx, fy = pKF->fy, cx = pKF->cx, cy = pKF->cy;
            std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
            double *pose = mapFixedPose[pKF];
            problem2.AddParameterBlock(pose, 6);
            problem2.SetParameterBlockConstant(pose);

            for (size_t j = 0; j < vpMPs.size(); ++j)
            {
                MapPoint *pMP = vpMPs[j];
                if (!pMP || pMP->isBad() || !mapMPPoint.count(pMP) || sBadMPs.count(pMP))
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
                problem2.AddResidualBlock(cost, loss, pose, mapMPPoint[pMP]);
            }
        }

        ceres::Solver::Summary summary2;
        ceres::Solve(options, &problem2, &summary2);
    }

    // ------------------------------------------------------------------
    // Step 5: 无论是否被打断，均将当前已优化的参数回写
    // ------------------------------------------------------------------
    for (int i = 0; i < nLocalKFs; ++i)
        ArrayToPose(vpLocalKFs[i], mapKFPose[vpLocalKFs[i]]);

    for (int i = 0; i < nLocalMPs; ++i)
    {
        MapPoint *pMP = vpLocalMPs[i];
        if (sBadMPs.count(pMP))
        {
            pMP->SetBadFlag(); // 将检验出的外点正式标记为坏点
            continue;
        }
        double *p = mapMPPoint[pMP];
        pMP->SetWorldPos(Eigen::Vector3f((float)p[0], (float)p[1], (float)p[2]));
    }

    // ------------------------------------------------------------------
    // Step 6: 释放内存
    // ------------------------------------------------------------------
    CleanupMemory();
}

void Optimizer::GlobalBundleAdjustment(std::shared_ptr<Map> pMap)
{
    (void)pMap;
}