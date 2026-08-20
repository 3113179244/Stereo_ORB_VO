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
/**
 * @brief 局部 BA 用的双目重投影误差（3D-2D，带右目视差约束）
 * 适用于：双目近点 (u_r >= 0 且 Z < mThDepth)
 */
struct LocalRepoErrorStereo
{
    LocalRepoErrorStereo(double fx, double fy, double cx, double cy, double mbf,
                         double u, double v, double u_r, double sqrtInvSigma2)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), mbf_(mbf),
          u_(u), v_(v), u_r_(u_r), sqrtInvSigma2_(sqrtInvSigma2) {}

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
        const T pred_ur = pred_u - T(mbf_) * invz;

        residuals[0] = (pred_u - T(u_)) * T(sqrtInvSigma2_);
        residuals[1] = (pred_v - T(v_)) * T(sqrtInvSigma2_);
        residuals[2] = (pred_ur - T(u_r_)) * T(sqrtInvSigma2_);
        return true;
    }

    static ceres::CostFunction *Create(double fx, double fy, double cx, double cy, double mbf,
                                       double u, double v, double u_r, double sqrtInvSigma2)
    {
        return new ceres::AutoDiffCostFunction<LocalRepoErrorStereo, 3, 6, 3>(
            new LocalRepoErrorStereo(fx, fy, cx, cy, mbf, u, v, u_r, sqrtInvSigma2));
    }

    double fx_, fy_, cx_, cy_, mbf_;
    double u_, v_, u_r_;
    double sqrtInvSigma2_;
};

// 6自由度 SE3 位姿图残差: 约束帧 i 到 帧 j 的相对测量 T_ij = T_iw * (T_jw)^-1
struct PoseGraphEdgeError {
    PoseGraphEdgeError(const Eigen::Matrix4f& T_ij_meas) {
        Eigen::Matrix3d R_ij = T_ij_meas.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d t_ij = T_ij_meas.block<3,1>(0,3).cast<double>();
        r_ij_ = Eigen::AngleAxisd(R_ij).angle() * Eigen::AngleAxisd(R_ij).axis();
        t_ij_ = t_ij;
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const {
        // pose: [rx, ry, rz, tx, ty, tz]
        T p_i[3] = {pose_i[0], pose_i[1], pose_i[2]};
        T t_i[3] = {pose_i[3], pose_i[4], pose_i[5]};
        T p_j[3] = {pose_j[0], pose_j[1], pose_j[2]};
        T t_j[3] = {pose_j[3], pose_j[4], pose_j[5]};

        // 1. 相对旋转 R_pred = R_i * R_j^T
        T q_i[4], q_j[4];
        ceres::AngleAxisToQuaternion(p_i, q_i);
        ceres::AngleAxisToQuaternion(p_j, q_j);
        
        // q_j_inv = [-q_j[1], -q_j[2], -q_j[3], q_j[0]]
        T q_j_inv[4] = {q_j[0], -q_j[1], -q_j[2], -q_j[3]};
        T q_pred[4];
        ceres::QuaternionProduct(q_i, q_j_inv, q_pred);

        // 2. 相对平移 t_pred = t_i - R_i * R_j^T * t_j
        T t_j_rotated[3];
        ceres::QuaternionRotatePoint(q_pred, t_j, t_j_rotated);
        T t_pred[3] = {t_i[0] - t_j_rotated[0], t_i[1] - t_j_rotated[1], t_i[2] - t_j_rotated[2]};

        // 平移误差
        residuals[0] = t_pred[0] - T(t_ij_[0]);
        residuals[1] = t_pred[1] - T(t_ij_[1]);
        residuals[2] = t_pred[2] - T(t_ij_[2]);

        // 旋转误差
        T q_meas[4];
        T r_meas[3] = {T(r_ij_[0]), T(r_ij_[1]), T(r_ij_[2])};
        ceres::AngleAxisToQuaternion(r_meas, q_meas);
        T q_meas_inv[4] = {q_meas[0], -q_meas[1], -q_meas[2], -q_meas[3]};

        T q_diff[4];
        ceres::QuaternionProduct(q_meas_inv, q_pred, q_diff);

        residuals[3] = T(2.0) * q_diff[1];
        residuals[4] = T(2.0) * q_diff[2];
        residuals[5] = T(2.0) * q_diff[3];

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix4f& T_ij_meas) {
        return new ceres::AutoDiffCostFunction<PoseGraphEdgeError, 6, 6, 6>(
            new PoseGraphEdgeError(T_ij_meas));
    }

    Eigen::Vector3d r_ij_;
    Eigen::Vector3d t_ij_;
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
        // 扩大固定关键帧共视搜索范围 (从 5 扩大到 10)
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

    const int nLocalKFs = static_cast<int>(vpLocalKFs.size());
    const int nLocalMPs = static_cast<int>(vpLocalMPs.size());

    if (nLocalKFs < 2 || nLocalMPs < 5)
        return;
    // 找出局部帧中 ID 最小的一帧（当没有固定关键帧时，作为局部地图坐标系的保底锚定帧）
    KeyFrame* pAnchorKF = nullptr;
    if (vpFixedKFs.empty() && !vpLocalKFs.empty())
    {
        pAnchorKF = vpLocalKFs[0];
        for (size_t i = 1; i < vpLocalKFs.size(); ++i)
        {
            if (vpLocalKFs[i]->mnId < pAnchorKF->mnId)
                pAnchorKF = vpLocalKFs[i];
        }
    }
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

    auto CleanupMemory = [&]()
    {
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
        if (pKF->mnId == 0 || pKF == pAnchorKF)
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
            const float u_r = pKF->mvuRight[j];
            const float depth = pKF->mvDepth[j];

            ceres::CostFunction *cost = nullptr;
            // 判断是否是双目近点（带右目视差约束）
            if (u_r >= 0.0f && depth > 0.0f && depth < pKF->mThDepth)
            {
                cost = LocalRepoErrorStereo::Create(fx, fy, cx, cy, pKF->mbf,
                                                    kp.pt.x, kp.pt.y, u_r, sqrtInvSigma2);
            }
            else
            {
                cost = LocalRepoError::Create(fx, fy, cx, cy, kp.pt.x, kp.pt.y, sqrtInvSigma2);
            }

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
            if (pKF->mnId == 0 || pKF == pAnchorKF)
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
                const float u_r = pKF->mvuRight[j];
                const float depth = pKF->mvDepth[j];

                ceres::CostFunction *cost = nullptr;
                // 双目近点约束
                if (u_r >= 0.0f && depth > 0.0f && depth < pKF->mThDepth)
                {
                    cost = LocalRepoErrorStereo::Create(fx, fy, cx, cy, pKF->mbf,
                                                        kp.pt.x, kp.pt.y, u_r, sqrtInvSigma2);
                }
                else
                {
                    cost = LocalRepoError::Create(fx, fy, cx, cy, kp.pt.x, kp.pt.y, sqrtInvSigma2);
                }

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
    if (pbStopFlag && *pbStopFlag)
    {
        CleanupMemory();
        return; // 被外部打断，直接放弃本次未收敛的结果，防止地图被半成品位姿污染
    }

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

void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF, const Eigen::Matrix4f& Tcw_loop)
{
    std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const int N = vpKFs.size();
    if (N < 2) return;

    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    // 1. 记录优化前所有关键帧的平滑位姿（生成树边必须基于该位姿计算相对变换）
    std::map<KeyFrame*, Eigen::Matrix4f> mapOldPoses;
    std::map<KeyFrame*, double*> mapPoses;

    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->mbBad) continue;
        double* pose = new double[6];
        
        Eigen::Matrix4f Tcw = pKF->GetPose();
        mapOldPoses[pKF] = Tcw;

        Eigen::Matrix3d R = Tcw.block<3,3>(0,0).cast<double>();
        Eigen::Vector3d t = Tcw.block<3,1>(0,3).cast<double>();
        Eigen::AngleAxisd aa(R);
        Eigen::Vector3d r = aa.angle() * aa.axis();

        pose[0] = r.x(); pose[1] = r.y(); pose[2] = r.z();
        pose[3] = t.x(); pose[4] = t.y(); pose[5] = t.z();
        
        mapPoses[pKF] = pose;
        problem.AddParameterBlock(pose, 6);

        // 固定闭环匹配关键帧作为绝对不动锚点
        if (pKF == pLoopKF || pKF->mnId == 0) {
            problem.SetParameterBlockConstant(pose);
        }
    }

    // 2. 添加生成树（Spanning Tree）边：使用优化前两帧之间连续记录的相对里程计
    for (KeyFrame* pKF : vpKFs) {
        if (!pKF || pKF->mbBad || pKF->mnId == 0) continue;
        KeyFrame* pParent = pKF->GetParent();
        if (!pParent || !mapPoses.count(pParent) || !mapPoses.count(pKF)) continue;

        // 真实未被破坏的相对变换: T_child_parent = T_child_w * (T_parent_w)^-1
        Eigen::Matrix4f T_child_parent = mapOldPoses[pKF] * mapOldPoses[pParent].inverse();
        
        problem.AddResidualBlock(PoseGraphEdgeError::Create(T_child_parent),
                                 loss_function,
                                 mapPoses[pKF], mapPoses[pParent]);
    }

    // 3. 添加回环闭合边 (Loop Edge): 由 PnP 解算得到的 Tcw_loop 与 pLoopKF 的相对位姿强约束
    if (mapPoses.count(pCurKF) && mapPoses.count(pLoopKF)) {
        // 计算当前帧与闭环帧之间精准的相对测量值: T_cur_loop = T_cur_loop_pnp * (T_loop_w)^-1
        Eigen::Matrix4f T_cur_loop = Tcw_loop * mapOldPoses[pLoopKF].inverse();
        
        problem.AddResidualBlock(PoseGraphEdgeError::Create(T_cur_loop),
                                 nullptr, // 闭环边作为强拉直约束，不加核函数
                                 mapPoses[pCurKF], mapPoses[pLoopKF]);
    }

    // 4. 配置并求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 30;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 5. 回写优化后的所有关键帧位姿
    std::map<KeyFrame*, Eigen::Matrix4f> mapNewPoses;
    for (auto& kv : mapPoses) {
        KeyFrame* pKF = kv.first;
        double* pose = kv.second;

        Eigen::Vector3d r(pose[0], pose[1], pose[2]);
        Eigen::Vector3d t(pose[3], pose[4], pose[5]);
        double angle = r.norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (angle > 1e-12) R = Eigen::AngleAxisd(angle, r.normalized()).toRotationMatrix();

        Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
        Tcw.block<3,3>(0,0) = R.cast<float>();
        Tcw.block<3,1>(0,3) = t.cast<float>();
        
        pKF->SetPose(Tcw);
        mapNewPoses[pKF] = Tcw;

        delete[] pose;
    }

    // 6. 根据各关键帧优化前后的位姿差，同步更新所有 3D 地图点
    std::vector<MapPoint*> vpAllMPs = pMap->GetAllMapPoints();
    for (MapPoint* pMP : vpAllMPs) {
        if (!pMP || pMP->isBad()) continue;
        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
        if (!pRefKF || !mapOldPoses.count(pRefKF) || !mapNewPoses.count(pRefKF)) continue;

        Eigen::Matrix4f T_old_cw = mapOldPoses[pRefKF];
        Eigen::Matrix4f T_new_wc = mapNewPoses[pRefKF].inverse();

        Eigen::Vector3f Pw_old = pMP->GetWorldPos();
        Eigen::Vector4f Pw_homo(Pw_old.x(), Pw_old.y(), Pw_old.z(), 1.0f);

        Eigen::Vector4f Pw_new = T_new_wc * (T_old_cw * Pw_homo);
        pMP->SetWorldPos(Pw_new.head<3>());
        pMP->UpdateNormalAndDepth();
    }
}

void Optimizer::GlobalBundleAdjustment(std::shared_ptr<Map> pMap)
{
    (void)pMap;
}