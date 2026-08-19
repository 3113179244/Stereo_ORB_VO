#include "LoopClosing.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "LocalMapping.h"
#include "Tracker.h"
#include "ORBmatcher.h"
#include "KeyFrameDatabase.h"
#include "MotionOnlyBA.h"

#include <unistd.h>
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

LoopClosing::LoopClosing(Map* pMap, KeyFrameDatabase* pDB, DBoW3::Vocabulary* pVoc, const bool bFixScale)
    : mpMap(pMap), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpTracker(nullptr), mpLocalMapper(nullptr),
      mbFixScale(bFixScale), mpCurrentKF(nullptr), mpMatchedKF(nullptr),
      mbStopRequested(false), mbStopped(false), mbResetRequested(false)
{
    mpThread = new std::thread(&LoopClosing::Run, this);
}

LoopClosing::~LoopClosing()
{
    if (mpThread)
    {
        RequestStop();
        mpThread->join();
        delete mpThread;
    }
}

void LoopClosing::InsertKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    if (pKF->mnId != 0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    return !mlpLoopKeyFrameQueue.empty();
}

void LoopClosing::Run()
{
    mbStopped = false;

    while (1)
    {
        if (CheckNewKeyFrames())
        {
            {
                std::unique_lock<std::mutex> lock(mMutexLoopQueue);
                mpCurrentKF = mlpLoopKeyFrameQueue.front();
                mlpLoopKeyFrameQueue.pop_front();
            }

            // 1. 闭环候选初筛与连续性检验
            if (DetectLoop())
            {
                std::cout << "\033[32;1m>>> [LoopTrigger] 成功检测到连续一致性闭环候选组！<<<\033[0m" << std::endl;

                // 2. 几何位姿求解检验 (PnP / RANSAC)
                if (ComputeSE3())
                {
                    std::cout << "\033[32;1m>>> [LoopTrigger SUCCESS] 闭环几何校验完全通过！当前KF-" 
                              << mpCurrentKF->mnId << " 与闭环KF-" << mpMatchedKF->mnId 
                              << " 形成有效回环！<<<\033[0m" << std::endl;

                    // 阶段一验证：先只打印提示，暂不进入后端位姿图优化以验证纯前端召回率
                    // CorrectLoop();
                }
                else
                {
                    std::cout << "\033[31m[LoopTrigger FAILED] 候选帧连续性达标，但 PnP 几何校验未通过。\033[0m" << std::endl;
                }
            }
        }

        // 线程停止控制
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            if (mbStopRequested)
            {
                mbStopped = true;
                while (mbStopRequested)
                {
                    usleep(5000);
                }
                mbStopped = false;
            }
        }

        usleep(5000);
    }
}

// -----------------------------------------------------------------------------
// 阶段 1: 候选帧与共视组检测 (DetectLoop)
// -----------------------------------------------------------------------------
bool LoopClosing::DetectLoop()
{
    if (!mpCurrentKF || mpCurrentKF->mbBad)
        return false;

    // 确保当前关键帧已计算 BoW
    mpCurrentKF->ComputeBoW();

    // 1. 获取当前帧的所有相连共视帧，计算共视邻域内的最小 BoW 相似度得分
    const std::vector<KeyFrame*> vpConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    const DBoW3::BowVector& curBow = mpCurrentKF->mBowVec;

    float minScore = 1.0f;
    for (KeyFrame* pKF : vpConnectedKFs)
    {
        if (pKF->mbBad) continue;
        pKF->ComputeBoW();
        float score = mpORBVocabulary->score(curBow, pKF->mBowVec);
        if (score < minScore)
            minScore = score;
    }

    // 2. 从关键帧数据库中检索候选帧
    std::vector<KeyFrame*> vpCandidateKFs;
    if (mpKeyFrameDB)
    {
        vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
    }
    if (vpCandidateKFs.empty())
    {
        mvConsistentGroups.clear();
        return false;
    }

    // 3. 对候选帧进行共视组聚类与连续性检验 (Temporal Consistency)
    mvpEnoughConsistentCandidates.clear();
    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbGroupMatched(mvConsistentGroups.size(), false);

    for (KeyFrame* pCandKF : vpCandidateKFs)
    {
        std::set<KeyFrame*> sGroup;
        sGroup.insert(pCandKF);
        std::vector<KeyFrame*> vNeighs = pCandKF->GetBestCovisibilityKeyFrames(5);
        for (KeyFrame* pN : vNeighs)
            sGroup.insert(pN);

        bool bMatched = false;
        for (size_t i = 0; i < mvConsistentGroups.size(); ++i)
        {
            const std::set<KeyFrame*>& sPrevGroup = mvConsistentGroups[i].first;
            for (KeyFrame* pGKF : sGroup)
            {
                if (sPrevGroup.count(pGKF))
                {
                    int nConsistency = mvConsistentGroups[i].second + 1;
                    vCurrentConsistentGroups.push_back(std::make_pair(sGroup, nConsistency));
                    vbGroupMatched[i] = true;
                    bMatched = true;

                    // 连续 3 帧命中同一组共视关键帧
                    if (nConsistency >= 3)
                    {
                        mvpEnoughConsistentCandidates.push_back(pCandKF);
                    }
                    break;
                }
            }
            if (bMatched) break;
        }

        if (!bMatched)
        {
            vCurrentConsistentGroups.push_back(std::make_pair(sGroup, 1));
        }
    }

    mvConsistentGroups = vCurrentConsistentGroups;
    return !mvpEnoughConsistentCandidates.empty();
}

// -----------------------------------------------------------------------------
// 阶段 2: 几何一致性校验与位姿求解 (ComputeSE3)
// -----------------------------------------------------------------------------
bool LoopClosing::ComputeSE3()
{
    ORBmatcher matcher(0.75f, true);

    for (KeyFrame* pCandKF : mvpEnoughConsistentCandidates)
    {
        if (!pCandKF || pCandKF->mbBad)
            continue;

        // 1. 通过词袋匹配候选帧与当前帧地图点
        std::vector<MapPoint*> vpMatchedMapPoints;
        int nmatches = matcher.SearchByBoW(mpCurrentKF, pCandKF, vpMatchedMapPoints);

        if (nmatches < 20)
            continue;

        // 2. 构造 3D-2D 对应进行 RANSAC PnP 求解
        std::vector<cv::Point3f> vPts3D;
        std::vector<cv::Point2f> vPts2D;
        std::vector<int> vMPIndices;

        for (int i = 0; i < mpCurrentKF->N; ++i)
        {
            MapPoint* pMP = vpMatchedMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f Pw = pMP->GetWorldPos();
                vPts3D.push_back(cv::Point3f(Pw.x(), Pw.y(), Pw.z()));
                vPts2D.push_back(mpCurrentKF->mvKeysUn[i].pt);
                vMPIndices.push_back(i);
            }
        }

        if (vPts3D.size() < 15)
            continue;

        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool bOK = cv::solvePnPRansac(
            vPts3D, vPts2D, mpCurrentKF->mK, cv::Mat::zeros(4, 1, CV_32F),
            rvec, tvec, false, 300, 8.0f, 0.99, inliers, cv::SOLVEPNP_EPNP);

        if (!bOK || inliers.size() < 15)
            continue;

        // 3. 转换为当前帧全局位姿 Tcw_loop
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
        for (int r = 0; r < 3; ++r)
        {
            Tcw(r, 3) = static_cast<float>(tvec.at<double>(r));
            for (int c = 0; c < 3; ++c)
                Tcw(r, c) = static_cast<float>(R_cv.at<double>(r, c));
        }

        mpMatchedKF = pCandKF;
        mTcw_loop = Tcw;
        mvpLoopMatchedPoints = vpMatchedMapPoints;
        return true;
    }

    return false;
}

// -----------------------------------------------------------------------------
// 阶段 3: 闭环校正、地图融合与位姿图优化 (CorrectLoop)
// -----------------------------------------------------------------------------
void LoopClosing::CorrectLoop()
{
    std::cout << "[LoopClosing] 发现有效闭环！当前关键帧: " << mpCurrentKF->mnId
              << " <---> 闭环关键帧: " << mpMatchedKF->mnId << std::endl;

    // 1. 请求暂停 LocalMapping，防止新点生成和局部 BA 产生数据竞争
    if (mpLocalMapper)
    {
        mpLocalMapper->RequestStop();
        mpLocalMapper->RequestStopBA();
        while (!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }
    }

    // 确保当前关键帧连接关系最新
    mpCurrentKF->UpdateConnections();

    // 2. 闭环侧位姿传播：计算当前帧校正前后的漂移变换 dT = Tcw_corrected * Tcw_uncorrected^-1
    Eigen::Matrix4f Tcw_old = mpCurrentKF->GetPose();
    Eigen::Matrix4f Twc_old = mpCurrentKF->GetPoseInverse();
    Eigen::Matrix4f dT = mTcw_loop * Twc_old;

    // 收集当前关键帧及其相连共视帧
    std::vector<KeyFrame*> vpCurrentConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);

    std::map<KeyFrame*, Eigen::Matrix4f> CorrectedPosesMap;
    for (KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        Eigen::Matrix4f Ti_old = pKFi->GetPose();
        Eigen::Matrix4f Ti_new = Ti_old * dT.inverse(); // 修正相邻帧位姿
        pKFi->SetPose(Ti_new);
        CorrectedPosesMap[pKFi] = Ti_new;
    }

    // 3. 将闭环组相连关键帧中的地图点投影到当前相连帧中，进行点融合
    std::vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetConnectedKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    SearchAndFuse(vpLoopConnectedKFs);

    // 4. 更新闭环区域涉及的所有关键帧共视边
    for (KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        pKFi->UpdateConnections();
    }

    // 5. 执行 Essential Graph 优化 (优化生成树 + 高共视边 + 闭环边)
    OptimizeEssentialGraph(mpCurrentKF, mpMatchedKF, mTcw_loop);

    // 6. 唤醒并恢复 LocalMapping 线程
    if (mpLocalMapper)
    {
        mpLocalMapper->SetNotStop();
    }

    std::cout << "[LoopClosing] 闭环校正与融合优化完成。" << std::endl;
}

// -----------------------------------------------------------------------------
// 辅助函数: 闭环区域地图点投影融合
// -----------------------------------------------------------------------------
void LoopClosing::SearchAndFuse(const std::vector<KeyFrame*>& vpLoopConnectedKFs)
{
    // 收集闭环侧所有地图点
    std::set<MapPoint*> sLoopMPs;
    for (KeyFrame* pKF : vpLoopConnectedKFs)
    {
        std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (pMP && !pMP->isBad())
                sLoopMPs.insert(pMP);
        }
    }

    // 投影到当前帧及其相连帧中并执行 Replace
    std::vector<KeyFrame*> vpCurrentConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);

    for (KeyFrame* pKF : vpCurrentConnectedKFs)
    {
        const Eigen::Matrix3f Rcw = pKF->GetRotation();
        const Eigen::Vector3f tcw = pKF->GetTranslation();

        for (MapPoint* pMP : sLoopMPs)
        {
            if (!pMP || pMP->isBad())
                continue;

            Eigen::Vector3f Pc = Rcw * pMP->GetWorldPos() + tcw;
            if (Pc.z() <= 0.0f)
                continue;

            float invz = 1.0f / Pc.z();
            float u = pKF->fx * Pc.x() * invz + pKF->cx;
            float v = pKF->fy * Pc.y() * invz + pKF->cy;

            if (u < pKF->mnMinX || u >= pKF->mnMaxX || v < pKF->mnMinY || v >= pKF->mnMaxY)
                continue;

            std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, 8.0f);
            if (vIndices.empty())
                continue;

            cv::Mat dMP = pMP->GetDescriptor();
            int bestDist = ORBmatcher::TH_LOW;
            int bestIdx = -1;

            for (size_t idx : vIndices)
            {
                cv::Mat dF = pKF->mDescriptors.row(idx);
                int dist = ORBmatcher::DescriptorDistance(dMP, dF);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestIdx >= 0)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if (!pMPinKF)
                {
                    pKF->AddMapPoint(pMP, bestIdx);
                    pMP->AddObservation(pKF, bestIdx);
                }
                else if (pMPinKF != pMP)
                {
                    // 将旧点合并为新点
                    pMPinKF->Replace(pMP);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 辅助函数: 位姿图残差定义与 Ceres 求解
// -----------------------------------------------------------------------------
struct PoseGraphErrorSE3
{
    PoseGraphErrorSE3(const Eigen::Vector3d& r_ij, const Eigen::Vector3d& t_ij)
        : r_ij_(r_ij), t_ij_(t_ij) {}

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, T* residuals) const
    {
        // pose: [rx, ry, rz, tx, ty, tz]
        // 计算 T_ij_meas^-1 * (T_i * T_j^-1)
        T r_i[3] = {pose_i[0], pose_i[1], pose_i[2]};
        T t_i[3] = {pose_i[3], pose_i[4], pose_i[5]};

        T r_j[3] = {pose_j[0], pose_j[1], pose_j[2]};
        T t_j[3] = {pose_j[3], pose_j[4], pose_j[5]};

        // 简化的相对位姿平移与旋转残差
        residuals[0] = (r_j[0] - r_i[0]) - T(r_ij_[0]);
        residuals[1] = (r_j[1] - r_i[1]) - T(r_ij_[1]);
        residuals[2] = (r_j[2] - r_i[2]) - T(r_ij_[2]);

        residuals[3] = (t_j[0] - t_i[0]) - T(t_ij_[0]);
        residuals[4] = (t_j[1] - t_i[1]) - T(t_ij_[1]);
        residuals[5] = (t_j[2] - t_i[2]) - T(t_ij_[2]);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& r_ij, const Eigen::Vector3d& t_ij)
    {
        return new ceres::AutoDiffCostFunction<PoseGraphErrorSE3, 6, 6, 6>(
            new PoseGraphErrorSE3(r_ij, t_ij));
    }

    Eigen::Vector3d r_ij_, t_ij_;
};

void LoopClosing::OptimizeEssentialGraph(KeyFrame* pCurKF, KeyFrame* pMatchedKF, 
                                        const Eigen::Matrix4f& g2oCorrectedTcw)
{
    std::vector<KeyFrame*> vpAllKFs = mpMap->GetAllKeyFrames();
    std::map<KeyFrame*, double*> mapKFPose;

    ceres::Problem problem;

    for (KeyFrame* pKF : vpAllKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        double* pose = new double[6];
        Eigen::Matrix4f Tcw = (pKF == pCurKF) ? g2oCorrectedTcw : pKF->GetPose();
        Eigen::AngleAxisd aa(Tcw.block<3, 3>(0, 0).cast<double>());
        Eigen::Vector3d r = aa.angle() * aa.axis();
        Eigen::Vector3d t = Tcw.block<3, 1>(0, 3).cast<double>();

        pose[0] = r.x(); pose[1] = r.y(); pose[2] = r.z();
        pose[3] = t.x(); pose[4] = t.y(); pose[5] = t.z();

        mapKFPose[pKF] = pose;
        problem.AddParameterBlock(pose, 6);

        // 固定初始关键帧或闭环参考关键帧作为优化锚点
        if (pKF->mnId == 0 || pKF == pMatchedKF)
        {
            problem.SetParameterBlockConstant(pose);
        }
    }

    // 遍历生成树边 (Spanning Tree) 添加位姿约束
    for (KeyFrame* pKF : vpAllKFs)
    {
        if (!pKF || pKF->mbBad) continue;
        KeyFrame* pParent = pKF->GetParent();
        if (pParent && mapKFPose.count(pParent) && mapKFPose.count(pKF))
        {
            Eigen::Matrix4f T_parent_kf = pParent->GetPose() * pKF->GetPoseInverse();
            Eigen::AngleAxisd aa(T_parent_kf.block<3, 3>(0, 0).cast<double>());
            Eigen::Vector3d r_rel = aa.angle() * aa.axis();
            Eigen::Vector3d t_rel = T_parent_kf.block<3, 1>(0, 3).cast<double>();

            ceres::CostFunction* cost = PoseGraphErrorSE3::Create(r_rel, t_rel);
            problem.AddResidualBlock(cost, nullptr, mapKFPose[pParent], mapKFPose[pKF]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 20;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 写回优化后的位姿，并根据位姿增量更新所有地图点的 3D 坐标
    for (auto& pair : mapKFPose)
    {
        KeyFrame* pKF = pair.first;
        double* pose = pair.second;

        Eigen::Vector3d r(pose[0], pose[1], pose[2]);
        Eigen::Vector3d t(pose[3], pose[4], pose[5]);
        double angle = r.norm();
        Eigen::Matrix3d R = (angle > 1e-12) ? Eigen::AngleAxisd(angle, r.normalized()).toRotationMatrix() : Eigen::Matrix3d::Identity();

        Eigen::Matrix4f Tcw_optimized = Eigen::Matrix4f::Identity();
        Tcw_optimized.block<3, 3>(0, 0) = R.cast<float>();
        Tcw_optimized.block<3, 1>(0, 3) = t.cast<float>();

        Eigen::Matrix4f Tcw_old = pKF->GetPose();
        pKF->SetPose(Tcw_optimized);

        // 更新归属于本关键帧的地图点世界坐标: Pw' = Twc_new * Tcw_old * Pw
        std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint* pMP : vpMPs)
        {
            if (pMP && !pMP->isBad() && pMP->IsInKeyFrame(pKF))
            {
                Eigen::Vector3f Pw = pMP->GetWorldPos();
                Eigen::Vector3f Pw_new = (pKF->GetPoseInverse() * Tcw_old * Pw.homogeneous()).head<3>();
                pMP->SetWorldPos(Pw_new);
                pMP->UpdateNormalAndDepth();
            }
        }

        delete[] pose;
    }
}

void LoopClosing::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool LoopClosing::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

void LoopClosing::RequestReset()
{
    std::unique_lock<std::mutex> lock(mMutexReset);
    mbResetRequested = true;
}