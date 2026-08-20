#include "LoopClosing.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "LocalMapping.h"
#include "Tracker.h"
#include "ORBmatcher.h"
#include "KeyFrameDatabase.h"
#include "MotionOnlyBA.h"
#include "Optimizer.h"
#include <unistd.h>
#include <algorithm>

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, DBoW3::Vocabulary *pVoc, const bool bFixScale)
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

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
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
                // std::cout << "\033[32;1m>>> [LoopTrigger] 成功检测到连续一致性闭环候选组！<<<\033[0m" << std::endl;

                // 2. 几何位姿求解检验 (PnP / RANSAC)
                if (ComputeSE3())
                {
                    std::cout << "\033[32;1m>>> [LoopTrigger SUCCESS] 闭环几何校验完全通过！当前KF-"
                              << mpCurrentKF->mnId << " 与闭环KF-" << mpMatchedKF->mnId
                              << " 形成有效回环！<<<\033[0m" << std::endl;

                    // 执行闭环融合
                    CorrectLoop();

                    // 仅清空当前候选，保留连续性分组由 DetectLoop 迭代更新
                    mvpEnoughConsistentCandidates.clear();
                }
                // else
                // {
                //     std::cout << "\033[31m[LoopTrigger FAILED] 候选帧连续性达标，但 PnP 几何校验未通过。\033[0m" << std::endl;
                // }
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

    mpCurrentKF->ComputeBoW();

    // 1. 获取共视邻域内的最小 BoW 得分
    const std::vector<KeyFrame *> vpConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    const DBoW3::BowVector &curBow = mpCurrentKF->mBowVec;

    float minScore = 1.0f;
    for (KeyFrame *pKF : vpConnectedKFs)
    {
        if (pKF->mbBad)
            continue;
        pKF->ComputeBoW();
        float score = mpORBVocabulary->score(curBow, pKF->mBowVec);
        if (score < minScore)
            minScore = score;
    }

    // 2. 数据库检索候选关键帧
    std::vector<KeyFrame *> vpCandidateKFs;
    if (mpKeyFrameDB)
    {
        vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
    }
    if (vpCandidateKFs.empty())
    {
        mvConsistentGroups.clear();
        return false;
    }

    // 3. ORB-SLAM2 官方一致性滑动窗口检查
    mvpEnoughConsistentCandidates.clear();
    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbGroupMatched(mvConsistentGroups.size(), false);

    for (KeyFrame *pCandKF : vpCandidateKFs)
    {
        std::set<KeyFrame *> sGroup;
        sGroup.insert(pCandKF);
        std::vector<KeyFrame *> vNeighs = pCandKF->GetBestCovisibilityKeyFrames(10);
        for (KeyFrame *pN : vNeighs)
            sGroup.insert(pN);

        int nConsistency = 1;
        bool bMatched = false;

        for (size_t i = 0; i < mvConsistentGroups.size(); ++i)
        {
            const std::set<KeyFrame *> &sPrevGroup = mvConsistentGroups[i].first;
            for (KeyFrame *pGKF : sGroup)
            {
                if (sPrevGroup.count(pGKF))
                {
                    nConsistency = mvConsistentGroups[i].second + 1;
                    vbGroupMatched[i] = true;
                    bMatched = true;
                    break;
                }
            }
            if (bMatched)
                break;
        }

        vCurrentConsistentGroups.push_back(std::make_pair(sGroup, nConsistency));

        // 达到 3 帧连续一致性阈值
        if (nConsistency >= 3)
        {
            mvpEnoughConsistentCandidates.push_back(pCandKF);
        }
    }

    // 将未匹配但历史存在的组降级保留（衰减机制，防止剧烈抖动清零）
    for (size_t i = 0; i < mvConsistentGroups.size(); ++i)
    {
        if (!vbGroupMatched[i] && mvConsistentGroups[i].second > 1)
        {
            vCurrentConsistentGroups.push_back(std::make_pair(mvConsistentGroups[i].first, 1));
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

    for (KeyFrame *pCandKF : mvpEnoughConsistentCandidates)
    {
        if (!pCandKF || pCandKF->mbBad)
            continue;

        // 阶段 1: BoW 粗匹配
        std::vector<MapPoint *> vpMatchedMapPoints;
        int nmatches = matcher.SearchByBoW(mpCurrentKF, pCandKF, vpMatchedMapPoints);

        if (nmatches < 20)
            continue;

        std::vector<cv::Point3f> vPts3D;
        std::vector<cv::Point2f> vPts2D;
        std::vector<int> vMPIndices;

        for (int i = 0; i < mpCurrentKF->N; ++i)
        {
            MapPoint *pMP = vpMatchedMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f Pw = pMP->GetWorldPos();
                vPts3D.push_back(cv::Point3f(Pw.x(), Pw.y(), Pw.z()));
                vPts2D.push_back(mpCurrentKF->mvKeysUn[i].pt);
                vMPIndices.push_back(i);
            }
        }

        if (vPts3D.size() < 20)
            continue;

        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        bool bOK = cv::solvePnPRansac(
            vPts3D, vPts2D, mpCurrentKF->mK, cv::Mat::zeros(4, 1, CV_32F),
            rvec, tvec, false, 300, 8.0f, 0.99, inliers, cv::SOLVEPNP_EPNP);

        if (!bOK || inliers.size() < 20)
            continue;

        // 提取粗估计位姿
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
        for (int r = 0; r < 3; ++r)
        {
            Tcw(r, 3) = static_cast<float>(tvec.at<double>(r));
            for (int c = 0; c < 3; ++c)
                Tcw(r, c) = static_cast<float>(R_cv.at<double>(r, c));
        }

        // 阶段 2: ORB-SLAM2 官方投影引导二次扩充匹配
        // 利用初解位姿将候选帧及共视邻居的地图点反投影到当前帧搜索更多匹配
        std::vector<KeyFrame *> vpCandNeighs = pCandKF->GetBestCovisibilityKeyFrames(10);
        vpCandNeighs.push_back(pCandKF);

        std::set<MapPoint *> sCandMPs;
        for (KeyFrame *pKFi : vpCandNeighs)
        {
            for (MapPoint *pMPi : pKFi->GetMapPointMatches())
            {
                if (pMPi && !pMPi->isBad())
                    sCandMPs.insert(pMPi);
            }
        }

        int nAdditionalMatches = 0;
        const Eigen::Matrix3f Rcw = Tcw.block<3, 3>(0, 0);
        const Eigen::Vector3f tcw = Tcw.block<3, 1>(0, 3);

        for (MapPoint *pMP : sCandMPs)
        {
            Eigen::Vector3f Pc = Rcw * pMP->GetWorldPos() + tcw;
            if (Pc.z() <= 0.0f)
                continue;

            float invz = 1.0f / Pc.z();
            float u = mpCurrentKF->fx * Pc.x() * invz + mpCurrentKF->cx;
            float v = mpCurrentKF->fy * Pc.y() * invz + mpCurrentKF->cy;

            if (u < mpCurrentKF->mnMinX || u >= mpCurrentKF->mnMaxX ||
                v < mpCurrentKF->mnMinY || v >= mpCurrentKF->mnMaxY)
                continue;

            std::vector<size_t> vIndices = mpCurrentKF->GetFeaturesInArea(u, v, 10.0f);
            if (vIndices.empty())
                continue;

            const cv::Mat &dMP = pMP->GetDescriptor();
            int bestDist = ORBmatcher::TH_LOW;
            int bestIdx = -1;

            for (size_t idx : vIndices)
            {
                if (vpMatchedMapPoints[idx])
                    continue;
                const cv::Mat &dF = mpCurrentKF->mDescriptors.row(idx);
                int dist = ORBmatcher::DescriptorDistance(dMP, dF);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestIdx >= 0)
            {
                vpMatchedMapPoints[bestIdx] = pMP;
                nAdditionalMatches++;
            }
        }

        // 阶段 3: 最终严格内点判定 (总匹配点 >= 40 彻底确认闭环)
        int nTotalMatches = inliers.size() + nAdditionalMatches;
        if (nTotalMatches >= 40)
        {
            mpMatchedKF = pCandKF;
            mTcw_loop = Tcw;
            mvpLoopMatchedPoints = vpMatchedMapPoints;
            return true;
        }
    }

    return false;
}

// -----------------------------------------------------------------------------
// 阶段 3: 闭环校正与地图融合 (无位姿图优化)
// -----------------------------------------------------------------------------
void LoopClosing::CorrectLoop()
{
    std::cout << "\033[33;1m[LoopClosing] 开始执行闭环校正 (ORB-SLAM2 标准局部修正)... 当前KF-" 
              << mpCurrentKF->mnId << " <---> 闭环KF-" << mpMatchedKF->mnId << "\033[0m" << std::endl;

    // 1. 请求暂停 LocalMapping 线程并中断局部 BA
    if (mpLocalMapper)
    {
        mpLocalMapper->RequestStop();
        mpLocalMapper->RequestStopBA();
        while (!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }
    }

    // 地图全局更新互斥锁
    std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);

    // 确保当前关键帧连接关系最新
    mpCurrentKF->UpdateConnections();

    // 2. ORB-SLAM2 官方策略：仅获取强共视邻居（权重 >= 15 或前 15 个最佳邻居），严防带入历史远端
    std::vector<KeyFrame*> vpCurrentConnectedKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(15);
    vpCurrentConnectedKFs.push_back(mpCurrentKF);

    // 3. 计算当前帧的闭环校正增量：T_correct = Tcw_loop * (Tcw_old)^-1
    Eigen::Matrix4f Tcw_old = mpCurrentKF->GetPose();
    Eigen::Matrix4f Tcw_loop = mTcw_loop; // 由 ComputeSE3 解得的位姿
    Eigen::Matrix4f T_correct = Tcw_loop * mpCurrentKF->GetPoseInverse();

    // 4. 计算当前共视组中所有局部关键帧的修正位姿（排除首帧和无效帧）
    std::map<KeyFrame*, Eigen::Matrix4f> CorrectedPoses;
    CorrectedPoses[mpCurrentKF] = Tcw_loop;

    for (KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if (pKFi == mpCurrentKF || !pKFi || pKFi->mbBad || pKFi->mnId == 0)
            continue;

        // 依据当前帧校正前后的相对变换进行位姿传递: Ti_w_corrected = Ti_c * T_c_w_loop
        Eigen::Matrix4f Ti_c = pKFi->GetPose() * mpCurrentKF->GetPoseInverse();
        Eigen::Matrix4f Ti_w_corrected = Ti_c * Tcw_loop;
        CorrectedPoses[pKFi] = Ti_w_corrected;
    }

    // 5. 对当前共视组拥有且以其为参考帧的地图点进行 3D 坐标校正
    for (auto& kv : CorrectedPoses)
    {
        KeyFrame* pKFi = kv.first;
        Eigen::Matrix4f Tcw_new = kv.second;
        Eigen::Matrix4f Twc_new = Tcw_new.inverse();
        Eigen::Matrix4f Tcw_prev = pKFi->GetPose();

        std::vector<MapPoint*> vpMPs = pKFi->GetMapPointMatches();
        for (size_t i = 0; i < vpMPs.size(); ++i)
        {
            MapPoint* pMP = vpMPs[i];
            if (!pMP || pMP->isBad())
                continue;

            // 仅当该地图点的参考关键帧是当前局部帧时才校正，避免点被多次变换
            if (pMP->GetReferenceKeyFrame() == pKFi)
            {
                Eigen::Vector3f Pw_old = pMP->GetWorldPos();
                Eigen::Vector4f Pw_homo(Pw_old.x(), Pw_old.y(), Pw_old.z(), 1.0f);
                
                Eigen::Vector4f Pw_new = Twc_new * (Tcw_prev * Pw_homo);
                pMP->SetWorldPos(Pw_new.head<3>());
                pMP->UpdateNormalAndDepth();
            }
        }

        // 正式更新当前局部关键帧位姿
        pKFi->SetPose(Tcw_new);
    }

    // 6. 闭环侧地图点投影融合 (SearchAndFuse)
    std::vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetBestCovisibilityKeyFrames(10);
    vpLoopConnectedKFs.push_back(mpMatchedKF);

    SearchAndFuse(vpLoopConnectedKFs);

    // 7. 更新共视关系，建立闭环侧与当前侧的高权重连接
    for (KeyFrame* pKFi : vpCurrentConnectedKFs)
    {
        if (pKFi && !pKFi->mbBad)
            pKFi->UpdateConnections();
    }
    for (KeyFrame* pKFm : vpLoopConnectedKFs)
    {
        if (pKFm && !pKFm->mbBad)
            pKFm->UpdateConnections();
    }
    if (mpTracker)
    {
        mpTracker->ResetVelocity();
    }
    // 8. 恢复 LocalMapping 线程
    if (mpLocalMapper)
    {
        mpLocalMapper->Release();
    }

    std::cout << "\033[32;1m[LoopClosing] ORB-SLAM2 阶段 1 & 2 校正完成！\033[0m" << std::endl;
}

// -----------------------------------------------------------------------------
// 辅助函数: 闭环区域地图点投影融合
// -----------------------------------------------------------------------------
void LoopClosing::SearchAndFuse(const std::vector<KeyFrame *> &vpLoopConnectedKFs)
{
    // 收集闭环侧所有地图点
    std::set<MapPoint *> sLoopMPs;
    for (KeyFrame *pKF : vpLoopConnectedKFs)
    {
        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint *pMP : vpMPs)
        {
            if (pMP && !pMP->isBad())
                sLoopMPs.insert(pMP);
        }
    }

    // 投影到当前帧及其相连帧中并执行 Replace
    std::vector<KeyFrame *> vpCurrentConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);

    for (KeyFrame *pKF : vpCurrentConnectedKFs)
    {
        const Eigen::Matrix3f Rcw = pKF->GetRotation();
        const Eigen::Vector3f tcw = pKF->GetTranslation();

        for (MapPoint *pMP : sLoopMPs)
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
                MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
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