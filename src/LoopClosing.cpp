#include "LoopClosing.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "LocalMapping.h"
#include "Tracker.h"
#include "ORBmatcher.h"
#include "KeyFrameDatabase.h"
#include "Optimizer.h"
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include "Sim3Solver.h"
LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, DBoW3::Vocabulary *pVoc, const bool bFixScale)
    : mpMap(pMap), mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpTracker(nullptr), mpLocalMapper(nullptr),
      mbFixScale(bFixScale), mpCurrentKF(nullptr), mpMatchedKF(nullptr),
      mbStopRequested(false), mbStopped(false), mbResetRequested(false)
{
    mpThread = new std::thread(&LoopClosing::Run, this);
}

LoopClosing::~LoopClosing()
{
    // 1. 停止闭环主线程
    if (mpThread)
    {
        RequestStop();
        mpThread->join();
        delete mpThread;
        mpThread = nullptr;
    }

    // 2. 终止并回收 GBA 线程
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbStopGBA = true; // 发出终止信号
    }
    if (mpThreadGBA)
    {
        if (mpThreadGBA->joinable())
            mpThreadGBA->join();
        delete mpThreadGBA;
        mpThreadGBA = nullptr;
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

        if (CheckFinish())
            break;

        usleep(5000);
    }
    SetFinish();
}

// 阶段 1: 候选帧与共视组检测 (DetectLoop)
bool LoopClosing::DetectLoop()
{
    if (!mpCurrentKF || mpCurrentKF->mbBad)
        return false;

    mpCurrentKF->ComputeBoW();

    // 1. ORB-SLAM2 官方做法：仅获取权重 >= 20 的共视邻域计算 minScore
    const std::vector<KeyFrame *> vpConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    const DBoW3::BowVector &curBow = mpCurrentKF->mBowVec;

    float minScore = 1.0f;
    if (!vpConnectedKFs.empty())
    {
        for (KeyFrame *pKF : vpConnectedKFs)
        {
            if (pKF->mbBad)
                continue;
            pKF->ComputeBoW();
            float score = mpORBVocabulary->score(curBow, pKF->mBowVec);
            if (score < minScore)
                minScore = score;
        }
    }
    else
    {
        minScore = 0.4f;
    }

    // 限制最高门槛，避免当前共视帧重合度过高导致历史候选回环被全部误杀
    minScore = std::min(minScore, 0.35f);

    // 2. 数据库检索候选关键帧
    std::vector<KeyFrame *> vpCandidateKFs;
    if (mpKeyFrameDB)
    {
        vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
    }

    // 【修改】：无候选帧时不要直接清空，按原版让历史组自然衰减
    if (vpCandidateKFs.empty())
    {
        std::vector<ConsistentGroup> vCurrentConsistentGroups;
        for (size_t i = 0; i < mvConsistentGroups.size(); ++i)
        {
            if (mvConsistentGroups[i].second > 1)
                vCurrentConsistentGroups.push_back(std::make_pair(mvConsistentGroups[i].first, 1));
        }
        mvConsistentGroups = vCurrentConsistentGroups;
        return false;
    }

    // 3. ORB-SLAM2 官方连续性检查
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

        for (size_t i = 0; i < mvConsistentGroups.size(); ++i)
        {
            if (vbGroupMatched[i])
                continue; // 防止同一组在同帧被多次匹配覆盖

            const std::set<KeyFrame *> &sPrevGroup = mvConsistentGroups[i].first;
            for (KeyFrame *pGKF : sGroup)
            {
                if (sPrevGroup.count(pGKF))
                {
                    nConsistency = mvConsistentGroups[i].second + 1;
                    vbGroupMatched[i] = true;
                    break;
                }
            }
            if (vbGroupMatched[i])
                break;
        }

        vCurrentConsistentGroups.push_back(std::make_pair(sGroup, nConsistency));

        // 达到 3 帧一致性要求
        if (nConsistency >= 3)
        {
            mvpEnoughConsistentCandidates.push_back(pCandKF);
        }
    }

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

// 阶段 2: 几何一致性校验与位姿求解 (ComputeSE3)
bool LoopClosing::ComputeSE3()
{
    ORBmatcher matcher(0.75f, true);

    for (KeyFrame *pCandKF : mvpEnoughConsistentCandidates)
    {
        if (!pCandKF || pCandKF->mbBad)
            continue;

        // 1. 词袋粗匹配
        std::vector<MapPoint *> vpMatchedMapPoints;
        int nmatches = matcher.SearchByBoW(mpCurrentKF, pCandKF, vpMatchedMapPoints);

        if (nmatches < 15)
            continue;

        // 2. 构造 Sim3Solver
        Sim3Solver solver(mpCurrentKF, pCandKF, vpMatchedMapPoints, mbFixScale);
        solver.SetRansacParameters(0.99, 15, 300);

        bool bNoMore = false;
        std::vector<bool> vbInliers;
        int nInliers = 0;
        bool bMatch = false;

        // RANSAC 跑满直到收敛或没有更多候选
        while (!bNoMore)
        {
            bMatch = solver.iterate(50, bNoMore, vbInliers, nInliers);
            if (bMatch)
                break;
        }

        if (!bMatch || nInliers < 15)
            continue;

        // 相机 2 (pCandKF) 到相机 1 (mpCurrentKF) 的变换 T_c_l
        Eigen::Matrix4f T_c_l = solver.GetEstimatedTransformation();

        // 3. 推导当前帧的估计位姿: T_c_w = T_c_l * T_l_w
        Eigen::Matrix4f T_l_w = pCandKF->GetPose();
        Eigen::Matrix4f T_cw_estimated = T_c_l * T_l_w;

        // 4. 利用初值投影扩充匹配点
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

        const Eigen::Matrix3f Rcw = T_cw_estimated.block<3, 3>(0, 0);
        const Eigen::Vector3f tcw = T_cw_estimated.block<3, 1>(0, 3);

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
                    bestIdx = static_cast<int>(idx);
                }
            }

            if (bestIdx >= 0)
            {
                vpMatchedMapPoints[bestIdx] = pMP;
            }
        }

        // 5. 构造临时帧进行仅位姿优化
        Frame tempFrame;
        tempFrame.mnId = mpCurrentKF->mnId;
        tempFrame.N = mpCurrentKF->N;
        tempFrame.mvKeysUn = mpCurrentKF->mvKeysUn;
        tempFrame.mvuRight = mpCurrentKF->mvuRight;
        tempFrame.mvDepth = mpCurrentKF->mvDepth;
        tempFrame.mK = mpCurrentKF->mK.clone();
        tempFrame.mbf = mpCurrentKF->mbf;
        tempFrame.mThDepth = mpCurrentKF->mThDepth;
        if (mpTracker)
            tempFrame.mpORBextractorLeft = mpTracker->GetORBextractorLeft();
        tempFrame.mvpMapPoints = vpMatchedMapPoints;
        tempFrame.mvbOutlier = std::vector<bool>(tempFrame.N, false);
        tempFrame.SetPose(T_cw_estimated);

        int numInliersBA = Optimizer::PoseOptimization(&tempFrame);

        if (numInliersBA >= 25)
        {
            mpMatchedKF = pCandKF;
            mTcw_loop = tempFrame.mTcw;

            for (int i = 0; i < tempFrame.N; ++i)
            {
                if (tempFrame.mvbOutlier[i])
                    vpMatchedMapPoints[i] = nullptr;
            }
            mvpLoopMatchedPoints = vpMatchedMapPoints;
            return true;
        }
    }

    return false;
}

// 阶段 3: 闭环校正与地图融合 (无位姿图优化)
void LoopClosing::CorrectLoop()
{
    std::cout << "\033[32;1m>>> [LoopClosing] Loop detected! 开始闭环融合与位姿图优化... <<<\033[0m" << std::endl;

    // ----------------- Step 1: 停止 LocalMapping -----------------
    if (mpLocalMapper)
    {
        mpLocalMapper->RequestStop();
        mpLocalMapper->RequestStopBA();
        while (!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }
    }

    // ----------------- Step 2: 检查并中断正在运行的旧 GBA (ORB-SLAM2 规范) -----------------
    if (isRunningGBA())
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbStopGBA = true;
        mnFullBAIdx++; // 递增索引，废弃此前的 GBA 结果
        lock.unlock();

        if (mpThreadGBA)
        {
            if (mpThreadGBA->joinable())
                mpThreadGBA->join();
            delete mpThreadGBA;
            mpThreadGBA = nullptr;
        }
    }

    // ----------------- Step 3~9: 闭环融合与位姿图优化 -----------------
    mpCurrentKF->UpdateConnections();

    // 收集共视组并设置保护
    std::vector<KeyFrame *> vpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    std::vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetBestCovisibilityKeyFrames(10);
    vpLoopConnectedKFs.push_back(mpMatchedKF);

    for (KeyFrame *pKFi : vpCurrentConnectedKFs)
        if (pKFi)
            pKFi->SetNotErase();
    for (KeyFrame *pKFi : vpLoopConnectedKFs)
        if (pKFi)
            pKFi->SetNotErase();

    Eigen::Matrix4f Twc_cur_old = mpCurrentKF->GetPoseInverse();
    std::map<KeyFrame *, Eigen::Matrix4f> CorrectedPoses;
    std::map<KeyFrame *, Eigen::Matrix4f> NonCorrectedPoses;

    std::vector<KeyFrame *> vpAllKFs = mpMap->GetAllKeyFrames();
    for (KeyFrame *pKF : vpAllKFs)
        if (pKF && !pKF->mbBad)
            NonCorrectedPoses[pKF] = pKF->GetPose();

    // 1. 先计算当前共视组的校正位姿
    for (KeyFrame *pKFi : vpCurrentConnectedKFs)
    {
        if (!pKFi || pKFi->mbBad)
            continue;
        Eigen::Matrix4f Tiw_old = NonCorrectedPoses[pKFi];
        if (pKFi == mpCurrentKF)
            CorrectedPoses[pKFi] = mTcw_loop;
        else
            CorrectedPoses[pKFi] = (Tiw_old * Twc_cur_old) * mTcw_loop;
    }

    {
        std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);
        // 校正当前共视组地图点
        for (KeyFrame *pKFi : vpCurrentConnectedKFs)
        {
            if (!pKFi || pKFi->mbBad)
                continue;
            Eigen::Matrix4f Tiw_corrected = CorrectedPoses[pKFi];
            Eigen::Matrix4f Tw_i_corrected = Tiw_corrected.inverse();
            Eigen::Matrix4f Tiw_old = NonCorrectedPoses[pKFi];

            std::vector<MapPoint *> vpMPsi = pKFi->GetMapPointMatches();
            for (size_t iMP = 0; iMP < vpMPsi.size(); iMP++)
            {
                MapPoint *pMPi = vpMPsi[iMP];
                if (!pMPi || pMPi->isBad() || pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
                    continue;

                Eigen::Vector3f Pw_old = pMPi->GetWorldPos();
                Eigen::Vector3f Pc = Tiw_old.block<3, 3>(0, 0) * Pw_old + Tiw_old.block<3, 1>(0, 3);
                Eigen::Vector3f Pw_corrected = Tw_i_corrected.block<3, 3>(0, 0) * Pc + Tw_i_corrected.block<3, 1>(0, 3);

                pMPi->SetWorldPos(Pw_corrected);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }
        }

        // 替换闭环匹配点
        for (size_t i = 0; i < mvpLoopMatchedPoints.size(); i++)
        {
            MapPoint *pLoopMP = mvpLoopMatchedPoints[i];
            if (pLoopMP && !pLoopMP->isBad())
            {
                MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
                if (pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP, i);
                    pLoopMP->AddObservation(mpCurrentKF, i);
                    pLoopMP->ComputeDistinctiveDescriptor();
                }
            }
        }
    }

    // 1. 先将所有校正后的位姿更新进关键帧！
    {
        std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);
        for (auto &mit : CorrectedPoses)
        {
            KeyFrame *pKFi = mit.first;
            if (pKFi && !pKFi->mbBad)
            {
                pKFi->SetPose(mit.second);
            }
        }
    }

    // 2. 关键帧位姿已校正，此时重投影才能落在正确的像平面像素区域进行融合
    SearchAndFuse(vpLoopConnectedKFs);

    // 3. 最后更新共视连接与添加闭环边
    std::map<KeyFrame *, std::set<KeyFrame *>> LoopConnections;
    {
        std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);
        for (KeyFrame *pKFi : vpCurrentConnectedKFs)
        {
            if (!pKFi || pKFi->mbBad)
                continue;
            std::vector<KeyFrame *> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();
            pKFi->UpdateConnections();

            std::vector<KeyFrame *> vConnected = pKFi->GetConnectedKeyFrames();
            std::set<KeyFrame *> sNewConnected(vConnected.begin(), vConnected.end());
            for (KeyFrame *pPrev : vpPreviousNeighbors)
                sNewConnected.erase(pPrev);
            for (KeyFrame *pCurConn : vpCurrentConnectedKFs)
                sNewConnected.erase(pCurConn);
            LoopConnections[pKFi] = sNewConnected;
        }

        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);
    }

    // 执行 Essential Graph 优化
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF,
                                      NonCorrectedPoses, CorrectedPoses, LoopConnections);

    {
        std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);
        if (mpTracker)
        {
            // 1. 重置匀速模型速度
            // mpTracker->ResetVelocity();
            // 2. 检查替换点
            mpTracker->CheckReplacedInLastFrame();
            // 3. 强制把上一帧的位姿更新为闭环校正后的位姿，消除阶跃断层
            mpTracker->mLastFrame.SetPose(mpCurrentKF->GetPose());
            // 4. 更新参考关键帧
            mpTracker->mCurrentFrame.mpReferenceKF = mpCurrentKF;
            mpTracker->UpdateLastFrame();
        }
    }

    for (KeyFrame *pKFi : vpCurrentConnectedKFs)
        if (pKFi)
            pKFi->SetErase();
    for (KeyFrame *pKFi : vpLoopConnectedKFs)
        if (pKFi)
            pKFi->SetErase();

    // 释放 LocalMapping 继续运行
    if (mpLocalMapper)
        mpLocalMapper->Release();

    // ----------------- Step 10: 触发新一轮全局 BA (Trigger Global BA) -----------------
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
    }

    // 后台非阻塞启动 Global BA 线程
    mpThreadGBA = new std::thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

    std::cout << "\033[32;1m>>> [LoopClosing] 回环校正与位姿图优化完成，已在后台触发全局 BA！<<<\033[0m" << std::endl;
}

// 辅助函数: 闭环区域地图点投影融合
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

bool LoopClosing::isRunningGBA()
{
    std::unique_lock<std::mutex> lock(mMutexGBA);
    return mbRunningGBA;
}

bool LoopClosing::isFinishedGBA()
{
    std::unique_lock<std::mutex> lock(mMutexGBA);
    return mbFinishedGBA;
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    std::cout << "[GBA] 后台全局 BA 开始执行, 目标关键帧 ID: " << nLoopKF << " ..." << std::endl;

    const unsigned long idx = mnFullBAIdx;

    // 调用 Optimizer::GlobalBundleAdjustment，传入 &mbStopGBA 标志指针支持内部提前中断
    Optimizer::GlobalBundleAdjustment(mpMap, 20, &mbStopGBA, nLoopKF, true);

    std::unique_lock<std::mutex> lock(mMutexGBA);
    // 如果中途未被请求停止，并且期间没有新的回环递增 mnFullBAIdx
    if (idx == mnFullBAIdx && !mbStopGBA)
    {
        mbFinishedGBA = true;
        std::cout << "\033[32;1m[GBA] 后台全局 BA 成功收敛并更新地图！\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[33;1m[GBA] 后台全局 BA 被新回环或外部请求中断 (Aborted)。\033[0m" << std::endl;
    }
    mbRunningGBA = false;
}

void LoopClosing::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

void LoopClosing::RequestStopGBA()
{
    std::unique_lock<std::mutex> lock(mMutexGBA);
    if (mbRunningGBA)
    {
        mbStopGBA = true;
    }
}
