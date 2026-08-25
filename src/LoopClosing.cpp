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
    // 等待并释放 GBA 线程
    if (mpThreadGBA)
    {
        mpThreadGBA->join();
        delete mpThreadGBA;
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
        minScore = 0.4f; // 避免保底阈值过低
    }

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

        // 阶段 1: BoW 粗匹配 (ORB-SLAM2 标准门槛: 至少 20 对匹配)
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

        // ORB-SLAM2 标准门槛: RANSAC 内点数必须 >= 20
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

        // 阶段 2: 投影引导二次扩充匹配
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
                    bestIdx = static_cast<int>(idx);
                }
            }

            if (bestIdx >= 0)
            {
                vpMatchedMapPoints[bestIdx] = pMP;
                nAdditionalMatches++;
            }
        }

        // ORB-SLAM2 标准门槛: 最终有效内点总数 >= 40
        int nTotalMatches = static_cast<int>(inliers.size()) + nAdditionalMatches;
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

// 阶段 3: 闭环校正与地图融合 (无位姿图优化)
void LoopClosing::CorrectLoop()
{
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

    // 2. 如果上一次的全局 GBA 还在运行，发送中断请求并回收线程
    if (isRunningGBA())
    {
        {
            std::unique_lock<std::mutex> lock(mMutexGBA);
            mbStopGBA = true;
        }

        if (mpThreadGBA)
        {
            mpThreadGBA->join();
            delete mpThreadGBA;
            mpThreadGBA = nullptr;
        }
    }

    // 地图全局更新互斥锁（仅保护闭环校正与位姿图优化阶段）
    {
        std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);

        // 3. 将闭环匹配点替换到当前关键帧
        for (size_t i = 0; i < mvpLoopMatchedPoints.size(); i++)
        {
            MapPoint *pLoopMP = mvpLoopMatchedPoints[i];
            if (pLoopMP && !pLoopMP->isBad())
            {
                MapPoint *pCurMP = mpCurrentKF->GetMapPoint(i);
                if (pCurMP && pCurMP != pLoopMP)
                {
                    pCurMP->Replace(pLoopMP);
                }
                else if (!pCurMP)
                {
                    mpCurrentKF->AddMapPoint(pLoopMP, i);
                    pLoopMP->AddObservation(mpCurrentKF, i);
                }
            }
        }

        // 4. 投影反向融合
        std::vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetBestCovisibilityKeyFrames(10);
        vpLoopConnectedKFs.push_back(mpMatchedKF);
        SearchAndFuse(vpLoopConnectedKFs);

        // 5. 更新共视关系
        std::vector<KeyFrame *> vpCurrentConnectedKFs = mpCurrentKF->GetBestCovisibilityKeyFrames(15);
        vpCurrentConnectedKFs.push_back(mpCurrentKF);

        for (KeyFrame *pKFi : vpCurrentConnectedKFs)
            if (pKFi && !pKFi->mbBad)
                pKFi->UpdateConnections();

        for (KeyFrame *pKFm : vpLoopConnectedKFs)
            if (pKFm && !pKFm->mbBad)
                pKFm->UpdateConnections();

        // 6. 执行 Essential Graph 位姿图优化
        Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, mTcw_loop);

        // 7. 重置 Tracker 速度模型
        if (mpTracker)
        {
            mpTracker->ResetVelocity();
            mpTracker->mCurrentFrame.SetPose(mpCurrentKF->GetPose());
            mpTracker->mLastFrame.SetPose(mpCurrentKF->GetPose());
        }
    } // 离开 lockMap 作用域，释放地图锁

    // 8. 恢复 LocalMapping 线程运行
    if (mpLocalMapper)
    {
        mpLocalMapper->Release();
    }

    // 9. 开启独立新线程在后台运行 GlobalBundleAdjustment
    mbStopGBA = false;
    mbRunningGBA = true;
    if (mpThreadGBA)
    {
        mpThreadGBA->join();
        delete mpThreadGBA;
    }
    mpThreadGBA = new std::thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);
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

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    std::cout << "\033[32m[GBA Thread] 开始在后台执行全局 BA (LoopKF ID: " << nLoopKF << ")...\033[0m" << std::endl;

    // 调用支持中断的 GBA
    Optimizer::GlobalBundleAdjustment(mpMap, 35, &mbStopGBA);

    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        if (mbStopGBA)
        {
            std::cout << "\033[33m[GBA Thread] 全局 BA 响应中断请求已提前终止。\033[0m" << std::endl;
        }
        else
        {
            std::cout << "\033[32;1m[GBA Thread] 全局 BA 优化完成并成功更新地图！\033[0m" << std::endl;
            mnFullBAIdx++;
        }
        mbRunningGBA = false;
    }
}