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
                std::cout << "\033[32;1m>>> [LoopTrigger] 成功检测到连续一致性闭环候选组！<<<\033[0m" << std::endl;

                // 2. 几何位姿求解检验 (PnP / RANSAC)
                if (ComputeSE3())
                {
                    std::cout << "\033[32;1m>>> [LoopTrigger SUCCESS] 闭环几何校验完全通过！当前KF-"
                              << mpCurrentKF->mnId << " 与闭环KF-" << mpMatchedKF->mnId
                              << " 形成有效回环！<<<\033[0m" << std::endl;

                    // 1. 真正执行闭环融合校正
                    CorrectLoop();

                    // 2. 状态重置：清空连续性组与候选缓存
                    mvConsistentGroups.clear();
                    mvpEnoughConsistentCandidates.clear();
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

    // 2. 从关键帧数据库中检索候选帧
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

    // 3. 对候选帧进行共视组聚类与连续性检验 (Temporal Consistency)
    mvpEnoughConsistentCandidates.clear();
    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbGroupMatched(mvConsistentGroups.size(), false);

    for (KeyFrame *pCandKF : vpCandidateKFs)
    {
        std::set<KeyFrame *> sGroup;
        sGroup.insert(pCandKF);
        std::vector<KeyFrame *> vNeighs = pCandKF->GetBestCovisibilityKeyFrames(5);
        for (KeyFrame *pN : vNeighs)
            sGroup.insert(pN);

        bool bMatched = false;
        for (size_t i = 0; i < mvConsistentGroups.size(); ++i)
        {
            const std::set<KeyFrame *> &sPrevGroup = mvConsistentGroups[i].first;
            for (KeyFrame *pGKF : sGroup)
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
            if (bMatched)
                break;
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

    for (KeyFrame *pCandKF : mvpEnoughConsistentCandidates)
    {
        if (!pCandKF || pCandKF->mbBad)
            continue;

        // 1. 通过词袋匹配候选帧与当前帧地图点
        std::vector<MapPoint *> vpMatchedMapPoints;
        int nmatches = matcher.SearchByBoW(mpCurrentKF, pCandKF, vpMatchedMapPoints);

        if (nmatches < 20)
            continue;

        // 2. 构造 3D-2D 对应进行 RANSAC PnP 求解
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
// 阶段 3: 闭环校正与地图融合 (无位姿图优化)
// -----------------------------------------------------------------------------
void LoopClosing::CorrectLoop()
{
    std::cout << "[LoopClosing] 发现有效闭环！当前关键帧: " << mpCurrentKF->mnId
              << " <---> 闭环关键帧: " << mpMatchedKF->mnId << std::endl;

    // 1. 请求暂停 LocalMapping
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

    // 2. 将闭环组相连关键帧中的地图点投影到当前相连帧中，进行点融合
    std::vector<KeyFrame *> vpLoopConnectedKFs = mpMatchedKF->GetConnectedKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    SearchAndFuse(vpLoopConnectedKFs);

    // 3. 重新获取当前关键帧及其相连帧，并更新共视边
    std::vector<KeyFrame *> vpCurrentConnectedKFs = mpCurrentKF->GetConnectedKeyFrames();
    vpCurrentConnectedKFs.push_back(mpCurrentKF);
    for (KeyFrame *pKFi : vpCurrentConnectedKFs)
    {
        pKFi->UpdateConnections();
    }

    // 4. 唤醒并恢复 LocalMapping 线程
    if (mpLocalMapper)
    {
        mpLocalMapper->Release();
    }

    std::cout << "[LoopClosing] 闭环校正与融合完成。" << std::endl;
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