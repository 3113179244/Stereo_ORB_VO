#include "LocalMapping.h"
#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Tracker.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include <Eigen/SVD>
#include <unistd.h>
#include <algorithm>
#include <cmath>

LocalMapping::LocalMapping(System *pSys, std::shared_ptr<Map> pMap)
    : mpSystem(pSys), mpMap(pMap), mpTracker(nullptr),
      mpCurrentKeyFrame(nullptr), mbStopRequested(false),
      mbStopped(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
    mpThread = new std::thread(&LocalMapping::Run, this);
}

LocalMapping::~LocalMapping()
{
    if (mpThread)
    {
        mpThread->join();
        delete mpThread;
    }
}

static double ComputeParallax(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    Eigen::Vector3f va = a;
    Eigen::Vector3f vb = b;
    if (va.norm() < 1e-8 || vb.norm() < 1e-8)
        return 0.0;
    double cosA = va.dot(vb) / (va.norm() * vb.norm());
    if (cosA > 1.0)
        cosA = 1.0;
    if (cosA < -1.0)
        cosA = -1.0;
    return acos(cosA) * 180.0 / M_PI;
}

static float CheckDistEpipolarLine(const KeyFrame *pKF1, const KeyFrame *pKF2,
                                   const Eigen::Matrix3f &F12,
                                   const int idx1, const int idx2)
{
    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
    Eigen::Vector3f p1(kp1.pt.x, kp1.pt.y, 1.0f);
    Eigen::Vector3f p2(kp2.pt.x, kp2.pt.y, 1.0f);

    Eigen::Vector3f l2 = F12 * p1;
    float sq = l2(0) * l2(0) + l2(1) * l2(1);
    if (sq < 1e-10f)
        return 1e6f;
    return std::fabs(p2.dot(l2)) / std::sqrt(sq);
}

static Eigen::Matrix3f ComputeFundamentalMatrix(const Eigen::Matrix3f &R1,
                                                const Eigen::Vector3f &t1,
                                                const Eigen::Matrix3f &R2,
                                                const Eigen::Vector3f &t2,
                                                const float fx, const float fy,
                                                const float cx, const float cy)
{
    Eigen::Matrix3f R12 = R2 * R1.transpose();
    Eigen::Vector3f t12 = -R12 * t1 + t2;

    Eigen::Matrix3f tx;
    tx << 0, -t12(2), t12(1),
        t12(2), 0, -t12(0),
        -t12(1), t12(0), 0;
    Eigen::Matrix3f E = tx * R12;

    Eigen::Matrix3f K;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;
    Eigen::Matrix3f Kinv = K.inverse();

    return Kinv.transpose() * E * Kinv;
}

static bool Triangulate(const Eigen::Matrix3f &R1, const Eigen::Vector3f &t1,
                        const Eigen::Matrix3f &R2, const Eigen::Vector3f &t2,
                        const Eigen::Vector2f &x1, const Eigen::Vector2f &x2,
                        Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
    T1.block<3, 3>(0, 0) = R1;
    T1.block<3, 1>(0, 3) = t1;
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2.block<3, 3>(0, 0) = R2;
    T2.block<3, 1>(0, 3) = t2;

    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    const float u1 = x1(0), v1 = x1(1);
    const float u2 = x2(0), v2 = x2(1);

    A.row(0) = u1 * T1.row(2) - T1.row(0);
    A.row(1) = v1 * T1.row(2) - T1.row(1);
    A.row(2) = u2 * T2.row(2) - T2.row(0);
    A.row(3) = v2 * T2.row(2) - T2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f Xh = svd.matrixV().col(3);
    if (std::fabs(Xh(3)) < 1e-10f)
        return false;
    x3D = Xh.head<3>() / Xh(3);
    return true;
}

void LocalMapping::Run()
{
    mbStopped = false;

    while (1)
    {
        if (CheckNewKeyFrames())
        {
            SetNotStop();

            // 1. 处理关键帧
            ProcessNewKeyFrame();

            // 2. 考核并剔除劣质地图点
            MapPointCulling();

            // 3. 三角化新地图点
            CreateNewMapPoints();

            // 4. 重复点融合
            if (!CheckNewKeyFrames())
            {
                SearchInNeighbors();
            }

            // 5. 局部 BA 优化
            // if (!CheckNewKeyFrames())
            //     Optimizer::LocalBundleAdjustment(mpMap);

            // 6. 剔除冗余关键帧
            KeyFrameCulling();

            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbNotStop = false;
            }
        }

        // 处理停止请求
        if (GetStopRequired())
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            mbStopped = true;
            while (isStopped())
            {
                usleep(3000);
            }
        }

        usleep(3000);
    }
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    mlNewKeyFrames.push_back(pKF);
}

bool LocalMapping::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    return !mlNewKeyFrames.empty();
}

bool LocalMapping::KeyframesInQueue()
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    return !mlNewKeyFrames.empty();
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    std::vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP && !pMP->isBad())
        {
            if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
            {
                pMP->AddObservation(mpCurrentKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptor();
            }

            // 观测数较少（<= 2）的新点加入待考核队列进行质量检验
            if (pMP->GetObservations().size() <= 2)
            {
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    auto lit = mlpRecentAddedMapPoints.begin();
    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;
        if (pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->GetFoundRatio() < 0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 观测数足够（> 2）说明质量稳定，直接通过考核，移出待考队列
        else if (pMP->GetObservations().size() > 2)
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
        }
    }
}

void LocalMapping::CreateNewMapPoints()
{
    if (mpMap->GetKeyFramesInMap() < 2)
        return;

    const int nn = 10;
    std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    if (vpNeighKFs.empty())
        return;

    const Eigen::Matrix3f Rcw0 = mpCurrentKeyFrame->GetRotation();
    const Eigen::Vector3f tcw0 = mpCurrentKeyFrame->GetTranslation();
    const Eigen::Vector3f Ow0 = mpCurrentKeyFrame->GetCameraCenter();

    const float fx = mpCurrentKeyFrame->fx;
    const float fy = mpCurrentKeyFrame->fy;
    const float cx = mpCurrentKeyFrame->cx;
    const float cy = mpCurrentKeyFrame->cy;

    std::vector<MapPoint *> vpMapPoints0 = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t n = 0; n < vpNeighKFs.size(); n++)
    {
        KeyFrame *pKF2 = vpNeighKFs[n];
        if (pKF2->mbBad)
            continue;

        const Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        const Eigen::Vector3f vBaseline = Ow2 - Ow0;
        if (vBaseline.norm() < 1e-5f)
            continue;

        const cv::Mat &Desc0 = mpCurrentKeyFrame->mDescriptors;
        const cv::Mat &Desc2 = pKF2->mDescriptors;
        const std::vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();

        for (int i = 0; i < mpCurrentKeyFrame->N; i++)
        {
            if (vpMapPoints0[i])
                continue;

            const cv::KeyPoint &kp0 = mpCurrentKeyFrame->mvKeysUn[i];
            const int level0 = kp0.octave;
            const float fx_norm = mpCurrentKeyFrame->fx / 500.0f;
            const float radius = 2.0f * (15.0f * std::max(1.0f, fx_norm)) * mpCurrentKeyFrame->mvScaleFactors[level0];
            const std::vector<size_t> vCandidates =
                pKF2->GetFeaturesInArea(kp0.pt.x, kp0.pt.y, radius, level0 - 1, level0 + 1);
            if (vCandidates.empty())
                continue;

            const cv::Mat &d0 = Desc0.row(i);
            int bestDist = ORBmatcher::TH_LOW;
            int bestIdx2 = -1;

            for (size_t c = 0; c < vCandidates.size(); c++)
            {
                const size_t j = vCandidates[c];
                if (vpMapPoints2[j])
                    continue;

                const cv::Mat &d2 = Desc2.row(j);
                const int dist = ORBmatcher::DescriptorDistance(d0, d2);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx2 = static_cast<int>(j);
                }
            }

            if (bestIdx2 < 0 || bestDist > ORBmatcher::TH_LOW)
                continue;

            const Eigen::Matrix3f Rcw1 = Rcw0;
            const Eigen::Vector3f tcw1 = tcw0;
            const Eigen::Matrix3f Rcw2 = pKF2->GetRotation();
            const Eigen::Vector3f tcw2 = pKF2->GetTranslation();

            Eigen::Matrix3f F12 = ComputeFundamentalMatrix(Rcw1, tcw1, Rcw2, tcw2, fx, fy, cx, cy);

            float distEpipolar = CheckDistEpipolarLine(mpCurrentKeyFrame, pKF2, F12, i, bestIdx2);
            const float sigma2 = mpCurrentKeyFrame->mvLevelSigma2[level0];
            if (distEpipolar > 3.841f * sigma2)
                continue;

            Eigen::Vector2f xpi0((kp0.pt.x - cx) / fx, (kp0.pt.y - cy) / fy);
            Eigen::Vector2f xpi2((pKF2->mvKeysUn[bestIdx2].pt.x - cx) / fx,
                                 (pKF2->mvKeysUn[bestIdx2].pt.y - cy) / fy);

            Eigen::Vector3f x3D;
            if (!Triangulate(Rcw1, tcw1, Rcw2, tcw2, xpi0, xpi2, x3D))
                continue;
            if (std::isnan(x3D(0)) || std::isnan(x3D(1)) || std::isnan(x3D(2)))
                continue;

            Eigen::Vector3f Pc0 = Rcw1 * x3D + tcw1;
            if (Pc0.z() <= 0.0f)
                continue;
            Eigen::Vector3f Pc2 = Rcw2 * x3D + tcw2;
            if (Pc2.z() <= 0.0f)
                continue;

            // 重投影误差检验
            const float u0 = fx * Pc0.x() / Pc0.z() + cx;
            const float v0 = fy * Pc0.y() / Pc0.z() + cy;
            const float reprojErr0 = (u0 - kp0.pt.x) * (u0 - kp0.pt.x) + (v0 - kp0.pt.y) * (v0 - kp0.pt.y);
            if (reprojErr0 > 5.991f * sigma2)
                continue;

            const float u2 = fx * Pc2.x() / Pc2.z() + cx;
            const float v2 = fy * Pc2.y() / Pc2.z() + cy;
            const float reprojErr2 = (u2 - pKF2->mvKeysUn[bestIdx2].pt.x) * (u2 - pKF2->mvKeysUn[bestIdx2].pt.x) +
                                     (v2 - pKF2->mvKeysUn[bestIdx2].pt.y) * (v2 - pKF2->mvKeysUn[bestIdx2].pt.y);
            if (reprojErr2 > 5.991f * sigma2)
                continue;

            const Eigen::Vector3f ray0 = (x3D - Ow0).normalized();
            const Eigen::Vector3f ray2 = (x3D - Ow2).normalized();
            if (ComputeParallax(ray0, ray2) < 1.0)
                continue;

            MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap.get());
            pMP->AddObservation(mpCurrentKeyFrame, i);
            pMP->AddObservation(pKF2, bestIdx2);
            mpCurrentKeyFrame->AddMapPoint(pMP, i);
            pKF2->AddMapPoint(pMP, bestIdx2);

            pMP->ComputeDistinctiveDescriptor();
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
            vpMapPoints0[i] = pMP;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // 1. 获取当前关键帧的最佳共视邻居关键帧（一级邻居）
    const int nn = 10;
    std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    if (vpNeighKFs.empty())
        return;

    std::vector<KeyFrame *> vpTargetKFs;
    std::set<MapPoint *> vpFuseCandidates;
    const int nn2 = 5;

    // 收集所有候选邻居关键帧（一级邻居 + 部分二级邻居）以及它们的地图点
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        KeyFrame *pKFi = vpNeighKFs[i];
        if (!pKFi || pKFi->mbBad)
            continue;

        vpTargetKFs.push_back(pKFi);
        std::vector<MapPoint *> vpMPi = pKFi->GetMapPointMatches();
        for (size_t j = 0; j < vpMPi.size(); j++)
        {
            if (vpMPi[j] && !vpMPi[j]->isBad())
                vpFuseCandidates.insert(vpMPi[j]);
        }

        // 扩展到二级邻居
        std::vector<KeyFrame *> vpNeigh2 = pKFi->GetBestCovisibilityKeyFrames(nn2);
        for (size_t j = 0; j < vpNeigh2.size(); j++)
        {
            KeyFrame *pKFi2 = vpNeigh2[j];
            if (!pKFi2 || pKFi2->mbBad || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            if (std::find(vpTargetKFs.begin(), vpTargetKFs.end(), pKFi2) != vpTargetKFs.end())
                continue;

            vpTargetKFs.push_back(pKFi2);
            std::vector<MapPoint *> vpMP2 = pKFi2->GetMapPointMatches();
            for (size_t k = 0; k < vpMP2.size(); k++)
            {
                if (vpMP2[k] && !vpMP2[k]->isBad())
                    vpFuseCandidates.insert(vpMP2[k]);
            }
        }
    }

    const float fx = mpCurrentKeyFrame->fx;
    const float fy = mpCurrentKeyFrame->fy;
    const float cx = mpCurrentKeyFrame->cx;
    const float cy = mpCurrentKeyFrame->cy;
    const Eigen::Matrix3f Rcw = mpCurrentKeyFrame->GetRotation();
    const Eigen::Vector3f tcw = mpCurrentKeyFrame->GetTranslation();

    // ==========================================
    // 阶段 A: 将邻居的地图点投影到当前关键帧融合
    // ==========================================
    std::vector<MapPoint *> vpLocalMapPoints = mpCurrentKeyFrame->GetMapPointMatches();

    for (std::set<MapPoint *>::iterator sit = vpFuseCandidates.begin();
         sit != vpFuseCandidates.end(); sit++)
    {
        MapPoint *pMP = *sit;
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3f Pc = Rcw * pMP->GetWorldPos() + tcw;
        if (Pc.z() <= 0.0f)
            continue;

        const float invz = 1.0f / Pc.z();
        const float u = fx * Pc.x() * invz + cx;
        const float v = fy * Pc.y() * invz + cy;

        if (u < mpCurrentKeyFrame->mnMinX || u >= mpCurrentKeyFrame->mnMaxX ||
            v < mpCurrentKeyFrame->mnMinY || v >= mpCurrentKeyFrame->mnMaxY)
            continue;

        // 根据特征点金字塔层级设置搜索半径
        const std::vector<size_t> vCandidates = mpCurrentKeyFrame->GetFeaturesInArea(u, v, 10.0f);
        if (vCandidates.empty())
            continue;

        const cv::Mat &dMP = pMP->GetDescriptor();
        int bestDist = ORBmatcher::TH_HIGH;
        int bestIdx = -1;

        for (size_t c = 0; c < vCandidates.size(); c++)
        {
            const size_t idx = vCandidates[c];
            const cv::Mat &dF = mpCurrentKeyFrame->mDescriptors.row(idx);
            const int dist = ORBmatcher::DescriptorDistance(dMP, dF);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = static_cast<int>(idx);
            }
        }

        if (bestIdx < 0 || bestDist > ORBmatcher::TH_HIGH)
            continue;

        MapPoint *pLocalMP = mpCurrentKeyFrame->GetMapPoint(bestIdx);
        if (!pLocalMP)
        {
            // 当前关键帧没有该点，直接建立观测关联并更新点属性
            mpCurrentKeyFrame->AddMapPoint(pMP, bestIdx);
            pMP->AddObservation(mpCurrentKeyFrame, bestIdx);
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptor();
        }
        else if (pLocalMP->mnId != pMP->mnId)
        {
            // 发生重复观测，选择观测更多的点进行融合替换
            if (pLocalMP->GetObservations().size() >= pMP->GetObservations().size())
            {
                pMP->Replace(pLocalMP);
            }
            else
            {
                pLocalMP->Replace(pMP);
            }
        }
    }

    // ==========================================
    // 阶段 B: 将当前关键帧的地图点反向投影到目标关键帧融合
    // ==========================================
    vpLocalMapPoints = mpCurrentKeyFrame->GetMapPointMatches(); // 重新获取更新后的点

    for (size_t i = 0; i < vpLocalMapPoints.size(); i++)
    {
        MapPoint *pMP = vpLocalMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        for (size_t k = 0; k < vpTargetKFs.size(); k++)
        {
            KeyFrame *pKF = vpTargetKFs[k];
            if (!pKF || pKF->mbBad)
                continue;

            const Eigen::Matrix3f Rk = pKF->GetRotation();
            const Eigen::Vector3f tk = pKF->GetTranslation();
            const Eigen::Vector3f Pk = Rk * pMP->GetWorldPos() + tk;
            if (Pk.z() <= 0.0f)
                continue;

            const float invzk = 1.0f / Pk.z();
            const float uk = pKF->fx * Pk.x() * invzk + pKF->cx;
            const float vk = pKF->fy * Pk.y() * invzk + pKF->cy;

            if (uk < pKF->mnMinX || uk >= pKF->mnMaxX ||
                vk < pKF->mnMinY || vk >= pKF->mnMaxY)
                continue;

            const std::vector<size_t> vCands = pKF->GetFeaturesInArea(uk, vk, 10.0f);
            if (vCands.empty())
                continue;

            const cv::Mat &dMPF = pMP->GetDescriptor();
            int bestDist = ORBmatcher::TH_HIGH;
            int bestIdx = -1;

            for (size_t c = 0; c < vCands.size(); c++)
            {
                const size_t idx = vCands[c];
                const cv::Mat &dF = pKF->mDescriptors.row(idx);
                const int dist = ORBmatcher::DescriptorDistance(dMPF, dF);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = static_cast<int>(idx);
                }
            }

            if (bestIdx < 0 || bestDist > ORBmatcher::TH_HIGH)
                continue;

            MapPoint *pNeighMP = pKF->GetMapPoint(bestIdx);
            if (!pNeighMP)
            {
                pKF->AddMapPoint(pMP, bestIdx);
                pMP->AddObservation(pKF, bestIdx);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptor();
            }
            else if (pNeighMP->mnId != pMP->mnId)
            {
                if (pMP->GetObservations().size() >= pNeighMP->GetObservations().size())
                {
                    pNeighMP->Replace(pMP);
                }
                else
                {
                    pMP->Replace(pNeighMP);
                }
            }
        }
    }

    // ==========================================
    // 阶段 C: 关键修复：统一清理坏点并重新计算所有更新点的几何与描述子
    // ==========================================
    // 1. 刷新当前帧的匹配点列表并更新点属性
    vpLocalMapPoints = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0; i < vpLocalMapPoints.size(); i++)
    {
        MapPoint *pMP = vpLocalMapPoints[i];
        if (pMP)
        {
            if (pMP->isBad())
            {
                mpCurrentKeyFrame->EraseMapPointMatch(i);
            }
            else
            {
                pMP->ComputeDistinctiveDescriptor();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // 2. 更新 Covisibility Graph 拓扑关系
    mpCurrentKeyFrame->UpdateConnections();
    for (size_t i = 0; i < vpTargetKFs.size(); i++)
    {
        if (vpTargetKFs[i] && !vpTargetKFs[i]->mbBad)
            vpTargetKFs[i]->UpdateConnections();
    }
}

void LocalMapping::KeyFrameCulling()
{
    std::vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKeyFrame->GetConnectedKeyFrames();

    for (auto pKF : vpConnectedKeyFrames)
    {
        if (!pKF || pKF->mbBad || pKF->mnId <= 1)
            continue;

        std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
        int nRedundantObservations = 0;
        int nTotalObservations = 0;

        for (size_t i = 0; i < vpMapPoints.size(); i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                nTotalObservations++;

                // 获取当前特征点的金字塔层级
                const int scaleLevel = pKF->mvKeysUn[i].octave;
                const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();

                int nObs = 0;
                for (auto mit = observations.begin(); mit != observations.end(); mit++)
                {
                    KeyFrame *pKFi = mit->first;
                    if (pKFi == pKF || pKFi->mbBad)
                        continue;

                    const int scaleLeveli = pKFi->mvKeysUn[mit->second].octave;
                    // 只有在邻居关键帧中的观测尺度优于或接近当前尺度时，才算作有效冗余观测
                    if (scaleLeveli <= scaleLevel + 1)
                    {
                        nObs++;
                        if (nObs >= 3)
                            break;
                    }
                }

                if (nObs >= 3)
                {
                    nRedundantObservations++;
                }
            }
        }

        if (nTotalObservations > 0 && (float)nRedundantObservations / nTotalObservations > 0.90f)
        {
            pKF->SetBadFlag();
        }
    }
}

void LocalMapping::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool LocalMapping::GetStopRequired()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool LocalMapping::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::SetNotStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (mbStopped)
        return false;
    mbNotStop = true;
    return true;
}