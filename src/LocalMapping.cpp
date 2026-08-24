#include "LocalMapping.h"
#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Tracker.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include "LoopClosing.h"
#include <Eigen/SVD>
#include <unistd.h>
#include <algorithm>
#include <cmath>

LocalMapping::LocalMapping(System *pSys, std::shared_ptr<Map> pMap)
    : mpSystem(pSys), mpMap(pMap), mpTracker(nullptr),
      mpCurrentKeyFrame(nullptr), mbStopRequested(false),
      mbStopped(false), mbNotStop(false), mbAcceptKeyFrames(true), mbAbortBA(false)
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
        // 【修改点 1】优先处理停止请求，避免卡死在内部死循环中
        if (GetStopRequired())
        {
            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbStopped = true;
            }

            // 使用 mbStopRequested 作为循环判断条件，配合外部的 Release() 唤醒
            while (GetStopRequired())
            {
                usleep(3000);
            }

            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbStopped = false;
            }
        }

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
            if (!CheckNewKeyFrames() && !GetStopRequired())
            {
                SearchInNeighbors();
            }

            // 5. 局部 BA 优化
            if (!CheckNewKeyFrames() && !GetStopRequired())
            {
                mbAbortBA = false;
                Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
            }

            // 6. 剔除冗余关键帧
            KeyFrameCulling();

            if (mpSystem && mpSystem->GetLoopCloser())
            {
                mpSystem->GetLoopCloser()->InsertKeyFrame(mpCurrentKeyFrame);
            }

            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbNotStop = false;
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

void LocalMapping::ProcessNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // 计算词袋向量
    mpCurrentKeyFrame->ComputeBoW();

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
            else
            {
                // 由 Tracking 线程在插帧时直接创建并加入的立体点，必须加入待考队列接受 MapPointCulling 检验
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}
/**
 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增地图点
 * mlpRecentAddedMapPoints：存储新增的待考核地图点
 */
void LocalMapping::MapPointCulling()
{
    // 待考核地图点队列为空直接返回
    if (mlpRecentAddedMapPoints.empty())
        return;

    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // 双目模式下，通过考核所需的关键帧观测数阈值为 3 (单目为 2)
    const int cnThObs = 3;

    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;

        if (pMP->isBad())
        {
            // 1. 已经是坏点的地图点，直接从待考核队列中移除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->GetFoundRatio() < 0.25f)
        {
            // 2. 跟踪到的帧数占视野预计可见帧数比例小于 25%，剔除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && static_cast<int>(pMP->GetObservations().size()) <= cnThObs)
        {
            // 3. 建立已满 2 个关键帧以上，但观测到该点的关键帧数仍 <= 3，判定为不稳定点剔除
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
        {
            // 4. 建立已达 3 个关键帧且通过上述测试，说明质量可靠，移出考核队列转正
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

/**
 * @brief 检查并融合当前关键帧与相邻关键帧（两级相邻）中重复的地图点
 */
void LocalMapping::SearchInNeighbors()
{
    // Step 1: 获得共视图中权重排名前 nn 的一级邻居（双目/RGB-D 取 20，单目取 10）
    const int nn = 20;
    const std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (vpNeighKFs.empty())
        return;

    // Step 2: 搜集一级相邻关键帧与二级相邻关键帧 (前 5 个共视邻居)
    std::vector<KeyFrame *> vpTargetKFs;
    vpTargetKFs.reserve(vpNeighKFs.size() * 3);

    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        KeyFrame *pKFi = vpNeighKFs[i];
        if (!pKFi || pKFi->mbBad || pKFi->mnId == mpCurrentKeyFrame->mnId)
            continue;

        if (std::find(vpTargetKFs.begin(), vpTargetKFs.end(), pKFi) == vpTargetKFs.end())
            vpTargetKFs.push_back(pKFi);

        // 扩充二级邻居 (前 5 个共视邻居)
        const std::vector<KeyFrame *> vpSecondNeighs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (size_t j = 0; j < vpSecondNeighs.size(); j++)
        {
            KeyFrame *pKFi2 = vpSecondNeighs[j];
            if (!pKFi2 || pKFi2->mbBad || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            if (std::find(vpTargetKFs.begin(), vpTargetKFs.end(), pKFi2) == vpTargetKFs.end())
                vpTargetKFs.push_back(pKFi2);
        }
    }

    if (vpTargetKFs.empty())
        return;

    // Step 3: 正向融合 —— 将当前关键帧的地图点投影到所有目标关键帧中融合
    std::vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (KeyFrame *pKF : vpTargetKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        const Eigen::Matrix3f Rcw = pKF->GetRotation();
        const Eigen::Vector3f tcw = pKF->GetTranslation();
        const Eigen::Vector3f Ow = pKF->GetCameraCenter();

        for (size_t i = 0; i < vpMapPointMatches.size(); i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if (!pMP || pMP->isBad())
                continue;

            // 视锥与可见性几何校验[cite: 2]
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            Eigen::Vector3f Pc = Rcw * Pw + tcw;
            if (Pc.z() <= 0.0f)
                continue;

            const float dist = (Pw - Ow).norm();
            if (dist < pMP->GetMinDistanceInvariance() * 0.8f || dist > pMP->GetMaxDistanceInvariance() * 1.2f)
                continue;

            Eigen::Vector3f Pn = (Pw - Ow).normalized();
            if (Pn.dot(pMP->GetNormal()) < 0.5f)
                continue;

            const float invz = 1.0f / Pc.z();
            const float u = pKF->fx * Pc.x() * invz + pKF->cx;
            const float v = pKF->fy * Pc.y() * invz + pKF->cy;

            if (u < pKF->mnMinX || u >= pKF->mnMaxX || v < pKF->mnMinY || v >= pKF->mnMaxY)
                continue;

            // 自适应金字塔搜索半径
            float ratio = dist / pMP->GetMaxDistanceInvariance();
            int nPredictedLevel = std::max(0, std::min(static_cast<int>(ratio * 4.0f), 3));
            const float radius = 3.0f * pKF->mvScaleFactors[nPredictedLevel];

            const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);
            if (vIndices.empty())
                continue;

            const cv::Mat &dMP = pMP->GetDescriptor();
            int bestDist = ORBmatcher::TH_LOW;
            int bestIdx = -1;

            for (size_t idx : vIndices)
            {
                const cv::Mat &dF = pKF->mDescriptors.row(idx);
                int distDesc = ORBmatcher::DescriptorDistance(dMP, dF);
                if (distDesc < bestDist)
                {
                    bestDist = distDesc;
                    bestIdx = static_cast<int>(idx);
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
                    if (pMP->GetObservations().size() >= pMPinKF->GetObservations().size())
                        pMPinKF->Replace(pMP);
                    else
                        pMP->Replace(pMPinKF);
                }
            }
        }
    }

    // Step 4: 反向融合 —— 搜集目标关键帧中的所有地图点，投影到当前帧中融合
    std::set<MapPoint *> sFuseCandidates;
    for (KeyFrame *pKF : vpTargetKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint *pMP : vpMPs)
        {
            if (pMP && !pMP->isBad())
                sFuseCandidates.insert(pMP);
        }
    }

    const Eigen::Matrix3f RcwCur = mpCurrentKeyFrame->GetRotation();
    const Eigen::Vector3f tcwCur = mpCurrentKeyFrame->GetTranslation();
    const Eigen::Vector3f OwCur = mpCurrentKeyFrame->GetCameraCenter();

    for (MapPoint *pMP : sFuseCandidates)
    {
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3f Pw = pMP->GetWorldPos();
        Eigen::Vector3f Pc = RcwCur * Pw + tcwCur;
        if (Pc.z() <= 0.0f)
            continue;

        const float dist = (Pw - OwCur).norm();
        if (dist < pMP->GetMinDistanceInvariance() * 0.8f || dist > pMP->GetMaxDistanceInvariance() * 1.2f)
            continue;

        Eigen::Vector3f Pn = (Pw - OwCur).normalized();
        if (Pn.dot(pMP->GetNormal()) < 0.5f)
            continue;

        const float invz = 1.0f / Pc.z();
        const float u = mpCurrentKeyFrame->fx * Pc.x() * invz + mpCurrentKeyFrame->cx;
        const float v = mpCurrentKeyFrame->fy * Pc.y() * invz + mpCurrentKeyFrame->cy;

        if (u < mpCurrentKeyFrame->mnMinX || u >= mpCurrentKeyFrame->mnMaxX ||
            v < mpCurrentKeyFrame->mnMinY || v >= mpCurrentKeyFrame->mnMaxY)
            continue;

        float ratio = dist / pMP->GetMaxDistanceInvariance();
        int nPredictedLevel = std::max(0, std::min(static_cast<int>(ratio * 4.0f), 3));
        const float radius = 3.0f * mpCurrentKeyFrame->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = mpCurrentKeyFrame->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty())
            continue;

        const cv::Mat &dMP = pMP->GetDescriptor();
        int bestDist = ORBmatcher::TH_LOW;
        int bestIdx = -1;

        for (size_t idx : vIndices)
        {
            const cv::Mat &dF = mpCurrentKeyFrame->mDescriptors.row(idx);
            int distDesc = ORBmatcher::DescriptorDistance(dMP, dF);
            if (distDesc < bestDist)
            {
                bestDist = distDesc;
                bestIdx = static_cast<int>(idx);
            }
        }

        if (bestIdx >= 0)
        {
            MapPoint *pLocalMP = mpCurrentKeyFrame->GetMapPoint(bestIdx);
            if (!pLocalMP)
            {
                mpCurrentKeyFrame->AddMapPoint(pMP, bestIdx);
                pMP->AddObservation(mpCurrentKeyFrame, bestIdx);
            }
            else if (pLocalMP != pMP)
            {
                if (pMP->GetObservations().size() >= pLocalMP->GetObservations().size())
                    pLocalMP->Replace(pMP);
                else
                    pMP->Replace(pLocalMP);
            }
        }
    }

    // Step 5: 刷新当前帧所有地图点的属性并更新共视连接
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
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

    mpCurrentKeyFrame->UpdateConnections();
    for (KeyFrame *pKF : vpTargetKFs)
    {
        if (pKF && !pKF->mbBad)
            pKF->UpdateConnections();
    }
}

/**
 * @brief 检测当前关键帧在共视图中的关键帧，根据地图点在共视图中的冗余程度剔除该共视关键帧
 * 冗余关键帧的判定：90%以上的地图点能被其他关键帧（至少3个）在相同或更优尺度下观测到
 */
void LocalMapping::KeyFrameCulling()
{
    // Step 1: 获取当前关键帧所有的共视关键帧
    std::vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetConnectedKeyFrames();

    // 对所有的共视关键帧进行遍历
    for (std::vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; ++vit)
    {
        KeyFrame *pKF = *vit;

        // 保护初始关键帧不被剔除，跳过坏帧
        if (!pKF || pKF->mnId == 0 || pKF->mbBad)
            continue;

        // Step 2: 提取该共视关键帧关联的所有地图点
        const std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        const int thObs = 3;            // 冗余观测次数门槛
        int nRedundantObservations = 0; // 记录冗余地图点数量
        int nMPs = 0;                   // 记录该帧有效近处地图点总数

        // Step 3: 遍历该关键帧下的所有地图点
        for (size_t i = 0; i < vpMapPoints.size(); i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                // 双目专属逻辑：仅考虑深度有效的近点来进行冗余评估（远点不作为主冗余依据）
                if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] <= 0.0f)
                    continue;

                nMPs++;

                // 地图点的观测关键帧数必须大于 3 才有可能冗余
                const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
                if (static_cast<int>(observations.size()) > thObs)
                {
                    const int &scaleLevel = pKF->mvKeysUn[i].octave;

                    int nObs = 0;
                    // 遍历观测到该地图点的所有关键帧
                    for (auto mit = observations.begin(), mend = observations.end(); mit != mend; ++mit)
                    {
                        KeyFrame *pKFi = mit->first;
                        if (pKFi == pKF || pKFi->mbBad)
                            continue;

                        const size_t idx_i = mit->second;
                        if (idx_i >= pKFi->mvKeysUn.size())
                            continue;

                        const int &scaleLeveli = pKFi->mvKeysUn[idx_i].octave;

                        // 尺度约束：只有其它关键帧的观测分辨率优于或等同于当前帧时（即 scaleLeveli <= scaleLevel + 1），才算作有效冗余观测
                        if (scaleLeveli <= scaleLevel + 1)
                        {
                            nObs++;
                            if (nObs >= thObs)
                                break;
                        }
                    }

                    // 该地图点在相同/更优尺度下被至少 3 个其它关键帧观测到
                    if (nObs >= thObs)
                    {
                        nRedundantObservations++;
                    }
                }
            }
        }

        // Step 4: 90% 以上的有效近点为冗余观测点时，删除该关键帧
        if (nMPs > 0 && (float)nRedundantObservations > 0.90f * (float)nMPs)
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

// 请求中断当前正在进行的局部 BA
void LocalMapping::RequestStopBA()
{
    mbAbortBA = true;
}

int LocalMapping::KeyframesInQueue()
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    return static_cast<int>(mlNewKeyFrames.size());
}

void LocalMapping::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = false;
    mbStopped = false;
    mbNotStop = false;
}