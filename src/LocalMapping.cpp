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
    // 启动 LocalMapping 独立线程
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

// 计算两个向量之间的视差角（角度制）
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

// 判断一对特征点是否满足对极几何约束（双目极线约束）
// 返回点到极线的整体误差（像素）
static float CheckDistEpipolarLine(const KeyFrame *pKF1, const KeyFrame *pKF2,
                                   const Eigen::Matrix3f &F12,
                                   const int idx1, const int idx2)
{
    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
    // 归一化坐标点（齐次）
    Eigen::Vector3f p1(kp1.pt.x, kp1.pt.y, 1.0f);
    Eigen::Vector3f p2(kp2.pt.x, kp2.pt.y, 1.0f);

    Eigen::Vector3f l2 = F12 * p1;
    float sq = l2(0) * l2(0) + l2(1) * l2(1);
    if (sq < 1e-10f)
        return 1e6f;
    float dist = std::fabs(p2.dot(l2)) / std::sqrt(sq);
    return dist;
}

// 计算两个相机位姿之间的本征矩阵所对应的基础矩阵 F12
static Eigen::Matrix3f ComputeFundamentalMatrix(const Eigen::Matrix3f &R1,
                                                const Eigen::Vector3f &t1,
                                                const Eigen::Matrix3f &R2,
                                                const Eigen::Vector3f &t2,
                                                const float fx, const float fy,
                                                const float cx, const float cy)
{
    // 相机1到相机2的变换
    Eigen::Matrix3f R12 = R2 * R1.transpose();
    Eigen::Vector3f t12 = -R12 * t1 + t2;

    // 本质矩阵 E = [t12]_x * R12
    Eigen::Matrix3f tx;
    tx << 0, -t12(2), t12(1),
        t12(2), 0, -t12(0),
        -t12(1), t12(0), 0;
    Eigen::Matrix3f E = tx * R12;

    // 内参矩阵 K 及其逆
    Eigen::Matrix3f K;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;
    Eigen::Matrix3f Kinv = K.inverse();

    // F = K^-T * E * K^-1
    Eigen::Matrix3f F = Kinv.transpose() * E * Kinv;
    return F;
}

// 用中点法（DLT 近似）三角化一对归一化观测
// 返回 true 表示三角化成功，输出 3D 点（世界坐标系）
static bool Triangulate(const Eigen::Matrix3f &R1, const Eigen::Vector3f &t1,
                        const Eigen::Matrix3f &R2, const Eigen::Vector3f &t2,
                        const Eigen::Vector2f &x1, const Eigen::Vector2f &x2,
                        Eigen::Vector3f &x3D)
{
    // 第一相机坐标系下的位姿
    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
    T1.block<3, 3>(0, 0) = R1;
    T1.block<3, 1>(0, 3) = t1;
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2.block<3, 3>(0, 0) = R2;
    T2.block<3, 1>(0, 3) = t2;

    // 构建 DLT 系数矩阵（4 个方程，最小二乘）
    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    const float u1 = x1(0), v1 = x1(1);
    const float u2 = x2(0), v2 = x2(1);

    // Row from view 1: x1 × (T1 X) = 0
    A.row(0) = u1 * T1.row(2) - T1.row(0);
    A.row(1) = v1 * T1.row(2) - T1.row(1);
    // Row from view 2: x2 × (T2 X) = 0
    A.row(2) = u2 * T2.row(2) - T2.row(0);
    A.row(3) = v2 * T2.row(2) - T2.row(1);

    // SVD 求解 Ax = 0，最小奇异值对应的右奇异向量即为解
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
        // 告知 Tracking 线程，LocalMapping 处于忙碌状态，暂时不要频繁插入关键帧
        SetNotStop();

        // 检查是否有新的关键帧等待处理
        if (CheckNewKeyFrames())
        {
            // 1. 处理队列中的首个关键帧（关联 MapPoint，更新共视图 Connections）
            ProcessNewKeyFrame();

            // 2. 剔除近期新增的质量不佳的 MapPoints（观测数不足或可见率低）
            MapPointCulling();

            // 3. 通过与相邻关键帧三角化，创建新的 MapPoints
            CreateNewMapPoints();

            // 4. 融合邻近关键帧中重复的 MapPoints
            SearchInNeighbors();

            // 5. 执行局部 BA 优化
            if (!CheckNewKeyFrames())
                Optimizer::LocalBundleAdjustment(mpMap);

            // 6. 剔除冗余的关键帧（如果某关键帧 90% 以上的地图点能被其他至少3个关键帧看到）
            KeyFrameCulling();
        }

        // 处理完毕，解除 NotStop 标记
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            mbNotStop = false;
        }

        // 检查外部是否有停止请求（如回环检测 LoopClosing 触发 pause）
        if (GetStopRequired())
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            mbStopped = true;
            while (isStopped())
            {
                usleep(3000); // 挂起线程
            }
        }

        usleep(3000); // 适当休眠，避免 CPU 空转
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

    // 1. 关联普通帧匹配生成的地图点到关键帧，并加入局部待检验列表
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
                // 将近期创建的地图点放入考核队列
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    // 2. 更新关键帧在共视图（Covisibility Graph）中的连接关系
    mpCurrentKeyFrame->UpdateConnections();

    // 3. 将关键帧插入全局地图
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // 对近期创建的地图点执行严格的质量考核：
    // 条件1: 被标记为 Bad 的直接从列表中剔除
    // 条件2: 被观测到的实际比例 (Found / Visible) < 25% 的判定为坏点并剔除
    // 条件3: 从创建起经过了连续 2 个关键帧后，观测到它的关键帧数量 < 2 (单目) 或 < 3 (双目) 则剔除
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
        else if (((int)mpCurrentKeyFrame->mnId - (int)pMP->mnId) >= 2 && pMP->GetObservations().size() <= 2)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)mpCurrentKeyFrame->mnId - (int)pMP->mnId) >= 3)
        {
            // 通过考核，不再是“近期新增点”，从待考列表中移除
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
    // 地图中关键帧少于 2 个时不具备三角化条件
    if (mpMap->GetKeyFramesInMap() < 2)
        return;

    // 1. 取共视程度最高的前 10 个邻居关键帧
    const int nn = 10;
    std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    if (vpNeighKFs.empty())
        return;

    // 2. 当前关键帧位姿与内参
    const Eigen::Matrix3f Rcw0 = mpCurrentKeyFrame->GetRotation();
    const Eigen::Vector3f tcw0 = mpCurrentKeyFrame->GetTranslation();
    const Eigen::Vector3f Ow0 = mpCurrentKeyFrame->GetCameraCenter();

    const float fx = mpCurrentKeyFrame->fx;
    const float fy = mpCurrentKeyFrame->fy;
    const float cx = mpCurrentKeyFrame->cx;
    const float cy = mpCurrentKeyFrame->cy;

    // 3. 每个邻居关键帧构造一个 matcher（new 对象便于将来替换成 ORBmatcher 专用匹配）
    // 这里直接复用静态 DescriptorDistance，无需实例化 matcher
    std::vector<MapPoint *> vpMapPoints0 = mpCurrentKeyFrame->GetMapPointMatches();

    unsigned int nNew = 0;

    // 遍历每个邻居关键帧
    for (size_t n = 0; n < vpNeighKFs.size(); n++)
    {
        KeyFrame *pKF2 = vpNeighKFs[n];
        if (pKF2->mbBad)
            continue;

        // --- 基线/视差角检查：两相机光心连线与特征的夹角要足够大 ---
        // 计算两相机光心连线向量
        const Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();
        const Eigen::Vector3f vBaseline = Ow2 - Ow0;

        // 邻居关键帧的观测尺度（用参考特征的平均距离做粗略基线判定）
        // 这里简化为：只检查光心距离不为0
        if (vBaseline.norm() < 1e-5f)
            continue;

        // 求出当前关键帧所有已建特征点中，沿两条光射线方向的夹角，用来筛掉纯平移
        // （更严谨做法见 ORB-SLAM 中基于 CheckRT 的判定，这里用最简可用的版本：
        //   先三角化，再用视差角验证）

        // 建立当前帧特征点索引 -> 是否可用来三角化的标记
        const cv::Mat &Desc0 = mpCurrentKeyFrame->mDescriptors;
        const cv::Mat &Desc2 = pKF2->mDescriptors;

        // 对每一个当前帧特征点，在邻居帧中搜索最相似的点
        const std::vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();

        // 遍历当前关键帧的所有特征点
        for (int i = 0; i < mpCurrentKeyFrame->N; i++)
        {
            // 只处理还没有地图点、且不是双目远点的情况
            if (vpMapPoints0[i])
                continue;

            const cv::KeyPoint &kp0 = mpCurrentKeyFrame->mvKeysUn[i];
            const int level0 = kp0.octave;

            // 邻居帧也在相同区域搜索（尺度相近）
            const float radius = 2.0f * 15.0f * mpCurrentKeyFrame->mvScaleFactors[level0];
            const std::vector<size_t> vCandidates =
                pKF2->GetFeaturesInArea(kp0.pt.x, kp0.pt.y, radius, level0 - 1, level0 + 1);
            if (vCandidates.empty())
                continue;

            const cv::Mat &d0 = Desc0.row(i);
            int bestDist = ORBmatcher::TH_LOW;
            int bestIdx2 = -1;

            // 在候选点里找描述子最近的那个
            for (size_t c = 0; c < vCandidates.size(); c++)
            {
                const size_t j = vCandidates[c];
                if (vpMapPoints2[j]) // 邻居帧该点已有地图点，跳过（避免重复建）
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

            // --- 对极约束检查 ---
            const Eigen::Matrix3f Rcw1 = Rcw0;
            const Eigen::Vector3f tcw1 = tcw0;
            const Eigen::Matrix3f Rcw2 = pKF2->GetRotation();
            const Eigen::Vector3f tcw2 = pKF2->GetTranslation();

            // 计算基础矩阵 F12
            // F12 把当前帧(k1)像素坐标映射到邻居帧(k2)的极线
            Eigen::Matrix3f F12 = ComputeFundamentalMatrix(
                Rcw1, tcw1, Rcw2, tcw2, fx, fy, cx, cy);

            float distEpipolar = CheckDistEpipolarLine(
                mpCurrentKeyFrame, pKF2, F12, i, bestIdx2);

            // 像素距离阈值：依据特征尺度放大
            const float sigma2 = mpCurrentKeyFrame->mvLevelSigma2[level0];
            const float epipolarTh = 3.841f * sigma2; // 单目时用 3.84
            if (distEpipolar > epipolarTh)
                continue;

            // --- 三角化 ---
            Eigen::Vector2f xpi0(kp0.pt.x, kp0.pt.y); // 已含内参，直接作为归一化观测（近似）
            Eigen::Vector2f xpi2(pKF2->mvKeysUn[bestIdx2].pt.x,
                                 pKF2->mvKeysUn[bestIdx2].pt.y);

            Eigen::Vector3f x3D;
            if (!Triangulate(Rcw1, tcw1, Rcw2, tcw2, xpi0, xpi2, x3D))
                continue;
            if (std::isnan(x3D(0)) || std::isnan(x3D(1)) || std::isnan(x3D(2)))
                continue;

            // --- 检查 3D 点在两个相机前的深度为正，且在有效尺度范围 ---
            // 转到当前帧相机坐标
            Eigen::Vector3f Pc0 = Rcw1 * x3D + tcw1;
            if (Pc0.z() <= 0.0f)
                continue;
            // 转到邻居帧相机坐标
            Eigen::Vector3f Pc2 = Rcw2 * x3D + tcw2;
            if (Pc2.z() <= 0.0f)
                continue;

            const float dist0 = (x3D - Ow0).norm();
            const float dist2 = (x3D - Ow2).norm();
            if (dist0 <= 0.0f || dist2 <= 0.0f)
                continue;

            // --- 重投影误差检查（以当前帧为准）---
            const float u0 = fx * Pc0.x() / Pc0.z() + cx;
            const float v0 = fy * Pc0.y() / Pc0.z() + cy;
            const float errU = u0 - kp0.pt.x;
            const float errV = v0 - kp0.pt.y;
            const float reprojErr = errU * errU + errV * errV;
            if (reprojErr > 5.991f * sigma2) // 2 自由度卡方阈值
                continue;

            // 在邻居帧同样检查重投影
            const float u2 = fx * Pc2.x() / Pc2.z() + cx;
            const float v2 = fy * Pc2.y() / Pc2.z() + cy;
            const float errU2 = u2 - pKF2->mvKeysUn[bestIdx2].pt.x;
            const float errV2 = v2 - pKF2->mvKeysUn[bestIdx2].pt.y;
            const float reprojErr2 = errU2 * errU2 + errV2 * errV2;

            // --- 视差角检查：利用两条从光心到地图点的射线夹角 ---
            const Eigen::Vector3f ray0 = (x3D - Ow0).normalized();
            const Eigen::Vector3f ray2 = (x3D - Ow2).normalized();
            const double parallax = ComputeParallax(ray0, ray2);
            // 近点需要更大视差角，远点允许更小；这里给一个保守下限
            if (parallax < 1.0) // 小于1度视为纯旋转/退化，跳过
                continue;

            // --- 通过所有检查，创建新的地图点 ---
            MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap.get());
            // 建立两帧的观测
            pMP->AddObservation(mpCurrentKeyFrame, i);
            pMP->AddObservation(pKF2, bestIdx2);
            // 反向建立关键帧到地图点的引用
            mpCurrentKeyFrame->AddMapPoint(pMP, i);
            pKF2->AddMapPoint(pMP, bestIdx2);

            // 计算描述子、法线、深度范围
            pMP->ComputeDistinctiveDescriptor();
            pMP->UpdateNormalAndDepth();

            // 加入地图和近期考核队列
            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            // 同步本地副本，防止后续特征点重复处理
            vpMapPoints0[i] = pMP;
            nNew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // 1. 获取当前关键帧的一级邻居（共视最高前 20 个）
    const int nn = 20;
    std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    if (vpNeighKFs.empty())
        return;

    // 2. 收集一、二级邻居的全部地图点（作为融合候选），并记录一级邻居
    std::vector<KeyFrame *> vpTargetKFs;   // 需要更新连接关系的关键帧集合
    std::set<MapPoint *> vpFuseCandidates; // 邻居的地图点（用于与当前帧融合）
    const int nn2 = 5;                     // 每个一级邻居取多少二级邻居

    // 当前关键帧自己的地图点
    std::vector<MapPoint *> vpLocalMapPoints = mpCurrentKeyFrame->GetMapPointMatches();

    // ------------- 阶段 A：融合（把邻居的地图点与当前帧融合） -------------
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        KeyFrame *pKFi = vpNeighKFs[i];
        if (pKFi->mbBad)
            continue;

        vpTargetKFs.push_back(pKFi);

        // 一级邻居的地图点作为融合候选
        std::vector<MapPoint *> vpMPi = pKFi->GetMapPointMatches();
        for (size_t j = 0; j < vpMPi.size(); j++)
            if (vpMPi[j] && !vpMPi[j]->isBad())
                vpFuseCandidates.insert(vpMPi[j]);

        // 二级邻居
        std::vector<KeyFrame *> vpNeigh2 = pKFi->GetBestCovisibilityKeyFrames(nn2);
        for (size_t j = 0; j < vpNeigh2.size(); j++)
        {
            KeyFrame *pKFi2 = vpNeigh2[j];
            if (pKFi2->mbBad || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;

            // 避免重复加入（已在集合中则跳过）
            if (std::find(vpTargetKFs.begin(), vpTargetKFs.end(), pKFi2) != vpTargetKFs.end())
                continue;

            vpTargetKFs.push_back(pKFi2);

            std::vector<MapPoint *> vpMP2 = pKFi2->GetMapPointMatches();
            for (size_t k = 0; k < vpMP2.size(); k++)
                if (vpMP2[k] && !vpMP2[k]->isBad())
                    vpFuseCandidates.insert(vpMP2[k]);
        }
    }

    // 当前帧的内参与位姿
    const float fx = mpCurrentKeyFrame->fx;
    const float fy = mpCurrentKeyFrame->fy;
    const float cx = mpCurrentKeyFrame->cx;
    const float cy = mpCurrentKeyFrame->cy;
    const Eigen::Matrix3f Rcw = mpCurrentKeyFrame->GetRotation();
    const Eigen::Vector3f tcw = mpCurrentKeyFrame->GetTranslation();

    // 把邻居的地图点投影到当前帧，与当前帧已有地图点/特征点做融合
    for (std::set<MapPoint *>::iterator sit = vpFuseCandidates.begin();
         sit != vpFuseCandidates.end(); sit++)
    {
        MapPoint *pMP = *sit;
        if (pMP->isBad())
            continue;

        // a) 投影到当前帧像素坐标
        Eigen::Vector3f Pc = Rcw * pMP->GetWorldPos() + tcw;
        if (Pc.z() <= 0.0f)
            continue;

        const float u = fx * Pc.x() / Pc.z() + cx;
        const float v = fy * Pc.y() / Pc.z() + cy;
        if (u < mpCurrentKeyFrame->mnMinX || u >= mpCurrentKeyFrame->mnMaxX ||
            v < mpCurrentKeyFrame->mnMinY || v >= mpCurrentKeyFrame->mnMaxY)
            continue;

        // b) 半径内搜索当前帧特征点
        const float radius = 10.0f;
        const std::vector<size_t> vCandidates =
            mpCurrentKeyFrame->GetFeaturesInArea(u, v, radius);
        if (vCandidates.empty())
            continue;

        // c) 找描述子最近、无冲突的候选特征
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

        MapPoint *pLocalMP = vpLocalMapPoints[bestIdx];
        if (!pLocalMP)
        {
            // 当前帧该特征点尚未有关联地图点 -> 直接关联邻居的地图点
            mpCurrentKeyFrame->AddMapPoint(pMP, bestIdx);
            pMP->AddObservation(mpCurrentKeyFrame, bestIdx);
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptor();
        }
        else if (pLocalMP->mnId != pMP->mnId)
        {
            // 两个地图点在该特征点投影处重合，属于重复点：
            // 保留被观测次数更多（更可信）的那个，另一个用 Replace 融合进来
            if (pLocalMP->GetObservations().size() >= pMP->GetObservations().size())
                pMP->Replace(pLocalMP);
            else
                pLocalMP->Replace(pMP);
        }
    }

    // ------------- 阶段 B：把当前帧的地图点投影到邻居关键帧做二次融合 -------------
    // 将当前帧的地图点与之前收集的邻居地图点互相融合（反向投影搜索）
    // 这样能消除邻居之间以及邻居与当前帧之间的重复点
    for (size_t i = 0; i < vpLocalMapPoints.size(); i++)
    {
        MapPoint *pMP = vpLocalMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        // 将该地图点投影到每个一级邻居做匹配
        // 若在邻居帧某特征点投影处发现相似描述子，且邻居该特征点已有别的地图点，
        // 则进行 Replace 融合（合并重复点）
        for (size_t k = 0; k < vpNeighKFs.size(); k++)
        {
            KeyFrame *pKF = vpNeighKFs[k];
            if (pKF->mbBad)
                continue;

            const Eigen::Matrix3f Rk = pKF->GetRotation();
            const Eigen::Vector3f tk = pKF->GetTranslation();
            const Eigen::Vector3f Pk = Rk * pMP->GetWorldPos() + tk;
            if (Pk.z() <= 0.0f)
                continue;

            const float uk = pKF->fx * Pk.x() / Pk.z() + pKF->cx;
            const float vk = pKF->fy * Pk.y() / Pk.z() + pKF->cy;
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
            }
            else if (pNeighMP->mnId != pMP->mnId)
            {
                // 邻居该特征点已有重复地图点，按观测数融合
                if (pMP->GetObservations().size() >= pNeighMP->GetObservations().size())
                    pNeighMP->Replace(pMP);
                else
                    pMP->Replace(pNeighMP);
            }
        }
    }

    // ------------- 阶段 C：剔除坏点并更新连接关系 -------------
    // 清除当前关键帧中变成坏点的地图点引用（Replace/SetBadFlag 后可能残留）
    {
        std::vector<MapPoint *> vpMP = mpCurrentKeyFrame->GetMapPointMatches();
        for (size_t i = 0; i < vpMP.size(); i++)
            if (vpMP[i] && vpMP[i]->isBad())
                mpCurrentKeyFrame->EraseMapPointMatch(i);
    }

    // 重新计算共视图连接关系（融合改变了对同一地图点的观测，权重可能变化）
    mpCurrentKeyFrame->UpdateConnections();
    for (size_t i = 0; i < vpTargetKFs.size(); i++)
        vpTargetKFs[i]->UpdateConnections();
}

// LocalMapping.cpp
void LocalMapping::KeyFrameCulling()
{
    // 替换为已有接口名：GetConnectedKeyFrames()
    std::vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKeyFrame->GetConnectedKeyFrames();

    for (auto pKF : vpConnectedKeyFrames)
    {
        if (pKF->mnId == 0)
            continue; // 保留初始化帧

        std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
        int nRedundantObservations = 0;
        int nTotalObservations = 0;

        for (size_t i = 0; i < vpMapPoints.size(); i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                nTotalObservations++;
                if (pMP->GetObservations().size() > 3)
                {
                    nRedundantObservations++;
                }
            }
        }

        if (nTotalObservations > 0 && (float)nRedundantObservations / nTotalObservations > 0.90f)
        {
            pKF->SetBadFlag(); // 现在已可正常调用
        }
    }
}

// 线程控制相关函数
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
