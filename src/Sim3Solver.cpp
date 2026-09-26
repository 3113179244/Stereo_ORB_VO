#include "Sim3Solver.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include <algorithm>

Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2,
                       const std::vector<MapPoint *> &vpMatched12,
                       const bool bFixScale)
    : mpKF1(pKF1), mpKF2(pKF2), mbFixScale(bFixScale),
      mIterations(0), mBestInliers(0), mBestScale(1.0f),
      mRng(std::random_device{}())
{
    mRng.seed(0);
    // 获取相机 1 与相机 2 的位姿，用于将地图点转换至各自相机坐标系
    Eigen::Matrix4f T1w = pKF1->GetPose();
    Eigen::Matrix3f R1w = T1w.block<3, 3>(0, 0);
    Eigen::Vector3f t1w = T1w.block<3, 1>(0, 3);

    Eigen::Matrix4f T2w = pKF2->GetPose();
    Eigen::Matrix3f R2w = T2w.block<3, 3>(0, 0);
    Eigen::Vector3f t2w = T2w.block<3, 1>(0, 3);

    const int N = vpMatched12.size();
    mvX3Dc1.reserve(N);
    mvX3Dc2.reserve(N);
    mvKeys1.reserve(N);
    mvKeys2.reserve(N);
    mvSigma2_1.reserve(N);
    mvSigma2_2.reserve(N);

    // 提取并转换两帧对应的局部相机系 3D 点与 2D 像素坐标
    for (int i1 = 0; i1 < N; ++i1)
    {
        MapPoint *pMP2 = vpMatched12[i1];
        if (!pMP2 || pMP2->isBad())
            continue;

        MapPoint *pMP1 = pKF1->GetMapPoint(i1);
        if (!pMP1 || pMP1->isBad())
            continue;

        int i2 = pMP2->GetIndexInKeyFrame(pKF2);
        if (i2 < 0)
            continue;

        // 相机系 3D 点
        Eigen::Vector3f P3D1c = R1w * pMP1->GetWorldPos() + t1w;
        Eigen::Vector3f P3D2c = R2w * pMP2->GetWorldPos() + t2w;

        if (P3D1c.z() <= 0.0f || P3D2c.z() <= 0.0f)
            continue;

        mvX3Dc1.push_back(P3D1c);
        mvX3Dc2.push_back(P3D2c);
        mvKeys1.push_back(pKF1->mvKeysUn[i1].pt);
        mvKeys2.push_back(pKF2->mvKeysUn[i2].pt);

        mvSigma2_1.push_back(pKF1->mvLevelSigma2[pKF1->mvKeysUn[i1].octave]);
        mvSigma2_2.push_back(pKF2->mvLevelSigma2[pKF2->mvKeysUn[i2].octave]);
    }

    mN1 = mvX3Dc1.size();
    mvAllIndices.resize(mN1);
    for (int i = 0; i < mN1; ++i)
        mvAllIndices[i] = i;

    SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mMinInliers = minInliers;
    mMaxIterations = maxIterations;
    mIterations = 0;
    mBestInliers = 0;

    if (mN1 < mMinInliers)
        return;

    // 根据期望置信度动态调整 RANSAC 迭代次数
    float epsilon = static_cast<float>(mMinInliers) / mN1;
    double denom = std::log(1.0 - std::pow(epsilon, 3));
    if (std::abs(denom) > 1e-6)
    {
        int nIterations = std::ceil(std::log(1.0 - probability) / denom);
        mMaxIterations = std::max(1, std::min(nIterations, maxIterations));
    }
}

// 核心闭式求解：严格的 Horn 绝对定向算法
void Sim3Solver::ComputeSim3(const std::vector<Eigen::Vector3f> &P1,
                             const std::vector<Eigen::Vector3f> &P2,
                             Eigen::Matrix3f &R12, Eigen::Vector3f &t12, float &s12)
{
    // 1. 计算质心
    Eigen::Vector3f O1 = (P1[0] + P1[1] + P1[2]) / 3.0f;
    Eigen::Vector3f O2 = (P2[0] + P2[1] + P2[2]) / 3.0f;

    // 2. 去质心坐标
    std::vector<Eigen::Vector3f> Pr1(3), Pr2(3);
    for (int i = 0; i < 3; ++i)
    {
        Pr1[i] = P1[i] - O1;
        Pr2[i] = P2[i] - O2;
    }

    // 3. 计算互协方差矩阵 M = \sum Pr2 * Pr1^T (与 ORB-SLAM2 官方标准定义一致)
    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    for (int i = 0; i < 3; ++i)
        M += Pr2[i] * Pr1[i].transpose();

    // 4. 构造标准 Horn 4x4 对称矩阵 N
    double N11 = M(0, 0) + M(1, 1) + M(2, 2);
    double N12 = M(1, 2) - M(2, 1);
    double N13 = M(2, 0) - M(0, 2);
    double N14 = M(0, 1) - M(1, 0);
    double N22 = M(0, 0) - M(1, 1) - M(2, 2);
    double N23 = M(0, 1) + M(1, 0);
    double N24 = M(2, 0) + M(0, 2);
    double N33 = -M(0, 0) + M(1, 1) - M(2, 2);
    double N34 = M(1, 2) + M(2, 1);
    double N44 = -M(0, 0) - M(1, 1) + M(2, 2);

    Eigen::Matrix4f N;
    N << N11, N12, N13, N14,
        N12, N22, N23, N24,
        N13, N23, N33, N34,
        N14, N24, N34, N44;

    // 特征值分解：最大特征值对应的特征向量即为最优旋转四元数
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> eigensolver(N);
    // col(3) 对应最大特征值
    Eigen::Vector4f q_vec = eigensolver.eigenvectors().col(3);

    // 标准四元数转换: Horn 矩阵中 col(3) 分布为 [w, x, y, z] -> [q_vec(0), q_vec(1), q_vec(2), q_vec(3)]
    Eigen::Quaternionf q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
    q.normalize();
    R12 = q.toRotationMatrix();

    // 5. 求解尺度 s (若固定尺度，则 s = 1.0)
    if (mbFixScale)
    {
        s12 = 1.0f;
    }
    else
    {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            Eigen::Vector3f P3 = R12 * Pr2[i];
            num += Pr1[i].dot(P3);
            den += P3.squaredNorm();
        }
        s12 = (den > 1e-8) ? static_cast<float>(num / den) : 1.0f;
    }

    // 6. 求解平移 t12 = O1 - s12 * R12 * O2
    t12 = O1 - s12 * R12 * O2;
}

// 双向重投影一致性检验
int Sim3Solver::CheckInliers(const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, const float s12,
                             std::vector<bool> &vbInliers)
{
    vbInliers.assign(mN1, false);
    int inliers = 0;

    const float fx1 = mpKF1->fx, fy1 = mpKF1->fy, cx1 = mpKF1->cx, cy1 = mpKF1->cy;
    const float fx2 = mpKF2->fx, fy2 = mpKF2->fy, cx2 = mpKF2->cx, cy2 = mpKF2->cy;

    // 计算逆变换 (相机 1 到相机 2)
    Eigen::Matrix3f R21 = R12.transpose();
    Eigen::Vector3f t21 = -R21 * t12 / s12;
    float s21 = 1.0f / s12;

    // 2 自由度重投影误差卡方检验阈值 (99% 显著性为 9.210，95% 显著性为 5.991)
    const float th = 9.210f;

    for (int i = 0; i < mN1; ++i)
    {
        // 1. 将相机 2 的点投影到相机 1
        Eigen::Vector3f P1 = s12 * R12 * mvX3Dc2[i] + t12;
        if (P1.z() <= 0.0f)
            continue;

        float invZ1 = 1.0f / P1.z();
        float u1 = fx1 * P1.x() * invZ1 + cx1;
        float v1 = fy1 * P1.y() * invZ1 + cy1;
        float dist1 = (u1 - mvKeys1[i].x) * (u1 - mvKeys1[i].x) + (v1 - mvKeys1[i].y) * (v1 - mvKeys1[i].y);

        if (dist1 > th * mvSigma2_1[i])
            continue;

        // 2. 将相机 1 的点反向投影到相机 2 (双向闭环强约束)
        Eigen::Vector3f P2 = s21 * R21 * mvX3Dc1[i] + t21;
        if (P2.z() <= 0.0f)
            continue;

        float invZ2 = 1.0f / P2.z();
        float u2 = fx2 * P2.x() * invZ2 + cx2;
        float v2 = fy2 * P2.y() * invZ2 + cy2;
        float dist2 = (u2 - mvKeys2[i].x) * (u2 - mvKeys2[i].x) + (v2 - mvKeys2[i].y) * (v2 - mvKeys2[i].y);

        if (dist2 > th * mvSigma2_2[i])
            continue;

        vbInliers[i] = true;
        inliers++;
    }

    return inliers;
}

bool Sim3Solver::iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    if (mN1 < mMinInliers)
    {
        bNoMore = true;
        return false;
    }

    std::vector<size_t> vAvailableIndices = mvAllIndices;

    int nit = 0;
    while (mIterations < mMaxIterations && nit < nIterations)
    {
        mIterations++;
        nit++;

        // 随机抽取 3 对无重复点
        std::vector<Eigen::Vector3f> P1(3), P2(3);
        std::vector<size_t> vCurrentIndices = vAvailableIndices;

        for (int i = 0; i < 3; ++i)
        {
            std::uniform_int_distribution<int> dist(0, vCurrentIndices.size() - 1);
            int randi = dist(mRng);
            size_t idx = vCurrentIndices[randi];

            P1[i] = mvX3Dc1[idx];
            P2[i] = mvX3Dc2[idx];

            vCurrentIndices[randi] = vCurrentIndices.back();
            vCurrentIndices.pop_back();
        }

        // 闭式解求出变换候选
        Eigen::Matrix3f R12;
        Eigen::Vector3f t12;
        float s12;
        ComputeSim3(P1, P2, R12, t12, s12);

        // 统计内点
        std::vector<bool> vbCurrentInliers;
        int inliers = CheckInliers(R12, t12, s12, vbCurrentInliers);

        if (inliers > mBestInliers)
        {
            mBestInliers = inliers;
            mBestRotation = R12;
            mBestTranslation = t12;
            mBestScale = s12;
            mvbBestInliers = vbCurrentInliers;

            if (inliers >= mMinInliers)
            {
                nInliers = mBestInliers;
                vbInliers = mvbBestInliers;
                return true; // 达到内点门槛，提前成功返回
            }
        }
    }

    if (mIterations >= mMaxIterations)
        bNoMore = true;

    return false;
}

Eigen::Matrix4f Sim3Solver::GetEstimatedTransformation() const
{
    Eigen::Matrix4f T12 = Eigen::Matrix4f::Identity();
    T12.block<3, 3>(0, 0) = mBestScale * mBestRotation;
    T12.block<3, 1>(0, 3) = mBestTranslation;
    return T12;
}