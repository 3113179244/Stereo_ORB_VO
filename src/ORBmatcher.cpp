#include "ORBmatcher.h"
#include "Frame.h"
#include <limits>
#include "ORBextractor.h"
#include <algorithm>
#include "KeyFrame.h"
#include "MapPoint.h"
// 初始化静态成员变量
const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 36;

ORBmatcher::ORBmatcher(float nnratio, bool checkOrientation)
    : mfNNratio(nnratio), mbCheckOrientation(checkOrientation) {}

/**
 * @brief 计算两个 ORB 描述子的汉明距离
 */
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    // ORB 描述子是 32 字节 (32 * 8 = 256 bits)。
    // 为了加速，将其强制转换为 8 个 32位整型 (int32_t) 进行处理 (8 * 32 = 256 bits)。
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    // 循环 8 次即可处理完 256 bit 的描述子
    for (int i = 0; i < 8; i++)
    {
        // 异或操作：对应的二进制位相同为0，不同为1
        unsigned int int_or = pa[i] ^ pb[i];
        // __builtin_popcount 是 GCC 内置函数，通过硬件指令(如 SSE4.2 的 POPCNT)快速统计整型数中 1 的个数
        dist += __builtin_popcount(int_or);
    }
    return dist;
}

/**
 * @brief 双目匹配，计算左目特征点对应的右目特征点，并恢复深度
 */
int ORBmatcher::ComputeStereoMatches(Frame &F)
{
    int nMatches = 0;

    // 初始化左右目匹配关系和深度信息为 -1
    // mvuRight 记录左目特征点匹配到的右目特征点的横坐标 (u)
    F.mvuRight = std::vector<float>(F.N, -1.0f);
    // mvDepth 记录通过视差计算得出的左目特征点的深度 (Z)
    F.mvDepth = std::vector<float>(F.N, -1.0f);

    // 获取图像的行数（Y坐标范围）。mbInitialComputations 可能表示是否是第一帧初始化。
    const int nRows = F.mbInitialComputations ? 480 : F.mnMaxY;

    // vRowIndices[i] 存储在图像第 i 行的所有右目特征点的索引 (ID)
    // 这种按行索引的数据结构可以极大加速双目极线搜索（只需在对应的行中寻找匹配点）
    std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());

    // 预分配内存，假设每行平均有 10 个特征点
    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(10);

    const int Nr = F.mvKeysRight.size(); // 右目图像提取到的特征点总数

    // 遍历所有右目特征点，将它们分配到对应的行索引中
    for (int iR = 0; iR < Nr; iR++)
    {
        const cv::KeyPoint &kp = F.mvKeysRight[iR];
        const float kpY = kp.pt.y; // 右目特征点的纵坐标

        // 考虑特征点所在的金字塔层级 (octave) 带来的尺度影响。
        // 层数越高，特征点对应的实际图像区域越大，搜索范围 (半径 r) 也应相应扩大。
        const float r = 2.0f * F.mpORBextractorRight->GetScaleFactors()[kp.octave];

        // 定义垂直方向的搜索范围 [minr, maxr]，允许存在一定的极线误差
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        // 将该右目特征点的索引 iR 放入对应行的集合中
        for (int yi = minr; yi <= maxr; yi++)
        {
            if (yi >= 0 && yi < nRows)
                vRowIndices[yi].push_back(iR);
        }
    }

    // 深度与视差的物理约束
    const float minZ = F.mb;         // 最小深度等于双目基线 (通常认为小于基线的物体无法稳定观测)
    const float minD = 0;            // 最小视差为 0 (对应无穷远)
    const float maxD = F.mbf / minZ; // 最大视差对应最小深度。 视差 d = bf / Z

    // 存储距离和索引的临时容器（此代码片段中未实际发挥大作用，可能由于原版截断）
    std::vector<std::pair<int, int>> vDistIdx;
    vDistIdx.reserve(F.N);

    // 用于旋转一致性检查的直方图（在此片段末尾未使用到该直方图的具体筛选逻辑，但预留了变量）
    std::vector<int> rotHistogram[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHistogram[i].reserve(F.N);

    const float factor = 1.0f / HISTO_LENGTH;

    // 遍历左目图像的所有特征点，在右目图像中寻找匹配
    for (int iL = 0; iL < F.N; iL++)
    {
        const cv::KeyPoint &kpL = F.mvKeys[iL];
        const int levelL = kpL.octave; // 左目特征点所在的金字塔层级
        const float vL = kpL.pt.y;     // 左目特征点纵坐标 (v)
        const float uL = kpL.pt.x;     // 左目特征点横坐标 (u)

        // 取出左目特征点同一行（或极线误差范围内）的所有右目候选特征点
        const std::vector<size_t> &vCandidates = vRowIndices[(int)vL];
        if (vCandidates.empty())
            continue; // 如果该行没有右目特征点，则直接跳过

        // 根据视差约束 [minD, maxD] 计算右目特征点可能存在的横坐标范围 [minU, maxU]
        // uL - uR = d  =>  uR = uL - d
        const float minU = uL - maxD;
        const float maxU = uL - minD;
        if (maxU < 0)
            continue; // 如果有效横坐标范围在图像外，则跳过

        int bestDist = TH_HIGH; // 初始化最佳匹配距离为高阈值
        size_t bestIdxR = 0;    // 最佳匹配的右目特征点索引

        // 获取左目特征点的描述子
        const cv::Mat &dL = F.mDescriptors.row(iL);

        // 遍历行内的所有右目候选点
        for (size_t iC = 0; iC < vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = F.mvKeysRight[iR];

            // 尺度一致性检查：左右目匹配的点，其所在的金字塔层级差异不能大于 1
            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;

            const float uR = kpR.pt.x;
            // 极线约束（水平方向）：右目特征点必须在理论的视差区间内
            if (uR >= minU && uR <= maxU)
            {
                // 获取右目特征点的描述子
                const cv::Mat &dR = F.mDescriptorsRight.row(iR);
                // 计算汉明距离
                const int dist = DescriptorDistance(dL, dR);

                // 寻找距离最小（最匹配）的点
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // 如果找到了满足阈值的匹配点
        if (bestDist < TH_HIGH)
        {
            const cv::KeyPoint &kpR = F.mvKeysRight[bestIdxR];
            const float uR = kpR.pt.x;
            const float disparity = uL - uR; // 计算视差

            // 视差必须大于等于0
            if (disparity >= 0)
            {
                // 根据视差计算深度: Z = (fb) / d
                // 注意：由于未做亚像素插值，这里的深度存在一定的量子化误差
                const float depth = F.mbf / disparity;

                // 记录匹配结果
                F.mvuRight[iL] = uR;
                F.mvDepth[iL] = depth;
                nMatches++; // 匹配成功数加一
            }
        }
    }

    return nMatches;
}

/**
 * @brief 找出直方图中票数（匹配点对数）前三名的索引
 * 该函数通常配合方向直方图使用。因为错误的匹配通常具有随机的相对旋转角度，
 * 而正确的匹配点对通常具有高度一致的相对旋转角度。
 * 保留前三大主方向的匹配点对，可以有效剔除误匹配。
 */
void ORBmatcher::ComputeThreeBestIdx(int *histo, const int L, int &idx1, int &idx2, int &idx3)
{
    int max1 = 0, max2 = 0, max3 = 0;

    // 遍历直方图的每一个 Bin，维护前三大的值(max)及其对应的索引(idx)
    for (int i = 0; i < L; i++)
    {
        const int n = histo[i]; // 当前 Bin 的票数
        if (n > max1)
        {
            // 如果比第一名大，则原第一名变第二名，原第二名变第三名
            max3 = max2;
            max2 = max1;
            max1 = n;
            idx3 = idx2;
            idx2 = idx1;
            idx1 = i;
        }
        else if (n > max2)
        {
            // 如果比第二名大，则原第二名变第三名
            max3 = max2;
            max2 = n;
            idx3 = idx2;
            idx2 = i;
        }
        else if (n > max3)
        {
            // 如果仅比第三名大，则直接更新第三名
            max3 = n;
            idx3 = i;
        }
    }

    // 阈值筛选：如果第二/三名包含的匹配对数量不到第一名的十分之一，
    // 则说明它们并不是一个明显的主方向（可能是随机噪声），将其废弃（设为-1）。
    if (max2 < 0.1f * max1)
    {
        idx2 = -1;
        idx3 = -1;
    }
    else if (max3 < 0.1f * max1)
    {
        idx3 = -1;
    }
}

int ORBmatcher::SearchByProjection(
    Frame &CurrentFrame, const Frame &LastFrame,
    const float th, const bool bMono)
{
    int nmatches = 0;

    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    const Eigen::Matrix3f Rcw = CurrentFrame.mTcw.block<3, 3>(0, 0);
    const Eigen::Vector3f tcw = CurrentFrame.mTcw.block<3, 1>(0, 3);

    for (int i = 0; i < LastFrame.N; ++i)
    {
        MapPoint *pMP = LastFrame.mvpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        const Eigen::Vector3f Pc = Rcw * pMP->GetWorldPos() + tcw;
        if (Pc.z() <= 0.0f)
            continue;

        const float invz = 1.0f / Pc.z();
        const float u = Frame::fx * Pc.x() * invz + Frame::cx;
        const float v = Frame::fy * Pc.y() * invz + Frame::cy;

        if (u < Frame::mnMinX || u >= Frame::mnMaxX ||
            v < Frame::mnMinY || v >= Frame::mnMaxY)
            continue;

        const int lastLevel = LastFrame.mvKeysUn[i].octave;
        const float radius = th * CurrentFrame.mpORBextractorLeft->GetScaleFactors()[lastLevel];

        const std::vector<size_t> candidates =
            CurrentFrame.GetFeaturesInArea(u, v, radius, lastLevel - 1, lastLevel + 1);

        if (candidates.empty())
            continue;

        const cv::Mat &lastDesc = LastFrame.mDescriptors.row(i);
        int bestDist = TH_HIGH;
        int secondBestDist = TH_HIGH;
        int bestIdx = -1;
        int bestLevel = -1;
        int secondBestLevel = -1;

        const float predictedUR = u - CurrentFrame.mbf * invz;

        for (size_t idx : candidates)
        {
            if (CurrentFrame.mvpMapPoints[idx])
                continue;

            if (!bMono)
            {
                const float ur = CurrentFrame.mvuRight[idx];
                if (ur >= 0.0f && std::fabs(ur - predictedUR) > radius)
                    continue;
            }

            const int dist = DescriptorDistance(
                lastDesc, CurrentFrame.mDescriptors.row(idx));

            if (dist < bestDist)
            {
                secondBestDist = bestDist;
                secondBestLevel = bestLevel;
                bestDist = dist;
                bestLevel = CurrentFrame.mvKeysUn[idx].octave;
                bestIdx = static_cast<int>(idx);
            }
            else if (dist < secondBestDist)
            {
                secondBestDist = dist;
                secondBestLevel = CurrentFrame.mvKeysUn[idx].octave;
            }
        }

        if (bestIdx < 0 || bestDist > TH_HIGH)
            continue;

        if (secondBestLevel >= 0 &&
            bestLevel == secondBestLevel &&
            static_cast<float>(bestDist) > mfNNratio * secondBestDist)
            continue;

        CurrentFrame.mvpMapPoints[bestIdx] = pMP;
        ++nmatches;

        if (mbCheckOrientation)
        {
            float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx].angle;
            if (rot < 0.0f)
                rot += 360.0f;

            int bin = cvRound(rot * rotFactor);
            if (bin == HISTO_LENGTH)
                bin = 0;

            rotHist[bin].push_back(bestIdx);
            ++histo[bin];
        }
    }

    if (mbCheckOrientation)
    {
        int idx1 = -1, idx2 = -1, idx3 = -1;
        ComputeThreeBestIdx(histo, HISTO_LENGTH, idx1, idx2, idx3);

        for (int bin = 0; bin < HISTO_LENGTH; ++bin)
        {
            if (bin == idx1 || bin == idx2 || bin == idx3)
                continue;

            for (int idx : rotHist[bin])
            {
                if (CurrentFrame.mvpMapPoints[idx])
                {
                    CurrentFrame.mvpMapPoints[idx] = nullptr;
                    --nmatches;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches)
{
    vpMapPointMatches = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(nullptr));

    const std::vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();
    const cv::Mat &DescriptorsKF = pKF->mDescriptors;

    pKF->ComputeBoW();
    F.ComputeBoW();

    int nmatches = 0;
    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;
    const DBoW3::FeatureVector &vFeatVecF = F.mFeatVec;

    auto KFit = vFeatVecKF.begin();
    auto Fit = vFeatVecF.begin();
    auto KFend = vFeatVecKF.end();
    auto Fend = vFeatVecF.end();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const std::vector<unsigned int> &vIndicesKF = KFit->second;
            const std::vector<unsigned int> &vIndicesF = Fit->second;

            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];
                MapPoint *pMP = vpMapPointsKF[realIdxKF];

                if (!pMP || pMP->isBad())
                    continue;

                const cv::Mat &dKF = DescriptorsKF.row(realIdxKF);

                int bestDist = TH_LOW;
                int secondBestDist = TH_LOW;
                int bestIdxF = -1;

                for (size_t iF = 0; iF < vIndicesF.size(); iF++)
                {
                    const unsigned int realIdxF = vIndicesF[iF];

                    if (vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);
                    const int dist = DescriptorDistance(dKF, dF);

                    if (dist < bestDist)
                    {
                        secondBestDist = bestDist;
                        bestDist = dist;
                        bestIdxF = realIdxF;
                    }
                    else if (dist < secondBestDist)
                    {
                        secondBestDist = dist;
                    }
                }

                if (bestDist < TH_LOW)
                {
                    if (static_cast<float>(bestDist) < mfNNratio * static_cast<float>(secondBestDist))
                    {
                        vpMapPointMatches[bestIdxF] = pMP;

                        if (mbCheckOrientation)
                        {
                            float rot = pKF->mvKeysUn[realIdxKF].angle - F.mvKeysUn[bestIdxF].angle;
                            if (rot < 0.0f) rot += 360.0f;
                            int bin = cvRound(rot * rotFactor);
                            if (bin == HISTO_LENGTH) bin = 0;

                            rotHist[bin].push_back(bestIdxF);
                            histo[bin]++;
                        }
                        nmatches++;
                    }
                }
            }
            KFit++;
            Fit++;
        }
        else if (KFit->first < Fit->first)
        {
            KFit++;
        }
        else
        {
            Fit++;  
        }
    }

    // 剔除非主方向误匹配
    if (mbCheckOrientation)
    {
        int idx1 = -1, idx2 = -1, idx3 = -1;
        ComputeThreeBestIdx(histo, HISTO_LENGTH, idx1, idx2, idx3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == idx1 || i == idx2 || i == idx3)
                continue;

            for (size_t j = 0; j < rotHist[i].size(); j++)
            {
                int idx = rotHist[i][j];
                if (vpMapPointMatches[idx])
                {
                    vpMapPointMatches[idx] = nullptr;
                    nmatches--;
                }
            }
        }
    }

    for (int i = 0; i < F.N; i++)
    {
        if (vpMapPointMatches[i])
            F.mvpMapPoints[i] = vpMapPointMatches[i];
    }

    return nmatches;
}