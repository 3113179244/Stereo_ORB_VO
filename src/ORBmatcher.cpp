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

    F.mvuRight = std::vector<float>(F.N, -1.0f);
    F.mvDepth = std::vector<float>(F.N, -1.0f);

    const int nRows = F.mImGrayLeft.rows;

    // 1. 按行建立右目特征点索引
    std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());
    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(10);

    const size_t Nr = F.mvKeysRight.size();
    for (size_t iR = 0; iR < Nr; iR++)
    {
        const cv::KeyPoint &kp = F.mvKeysRight[iR];
        const float kpY = kp.pt.y;
        const float r = 2.0f * F.mpORBextractorRight->GetScaleFactors()[kp.octave];

        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        for (int yi = minr; yi <= maxr; yi++)
        {
            if (yi >= 0 && yi < nRows)
                vRowIndices[yi].push_back(iR);
        }
    }

    const float minZ = F.mb;
    const float minD = 0.0f;
    const float maxD = F.mbf / minZ;

    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2; // 默认匹配阈值 75

    // 旋转直方图，用于过滤误匹配
    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    std::vector<std::pair<int, int>> vDistIdx(F.N, std::make_pair(INT_MAX, -1));

    // 2. 粗匹配：极线范围内的描述子汉明距离比对
    for (int iL = 0; iL < F.N; iL++)
    {
        const cv::KeyPoint &kpL = F.mvKeys[iL];
        const int levelL = kpL.octave;
        const float vL = kpL.pt.y;
        const float uL = kpL.pt.x;

        const std::vector<size_t> &vCandidates = vRowIndices[cvRound(vL)];
        if (vCandidates.empty())
            continue;

        const float minU = uL - maxD;
        const float maxU = uL - minD;
        if (maxU < 0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = F.mDescriptors.row(iL);

        for (size_t iC = 0; iC < vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = F.mvKeysRight[iR];

            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;

            const float uR = kpR.pt.x;
            if (uR >= minU && uR <= maxU)
            {
                const cv::Mat &dR = F.mDescriptorsRight.row(iR);
                const int dist = DescriptorDistance(dL, dR);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // 3. 精细匹配：SAD 滑动窗口 + 亚像素抛物线拟合
        if (bestDist < thOrbDist)
        {
            const cv::KeyPoint &kpR = F.mvKeysRight[bestIdxR];
            const float uR0 = kpR.pt.x;

            const int w = 5;
            // 边缘保护，防止滑窗越界
            if (uL - w < 0 || uL + w >= F.mImGrayLeft.cols ||
                vL - w < 0 || vL + w >= F.mImGrayLeft.rows ||
                uR0 - w - 2 < 0 || uR0 + w + 2 >= F.mImGrayRight.cols)
                continue;

            int bestL = 0;
            int distSubPixelMin = INT_MAX;
            int vIdxToCost[5] = {0};

            // 在 uR0 附近 [-2, 2] 像素范围内计算 11x11 窗口的 SAD
            for (int incR = -2; incR <= 2; incR++)
            {
                int distSubPixel = 0;
                for (int wy = -w; wy <= w; wy++)
                {
                    const uchar *pL = F.mImGrayLeft.ptr<uchar>(cvRound(vL) + wy);
                    const uchar *pR = F.mImGrayRight.ptr<uchar>(cvRound(vL) + wy);

                    for (int wx = -w; wx <= w; wx++)
                    {
                        distSubPixel += std::abs(pL[cvRound(uL) + wx] - pR[cvRound(uR0) + incR + wx]);
                    }
                }

                vIdxToCost[incR + 2] = distSubPixel;

                if (distSubPixel < distSubPixelMin)
                {
                    distSubPixelMin = distSubPixel;
                    bestL = incR;
                }
            }

            // 极小值不能落在边界点（必须在 -1, 0, 1 内才能拟合极值点）
            if (bestL == -2 || bestL == 2)
                continue;

            // 抛物线插值求极小值点偏移 delta
            const float dist1 = vIdxToCost[bestL + 1]; // 左侧点
            const float dist2 = vIdxToCost[bestL + 2]; // 极小值点
            const float dist3 = vIdxToCost[bestL + 3]; // 右侧点

            const float delta = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

            if (delta < -1.0f || delta > 1.0f)
                continue;

            // 亚像素右图横坐标与视差
            const float bestuR = uR0 + bestL + delta;
            float disparity = uL - bestuR;

            if (disparity >= minD && disparity < maxD)
            {
                if (disparity <= 0)
                    disparity = 0.01f;

                F.mvDepth[iL] = F.mbf / disparity;
                F.mvuRight[iL] = bestuR;
                vDistIdx[iL] = std::make_pair(bestDist, bestIdxR);
                nMatches++;

                // 记录方向差直方图
                if (mbCheckOrientation)
                {
                    float rot = kpL.angle - kpR.angle;
                    if (rot < 0.0f) rot += 360.0f;
                    int bin = cvRound(rot * rotFactor);
                    if (bin == HISTO_LENGTH) bin = 0;
                    rotHist[bin].push_back(iL);
                    histo[bin]++;
                }
            }
        }
    }

    // 4. 旋转一致性直方图剔除杂乱方向的误匹配
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
                const int idx = rotHist[i][j];
                F.mvDepth[idx] = -1.0f;
                F.mvuRight[idx] = -1.0f;
                nMatches--;
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