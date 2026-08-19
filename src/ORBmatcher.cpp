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
    const Eigen::Matrix3f Rlw = LastFrame.mTcw.block<3, 3>(0, 0);

    // =========================================================================
    // 1. 计算运动/角速度自适应缩放因子 (Motion & Rotation Adaptiveness)
    // =========================================================================
    // 计算上一帧到当前帧的相对旋转 R_c_last = R_cw * R_lw^T
    Eigen::Matrix3f R_c_last = Rcw * Rlw.transpose();
    
    // 计算相对旋转角度 delta_theta (弧度)
    double cos_angle = 0.5 * (R_c_last.trace() - 1.0);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    double delta_theta_rad = std::acos(cos_angle);
    double delta_theta_deg = delta_theta_rad * 180.0 / M_PI; // 角度制

    // 基础运动放大因子：
    // - 直线时 (delta_theta 近似 0)，motion_factor ≈ 1.0
    // - 转弯时 (如旋转 3°~10°)，搜索半径平滑线性放大 1.2x ~ 2.5x，并设置上限 3.0x
    float motion_factor = 1.0f + static_cast<float>(delta_theta_deg) * 0.15f;
    motion_factor = std::min(std::max(motion_factor, 1.0f), 3.0f);

    const float adaptive_th = th * motion_factor;

    // =========================================================================
    // 2. 遍历上一帧地图点，利用自适应半径进行投影搜索
    // =========================================================================
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
        // 结合【运动自适应】与【金字塔尺度自适应】
        const float radius = adaptive_th * CurrentFrame.mpORBextractorLeft->GetScaleFactors()[lastLevel];

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

        // Ratio Test 过滤
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

    // 旋转一致性直方图剔除杂点
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

int ORBmatcher::SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches = 0;

    const Eigen::Matrix3f Rcw = F.mTcw.block<3, 3>(0, 0);
    const Eigen::Vector3f tcw = F.mTcw.block<3, 1>(0, 3);
    const Eigen::Vector3f Ow  = -Rcw.transpose() * tcw;

    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); ++iMP)
    {
        MapPoint *pMP = vpMapPoints[iMP];
        if (!pMP || pMP->isBad())
            continue;

        // 1. 3D 点投影到相机坐标系
        const Eigen::Vector3f Pw = pMP->GetWorldPos();
        const Eigen::Vector3f Pc = Rcw * Pw + tcw;

        // 必须在相机前方
        if (Pc.z() <= 0.0f)
            continue;

        // 2. 视锥与距离检查 (Frustum & Distance Check)
        const float dist = (Pw - Ow).norm();
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        if (dist < minDistance * 0.8f || dist > maxDistance * 1.2f)
            continue;

        // 视角夹角检查 (Viewing angle > 60° 剔除)
        Eigen::Vector3f Pn = (Pw - Ow).normalized();
        if (Pn.dot(pMP->GetNormal()) < 0.5f)
            continue;

        // 3. 计算图像像素坐标
        const float invz = 1.0f / Pc.z();
        const float u = Frame::fx * Pc.x() * invz + Frame::cx;
        const float v = Frame::fy * Pc.y() * invz + Frame::cy;

        if (u < Frame::mnMinX || u >= Frame::mnMaxX ||
            v < Frame::mnMinY || v >= Frame::mnMaxY)
            continue;

        // 4. 根据当前距离估算金字塔层级与搜索半径
        // 距离越远，层级越高，搜索半径越大
        float ratio = dist / maxDistance;
        int predictedLevel = 0;
        if (ratio < 0.25f) predictedLevel = 0;
        else if (ratio < 0.5f) predictedLevel = 1;
        else if (ratio < 0.75f) predictedLevel = 2;
        else predictedLevel = 3;

        const float radius = th * F.mpORBextractorLeft->GetScaleFactors()[predictedLevel];
        const std::vector<size_t> candidates =
            F.GetFeaturesInArea(u, v, radius, predictedLevel - 1, predictedLevel + 1);

        if (candidates.empty())
            continue;

        const cv::Mat &dMP = pMP->GetDescriptor();
        int bestDist = TH_LOW;
        int secondBestDist = TH_LOW;
        int bestIdx = -1;

        for (size_t c = 0; c < candidates.size(); ++c)
        {
            const size_t idx = candidates[c];
            // 若当前特征点已经被匹配了，跳过
            if (F.mvpMapPoints[idx])
                continue;

            const cv::Mat &dF = F.mDescriptors.row(idx);
            const int distDesc = DescriptorDistance(dMP, dF);

            if (distDesc < bestDist)
            {
                secondBestDist = bestDist;
                bestDist = distDesc;
                bestIdx = static_cast<int>(idx);
            }
            else if (distDesc < secondBestDist)
            {
                secondBestDist = distDesc;
            }
        }

        if (bestIdx >= 0 && bestDist < TH_LOW)
        {
            if (static_cast<float>(bestDist) < mfNNratio * static_cast<float>(secondBestDist))
            {
                F.mvpMapPoints[bestIdx] = pMP;
                ++nmatches;

                if (mbCheckOrientation)
                {
                    // 统计方向直方图
                    float rot = F.mvKeysUn[bestIdx].angle;
                    int bin = cvRound(rot * rotFactor);
                    if (bin == HISTO_LENGTH) bin = 0;
                    rotHist[bin].push_back(bestIdx);
                    histo[bin]++;
                }
            }
        }
    }

    // 旋转一致性直方图剔除
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
                if (F.mvpMapPoints[idx])
                {
                    F.mvpMapPoints[idx] = nullptr;
                    --nmatches;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12)
{
    const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const std::vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = std::vector<MapPoint*>(pKF1->N, static_cast<MapPoint*>(nullptr));

    pKF1->ComputeBoW();
    pKF2->ComputeBoW();

    int nmatches = 0;
    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    auto f1it = vFeatVec1.begin();
    auto f2it = vFeatVec2.begin();
    auto f1end = vFeatVec1.end();
    auto f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            const std::vector<unsigned int> &vIndices1 = f1it->second;
            const std::vector<unsigned int> &vIndices2 = f2it->second;

            for (size_t i1 = 0; i1 < vIndices1.size(); i1++)
            {
                const unsigned int realIdx1 = vIndices1[i1];
                MapPoint *pMP1 = vpMapPoints1[realIdx1];
                if (!pMP1 || pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(realIdx1);

                int bestDist = TH_LOW;
                int secondBestDist = TH_LOW;
                int bestIdx2 = -1;

                for (size_t i2 = 0; i2 < vIndices2.size(); i2++)
                {
                    const unsigned int realIdx2 = vIndices2[i2];
                    MapPoint *pMP2 = vpMapPoints2[realIdx2];
                    if (!pMP2 || pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(realIdx2);
                    int dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist)
                    {
                        secondBestDist = bestDist;
                        bestDist = dist;
                        bestIdx2 = realIdx2;
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
                        vpMatches12[realIdx1] = vpMapPoints2[bestIdx2];

                        if (mbCheckOrientation)
                        {
                            float rot = pKF1->mvKeysUn[realIdx1].angle - pKF2->mvKeysUn[bestIdx2].angle;
                            if (rot < 0.0f) rot += 360.0f;
                            int bin = cvRound(rot * rotFactor);
                            if (bin == HISTO_LENGTH) bin = 0;
                            rotHist[bin].push_back(realIdx1);
                            histo[bin]++;
                        }
                        nmatches++;
                    }
                }
            }
            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it++;
        }
        else
        {
            f2it++;
        }
    }

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
                if (vpMatches12[idx])
                {
                    vpMatches12[idx] = nullptr;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}