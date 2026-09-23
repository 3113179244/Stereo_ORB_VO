#include "ORBmatcher.h"
#include "Frame.h"
#include <limits>
#include "ORBextractor.h"
#include <algorithm>
#include "KeyFrame.h"
#include "MapPoint.h"

// 描述子汉明距离判定门限：
// TH_HIGH(100): 较宽松门限，适用于连续帧运动跟踪（相邻帧图像畸变和视差小）
// TH_LOW(50):   较严格门限，适用于回环检测、重定位以及双关键帧三角化
const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
// 旋转一致性直方图分割槽数：360度划分为 36 个 Bin，每个 Bin 跨越 10 度
const int ORBmatcher::HISTO_LENGTH = 36;

/**
 * @brief 构造函数
 * @param nnratio 最优与次优距离的比例阈值（Lowe's ratio test），通常在 0.6 ~ 0.8
 * @param checkOrientation 是否开启方向一致性过滤检查
 */
ORBmatcher::ORBmatcher(float nnratio, bool checkOrientation)
    : mfNNratio(nnratio), mbCheckOrientation(checkOrientation) {}

/**
 * @brief 计算两个 256-bit ORB 描述子之间的汉明距离 (Hamming Distance)
 * @details 采用 CPU 原生 POPCNT 指令并行处理，提升二进制位运算效率
 */
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    // ORB 描述子是 32 字节 (32 * 8 = 256 bits)。
    // 强制转换为 8 个 32位整型 (uint32_t/int32_t) 进行批处理 (8 * 32 = 256 bits)。
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    // 展开为 8 次迭代处理完 256 bit 的二进制描述子
    for (int i = 0; i < 8; i++)
    {
        // 异或运算：比特位相同输出 0，不同输出 1
        unsigned int int_or = pa[i] ^ pb[i];
        // __builtin_popcount GCC 内置指令：通过硬件指令 (如 x86 SSE4.2 的 POPCNT) 单周期统计二进制 1 的个数
        dist += __builtin_popcount(int_or);
    }
    return dist;
}

/**
 * @brief 统计直方图中票数（匹配对数量）排名前三的主峰 Bin 索引
 * @details 正确的特征匹配在物理相机旋转下具有全局一致的旋转差；而随机误匹配的角度差呈均匀分布。
 *          保留主旋转角度对应的特征匹配，剔除离群散点。
 * @param histo 直方图数组
 * @param L 直方图长度 (36)
 * @param[out] idx1 票数第一的主峰 Bin 索引
 * @param[out] idx2 票数第二的主峰 Bin 索引
 * @param[out] idx3 票数第三的主峰 Bin 索引
 */
void ORBmatcher::ComputeThreeBestIdx(int *histo, const int L, int &idx1, int &idx2, int &idx3)
{
    int max1 = 0, max2 = 0, max3 = 0;

    // 遍历所有角度 Bin，维护前三大值及对应索引
    for (int i = 0; i < L; i++)
    {
        const int n = histo[i];
        if (n > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = n;
            idx3 = idx2;
            idx2 = idx1;
            idx1 = i;
        }
        else if (n > max2)
        {
            max3 = max2;
            max2 = n;
            idx3 = idx2;
            idx2 = i;
        }
        else if (n > max3)
        {
            max3 = n;
            idx3 = i;
        }
    }

    // 显著性门限筛选：若次峰/第三峰的票数达不到主峰的 10%，视作环境噪点并废弃（置 -1）
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

/**
 * @brief 跟踪上一帧：根据上一帧已关联的 MapPoints 投影到当前帧搜索匹配点
 * @details 包含角速度自适应搜索半径推导、金字塔尺度缩放及双目视差几何校验
 * @param CurrentFrame 当前帧（待填充匹配）
 * @param LastFrame 上一帧
 * @param th 基础搜索半径像素阈值
 * @param bMono 是否为单目模式（双目模式会增加右目视差校验）
 * @return int 成功建立匹配的特征对数
 */
int ORBmatcher::SearchByProjection(
    Frame &CurrentFrame, const Frame &LastFrame,
    const float th, const bool bMono)
{
    int nmatches = 0;

    // 旋转直方图初始化：每个 bin 存储属于该角度差的当前帧特征点索引
    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    // 获取当前帧和上一帧的世界到相机坐标系旋转与平移
    const Eigen::Matrix3f Rcw = CurrentFrame.mTcw.block<3, 3>(0, 0);
    const Eigen::Vector3f tcw = CurrentFrame.mTcw.block<3, 1>(0, 3);
    const Eigen::Matrix3f Rlw = LastFrame.mTcw.block<3, 3>(0, 0);

    // 1. 运动角速度自适应调节 (Motion & Rotation Adaptiveness)
    // 计算两帧间的相对旋转矩阵: R_c_last = R_cw * R_lw^T
    Eigen::Matrix3f R_c_last = Rcw * Rlw.transpose();

    // 利用李代数/旋转矩阵迹公式计算相对旋转角: Tr(R) = 1 + 2*cos(theta)
    double cos_angle = 0.5 * (R_c_last.trace() - 1.0);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle)); // 数值截断防止 acos 越界返回 NaN
    double delta_theta_rad = std::acos(cos_angle);
    double delta_theta_deg = delta_theta_rad * 180.0 / M_PI;

    // 旋转动态补偿：当相机快速转弯时，匀速运动模型预测往往偏离实际位置，动态扩大搜索圆域
    // 动态倍率范围钳制在 [1.0, 3.0] 倍之间
    float motion_factor = 1.0f + static_cast<float>(delta_theta_deg) * 0.15f;
    motion_factor = std::min(std::max(motion_factor, 1.0f), 3.0f);

    const float adaptive_th = th * motion_factor;

    // 2. 遍历上一帧中的有效地图点进行投影搜索
    for (int i = 0; i < LastFrame.N; ++i)
    {
        MapPoint *pMP = LastFrame.mvpMapPoints[i];
        if (!pMP || pMP->isBad())
            continue;

        // 将地图点从世界坐标系投影至当前相机坐标系: Pc = Rcw * Pw + tcw
        const Eigen::Vector3f Pc = Rcw * pMP->GetWorldPos() + tcw;
        if (Pc.z() <= 0.0f) // 深度必须在相机前方
            continue;

        // 透视除法，映射至当前帧像平面坐标 (u, v)
        const float invz = 1.0f / Pc.z();
        const float u = Frame::fx * Pc.x() * invz + Frame::cx;
        const float v = Frame::fy * Pc.y() * invz + Frame::cy;

        // 边界保护：落在有效图像边界外则跳过
        if (u < Frame::mnMinX || u >= Frame::mnMaxX ||
            v < Frame::mnMinY || v >= Frame::mnMaxY)
            continue;

        const int lastLevel = LastFrame.mvKeysUn[i].octave;
        // 搜索半径同时受【角速度动态因子】与【图像金字塔缩放因子】加权调节
        const float radius = adaptive_th * CurrentFrame.mpORBextractorLeft->GetScaleFactors()[lastLevel];

        // 基于网格划分快速搜索目标区域相邻金字塔层级的候选特征点 [lastLevel-1, lastLevel+1]
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

        // 双目预测基准视差 (ur): u - (fx * b) / Z
        const float predictedUR = u - CurrentFrame.mbf * invz;

        // 遍历所有圆域内的候选特征点，寻找最小汉明距离匹配
        for (size_t idx : candidates)
        {
            // 如果该特征点已经关联了其他地图点，避免竞争重复分配
            if (CurrentFrame.mvpMapPoints[idx])
                continue;

            // 双目几何一致性检验：右目视差水平坐标容差约束
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

        // 汉明距离绝对阈值判定
        if (bestIdx < 0 || bestDist > TH_HIGH)
            continue;

        // 严格 Lowe's Ratio Test：若处于同一尺度金字塔层级，最优距离需显著优于次优距离
        if (secondBestLevel >= 0 &&
            bestLevel == secondBestLevel &&
            static_cast<float>(bestDist) > mfNNratio * secondBestDist)
            continue;

        // 匹配成功：为当前帧绑定地图点
        CurrentFrame.mvpMapPoints[bestIdx] = pMP;
        ++nmatches;

        // 统计特征点旋转差，注入直方图槽位
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

    // 3. 方向一致性直方图剔除离群误匹配
    if (mbCheckOrientation)
    {
        int idx1 = -1, idx2 = -1, idx3 = -1;
        ComputeThreeBestIdx(histo, HISTO_LENGTH, idx1, idx2, idx3);

        // 剔除非前三大主峰 Bin 中的所有匹配对
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

/**
 * @brief 通过词袋 (BoW) 树结构加速，在关键帧与普通帧之间进行特征匹配
 * @details 利用视觉词袋树的节点聚类，仅比对属于同一 Visual Node 的特征，将复杂度从 O(N*M) 降为线性匹配
 * @param pKF 关键帧
 * @param F 普通当前帧
 * @param[out] vpMapPointMatches 当前帧各特征点匹配到的地图点
 * @return int 成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint *> &vpMapPointMatches)
{
    vpMapPointMatches = std::vector<MapPoint *>(F.N, static_cast<MapPoint *>(nullptr));

    const std::vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();
    const cv::Mat &DescriptorsKF = pKF->mDescriptors;

    // 确保两帧均计算了 BoW 词袋向量与节点倒排特征索引 (FeatureVector)
    pKF->ComputeBoW();
    F.ComputeBoW();

    int nmatches = 0;
    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    // FeatureVector 数据结构: std::map<NodeId, std::vector<unsigned int>>
    // 保存了每个词袋树特定层级 Node 下归属的特征点序列
    const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;
    const DBoW3::FeatureVector &vFeatVecF = F.mFeatVec;

    auto KFit = vFeatVecKF.begin();
    auto Fit = vFeatVecF.begin();
    auto KFend = vFeatVecKF.end();
    auto Fend = vFeatVecF.end();

    // 双指针归并遍历有序 map，仅在 NodeId 相同（视觉相似聚类簇）内进行特征匹配
    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            const std::vector<unsigned int> &vIndicesKF = KFit->second;
            const std::vector<unsigned int> &vIndicesF = Fit->second;

            // 遍历关键帧中落入该 Node 的所有特征点
            for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++)
            {
                const unsigned int realIdxKF = vIndicesKF[iKF];
                MapPoint *pMP = vpMapPointsKF[realIdxKF];

                // 排除无效或已标记为淘汰的地图点
                if (!pMP || pMP->isBad())
                    continue;

                const cv::Mat &dKF = DescriptorsKF.row(realIdxKF);

                int bestDist = TH_LOW;
                int secondBestDist = TH_LOW;
                int bestIdxF = -1;

                // 在普通帧同一 Node 集合内寻找距离最优与次优的特征点
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

                // 阈值筛选：绝对距离必须小于严格门限 TH_LOW (50)
                if (bestDist < TH_LOW)
                {
                    // 满足 Lowe's Ratio 校验
                    if (static_cast<float>(bestDist) < mfNNratio * static_cast<float>(secondBestDist))
                    {
                        vpMapPointMatches[bestIdxF] = pMP;

                        // 统计旋转一致性直方图
                        if (mbCheckOrientation)
                        {
                            float rot = pKF->mvKeysUn[realIdxKF].angle - F.mvKeysUn[bestIdxF].angle;
                            if (rot < 0.0f)
                                rot += 360.0f;
                            int bin = cvRound(rot * rotFactor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;

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

    // 剔除非主方向误匹配点
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

    // 将匹配成功的指针同步至当前帧对应特征点
    for (int i = 0; i < F.N; i++)
    {
        if (vpMapPointMatches[i])
            F.mvpMapPoints[i] = vpMapPointMatches[i];
    }

    return nmatches;
}

/**
 * @brief 局部地图跟踪（TrackLocalMap）：将全局/局部地图点投影至当前帧进行特征关联
 * @details 包含严格的三维可视锥体、视角入射角、尺度不变性范围判定
 * @param F 当前帧
 * @param vpMapPoints 待匹配的局部候选地图点集
 * @param th 基础搜索半径
 * @return int 成功建立匹配的地图点数
 */
int ORBmatcher::SearchByProjection(Frame &F, const std::vector<MapPoint *> &vpMapPoints, const float th)
{
    int nmatches = 0;

    const Eigen::Matrix3f Rcw = F.mTcw.block<3, 3>(0, 0);
    const Eigen::Vector3f tcw = F.mTcw.block<3, 1>(0, 3);
    // 当前相机光心在世界坐标系下的位置: Ow = -Rcw^T * tcw
    const Eigen::Vector3f Ow = -Rcw.transpose() * tcw;

    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    for (size_t iMP = 0; iMP < vpMapPoints.size(); ++iMP)
    {
        MapPoint *pMP = vpMapPoints[iMP];
        if (!pMP || pMP->isBad())
            continue;

        // 1. 坐标变换：世界系 -> 当前相机坐标系
        const Eigen::Vector3f Pw = pMP->GetWorldPos();
        const Eigen::Vector3f Pc = Rcw * Pw + tcw;

        if (Pc.z() <= 0.0f) // 剔除位于相机后方的点
            continue;

        // 2. 尺度不变性距离范围检验 (Scale Invariance Region)
        // 观察距离超出 ORB 特征可稳定提取的边界范围时剔除（加 20% 容差缓冲）
        const float dist = (Pw - Ow).norm();
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        if (dist < minDistance * 0.8f || dist > maxDistance * 1.2f)
            continue;

        // 3. 视线夹角约束 (Viewing Angle Check)
        // 观测向量与该地图点的平均观测法向量夹角需小于 60° (cos(60°) = 0.5)
        Eigen::Vector3f Pn = (Pw - Ow).normalized();
        if (Pn.dot(pMP->GetNormal()) < 0.5f)
            continue;

        // 4. 针孔相机透视投影至像素坐标
        const float invz = 1.0f / Pc.z();
        const float u = Frame::fx * Pc.x() * invz + Frame::cx;
        const float v = Frame::fy * Pc.y() * invz + Frame::cy;

        if (u < Frame::mnMinX || u >= Frame::mnMaxX ||
            v < Frame::mnMinY || v >= Frame::mnMaxY)
            continue;

        // 【关键修复 1】：只要几何投影在当前帧有效视锥体内，更新该地图点的可见统计 (Visible Counter)
        pMP->IncreaseVisible(1);

        // 5. 根据物距推算当前地图点在图像金字塔中对应的理论层级
        const float ratio = dist / pMP->GetMinDistanceInvariance();
        int predictedLevel = cvRound(std::log(ratio) / std::log(F.mpORBextractorLeft->GetScaleFactor()));

        const int nLevels = F.mpORBextractorLeft->GetLevels();
        if (predictedLevel < 0)
            predictedLevel = 0;
        else if (predictedLevel >= nLevels)
            predictedLevel = nLevels - 1;

        // 6. 依据预测尺度层级设定自适应网格搜索半径
        const float radius = th * F.mpORBextractorLeft->GetScaleFactors()[predictedLevel];
        const std::vector<size_t> candidates =
            F.GetFeaturesInArea(u, v, radius, predictedLevel - 1, predictedLevel + 1);

        if (candidates.empty())
            continue;

        const cv::Mat &dMP = pMP->GetDescriptor();
        int bestDist = TH_LOW;
        int secondBestDist = TH_LOW;
        int bestIdx = -1;

        // 寻找最优匹配描述子
        for (size_t c = 0; c < candidates.size(); ++c)
        {
            const size_t idx = candidates[c];
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

        // 7. 匹配判定与双向绑定
        if (bestIdx >= 0 && bestDist < TH_LOW)
        {
            if (static_cast<float>(bestDist) < mfNNratio * static_cast<float>(secondBestDist))
            {
                F.mvpMapPoints[bestIdx] = pMP;
                // 【关键修复 2】：匹配成功，递增地图点的有效匹配计数 (Found Counter)，供地图剪枝决策
                pMP->IncreaseFound(1);
                ++nmatches;

                if (mbCheckOrientation)
                {
                    float rot = F.mvKeysUn[bestIdx].angle;
                    int bin = cvRound(rot * rotFactor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    rotHist[bin].push_back(bestIdx);
                    histo[bin]++;
                }
            }
        }
    }

    // 8. 旋转一致性直方图剔除
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

/**
 * @brief 通过 BoW 在两个关键帧之间匹配共视地图点（常用于回环检测、重定位及共视关系融合）
 * @param pKF1 关键帧 1
 * @param pKF2 关键帧 2
 * @param[out] vpMatches12 pKF1 各特征点在 pKF2 中所对应的 MapPoint 数组
 * @return int 成功匹配的特征点对数
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12)
{
    const std::vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    const std::vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = std::vector<MapPoint *>(pKF1->N, static_cast<MapPoint *>(nullptr));
    // 用于确保 pKF2 中的地图点至多被一个 pKF1 特征点所绑定，保证单射
    std::vector<bool> vbMatched2(pKF2->N, false);

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

    // 词袋树节点特征集合双指针比对
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

                    // 【防多对一抢占】：如果该特征点已被匹配，跳过
                    if (vbMatched2[realIdx2])
                        continue;

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

                // Lowe's 比例过滤与距离门限过滤
                if (bestDist < TH_LOW)
                {
                    if (static_cast<float>(bestDist) < mfNNratio * static_cast<float>(secondBestDist))
                    {
                        vpMatches12[realIdx1] = vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2] = true; // 锁定目标点占用状态

                        if (mbCheckOrientation)
                        {
                            float rot = pKF1->mvKeysUn[realIdx1].angle - pKF2->mvKeysUn[bestIdx2].angle;
                            if (rot < 0.0f)
                                rot += 360.0f;
                            int bin = cvRound(rot * rotFactor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
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

    // 旋转直方图滤波
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

/**
 * @brief 在两个关键帧未匹配的孤立特征点间进行搜索，利用对极几何约束寻找待三角化生成新 MapPoint 的特征对
 * @param pKF1 关键帧 1
 * @param pKF2 关键帧 2
 * @param F12 两帧的基础矩阵 (Fundamental Matrix)，满足 x2^T * F12 * x1 = 0
 * @param[out] vMatchedPairs 输出的匹配索引点对 <idx_KF1, idx_KF2>
 * @param bOnlyStereo 是否只在已有双目深度的特征点间寻找匹配
 * @return int 成功配对用于三角化的数量
 */
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
                                       const bool bOnlyStereo)
{
    const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    // 1. 计算对极几何关键几何元：极点 (Epipole)
    Eigen::Matrix3f Rcw2_eig = pKF2->GetRotation();
    Eigen::Vector3f tcw2_eig = pKF2->GetTranslation();
    Eigen::Vector3f Ow1_eig = pKF1->GetCameraCenter();

    // 关键帧 1 的光心在关键帧 2 相机坐标系下的投影即为极点 e2: e2 = Rcw2 * Ow1 + tcw2
    Eigen::Vector3f e2 = Rcw2_eig * Ow1_eig + tcw2_eig;
    float e2x = e2.x();
    float e2y = e2.y();
    float e2z = e2.z();
    float invz = 1.0f / e2z;
    // 投影至像平面像素坐标 (ex, ey)
    float ex = pKF2->fx * e2x * invz + pKF2->cx;
    float ey = pKF2->fy * e2y * invz + pKF2->cy;

    // 双向互斥匹配数组，维护最佳匹配与其汉明距离
    std::vector<int> vMatched2(pKF2->N, -1);
    std::vector<int> vMatched1(pKF1->N, -1);
    std::vector<int> vMatchedDistance(pKF2->N, INT_MAX);

    std::vector<int> rotHist[HISTO_LENGTH];
    int histo[HISTO_LENGTH] = {0};
    const float rotFactor = static_cast<float>(HISTO_LENGTH) / 360.0f;

    int nmatches = 0;

    auto f1it = vFeatVec1.begin();
    auto f2it = vFeatVec2.begin();
    auto f1end = vFeatVec1.end();
    auto f2end = vFeatVec2.end();

    // 2. 利用词袋树节点约束加速特征配对
    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0; i1 < f1it->second.size(); i1++)
            {
                const size_t idx1 = f1it->second[i1];
                MapPoint *pMP1 = pKF1->GetMapPoint(idx1);
                // 仅寻找没有关联 3D 点的未匹配特征点进行三角化
                if (pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1] >= 0;
                if (bOnlyStereo && !bStereo1)
                    continue;

                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                for (size_t i2 = 0; i2 < f2it->second.size(); i2++)
                {
                    const size_t idx2 = f2it->second[i2];
                    MapPoint *pMP2 = pKF2->GetMapPoint(idx2);
                    if (pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2] >= 0;
                    if (bOnlyStereo && !bStereo2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    const int dist = DescriptorDistance(d1, d2);
                    if (dist > TH_LOW || dist > bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    // 几何检查 A: 极点避障校验
                    // 单目下，若特征点太靠近对极点，会导致极线方向高度退化，三角化几何解不稳定
                    if (!bStereo1 && !bStereo2)
                    {
                        float distex = kp2.pt.x - ex;
                        float distey = kp2.pt.y - ey;
                        if (distex * distex + distey * distey < 100.0f * pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    // 几何检查 B: 对极几何对称极线距离约束 (Symmetric Epipolar Distance)
                    // 极线方程: l = F12 * p1 = (a, b, c)^T
                    const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
                    const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
                    const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

                    // 点 kp2 到极线 l 的欧氏几何距离平方: d^2 = (a*x + b*y + c)^2 / (a^2 + b^2)
                    const float num = a * kp2.pt.x + b * kp2.pt.y + c;
                    const float den = a * a + b * b;
                    if (den == 0.0f)
                        continue;

                    // 卡方检验：单自由度在 95% 置信度下的阈值为 3.841，并按金字塔尺度方差缩放
                    if ((num * num / den) > 3.841f * pKF2->mvLevelSigma2[kp2.octave])
                        continue;

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestIdx2 = idx2;
                    }
                }
                // 3. 双向最优竞争匹配（保证双向一对一投影最优
                if (bestIdx2 >= 0)
                {
                    if (vMatchedDistance[bestIdx2] > bestDist)
                    {
                        // 若被之前别的点匹配过，发生冲突时解除旧关系
                        if (vMatched2[bestIdx2] >= 0)
                            vMatched1[vMatched2[bestIdx2]] = -1;
                        else
                            nmatches++;

                        vMatched2[bestIdx2] = idx1;
                        vMatched1[idx1] = bestIdx2;
                        vMatchedDistance[bestIdx2] = bestDist;

                        if (mbCheckOrientation)
                        {
                            float rot = pKF1->mvKeysUn[idx1].angle - pKF2->mvKeysUn[bestIdx2].angle;
                            if (rot < 0.0f)
                                rot += 360.0f;
                            int bin = cvRound(rot * rotFactor);
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            rotHist[bin].push_back(idx1);
                            histo[bin]++;
                        }
                    }
                }
            }
            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
            f1it++;
        else
            f2it++;
    }

    // 4. 旋转一致性直方图剔除杂点
    if (mbCheckOrientation)
    {
        int ind1 = -1, ind2 = -1, ind3 = -1;
        ComputeThreeBestIdx(histo, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0; j < rotHist[i].size(); j++)
            {
                int idx1 = rotHist[i][j];
                if (vMatched1[idx1] >= 0)
                {
                    vMatched2[vMatched1[idx1]] = -1;
                    vMatched1[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    // 整理最终用于三角化的特征点对
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);
    for (int i = 0; i < pKF1->N; i++)
    {
        if (vMatched1[i] >= 0)
            vMatchedPairs.push_back(std::make_pair(i, vMatched1[i]));
    }

    return nmatches;
}