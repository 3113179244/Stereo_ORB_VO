#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include <algorithm>
#include "ORBextractor.h"
#include "Map.h"

// 静态变量初始化：保证每个关键帧生成的 ID 全局唯一递增
long unsigned int KeyFrame::nNextId = 0;

// 构造函数：从普通帧 (Frame) 拷贝信息并生成 KeyFrame
KeyFrame::KeyFrame(Frame &F, Map *pMap)
    : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp),
      fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
      mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), mK(F.mK.clone()),
      N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mvuRight(F.mvuRight), mvDepth(F.mvDepth),
      mDescriptors(F.mDescriptors.clone()),
      mnScaleLevels(F.mpORBextractorLeft->GetLevels()), mfScaleFactor(F.mpORBextractorLeft->GetScaleFactor()),
      mvScaleFactors(F.mpORBextractorLeft->GetScaleFactors()),
      mvLevelSigma2(F.mpORBextractorLeft->GetScaleSigmaSquares()),
      mvInvLevelSigma2(F.mpORBextractorLeft->GetInverseScaleSigmaSquares()),
      mbBad(false), mpMap(pMap), mpORBvocabulary(F.mpORBvocabulary), mpParent(nullptr)
{
    mnId = nNextId++;              // 分配新的关键帧 ID
    mvpMapPoints = F.mvpMapPoints; // 继承普通帧中已经匹配好的 3D 地图点
    SetPose(F.mTcw);               // 设置关键帧的初始位姿
    AssignFeaturesToGrid();
}

// 线程安全地设置位姿，并同步更新旋转、平移以及相机光心坐标
void KeyFrame::SetPose(const Eigen::Matrix4f &Tcw_)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Tcw = Tcw_;
    // 提取旋转矩阵 3x3
    Rcw = Tcw.block<3, 3>(0, 0);
    // 提取平移向量 3x1
    tcw = Tcw.block<3, 1>(0, 3);
    // 计算旋转矩阵的逆 (因为是正交矩阵，转置即为逆)
    Rwc = Rcw.transpose();
    // 计算相机光心在世界坐标系下的坐标: Ow = -Rcw^T * tcw
    Ow = -Rwc * tcw;
}

// 线程安全获取世界到相机的变换矩阵
Eigen::Matrix4f KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Tcw;
}

// 线程安全获取相机到世界的变换矩阵 (Tcw的逆)
Eigen::Matrix4f KeyFrame::GetPoseInverse()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();
    Twc.block<3, 3>(0, 0) = Rwc;
    Twc.block<3, 1>(0, 3) = Ow;
    return Twc;
}

// 线程安全获取相机光心坐标
Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Ow;
}

// 线程安全获取旋转矩阵 Rcw
Eigen::Matrix3f KeyFrame::GetRotation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Rcw;
}

// 线程安全获取平移向量 tcw
Eigen::Vector3f KeyFrame::GetTranslation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return tcw;
}

// 重新计算并更新共视连接关系（遍历所有的观测点，统计与其它关键帧的共视情况）
void KeyFrame::UpdateConnections()
{
    // 1. 统计当前关键帧与其它关键帧的共视地图点数量
    std::map<KeyFrame *, int> KFcounter;
    std::vector<MapPoint *> vpMP;

    {
        std::unique_lock<std::mutex> lockMatches(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (!pMP || pMP->isBad())
            continue;

        std::map<KeyFrame *, size_t> observations = pMP->GetObservations();

        for (auto mit = observations.begin(); mit != observations.end(); mit++)
        {
            KeyFrame *pKF = mit->first;
            if (pKF->mnId == mnId)
                continue;
            KFcounter[pKF]++;
        }
    }

    if (KFcounter.empty())
        return;

    // 2. 找出共视点最多（nmax）的关键帧，作为无门槛邻居时的保底连接
    int nmax = 0;
    KeyFrame *pKFmax = nullptr;
    const int th = 15; // 共视门槛：至少共享 15 个地图点才算有效连接

    std::vector<std::pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(KFcounter.size());

    for (auto mit = KFcounter.begin(); mit != KFcounter.end(); mit++)
    {
        if (mit->second > nmax)
        {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th)
            vPairs.push_back(std::make_pair(mit->second, mit->first));
    }

    // 没有满足门槛的邻居时，强制与共视最多的那个建立连接（保证连通性）
    if (vPairs.empty() && pKFmax)
        vPairs.push_back(std::make_pair(nmax, pKFmax));

    // 3. 升序排序后逆序存入 → 得到降序
    std::sort(vPairs.begin(), vPairs.end());

    std::vector<KeyFrame *> vNeighbors;
    std::vector<int> vWeights;
    vNeighbors.reserve(vPairs.size());
    vWeights.reserve(vPairs.size());

    for (int i = static_cast<int>(vPairs.size()) - 1; i >= 0; i--)
    {
        vNeighbors.push_back(vPairs[i].second);
        vWeights.push_back(vPairs[i].first);
    }

    // 4. 更新自身的连接关系（只保留 >= 15 或保底的邻居，与 ORB-SLAM2 原版对齐）
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights.clear();
        for (size_t i = 0; i < vNeighbors.size(); i++)
            mConnectedKeyFrameWeights[vNeighbors[i]] = vWeights[i];

        mvpOrderedConnectedKeyFrames = vNeighbors; // 降序
        mvOrderedWeights = vWeights;
    }

    // 5. 反向注册 + 生成树建立（在锁外执行避免死锁）
    for (size_t i = 0; i < vNeighbors.size(); i++)
        vNeighbors[i]->AddConnection(this, vWeights[i]);

    if (mpParent == nullptr && pKFmax != nullptr)
        SetParent(pKFmax);
}

// 主动添加或修改一条共视连接
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        // 如果该关键帧尚未记录，则新增
        if (!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        // 如果已记录但权重发生变化，则更新权重
        else if (mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return; // 无变化则直接返回，无需重排序
    }

    // 权重有更新，重新整理共视最高序列
    UpdateBestCovisibles();
}

// 对目前所有的共视连接进行排序，更新高共视性列表
void KeyFrame::UpdateBestCovisibles()
{
    std::vector<std::pair<int, KeyFrame *>> vPairs;

    // 1. 在锁保护下读取连接权重（与 mConnectedKeyFrameWeights 的其它读写保持一致）
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        vPairs.reserve(mConnectedKeyFrameWeights.size());

        // 提取所有连接关系
        for (auto mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
            vPairs.push_back(std::make_pair(mit->second, mit->first));
    }

    // 2. 在锁外排序，尽量缩短持锁时间
    // 默认按 pair 的第一个元素（权重）进行升序排序
    std::sort(vPairs.begin(), vPairs.end());

    std::vector<KeyFrame *> vNeighbors;
    vNeighbors.reserve(vPairs.size());
    std::vector<int> vWeights;
    vWeights.reserve(vPairs.size());

    // 逆序遍历，使得最终保存的列表按权重降序排列（权重最大的排在最前面）
    for (int i = vPairs.size() - 1; i >= 0; i--)
    {
        vNeighbors.push_back(vPairs[i].second);
        vWeights.push_back(vPairs[i].first);
    }

    // 3. 在锁保护下写回内部有序列表（与读取方 GetBestCoviscibilityKeyFrames 等保持一致，
    //    避免对 mvpOrderedConnectedKeyFrames / mvOrderedWeights 的并发读写造成数据竞争）
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        mvpOrderedConnectedKeyFrames = vNeighbors;
        mvOrderedWeights = vWeights;
    }
}

// 提取共视程度排名前 N 的关键帧
std::vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    // 如果总连接数不足 N 个，则返回全部
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        // 截取前 N 个返回
        return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

// 添加特征点与 3D 地图点之间的绑定关联
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

// 根据特征点索引解除匹配关系
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = nullptr;
}

// 根据地图点指针解除匹配关系
void KeyFrame::EraseMapPointMatch(MapPoint *pMP)
{
    // 先获取该地图点在当前关键帧中的索引
    int idx = pMP->GetIndexInKeyFrame(this);
    if (idx >= 0)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = nullptr;
    }
}

// 替换对应位置的地图点匹配（如闭环融合后将旧点换成新点）
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

// 获取本关键帧中所有的地图点匹配列表
std::vector<MapPoint *> KeyFrame::GetMapPointMatches()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

// 获取某个特定特征点关联的地图点
MapPoint *KeyFrame::GetMapPoint(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::SetBadFlag()
{
    std::map<KeyFrame *, int> connectedKFs;
    std::vector<MapPoint *> vpMP;
    KeyFrame *pParent = nullptr;
    std::set<KeyFrame *> vChildren;

    // 1. 锁保护检查并提取生成树和共视关系
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        if (mbBad || mnId == 0)
            return;

        if (mbNotErase)
        {
            mbToBeErased = true;
            return;
        }

        mbBad = true;
        connectedKFs = mConnectedKeyFrameWeights;
        pParent = mpParent;
        vChildren = mspChildren;
    }

    // 2. 避免锁嵌套：在单独获取位姿锁的逻辑下计算相对位姿
    if (pParent)
    {
        Eigen::Matrix4f T_c_w = GetPose();
        Eigen::Matrix4f T_p_w = pParent->GetPoseInverse();
        std::unique_lock<std::mutex> lockPose(mMutexPose);
        mTcp = T_c_w * T_p_w;
    }

    {
        std::unique_lock<std::mutex> lockFeat(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    // 3. 维护生成树（Spanning Tree）：ORB-SLAM2 级防成环重连策略
    // 收集所有候选的新父节点：优先考虑当前帧的父节点，其次是子节点的共视帧
    std::vector<KeyFrame *> vpCandidateParents;
    if (pParent && !pParent->mbBad)
        vpCandidateParents.push_back(pParent);

    for (KeyFrame *pChild : vChildren)
    {
        if (!pChild || pChild->mbBad)
            continue;

        KeyFrame *pNewParent = nullptr;

        // 策略 A: 先在候选集合（即当前帧的父节点）中匹配
        for (KeyFrame *pCand : vpCandidateParents)
        {
            if (pCand != pChild)
            {
                pNewParent = pCand;
                break;
            }
        }

        // 策略 B: 若没有，则从子节点自身的共视帧中选取（排除当前正在被删的帧和待处理子节点自身）
        if (!pNewParent)
        {
            std::vector<KeyFrame *> vpCov = pChild->GetBestCovisibilityKeyFrames(20);
            for (KeyFrame *pCov : vpCov)
            {
                if (pCov && !pCov->mbBad && pCov != this && pCov != pChild)
                {
                    // 成环检测：沿着 pCov 往上爬，如果中途遇到了 pChild，则不能选它，否则成环
                    bool bLoop = false;
                    KeyFrame *pCurrAncestor = pCov->GetParent();
                    while (pCurrAncestor)
                    {
                        if (pCurrAncestor == pChild)
                        {
                            bLoop = true;
                            break;
                        }
                        pCurrAncestor = pCurrAncestor->GetParent();
                    }

                    if (!bLoop)
                    {
                        pNewParent = pCov;
                        break;
                    }
                }
            }
        }

        // 策略 C: 实在找不到，保底沿用当前帧的父节点
        if (!pNewParent && pParent)
            pNewParent = pParent;

        if (pNewParent)
        {
            pChild->ChangeParent(pNewParent);
            vpCandidateParents.push_back(pChild); // 该子节点重连成功后，也可以作为后续其他兄弟子节点的候选父节点
        }
    }

    if (pParent)
        pParent->EraseChild(this);

    // 4. 断开相连关键帧的双向共视连接
    for (auto mit = connectedKFs.begin(); mit != connectedKFs.end(); mit++)
        mit->first->EraseConnection(this);

    // 5. 解除地图点对该关键帧的观测
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        if (vpMP[i])
            vpMP[i]->EraseObservation(this);
    }

    // 6. 清空连接记录（保留 mpParent 供回溯相对位姿 mTcp）
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();
        mvOrderedWeights.clear();
        mspChildren.clear();
        mspLoopEdges.clear();
    }

    // 7. 从全局地图中删除
    mpMap->EraseKeyFrame(this);
}

void KeyFrame::EraseConnection(KeyFrame *pKF)
{
    bool bUpdate = false;
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if (mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate = true;
        }
    }

    if (bUpdate)
        UpdateBestCovisibles();
}

std::vector<KeyFrame *> KeyFrame::GetConnectedKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    std::vector<KeyFrame *> vKF;
    vKF.reserve(mConnectedKeyFrameWeights.size());
    for (auto mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
    {
        vKF.push_back(mit->first);
    }
    return vKF;
}

// 获取权重（共视点数）大于等于指定值 w 的所有相连关键帧（按权重降序返回）
// 与 ORB-SLAM2 原版语义一致：内部循环按权重降序扫描，碰到第一个小于 w 的就停止
std::vector<KeyFrame *> KeyFrame::GetCovisibleByWeight(const int &w)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
        return std::vector<KeyFrame *>();

    std::vector<KeyFrame *> vKFs;
    const int N = mvOrderedWeights.size();
    for (int i = 0; i < N; i++)
    {
        if (mvOrderedWeights[i] < w)
            break; // 权重降序，遇到小于阈值的直接停止
        vKFs.push_back(mvpOrderedConnectedKeyFrames[i]);
    }
    return vKFs;
}

// 获取当前关键帧与指定关键帧之间的共视权重（共享地图点数量）
// 若两者之间没有连接，返回 0
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    auto it = mConnectedKeyFrameWeights.find(pKF);
    if (it != mConnectedKeyFrameWeights.end())
        return it->second;
    return 0;
}

void KeyFrame::ComputeBoW()
{
    if (!mpORBvocabulary)
    {
        std::cerr << "[ERROR] KeyFrame::ComputeBoW(): mpORBvocabulary is nullptr!" << std::endl;
        return;
    }

    if (mDescriptors.empty())
    {
        return;
    }

    if (mBowVec.empty())
    {
        // 将 cv::Mat 矩阵按行转换为 DBoW3 要求的 std::vector<cv::Mat>
        std::vector<cv::Mat> vCurrentDesc;
        vCurrentDesc.reserve(mDescriptors.rows);
        for (int i = 0; i < mDescriptors.rows; i++)
        {
            vCurrentDesc.push_back(mDescriptors.row(i).clone());
        }

        // 调用 DBoW3 的 transform 接口
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void KeyFrame::AssignFeaturesToGrid()
{
    // 修正图像边界来源：直接使用普通帧已经计算好的去畸变图像边界
    // （Frame::ComputeImageBounds 用真实的 imLeft.cols/rows 设定，位于图像网格内）。
    // 之前用 2.0f*cx / 2.0f*cy 推导会导致主点 cx != cols/2 时边界不准确，
    // 使图像边缘的特征点被漏搜。
    mnMinX = Frame::mnMinX;
    mnMinY = Frame::mnMinY;
    mnMaxX = Frame::mnMaxX;
    mnMaxY = Frame::mnMaxY;
    mfGridElementWidthInv = static_cast<float>(mnGridCols) / (mnMaxX - mnMinX);
    mfGridElementHeightInv = static_cast<float>(mnGridRows) / (mnMaxY - mnMinY);

    // 预分配
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(static_cast<int>(0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS)));

    for (int i = 0; i < N; i++)
    {
        int nGridPosX, nGridPosY;
        if (PosInGrid(mvKeysUn[i], nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

bool KeyFrame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;
    return true;
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(
    const float &x, const float &y, const float &r,
    const int minLevel, const int maxLevel) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = std::max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;
    const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = std::max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;
    const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> &vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;
            for (size_t j = 0; j < vCell.size(); j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if (minLevel < 0 || (kpUn.octave >= minLevel && kpUn.octave <= maxLevel))
                {
                    if (std::fabs(kpUn.pt.x - x) < r && std::fabs(kpUn.pt.y - y) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }
    }
    return vIndices;
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    if (pKF == this)
        return;

    KeyFrame *pOldParent = nullptr;
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if (mpParent == pKF)
            return;
        pOldParent = mpParent;
        mpParent = pKF;
    }

    if (pOldParent)
        pOldParent->EraseChild(this);

    if (pKF)
        pKF->AddChild(this);
}

void KeyFrame::SetParent(KeyFrame *pKF)
{
    if (pKF == this)
        return;
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        mpParent = pKF;
    }

    if (pKF)
        pKF->AddChild(this);
}

KeyFrame *KeyFrame::GetParent()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mpParent;
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    mspChildren.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    mspChildren.erase(pKF);
}

std::set<KeyFrame *> KeyFrame::GetChilds()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mspChildren;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    int nPoints = 0;
    const bool bCheckObs = minObs > 0;
    for (int i = 0; i < N; i++)
    {
        MapPoint *pMP = mvpMapPoints[i];
        if (pMP && !pMP->isBad())
        {
            if (bCheckObs)
            {
                if (pMP->GetObservations().size() >= minObs)
                    nPoints++;
            }
            else
                nPoints++;
        }
    }
    return nPoints;
}

Eigen::Matrix4f KeyFrame::GetRelativePoseToParent()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcp;
}

std::vector<KeyFrame *> KeyFrame::GetVectorCovisibleKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

Eigen::Vector3f KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if (z > 0.0f)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        return GetPoseInverse().block<3, 3>(0, 0) * x3Dc + GetCameraCenter();
    }
    return Eigen::Vector3f::Zero();
}

void KeyFrame::SetNotErase()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        if (mbToBeErased)
        {
            mbToBeErased = false;
            mbNotErase = false;
            SetBadFlag();
            return;
        }
        mbNotErase = false;
    }
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    mspLoopEdges.insert(pKF);
}

std::set<KeyFrame *> KeyFrame::GetLoopEdges()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mspLoopEdges;
}