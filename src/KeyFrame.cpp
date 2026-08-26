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
        vpMP = mvpMapPoints; // 拷贝地图点列表（受 mMutexFeatures 保护）
    }

    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (!pMP || pMP->isBad())
            continue;

        // 获取观测到该地图点的所有关键帧（返回副本，内部已加锁）
        std::map<KeyFrame *, size_t> observations = pMP->GetObservations();

        for (auto mit = observations.begin(); mit != observations.end(); mit++)
        {
            KeyFrame *pKF = mit->first;
            if (pKF->mnId == mnId) // 跳过自身
                continue;
            KFcounter[pKF]++; // 累加共视地图点数
        }
    }

    if (KFcounter.empty())
        return;

    // 2. 找出共视点最多（nmax）的关键帧，作为无门槛邻居时的保底连接
    int nmax = 0;
    KeyFrame *pKFmax = nullptr;
    const int th = 15; // 共视门槛：至少共享15个地图点才算有效连接

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

    // 3. 升序排序后逆序存入 → 得到降序（权重最大在前），与 UpdateBestCovisibles 一致
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

    // 4. 更新自身的连接关系（加锁，持锁时间尽量短，只更新自己的数据）
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vNeighbors; // 降序
        mvOrderedWeights = vWeights;
    }

    // 5. 反向注册 + 生成树（Spanning Tree）建立。
    //    这两步都会获取【其它】关键帧的锁，因此必须放在锁外执行，
    //    否则会出现“持 this 锁 + 等 neighbor 锁”的嵌套解锁顺序，
    //    与另一线程反向执行时构成 ABBA 死锁（程序卡死）。
    for (size_t i = 0; i < vNeighbors.size(); i++)
        vNeighbors[i]->AddConnection(this, KFcounter[vNeighbors[i]]);

    //    生成树建立：只有在本关键帧还没有父节点时，才把它共视程度最高的
    //    关键帧设为父节点。这样生成树会随关键帧插入逐步生长、保持连通，
    //    且每个关键帧有且仅有一个父节点（不构成环）。
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

    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        if (mbBad)
            return;
        mbBad = true;
        connectedKFs = mConnectedKeyFrameWeights;
        pParent = mpParent;
        vChildren = mspChildren;
        if (pParent)
        {
            mTcp = GetPose() * pParent->GetPoseInverse();
        }
    }
    {
        std::unique_lock<std::mutex> lockFeat(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    // 1. 维护生成树（Spanning Tree）：把子节点重连给父节点或最佳共视帧
    for (auto it = vChildren.begin(); it != vChildren.end(); it++)
    {
        KeyFrame *pChild = *it;
        if (!pChild || pChild->mbBad)
            continue;

        KeyFrame *pNewParent = pParent;
        if (!pNewParent || pNewParent->mbBad)
        {
            std::vector<KeyFrame *> vpCovisibles = pChild->GetBestCovisibilityKeyFrames(10);
            for (KeyFrame *pCov : vpCovisibles)
            {
                if (pCov && !pCov->mbBad && pCov != this)
                {
                    pNewParent = pCov;
                    break;
                }
            }
        }
        pChild->SetParent(pNewParent);
    }

    if (pParent)
        pParent->EraseChild(this);

    // 2. 断开相连关键帧的双向共视连接
    for (auto mit = connectedKFs.begin(); mit != connectedKFs.end(); mit++)
        mit->first->EraseConnection(this);

    // 3. 解除地图点对该关键帧的观测
    for (size_t i = 0; i < vpMP.size(); i++)
        if (vpMP[i])
            vpMP[i]->EraseObservation(this);

    // 4. 清空连接记录
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();
        mvOrderedWeights.clear();
        // 【关键改动】：务必注释掉下面这行！保留自己的 mpParent 指针供普通帧回溯位姿
        // mpParent = nullptr;
        mspChildren.clear();
    }

    // 5. 从全局地图中删除
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

void KeyFrame::SetParent(KeyFrame *pKF)
{
    if (pKF == this) return; // 避免自环
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