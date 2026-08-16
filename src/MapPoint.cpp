#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"

// 静态变量初始化，用于分配全局递增的地图点ID
long unsigned int MapPoint::nNextId = 0;
std::mutex MapPoint::mGlobalMutex;

// 构造函数，初始化世界坐标、参考关键帧、地图、可见/匹配次数等，并将 bad 标志位置为 false
MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap)
    : mWorldPos(Pos), mpRefKF(pRefKF), mpMap(pMap), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(nullptr)
{
    mnId = nNextId++; // 赋予独立ID
    mNormalVector.setZero(); // 法向量初始化为0
    mnFirstKFid = pRefKF ? pRefKF->mnId : 0; // 记录是由哪个关键帧创建的
}

// 设置世界坐标，加锁保护以确保线程安全
void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos)
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

// 获取世界坐标，同样需要加锁保护读取过程
Eigen::Vector3f MapPoint::GetWorldPos()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos;
}

// 获取该点的平均法向量
Eigen::Vector3f MapPoint::GetNormal()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mNormalVector;
}

// 记录哪个关键帧的哪个特征点观测到了该地图点
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    // 如果已经存在该关键帧的观测，则直接返回
    if(mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
}

// 抹除某个关键帧对该地图点的观测记录
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            mObservations.erase(pKF); // 移除观测
            
            // 如果移除的恰好是参考关键帧，则需要重新指定一个存在的关键帧作为参考关键帧
            if(mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // 当观测到该点的关键帧少于等于 1 个时，该点失去了三角化或者稳定约束的意义，标记为坏点
            if(mObservations.size() <= 1)
                bBad = true;
        }
    }

    // 只有出了上面的作用域解锁后，再调用 SetBadFlag 以避免潜在的死锁
    if(bBad)
        SetBadFlag();
}

// 返回所有的观测字典
std::map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

// 计算代表性描述子：找到与其他所有观测描述子汉明距离中位数最小的那个描述子
// 因为不同角度、距离观测到的描述子会有微小差异，选“中位数最小”即选最具代表性（处于聚类中心）的一个
void MapPoint::ComputeDistinctiveDescriptor()
{
    std::vector<cv::Mat> vDescriptors;
    std::map<KeyFrame*, size_t> observations;

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mbBad) return; // 如果已经是坏点直接返回
        observations = mObservations;
    }

    if(observations.empty()) return;

    vDescriptors.reserve(observations.size());

    // 遍历所有观测，将未被判定为坏点的关键帧中对应的特征描述子提取出来
    for(auto mit = observations.begin(); mit != observations.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        if(!pKF->mbBad)
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty()) return;

    // 计算两两之间的距离矩阵
    const size_t N = vDescriptors.size();
    float Distances[N][N];
    for(size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0; // 自身距离为0
        for(size_t j = i + 1; j < N; j++)
        {
            // 使用 OpenCV 计算汉明距离
            int dist = cv::norm(vDescriptors[i], vDescriptors[j], cv::NORM_HAMMING);
            Distances[i][j] = dist;
            Distances[j][i] = dist; // 对称矩阵
        }
    }

    int BestMedian = INT_MAX;
    int BestIdx = 0;
    // 遍历所有描述子，计算各自与其余描述子距离的中位数
    for(size_t i = 0; i < N; i++)
    {
        std::vector<int> vDists(Distances[i], Distances[i] + N);
        std::sort(vDists.begin(), vDists.end()); // 排序以方便寻找中位数
        int median = vDists[0.5 * (N - 1)];      // 获取中位数

        // 记录中位数最小的那个描述子索引
        if(median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone(); // 保存最具代表性的描述子
    }
}

// 线程安全地获取描述子
cv::Mat MapPoint::GetDescriptor()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

// 更新平均观测方向向量与尺度（距离）下限/上限
void MapPoint::UpdateNormalAndDepth()
{
    std::map<KeyFrame*, size_t> observations;
    KeyFrame* pRefKF;
    Eigen::Vector3f Pos;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        if(mbBad) return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if(observations.empty()) return;

    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    int n=0;

    // 遍历所有观测，累加观测方向的单位向量
    for(auto mit = observations.begin(); mit != observations.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        Eigen::Vector3f Owc = pKF->GetCameraCenter(); // 获取相机的光心
        Eigen::Vector3f normali = Pos - Owc;          // 从光心指向地图点的向量
        normal += normali.normalized();               // 归一化后累加
        n++;
    }

    // 计算参考关键帧到地图点的距离
    Eigen::Vector3f Owc = pRefKF->GetCameraCenter();
    Eigen::Vector3f dist = Pos - Owc;
    const float distRef = dist.norm();

    // 结合特征金字塔，估算该点在哪个距离范围内依然能够被金字塔层级探测到
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave; // 特征点所在金字塔层级
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];    // 该层级的缩放因子
    const int nLevels = pRefKF->mnScaleLevels;                       // 总层数

    {
        std::unique_lock<std::mutex> lock3(mMutexPos);
        // 推算能观测到该特征点的最小与最大距离范围，用于后续做投影匹配时的剔除判断
        mfMinDistance = distRef / levelScaleFactor;
        mfMaxDistance = mfMinDistance * pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal.normalized(); // 最终平均法向量
    }
}

// 设置坏点标志
void MapPoint::SetBadFlag()
{
    std::map<KeyFrame*, size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        mbBad = true; // 标记为 bad
        obs = mObservations;
        mObservations.clear(); // 清空本点的观测记录
    }
    
    // 通知曾经观测过这个点的关键帧：抹除关键帧端指向这里的匹配引用
    for(auto mit = obs.begin(); mit != obs.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    // 从地图系统中剔除自己
    mpMap->EraseMapPoint(this);
}

// 线程安全地返回当前是否为坏点
bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mbBad;
}

// 用另外一个地图点替换本点 (主要发生于局部建图去重、或者闭环检测融合阶段)
void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId == this->mnId) return; // 自己无需替换自己

    std::map<KeyFrame*, size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear(); // 转移观测之前，先清空自身的观测
        mbBad = true;          // 本点被抛弃
        mpReplaced = pMP;      // 记录被哪个点顶替了
    }

    // 处理原来所有能看到当前旧点的关键帧
    for(auto mit = obs.begin(); mit != obs.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        // 如果顶替它的新点还没有被该关键帧观测到
        if(!pMP->IsInKeyFrame(pKF))
        {
            // 让关键帧把对应的地图点指针更新为新的点
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            // 新点增加对该关键帧的观测记录
            pMP->AddObservation(pKF, mit->second);
        }
        else
        {
            // 如果新点本来就已经在这个关键帧里有观测了，说明产生了冲突，则直接抹除关键帧里原来的旧匹配
            pKF->EraseMapPointMatch(mit->second);
        }
    }

    // 由于新点吸收了旧点的观测，所以需要重新计算它的最优描述子和法向量深度
    pMP->ComputeDistinctiveDescriptor();
    pMP->UpdateNormalAndDepth();

    // 最后让地图删掉原来的自己
    mpMap->EraseMapPoint(this);
}

// 查询本点是否在给定的关键帧视野内被提取出来了
bool MapPoint::IsInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.count(pKF);
}

// 查询本点对应给定关键帧上的第几个特征点，若不存在返回-1
int MapPoint::GetIndexInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
        return mObservations[pKF];
    }
    return -1;
}

float MapPoint::GetFoundRatio()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound) / static_cast<float>(mnVisible);
}

