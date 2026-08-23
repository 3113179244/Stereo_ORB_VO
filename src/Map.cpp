#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"

// 构造函数
Map::Map() {}

/**
 * @brief 向地图添加关键帧
 * @details 使用 unique_lock 加锁，防止多线程竞争
 */
void Map::AddKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
}

/**
 * @brief 向地图添加地图点
 */
void Map::AddMapPoint(MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

/**
 * @brief 从地图删除指定的地图点
 */
void Map::EraseMapPoint(MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
}

/**
 * @brief 从地图删除指定的关键帧
 */
void Map::EraseKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
}

/**
 * @brief 获取地图中所有关键帧的副本
 * @return 将 std::set 转换为 std::vector 返回，避免外部直接修改内部集合
 */
std::vector<KeyFrame*> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}

/**
 * @brief 获取地图中所有地图点的副本
 */
std::vector<MapPoint*> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}

/**
 * @brief 获取参考地图点（局部地图点）的副本
 */
std::vector<MapPoint*> Map::GetReferenceMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

/**
 * @brief 设置参考地图点（由 Tracker 更新）
 */
void Map::SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

/**
 * @brief 获取当前地图点的总数
 */
long unsigned int Map::GetMapPointsInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

/**
 * @brief 获取当前关键帧的总数
 */
long unsigned int Map::GetKeyFramesInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

/**
 * @brief 清空地图数据
 * @details 清除关键帧、地图点以及参考地图点容器中的所有元素
 */
void Map::Clear()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mvpReferenceMapPoints.clear();
}