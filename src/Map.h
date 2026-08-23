#ifndef MAP_H
#define MAP_H

#include <set>
#include <vector>
#include <mutex>

// 前向声明，避免循环引用头文件
class KeyFrame;
class MapPoint;

/**
 * @brief 地图类（Map）
 * @details 负责管理 SLAM 系统中的所有关键帧（KeyFrame）和地图点（MapPoint），
 *          并提供线程安全的插入、删除、查询等操作。
 */
class Map
{
public:
    /**
     * @brief 构造函数
     */
    Map();

    /**
     * @brief 向地图中添加一个新的关键帧
     * @param pKF 待添加的关键帧指针
     */
    void AddKeyFrame(KeyFrame* pKF);

    /**
     * @brief 向地图中添加一个新的地图点
     * @param pMP 待添加的地图点指针
     */
    void AddMapPoint(MapPoint* pMP);

    /**
     * @brief 从地图中删除指定的地图点
     * @param pMP 待删除的地图点指针
     */
    void EraseMapPoint(MapPoint* pMP);

    /**
     * @brief 从地图中删除指定的关键帧
     * @param pKF 待删除的关键帧指针
     */
    void EraseKeyFrame(KeyFrame* pKF);
    
    /**
     * @brief 获取地图中所有的关键帧
     * @return 包含所有关键帧指针的 vector 容器
     */
    std::vector<KeyFrame*> GetAllKeyFrames();

    /**
     * @brief 获取地图中所有的地图点
     * @return 包含所有地图点指针的 vector 容器
     */
    std::vector<MapPoint*> GetAllMapPoints();

    /**
     * @brief 获取参考地图点（局部地图点）
     * @return 包含参考地图点指针的 vector 容器
     */
    std::vector<MapPoint*> GetReferenceMapPoints();

    /**
     * @brief 设置参考地图点（局部地图点）
     * @param vpMPs 参考地图点指针容器
     */
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

    /**
     * @brief 获取地图中当前地图点的总数量
     * @return 地图点数量
     */
    long unsigned int GetMapPointsInMap();

    /**
     * @brief 获取地图中当前关键帧的总数量
     * @return 关键帧数量
     */
    long unsigned int GetKeyFramesInMap();

    /**
     * @brief 清空地图中的所有数据（包括关键帧、地图点和参考点）
     */
    void Clear();
    // 提供给外部的地图级互斥锁（保护整个地图拓扑与批量更新一致性）
    std::mutex mMutexMapUpdate;
private:
    // 存储地图中所有地图点的集合（使用 std::set 保证唯一性）
    std::set<MapPoint*> mspMapPoints;

    // 存储地图中所有关键帧的集合
    std::set<KeyFrame*> mspKeyFrames;

    // 存储参考地图点（通常用于局部地图跟踪或可视化展示）
    std::vector<MapPoint*> mvpReferenceMapPoints;

    // 互斥锁，用于保证多线程环境下对地图数据访问的线程安全
    std::mutex mMutexMap;
};

#endif // MAP_H