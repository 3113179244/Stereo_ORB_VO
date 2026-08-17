#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/opencv.hpp>

class Frame; 
class KeyFrame;
class MapPoint;
/**
 * @brief ORB 特征匹配器
 * 用于在双目图像之间或连续帧之间进行 ORB 特征点的匹配
 */
class ORBmatcher
{
public:
    /**
     * @brief 构造函数
     * @param nnratio            最优与次优距离的比值阈值（Nearest Neighbor Ratio）。用于剔除模糊匹配（如纹理相似区域），通常取 0.6~0.8。比值越小，匹配越严格。
     * @param checkOrientation   是否检查旋转一致性。开启后会通过方向直方图统计，剔除方向变化不一致的误匹配。
     */
    ORBmatcher(float nnratio = 0.6f, bool checkOrientation = true);
    ~ORBmatcher() = default;

    /**
     * @brief 核心计算工具：计算两个 ORB 描述子的汉明距离 (Hamming Distance)
     * ORB 描述子是 256 bit 的二进制向量。这里利用 CPU 硬件指令 popcount 进行高性能加速。
     * @param a 描述子 A (cv::Mat, CV_8U, 32列)
     * @param b 描述子 B (cv::Mat, CV_8U, 32列)
     * @return 汉明距离（对应位不同的数量）
     */
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    /**
     * @brief 双目极线搜索匹配：在左右目图像中匹配特征点，并直接计算视差 (Disparity) 和深度 (Depth)
     * 假设图像已经过极线矫正，因此匹配点只在同一行（或相近行）中搜索。
     * @param F 包含左右图像特征信息的 Frame 引用
     * @return 成功建立双目匹配的特征点数量
     */
    int ComputeStereoMatches(Frame &F);

    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono = false);
    
    /**
     * @brief 通过词袋 (BoW) 匹配 KeyFrame 与 Frame 中的特征点
     * @param pKF 关键帧指针
     * @param F 当前帧
     * @param vpMapPointMatches 匹配到的地图点输出数组
     * @return 成功匹配的数量
     */
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    
    static const int TH_LOW;       // 匹配距离较低阈值，用于要求较高的匹配场景（如非连续帧或宽基线）
    static const int TH_HIGH;      // 匹配距离较高阈值，用于要求较宽松的场景（如连续帧追踪）
    static const int HISTO_LENGTH; // 方向直方图的 Bin 数量 (通常为 36 个 bin，每 10 度划分一个)

private:
    /**
     * @brief 计算旋转直方图中前三个最大值的索引
     * 用于在方向一致性检测时，保留包含匹配点最多的三个主方向，剔除其他离群方向。
     * @param histo 长度为 L 的直方图数组
     * @param L     直方图长度（通常为 36）
     * @param idx1  输出：包含匹配点最多的 Bin 索引
     * @param idx2  输出：包含匹配点第二多的 Bin 索引
     * @param idx3  输出：包含匹配点第三多的 Bin 索引
     */
    void ComputeThreeBestIdx(int *histo, const int L, int &idx1, int &idx2, int &idx3);

    float mfNNratio;         // 最优/次优距离比率阈值
    bool mbCheckOrientation; // 角度（方向）一致性检查标志位
};

#endif // ORBMATCHER_H