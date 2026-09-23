#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

/**
 * @brief 四叉树分配节点类
 * 用于在图像金字塔的各层中通过四叉树分割特征点，保证特征点在空间上分布均匀。
 */
class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}

    /**
     * @brief 将当前节点划分为4个子节点（UL: 左上, UR: 右上, BL: 左下, BR: 右下）
     */
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;          // 当前节点（区域）内包含的特征点集合
    cv::Point2i UL, UR, BL, BR;               // 当前节点的四个角点坐标（左上、右上、左下、右下）
    std::list<ExtractorNode>::iterator lit;   // 指向当前节点在父链表中的迭代器，便于快速访问和删除
    bool bNoMore;                             // 标记该节点是否无法再细分（例如节点内仅有1个特征点或区域过小时）
};

/**
 * @brief ORB 特征提取器类
 * 负责构建图像金字塔、使用四叉树均匀分配 FAST 特征点、计算特征点方向以及提取 BRIEF 描述子。
 */
class ORBextractor
{
public:
    // 特征点打分机制类型枚举
    enum { 
        HARRIS_SCORE = 0,  // 使用 Harris 响应值打分（排序更准确但计算稍耗时）
        FAST_SCORE = 1     // 使用 FAST 阈值差打分（计算速度更快）
    };

    /**
     * @brief 构造函数
     * @param nfeatures       总共期望提取的特征点数量
     * @param scaleFactor     图像金字塔每层之间的缩放因子（通常取 1.2）
     * @param nlevels         图像金字塔的层数（通常取 8 层）
     * @param iniThFAST       FAST 角点提取的初始默认阈值（较严格，优先提取强角点）
     * @param minThFAST       FAST 角点提取的最小阈值（当初始阈值提取不到足够点时降级使用）
     */
    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
    
    ~ORBextractor() = default;

    /**
     * @brief 重载括号操作符，用于执行特征点和描述子的提取流程
     * @param image           输入图像
     * @param mask            掩膜图像（可用于指定感兴趣区域，若无则传空）
     * @param keypoints       输出提取到的关键点集合
     * @param descriptors     输出计算好的特征描述子（矩阵形式）
     */
    void operator()(cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    // 常用属性的 Getter 接口
    int GetLevels() const { return nlevels; }                                          // 获取金字塔层数
    float GetScaleFactor() const { return scaleFactor; }                              // 获取每层缩放倍率
    const std::vector<float>& GetScaleFactors() const { return mvScaleFactor; }       // 获取各层相对于原图的尺度缩放因子
    const std::vector<float>& GetInverseScaleFactors() const { return mvInvScaleFactor; } // 获取各层缩放因子的倒数
    const std::vector<float>& GetScaleSigmaSquares() const { return mvLevelSigma2; } // 获取各层尺度的方差（尺度平方，用于重投影误差等加权）
    const std::vector<float>& GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; } // 获取尺度方差的倒数
    const std::vector<cv::Mat>& GetImagePyramid() const { return mvImagePyramid; }   // 获取图像金字塔图像缓存
        
protected:
    /**
     * @brief 构建高斯/下采样图像金字塔
     * @param image 原始输入灰度图
     */
    void ComputePyramid(cv::Mat image);

    /**
     * @brief 使用四叉树法在金字塔每一层中提取并分配均匀分布的关键点
     * @param allKeypoints 输出各层关键点的二维数组（第一维代表金字塔层数）
     */
    void ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);

    /**
     * @brief 具体的四叉树分割与特征点均匀分配算法
     * @param vToDistributeKeys 待分配的候选特征点集合
     * @param minX              区域最小 X 坐标
     * @param maxX              区域最大 X 坐标
     * @param minY              区域最小 Y 坐标
     * @param maxY              区域最大 Y 坐标
     * @param N                 本区域最终期望保留的目标特征点数量
     * @param level             当前处理的金字塔层数
     * @return std::vector<cv::KeyPoint> 筛选后分布均匀的特征点集合
     */
    std::vector<cv::KeyPoint> DistributeOctree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                               const int &minX, const int &maxX,
                                               const int &minY, const int &maxY,
                                               const int &N, const int &level);

    int nfeatures;        // 期望提取的总特征点数
    float scaleFactor;    // 金字塔层间缩放比例
    int nlevels;          // 金字塔总层数
    int iniThFAST;        // FAST 初始阈值
    int minThFAST;        // FAST 备选最低阈值

    std::vector<cv::Mat> mvImagePyramid;       // 图像金字塔容器，每一层对应一幅图像
    std::vector<int> mnFeaturesPerLevel;       // 预先分配给每一层的特征点数量
    std::vector<float> mvScaleFactor;          // 每一层的尺度因子：scaleFactor^i
    std::vector<float> mvInvScaleFactor;       // 每一层尺度因子的倒数：1 / scaleFactor^i
    std::vector<float> mvLevelSigma2;          // 每一层尺度方差：scaleFactor^(2*i)
    std::vector<float> mvInvLevelSigma2;       // 每一层尺度方差的倒数：1 / scaleFactor^(2*i)

    std::vector<cv::Point> pattern;            // 预定义好的 BRIEF 描述子点对采样模板（共 512 个点，组成 256 组比对点对）
    std::vector<int> umax;                     // 圆形邻域的边界表，用于灰度质心法（IC_Angle）快速确定关键点的主方向
};

#endif // ORBEXTRACTOR_H