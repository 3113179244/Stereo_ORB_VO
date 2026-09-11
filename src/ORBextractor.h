#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

/**
 * @brief 四叉树节点类：用于管理特征点在图像空间上的均匀分布
 * 
 * 在 ORB 提取中，为了防止特征点扎堆在纹理丰富的区域，
 * 会通过四叉树将图像划分为多个区域，确保特征点在整张图像上分布均匀。
 */
class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}
    
    // 将当前节点对应的图像区域一分为四（左上、右上、左下、右下四个子节点）
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;        // 当前节点区域内包含的候选特征点集合
    cv::Point2f UL, UR, BL, BR;             // 节点对应图像区域的四个顶点坐标（Up-Left, Up-Right, Bottom-Left, Bottom-Right）
    std::list<ExtractorNode>::iterator lit; // 指向该节点在全局链表中位置的迭代器，方便快速删除
    bool bNoMore;                           // 标记该节点是否不可再分割（例如该区域内仅剩1个或0个特征点）
};

/**
 * @brief ORB 特征提取器主类
 */
class ORBextractor
{
public:
    // 特征点响应值计算方法的枚举（目前主要使用 FAST 角点自带的响应值，或者 Harris 响应值）
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    /**
     * @brief 构造函数：初始化 ORB 提取器参数与图像金字塔层级信息
     * @param nfeatures     期望提取的特征点总数 N
     * @param scaleFactor   金字塔层间缩放因子（通常取 1.2）
     * @param nlevels       金字塔总层数（通常取 8）
     * @param iniThFAST     初始 FAST 角点检测阈值（较严格，用于提取高质量角点，通常为 20）
     * @param minThFAST     最小 FAST 角点检测阈值（较宽松，当初始阈值提取不到角点时作为退化策略使用，通常为 7）
     */
    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
    ~ORBextractor() = default;

    /**
     * @brief 重载 () 运算符（仿函数）：完成特征提取的核心接口
     * @param image        输入单目图像矩阵
     * @param mask         掩码矩阵（指定只在特定区域提取特征，通常为空）
     * @param keypoints    输出提取到的关键点集合
     * @param descriptors  输出计算得到的 ORB 描述子矩阵，维度为 N x 32 (格式为 CV_8UC1，每行 32 字节即 256 bits)
     */
    void operator()(cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    // Getters 接口
    int GetLevels() const { return nlevels; }
    float GetScaleFactor() const { return scaleFactor; }
    const std::vector<float>& GetScaleFactors() const { return mvScaleFactor; }
    const std::vector<float>& GetInverseScaleFactors() const { return mvInvScaleFactor; }
    const std::vector<float>& GetScaleSigmaSquares() const { return mvLevelSigma2; }
    const std::vector<float>& GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }
    const std::vector<cv::Mat> &GetImagePyramid() const { return mvImagePyramid; }

private:
    // 构建图像金字塔（生成多尺度图像以实现特征的尺度不变性）
    void ComputePyramid(const cv::Mat& image); 
    
    // 遍历金字塔各层，提取 FAST 角点并将其分配给四叉树进行均匀化
    void ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints); 
    
    // 使用四叉树分割算法，剔除冗余点，确保最终保留的特征点在空间上分布均匀
    std::vector<cv::KeyPoint> DistributeOctree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                               const int &minX, const int &maxX,
                                               const int &minY, const int &maxY,
                                               const int &nFeatures, const int &level);

    // 使用灰度质心法（Intensity Centroid）计算特征点的主方向（用于实现旋转不变性）
    void ComputeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);

    int nfeatures;        // 期望提取的总特征点数
    float scaleFactor;    // 金字塔层间缩放比例 (比如 1.2)
    int nlevels;          // 金字塔总层数
    int iniThFAST;        // 初始 FAST 阈值
    int minThFAST;        // 备用 FAST 阈值

    std::vector<cv::Mat> mvImagePyramid;       // 存储每层金字塔的图像数据
    std::vector<int> mnFeaturesPerLevel;       // 每层金字塔按面积比例分配的期望特征点数量
    std::vector<float> mvScaleFactor;          // 每层相对于第 0 层的累计缩放因子
    std::vector<float> mvInvScaleFactor;       // 缩放因子的倒数（用于将金字塔坐标投影回第 0 层）
    std::vector<float> mvLevelSigma2;          // 尺度方差（缩放因子的平方，表示信息量缩减比例）
    std::vector<float> mvInvLevelSigma2;       // 尺度方差的倒数（用于信息矩阵/协方差的权重分配）

    std::vector<cv::Point> pattern;            // 用于计算 BRIEF 描述子的随机采样点对模式（共 256 对，即 512 个点）
    std::vector<int> umax;                     // 用于计算灰度质心时，半径为 R 的圆域内每一行（不同 v 坐标）对应的最大 u 坐标偏移量，形成一个圆形掩码边界
};

#endif // ORBEXTRACTOR_H