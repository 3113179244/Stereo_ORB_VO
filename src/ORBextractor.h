#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
    ~ORBextractor() = default;

    void operator()(cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    int GetLevels() const { return nlevels; }
    float GetScaleFactor() const { return scaleFactor; }
    const std::vector<float>& GetScaleFactors() const { return mvScaleFactor; }
    const std::vector<float>& GetInverseScaleFactors() const { return mvInvScaleFactor; }
    const std::vector<float>& GetScaleSigmaSquares() const { return mvLevelSigma2; }
    const std::vector<float>& GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }
    const std::vector<cv::Mat>& GetImagePyramid() const { return mvImagePyramid; }

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                               const int &minX, const int &maxX,
                                               const int &minY, const int &maxY,
                                               const int &N, const int &level);

    int nfeatures;
    float scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<int> mnFeaturesPerLevel;
    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<cv::Point> pattern;
    std::vector<int> umax;
};

#endif // ORBEXTRACTOR_H