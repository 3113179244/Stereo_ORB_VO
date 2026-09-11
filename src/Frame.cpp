#include "Frame.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include <thread>
#include <cmath>
#include "Config.h"
#include <algorithm>
#include <limits>
#include "MapPoint.h"
long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::fx = 0, Frame::fy = 0, Frame::cx = 0, Frame::cy = 0, Frame::invfx = 0, Frame::invfy = 0;
float Frame::mnMinX = 0, Frame::mnMinY = 0, Frame::mnMaxX = 0, Frame::mnMaxY = 0;
float Frame::mfGridElementWidthInv = 0, Frame::mfGridElementHeightInv = 0;

Frame::Frame() {}

// 双目帧构造函数实现
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
             ORBextractor *extractorLeft, ORBextractor *extractorRight,
             ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    : mTimeStamp(timeStamp), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
      mpORBvocabulary(voc), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // 分配唯一的帧 ID，并递增计数器
    mnId = nNextId++;

    // 多线程并发提取左右图 ORB 特征，以加速前端处理
    mImGrayLeft = imLeft.clone();
    mImGrayRight = imRight.clone();

    // 仅为右图启动一个子线程
    std::thread threadRight(&Frame::ExtractORB, this, 1, mImGrayRight);

    // 当前主线程直接负责提取左图
    ExtractORB(0, mImGrayLeft);

    // 等待右图提取完成
    threadRight.join();

    // 记录特征点总数
    N = mvKeys.size();
    if (mvKeys.empty())
        return;

    // 此处简化了去畸变过程。在实际系统中，若输入图像已极线校正，可直接拷贝 mvKeysUn = mvKeys
    mvKeysUn = mvKeys;
    // mImGrayLeft = imLeft.clone();
    // mImGrayRight = imRight.clone();
    mb = mbf / mK.at<float>(0, 0);
    mThDepth = thDepth * mb;
    // 双目匹配，通过左右目特征点匹配计算视差，进而获得深度信息
    ComputeStereoMatches();

    // 初始化地图点和外点标记数组，大小为特征点总数 N
    mvpMapPoints = std::vector<MapPoint *>(N, nullptr);
    mvbOutlier = std::vector<bool>(N, false);

    // 初始化图像边界和相机内参。因为是静态变量，只需在程序启动时(第一帧)计算一次
    if (mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        // 计算网格宽度和高度的倒数，用于后续将坐标快速映射到网格索引 (乘法比除法快)
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        // 提取相机内参矩阵中的参数
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    // 将特征点划分到网格中，加速局部区域特征匹配搜索
    AssignFeaturesToGrid();
}

// 提取 ORB 特征
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if (flag == 0)
    {
        // 左图：使用左目提取器，提取特征点存入 mvKeys，描述子存入 mDescriptors
        if (mpORBextractorLeft)
        {
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        }
    }
    else
    {
        // 右图：使用右目提取器，结果存入 mvKeysRight 和 mDescriptorsRight
        if (mpORBextractorRight)
        {
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
        }
    }
}

// 设置相机位姿矩阵
void Frame::SetPose(const Eigen::Matrix4f &Tcw)
{
    mTcw = Tcw;
    UpdatePoseMatrices(); // 位姿更新后，同步更新其分解的各个矩阵和向量
}

// 依据最新的 $T_{cw}$ 更新旋转平移矩阵
void Frame::UpdatePoseMatrices()
{
    // 从 4x4 变换矩阵中提取 3x3 旋转矩阵 $R_{cw}$
    mRcw = mTcw.block<3, 3>(0, 0);
    // 计算其转置 (即逆矩阵) $R_{wc}$
    mRwc = mRcw.transpose();
    // 从 4x4 变换矩阵中提取 3x1 平移向量 $t_{cw}$
    mtcw = mTcw.block<3, 1>(0, 3);
    // 计算相机光心在世界坐标系下的 3D 坐标: $O_w = -R_{cw}^T \cdot t_{cw} = -R_{wc} \cdot t_{cw}$
    mOw = -mRwc * mtcw;
}

// 计算双目匹配以获取深度 (此处为框架示意代码)
void Frame::ComputeStereoMatches()
{
    mvuRight = std::vector<float>(N, -1.0f);
    mvDepth = std::vector<float>(N, -1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

    // 金字塔 0 层图像高度
    const int nRows = mpORBextractorLeft->GetScaleFactors().empty() ? mImGrayLeft.rows : mImGrayLeft.rows;
    std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());
    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    // 1. 建立右图特征点的行索引（考虑尺度半径扩展）
    for (int iR = 0; iR < Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f * mpORBextractorLeft->GetScaleFactors()[kp.octave];
        const int maxr = std::ceil(kpY + r);
        const int minr = std::floor(kpY - r);

        for (int yi = minr; yi <= maxr; yi++)
        {
            if (yi >= 0 && yi < nRows)
                vRowIndices[yi].push_back(iR);
        }
    }

    // 2. 视差范围约束
    const float minZ = mb;
    const float minD = 0.0f;
    const float maxD = mbf / minZ;

    std::vector<std::pair<int, int>> vDistIdx;
    vDistIdx.reserve(N);

    // 3. 遍历左图特征点，搜索右图匹配
    for (int iL = 0; iL < N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const int vL_round = cvRound(vL);
        if (vL_round < 0 || vL_round >= nRows)
            continue;

        const std::vector<size_t> &vCandidates = vRowIndices[vL_round];
        if (vCandidates.empty())
            continue;

        const float minU = uL - maxD;
        const float maxU = uL - minD;
        if (maxU < 0.0f)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;
        const cv::Mat &dL = mDescriptors.row(iL);

        // 粗匹配：在极线范围内基于描述子汉明距离找最近邻
        for (size_t iC = 0; iC < vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
                continue;

            const float &uR = kpR.pt.x;
            if (uR >= minU && uR <= maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL, dR);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // 4. 精确匹配：SAD 块滑动窗口与亚像素抛物线拟合
        if (bestDist < thOrbDist)
        {
            const cv::KeyPoint &kpR = mvKeysRight[bestIdxR];
            const float uR0 = kpR.pt.x;

            // 映射到金字塔当前层尺度
            const float scaleFactor = mpORBextractorLeft->GetInverseScaleFactors()[levelL];
            const float scaleduL = cvRound(kpL.pt.x * scaleFactor);
            const float scaledvL = cvRound(kpL.pt.y * scaleFactor);
            const float scaleduR0 = cvRound(uR0 * scaleFactor);

            const int w = 5; // 窗口半宽 (11x11 patch)
            const int L = 5; // 滑动范围 [-L, L]

            // 边界检查：确保左右图当前层 patch 不越界
            // 采用原图与金字塔下采样图尺寸做校验
            const int colsL = cvRound(mImGrayLeft.cols * scaleFactor);
            const int rowsL = cvRound(mImGrayLeft.rows * scaleFactor);
            const int colsR = cvRound(mImGrayRight.cols * scaleFactor);
            const int rowsR = cvRound(mImGrayRight.rows * scaleFactor);

            const int iniu = scaleduR0 - L - w;
            const int endu = scaleduR0 + L + w + 1;

            if (scaleduL - w < 0 || scaleduL + w + 1 > colsL ||
                scaledvL - w < 0 || scaledvL + w + 1 > rowsL ||
                iniu < 0 || endu > colsR)
                continue;

            // 采样左图像块并去均值归一化 (直接在原灰度图按对应缩放比例采样像素，等价于原版金字塔抽取)
            cv::Mat IL(2 * w + 1, 2 * w + 1, CV_32F);
            const float invScale = mpORBextractorLeft->GetScaleFactors()[levelL];
            for (int r = -w; r <= w; ++r)
            {
                int origY = cvRound((scaledvL + r) * invScale);
                origY = std::max(0, std::min(origY, mImGrayLeft.rows - 1));
                const uchar *pL = mImGrayLeft.ptr<uchar>(origY);
                float *pDst = IL.ptr<float>(r + w);

                for (int c = -w; c <= w; ++c)
                {
                    int origX = cvRound((scaleduL + c) * invScale);
                    origX = std::max(0, std::min(origX, mImGrayLeft.cols - 1));
                    pDst[c + w] = static_cast<float>(pL[origX]);
                }
            }
            IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

            float bestSadDist = std::numeric_limits<float>::max();
            int bestincR = 0;
            std::vector<float> vDists(2 * L + 1, 0.0f);

            // 在 [-L, L] 范围内滑动
            for (int incR = -L; incR <= L; incR++)
            {
                cv::Mat IR(2 * w + 1, 2 * w + 1, CV_32F);
                for (int r = -w; r <= w; ++r)
                {
                    int origY = cvRound((scaledvL + r) * invScale);
                    origY = std::max(0, std::min(origY, mImGrayRight.rows - 1));
                    const uchar *pR = mImGrayRight.ptr<uchar>(origY);
                    float *pDst = IR.ptr<float>(r + w);

                    for (int c = -w; c <= w; ++c)
                    {
                        int origX = cvRound((scaleduR0 + incR + c) * invScale);
                        origX = std::max(0, std::min(origX, mImGrayRight.cols - 1));
                        pDst[c + w] = static_cast<float>(pR[origX]);
                    }
                }
                IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

                float sadDist = cv::norm(IL, IR, cv::NORM_L1);
                vDists[L + incR] = sadDist;

                if (sadDist < bestSadDist)
                {
                    bestSadDist = sadDist;
                    bestincR = incR;
                }
            }

            // 极值落在边界则抛弃
            if (bestincR == -L || bestincR == L)
                continue;

            // 抛物线亚像素插值
            const float dist1 = vDists[L + bestincR - 1];
            const float dist2 = vDists[L + bestincR];
            const float dist3 = vDists[L + bestincR + 1];

            const float denom = 2.0f * (dist1 + dist3 - 2.0f * dist2);
            if (std::abs(denom) < 1e-5f)
                continue;

            const float deltaR = (dist1 - dist3) / denom;
            if (deltaR < -1.0f || deltaR > 1.0f)
                continue;

            // 还原回原图尺度
            float bestuR = invScale * (scaleduR0 + bestincR + deltaR);
            float disparity = uL - bestuR;

            if (disparity >= minD && disparity < maxD)
            {
                if (disparity <= 0.0f)
                {
                    disparity = 0.01f;
                    bestuR = uL - 0.01f;
                }

                mvDepth[iL] = mbf / disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(std::make_pair(static_cast<int>(bestSadDist), iL));
            }
        }
    }

    // 5. ORB-SLAM2 官方外点剔除：基于中位数 SAD 过滤
    if (!vDistIdx.empty())
    {
        std::sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for (int i = static_cast<int>(vDistIdx.size()) - 1; i >= 0; i--)
        {
            if (vDistIdx[i].first < thDist)
                break;
            else
            {
                mvuRight[vDistIdx[i].second] = -1.0f;
                mvDepth[vDistIdx[i].second] = -1.0f;
            }
        }
    }
}

// 将指定特征点反投影为 3D 世界坐标
Eigen::Vector3f Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if (z > 0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;

        // 像素坐标转相机坐标：
        // $X_c = \frac{(u - c_x) \cdot Z}{f_x}$
        // $Y_c = \frac{(v - c_y) \cdot Z}{f_y}$
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        // 相机坐标转世界坐标: $P_{world} = R_{wc} \cdot P_{camera} + O_w$
        return mRwc * x3Dc + mOw;
    }
    return Eigen::Vector3f::Zero(); // 如果没有有效的深度值，则返回全零向量
}

// 获取图像有效区域边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 在这里假设图像已去畸变或不需要考虑畸变引起的边界收缩问题，边界即为图像本身尺寸
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
}

// 将图像中的特征点分配到离散网格中
void Frame::AssignFeaturesToGrid()
{
    // 预分配每个网格点的内存空间以避免动态扩容带来的时间开销
    // 假设特征点是均匀分布的，每个网格大约会有 0.5 * N / (Rows * Cols) 个点
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    // 遍历所有的去畸变特征点，计算其对应的网格坐标并存入
    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];
        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i); // 将该特征点的索引加入对应网格
    }
}

// 根据特征点像素坐标，计算对应的网格索引，判断是否在图像网格范围内
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    // 检查是否越界
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

// 快速查找指定区域内所有的特征点索引 (主要用于特征追踪阶段)
std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    // 计算包含搜索圆形的最小和最大网格索引，限定在合法范围内
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

    // 双层循环：仅遍历落入上述边界范围内的网格区域
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> &vCell = mGrid[ix][iy]; // 取出该网格内的所有特征点索引
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                // 判断欧氏距离的近似条件：通过方盒模型初步筛选落入搜索半径 r 的点
                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}

void Frame::ComputeBoW()
{
    if (!mpORBvocabulary)
    {
        std::cerr << "[ERROR] Frame::ComputeBoW(): mpORBvocabulary is nullptr!" << std::endl;
        return;
    }

    if (mDescriptors.empty())
    {
        return;
    }

    if (mBowVec.empty())
    {
        // 转换格式适配 DBoW3
        std::vector<cv::Mat> vCurrentDesc;
        vCurrentDesc.reserve(mDescriptors.rows);
        for (int i = 0; i < mDescriptors.rows; i++)
        {
            vCurrentDesc.push_back(mDescriptors.row(i).clone());
        }

        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

bool Frame::isNear(int i) const
{
    if (i < 0 || i >= N)
        return false;

    // 优先使用当前帧已经算好的双目/RGBD深度值
    float z = mvDepth[i];
    if (z > 0.0f)
    {
        return z < mThDepth;
    }

    // 若当前特征点关联了地图点，但没有直接深度，则通过地图点与当前相机中心计算深度
    if (mvpMapPoints[i])
    {
        return isMapPointNear(mvpMapPoints[i]);
    }

    return false;
}

bool Frame::isFar(int i) const
{
    if (i < 0 || i >= N)
        return false;

    float z = mvDepth[i];
    if (z > 0.0f)
    {
        return z >= mThDepth;
    }

    if (mvpMapPoints[i])
    {
        return isMapPointFar(mvpMapPoints[i]);
    }

    return false;
}

bool Frame::isMapPointNear(MapPoint *pMP) const
{
    if (!pMP || pMP->isBad())
        return false;

    // 计算地图点在当前相机坐标系下的坐标 P_c = R_cw * P_w + t_cw
    Eigen::Vector3f P_w = pMP->GetWorldPos();
    Eigen::Vector3f P_c = mRcw * P_w + mtcw;

    // Z 轴深度大于0且小于远近点阈值即为近点
    return (P_c.z() > 0.0f && P_c.z() < mThDepth);
}

bool Frame::isMapPointFar(MapPoint *pMP) const
{
    if (!pMP || pMP->isBad())
        return false;

    Eigen::Vector3f P_w = pMP->GetWorldPos();
    Eigen::Vector3f P_c = mRcw * P_w + mtcw;

    // Z 轴深度大于等于远近点阈值即为远点
    return (P_c.z() >= mThDepth);
}