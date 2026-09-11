#include "LocalMapping.h"
#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Tracker.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include "LoopClosing.h"
#include <Eigen/SVD>
#include <unistd.h>
#include <algorithm>
#include <cmath>

LocalMapping::LocalMapping(System *pSys, std::shared_ptr<Map> pMap)
    : mpSystem(pSys), mpMap(pMap), mpTracker(nullptr),
      mpCurrentKeyFrame(nullptr), mbStopRequested(false),
      mbStopped(false), mbNotStop(false), mbAcceptKeyFrames(true), mbAbortBA(false)
{
    mpThread = new std::thread(&LocalMapping::Run, this);
}

LocalMapping::~LocalMapping()
{
    if (mpThread)
    {
        mpThread->join();
        delete mpThread;
    }
}

static double ComputeParallax(const Eigen::Vector3f &a, const Eigen::Vector3f &b)
{
    Eigen::Vector3f va = a;
    Eigen::Vector3f vb = b;
    if (va.norm() < 1e-8 || vb.norm() < 1e-8)
        return 0.0;
    double cosA = va.dot(vb) / (va.norm() * vb.norm());
    if (cosA > 1.0)
        cosA = 1.0;
    if (cosA < -1.0)
        cosA = -1.0;
    return acos(cosA) * 180.0 / M_PI;
}

static float CheckDistEpipolarLine(const KeyFrame *pKF1, const KeyFrame *pKF2,
                                   const Eigen::Matrix3f &F12,
                                   const int idx1, const int idx2)
{
    const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
    Eigen::Vector3f p1(kp1.pt.x, kp1.pt.y, 1.0f);
    Eigen::Vector3f p2(kp2.pt.x, kp2.pt.y, 1.0f);

    Eigen::Vector3f l2 = F12 * p1;
    float sq = l2(0) * l2(0) + l2(1) * l2(1);
    if (sq < 1e-10f)
        return 1e6f;
    return std::fabs(p2.dot(l2)) / std::sqrt(sq);
}

static Eigen::Matrix3f ComputeFundamentalMatrix(const Eigen::Matrix3f &R1,
                                                const Eigen::Vector3f &t1,
                                                const Eigen::Matrix3f &R2,
                                                const Eigen::Vector3f &t2,
                                                const float fx, const float fy,
                                                const float cx, const float cy)
{
    Eigen::Matrix3f R12 = R2 * R1.transpose();
    Eigen::Vector3f t12 = -R12 * t1 + t2;

    Eigen::Matrix3f tx;
    tx << 0, -t12(2), t12(1),
        t12(2), 0, -t12(0),
        -t12(1), t12(0), 0;
    Eigen::Matrix3f E = tx * R12;

    Eigen::Matrix3f K;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;
    Eigen::Matrix3f Kinv = K.inverse();

    return Kinv.transpose() * E * Kinv;
}

static bool Triangulate(const Eigen::Matrix3f &R1, const Eigen::Vector3f &t1,
                        const Eigen::Matrix3f &R2, const Eigen::Vector3f &t2,
                        const Eigen::Vector2f &x1, const Eigen::Vector2f &x2,
                        Eigen::Vector3f &x3D)
{
    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
    T1.block<3, 3>(0, 0) = R1;
    T1.block<3, 1>(0, 3) = t1;
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2.block<3, 3>(0, 0) = R2;
    T2.block<3, 1>(0, 3) = t2;

    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    const float u1 = x1(0), v1 = x1(1);
    const float u2 = x2(0), v2 = x2(1);

    A.row(0) = u1 * T1.row(2) - T1.row(0);
    A.row(1) = v1 * T1.row(2) - T1.row(1);
    A.row(2) = u2 * T2.row(2) - T2.row(0);
    A.row(3) = v2 * T2.row(2) - T2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4f Xh = svd.matrixV().col(3);
    if (std::fabs(Xh(3)) < 1e-10f)
        return false;
    x3D = Xh.head<3>() / Xh(3);
    return true;
}

void LocalMapping::Run()
{
    mbStopped = false;
    while (1)
    {
        // 【修改点 1】优先处理停止请求，避免卡死在内部死循环中
        if (GetStopRequired())
        {
            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbStopped = true;
            }

            // 使用 mbStopRequested 作为循环判断条件，配合外部的 Release() 唤醒
            while (GetStopRequired())
            {
                usleep(3000);
            }

            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbStopped = false;
            }
        }

        if (CheckNewKeyFrames())
        {
            SetNotStop();

            // 1. 处理关键帧
            ProcessNewKeyFrame();

            // 2. 考核并剔除劣质地图点
            MapPointCulling();

            // 3. 三角化新地图点
            CreateNewMapPoints();

            // 4. 重复点融合
            if (!CheckNewKeyFrames() && !GetStopRequired())
            {
                SearchInNeighbors();
            }

            // 5. 局部 BA 优化
            if (!CheckNewKeyFrames() && !GetStopRequired())
            {
                mbAbortBA = false;
                Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
            }

            // 6. 剔除冗余关键帧
            KeyFrameCulling();

            if (mpSystem && mpSystem->GetLoopCloser())
            {
                mpSystem->GetLoopCloser()->InsertKeyFrame(mpCurrentKeyFrame);
            }

            {
                std::unique_lock<std::mutex> lock(mMutexStop);
                mbNotStop = false;
            }
        }

        usleep(3000);
    }
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    mlNewKeyFrames.push_back(pKF);
}

bool LocalMapping::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    return !mlNewKeyFrames.empty();
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // 计算词袋向量
    mpCurrentKeyFrame->ComputeBoW();

    std::vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP && !pMP->isBad())
        {
            if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
            {
                pMP->AddObservation(mpCurrentKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptor();
            }
            else
            {
                // 由 Tracking 线程直接创建的立体点，确保其起始关键帧 ID 设为当前关键帧
                pMP->mnFirstKFid = mpCurrentKeyFrame->mnId;
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}
/**
 * @brief 检查新增地图点，根据地图点的观测情况剔除质量不好的新增地图点
 * mlpRecentAddedMapPoints：存储新增的待考核地图点
 */
void LocalMapping::MapPointCulling()
{
    if (mlpRecentAddedMapPoints.empty())
        return;

    auto lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    // 单目为 2，双目 / RGB-D 为 3
    const int cnThObs = 3;

    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;

        if (pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 条件 1: 跟踪比例小于 25% 剔除
        else if (pMP->GetFoundRatio() < 0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 条件 2: 建立已过至少 2 个关键帧，但观测数 <= cnThObs 则判定为劣质点剔除
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 &&
                 static_cast<int>(pMP->GetObservations().size()) <= cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        // 条件 3: 连续存活 3 帧以上，考核通过转正
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
        }
    }
}

static cv::Mat ComputeF12(KeyFrame *pKF1, KeyFrame *pKF2)
{
    Eigen::Matrix3f R1w_eig = pKF1->GetRotation();
    Eigen::Vector3f t1w_eig = pKF1->GetTranslation();
    Eigen::Matrix3f R2w_eig = pKF2->GetRotation();
    Eigen::Vector3f t2w_eig = pKF2->GetTranslation();

    Eigen::Matrix3f R12_eig = R1w_eig * R2w_eig.transpose();
    Eigen::Vector3f t12_eig = -R1w_eig * R2w_eig.transpose() * t2w_eig + t1w_eig;

    cv::Mat R12(3, 3, CV_32F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R12.at<float>(r, c) = R12_eig(r, c);

    cv::Mat t12x = (cv::Mat_<float>(3, 3) << 0.0f, -t12_eig(2), t12_eig(1),
                    t12_eig(2), 0.0f, -t12_eig(0),
                    -t12_eig(1), t12_eig(0), 0.0f);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::CreateNewMapPoints()
{
    // 双目系统固定取权重排名前 10 的共视邻居
    const int nn = 10;
    const std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6f, false);

    Eigen::Matrix3f Rcw1_eig = mpCurrentKeyFrame->GetRotation();
    Eigen::Matrix3f Rwc1_eig = Rcw1_eig.transpose();
    Eigen::Vector3f tcw1_eig = mpCurrentKeyFrame->GetTranslation();
    Eigen::Vector3f Ow1_eig = mpCurrentKeyFrame->GetCameraCenter();

    cv::Mat Rcw1(3, 3, CV_32F), Rwc1(3, 3, CV_32F), tcw1(3, 1, CV_32F), Ow1(3, 1, CV_32F);
    for (int r = 0; r < 3; ++r)
    {
        tcw1.at<float>(r) = tcw1_eig(r);
        Ow1.at<float>(r) = Ow1_eig(r);
        for (int c = 0; c < 3; ++c)
        {
            Rcw1.at<float>(r, c) = Rcw1_eig(r, c);
            Rwc1.at<float>(r, c) = Rwc1_eig(r, c);
        }
    }

    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && CheckNewKeyFrames())
            return;

        KeyFrame *pKF2 = vpNeighKFs[i];
        if (!pKF2 || pKF2->mbBad)
            continue;

        // 双目运动基线检查：相机移动距离小于物理基线时跳过
        Eigen::Vector3f Ow2_eig = pKF2->GetCameraCenter();
        cv::Mat Ow2 = (cv::Mat_<float>(3, 1) << Ow2_eig.x(), Ow2_eig.y(), Ow2_eig.z());
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = cv::norm(vBaseline);
        if (baseline < pKF2->mb)
            continue;

        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        std::vector<std::pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        Eigen::Matrix3f Rcw2_eig = pKF2->GetRotation();
        Eigen::Matrix3f Rwc2_eig = Rcw2_eig.transpose();
        Eigen::Vector3f tcw2_eig = pKF2->GetTranslation();

        cv::Mat Rcw2(3, 3, CV_32F), Rwc2(3, 3, CV_32F), tcw2(3, 1, CV_32F);
        for (int r = 0; r < 3; ++r)
        {
            tcw2.at<float>(r) = tcw2_eig(r);
            for (int c = 0; c < 3; ++c)
            {
                Rcw2.at<float>(r, c) = Rcw2_eig(r, c);
                Rwc2.at<float>(r, c) = Rwc2_eig(r, c);
            }
        }

        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        for (size_t ikp = 0; ikp < vMatchedIndices.size(); ikp++)
        {
            const int idx1 = vMatchedIndices[ikp].first;
            const int idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            const bool bStereo1 = (kp1_ur >= 0.0f);

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            const bool bStereo2 = (kp2_ur >= 0.0f);

            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0f);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0f);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays + 1.0f;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            // 独立评估两帧的双目等效视差角（修复 else-if 的单向遮蔽）
            if (bStereo1)
                cosParallaxStereo1 = cos(2.0f * atan2(mpCurrentKeyFrame->mb / 2.0f, mpCurrentKeyFrame->mvDepth[idx1]));
            if (bStereo2)
                cosParallaxStereo2 = cos(2.0f * atan2(pKF2->mb / 2.0f, pKF2->mvDepth[idx2]));

            cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            // 视差角合适时使用两帧三角化，否则优先选用视差角更大且满足视差阈值的双目反投影点
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0.0f && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998f))
            {
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();
                if (x3D.at<float>(3) == 0.0f)
                    continue;

                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
            }
            else if (bStereo1 && cosParallaxStereo1 < 0.9998f && (cosParallaxStereo1 < cosParallaxStereo2 || !bStereo2))
            {
                Eigen::Vector3f x3D_eig = mpCurrentKeyFrame->UnprojectStereo(idx1);
                x3D = (cv::Mat_<float>(3, 1) << x3D_eig.x(), x3D_eig.y(), x3D_eig.z());
            }
            else if (bStereo2 && cosParallaxStereo2 < 0.9998f)
            {
                Eigen::Vector3f x3D_eig = pKF2->UnprojectStereo(idx2);
                x3D = (cv::Mat_<float>(3, 1) << x3D_eig.x(), x3D_eig.y(), x3D_eig.z());
            }
            else
            {
                continue;
            }

            cv::Mat x3Dt = x3D.t();

            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0.0f)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0.0f)
                continue;

            // 检查当前关键帧重投影误差
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            const float invz1 = 1.0f / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991f * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.815f * sigmaSquare1)
                    continue;
            }

            // 检查相邻关键帧重投影误差（使用 pKF2 对应的 mbf）
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            const float invz2 = 1.0f / z2;

            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991f * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - pKF2->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.815f * sigmaSquare2)
                    continue;
            }

            // 尺度一致性检验
            cv::Mat normal1 = x3D - Ow1;
            float dist1 = cv::norm(normal1);
            cv::Mat normal2 = x3D - Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0.0f || dist2 == 0.0f)
                continue;

            const float ratioDist = dist2 / dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // 创建并注册新地图点
            Eigen::Vector3f x3D_pos(x3D.at<float>(0), x3D.at<float>(1), x3D.at<float>(2));
            MapPoint *pMP = new MapPoint(x3D_pos, mpCurrentKeyFrame, mpMap.get());

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptor();
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}

/**
 * @brief 检查并融合当前关键帧与相邻关键帧（两级相邻）中重复的地图点
 */
void LocalMapping::SearchInNeighbors()
{
    // Step 1: 获得共视图中权重排名前 nn 的一级邻居（双目/RGB-D 取 20，单目取 10）
    const int nn = 20;
    const std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    if (vpNeighKFs.empty())
        return;

    // Step 2: 搜集一级相邻关键帧与二级相邻关键帧 (前 5 个共视邻居)
    std::vector<KeyFrame *> vpTargetKFs;
    vpTargetKFs.reserve(vpNeighKFs.size() * 3);

    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        KeyFrame *pKFi = vpNeighKFs[i];
        if (!pKFi || pKFi->mbBad || pKFi->mnId == mpCurrentKeyFrame->mnId)
            continue;

        if (std::find(vpTargetKFs.begin(), vpTargetKFs.end(), pKFi) == vpTargetKFs.end())
            vpTargetKFs.push_back(pKFi);

        // 扩充二级邻居 (前 5 个共视邻居)
        const std::vector<KeyFrame *> vpSecondNeighs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (size_t j = 0; j < vpSecondNeighs.size(); j++)
        {
            KeyFrame *pKFi2 = vpSecondNeighs[j];
            if (!pKFi2 || pKFi2->mbBad || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            if (std::find(vpTargetKFs.begin(), vpTargetKFs.end(), pKFi2) == vpTargetKFs.end())
                vpTargetKFs.push_back(pKFi2);
        }
    }

    if (vpTargetKFs.empty())
        return;

    // Step 3: 正向融合 —— 将当前关键帧的地图点投影到所有目标关键帧中融合
    std::vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (KeyFrame *pKF : vpTargetKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        const Eigen::Matrix3f Rcw = pKF->GetRotation();
        const Eigen::Vector3f tcw = pKF->GetTranslation();
        const Eigen::Vector3f Ow = pKF->GetCameraCenter();

        for (size_t i = 0; i < vpMapPointMatches.size(); i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if (!pMP || pMP->isBad())
                continue;

            // 视锥与可见性几何校验[cite: 2]
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            Eigen::Vector3f Pc = Rcw * Pw + tcw;
            if (Pc.z() <= 0.0f)
                continue;

            const float dist = (Pw - Ow).norm();
            if (dist < pMP->GetMinDistanceInvariance() * 0.8f || dist > pMP->GetMaxDistanceInvariance() * 1.2f)
                continue;

            Eigen::Vector3f Pn = (Pw - Ow).normalized();
            if (Pn.dot(pMP->GetNormal()) < 0.5f)
                continue;

            const float invz = 1.0f / Pc.z();
            const float u = pKF->fx * Pc.x() * invz + pKF->cx;
            const float v = pKF->fy * Pc.y() * invz + pKF->cy;

            if (u < pKF->mnMinX || u >= pKF->mnMaxX || v < pKF->mnMinY || v >= pKF->mnMaxY)
                continue;

            // 自适应金字塔搜索半径
            float ratio = dist / pMP->GetMaxDistanceInvariance();
            int nPredictedLevel = std::max(0, std::min(static_cast<int>(ratio * 4.0f), 3));
            const float radius = 5.0f * pKF->mvScaleFactors[nPredictedLevel];

            const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);
            if (vIndices.empty())
                continue;

            const cv::Mat &dMP = pMP->GetDescriptor();
            int bestDist = ORBmatcher::TH_LOW;
            int bestIdx = -1;

            for (size_t idx : vIndices)
            {
                const cv::Mat &dF = pKF->mDescriptors.row(idx);
                int distDesc = ORBmatcher::DescriptorDistance(dMP, dF);
                if (distDesc < bestDist)
                {
                    bestDist = distDesc;
                    bestIdx = static_cast<int>(idx);
                }
            }

            if (bestIdx >= 0)
            {
                MapPoint *pMPinKF = pKF->GetMapPoint(bestIdx);
                if (!pMPinKF)
                {
                    pKF->AddMapPoint(pMP, bestIdx);
                    pMP->AddObservation(pKF, bestIdx);
                }
                else if (pMPinKF != pMP)
                {
                    if (pMP->GetObservations().size() >= pMPinKF->GetObservations().size())
                        pMPinKF->Replace(pMP);
                    else
                        pMP->Replace(pMPinKF);
                }
            }
        }
    }

    // Step 4: 反向融合 —— 搜集目标关键帧中的所有地图点，投影到当前帧中融合
    std::set<MapPoint *> sFuseCandidates;
    for (KeyFrame *pKF : vpTargetKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        const std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint *pMP : vpMPs)
        {
            if (pMP && !pMP->isBad())
                sFuseCandidates.insert(pMP);
        }
    }

    const Eigen::Matrix3f RcwCur = mpCurrentKeyFrame->GetRotation();
    const Eigen::Vector3f tcwCur = mpCurrentKeyFrame->GetTranslation();
    const Eigen::Vector3f OwCur = mpCurrentKeyFrame->GetCameraCenter();

    for (MapPoint *pMP : sFuseCandidates)
    {
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3f Pw = pMP->GetWorldPos();
        Eigen::Vector3f Pc = RcwCur * Pw + tcwCur;
        if (Pc.z() <= 0.0f)
            continue;

        const float dist = (Pw - OwCur).norm();
        if (dist < pMP->GetMinDistanceInvariance() * 0.8f || dist > pMP->GetMaxDistanceInvariance() * 1.2f)
            continue;

        Eigen::Vector3f Pn = (Pw - OwCur).normalized();
        if (Pn.dot(pMP->GetNormal()) < 0.5f)
            continue;

        const float invz = 1.0f / Pc.z();
        const float u = mpCurrentKeyFrame->fx * Pc.x() * invz + mpCurrentKeyFrame->cx;
        const float v = mpCurrentKeyFrame->fy * Pc.y() * invz + mpCurrentKeyFrame->cy;

        if (u < mpCurrentKeyFrame->mnMinX || u >= mpCurrentKeyFrame->mnMaxX ||
            v < mpCurrentKeyFrame->mnMinY || v >= mpCurrentKeyFrame->mnMaxY)
            continue;

        float ratio = dist / pMP->GetMaxDistanceInvariance();
        int nPredictedLevel = std::max(0, std::min(static_cast<int>(ratio * 4.0f), 3));
        const float radius = 3.0f * mpCurrentKeyFrame->mvScaleFactors[nPredictedLevel];

        const std::vector<size_t> vIndices = mpCurrentKeyFrame->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty())
            continue;

        const cv::Mat &dMP = pMP->GetDescriptor();
        int bestDist = ORBmatcher::TH_LOW;
        int bestIdx = -1;

        for (size_t idx : vIndices)
        {
            const cv::Mat &dF = mpCurrentKeyFrame->mDescriptors.row(idx);
            int distDesc = ORBmatcher::DescriptorDistance(dMP, dF);
            if (distDesc < bestDist)
            {
                bestDist = distDesc;
                bestIdx = static_cast<int>(idx);
            }
        }

        if (bestIdx >= 0)
        {
            MapPoint *pLocalMP = mpCurrentKeyFrame->GetMapPoint(bestIdx);
            if (!pLocalMP)
            {
                mpCurrentKeyFrame->AddMapPoint(pMP, bestIdx);
                pMP->AddObservation(mpCurrentKeyFrame, bestIdx);
            }
            else if (pLocalMP != pMP)
            {
                if (pMP->GetObservations().size() >= pLocalMP->GetObservations().size())
                    pLocalMP->Replace(pMP);
                else
                    pMP->Replace(pLocalMP);
            }
        }
    }

    // Step 5: 刷新当前帧所有地图点的属性并更新共视连接
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (pMP->isBad())
            {
                mpCurrentKeyFrame->EraseMapPointMatch(i);
            }
            else
            {
                pMP->ComputeDistinctiveDescriptor();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
    for (KeyFrame *pKF : vpTargetKFs)
    {
        if (pKF && !pKF->mbBad)
            pKF->UpdateConnections();
    }
}

/**
 * @brief 检测当前关键帧在共视图中的关键帧，根据地图点在共视图中的冗余程度剔除该共视关键帧
 * 冗余关键帧的判定：90%以上的地图点能被其他关键帧（至少3个）在相同或更优尺度下观测到
 */
void LocalMapping::KeyFrameCulling()
{
    // 获取按共视权重降序排列的共视关键帧列表
    const std::vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (std::vector<KeyFrame *>::const_iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; ++vit)
    {
        KeyFrame *pKF = *vit;
        if (!pKF || pKF->mnId == 0 || pKF->mbBad)
            continue;

        const std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        const int thObs = 3;
        int nRedundantObservations = 0;
        int nMPs = 0;

        for (size_t i = 0; i < vpMapPoints.size(); ++i)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                // 对齐 ORB-SLAM2 官方双目逻辑：跳过深度无效 (<=0) 或超过近点阈值的远点
                const float &z = pKF->mvDepth[i];
                if (z <= 0.0f || z > pKF->mThDepth)
                    continue;

                nMPs++;

                // 地图点总观测数 > 3 才可能冗余
                const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
                if (static_cast<int>(observations.size()) > thObs)
                {
                    const int scaleLevel = pKF->mvKeysUn[i].octave;
                    int nObs = 0;

                    for (auto mit = observations.begin(), mend = observations.end(); mit != mend; ++mit)
                    {
                        KeyFrame *pKFi = mit->first;
                        if (pKFi == pKF || pKFi->mbBad)
                            continue;

                        const size_t idx_i = mit->second;
                        if (idx_i >= pKFi->mvKeysUn.size())
                            continue;

                        const int scaleLeveli = pKFi->mvKeysUn[idx_i].octave;

                        // 尺度条件：在相同或更优尺度层级 (scaleLeveli <= scaleLevel + 1) 下被观测
                        if (scaleLeveli <= scaleLevel + 1)
                        {
                            nObs++;
                            if (nObs >= thObs)
                                break;
                        }
                    }

                    if (nObs >= thObs)
                        nRedundantObservations++;
                }
            }
        }

        // 冗余点超过 90% 则标记剔除
        if (nMPs > 0 && (float)nRedundantObservations > 0.90f * (float)nMPs)
        {
            pKF->SetBadFlag();
        }
    }
}

void LocalMapping::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool LocalMapping::GetStopRequired()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool LocalMapping::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::SetNotStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (mbStopped)
        return false;
    mbNotStop = true;
    return true;
}

// 请求中断当前正在进行的局部 BA
void LocalMapping::RequestStopBA()
{
    mbAbortBA = true;
}

int LocalMapping::KeyframesInQueue()
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    return static_cast<int>(mlNewKeyFrames.size());
}

void LocalMapping::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = false;
    mbStopped = false;
    mbNotStop = false;
}

bool LocalMapping::AcceptKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}