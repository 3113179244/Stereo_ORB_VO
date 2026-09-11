#include "Tracker.h"
#include "Config.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FrameDrawer.h"
#include <algorithm>
#include <iostream>
#include "ORBmatcher.h"
#include "MotionOnlyBA.h"
#include "LocalMapping.h"
#include "KeyFrameDatabase.h"
#include "Viewer.h"
#include "Frame.h"
Tracker::Tracker(System *pSys, ORBVocabulary *pVoc, KeyFrameDatabase *pKFDB, std::shared_ptr<Map> pMap, System::eSensor sensor)
    : mpSystem(pSys), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB), mpMap(pMap), mState(NO_IMAGES_YET), mVelocity(Eigen::Matrix4f::Identity()), mpReferenceKF(nullptr), mpLocalMapper(nullptr), mnLastRelocFrameId(0)
{
    // 从 Config 类中加载 ORB 提取器参数
    int nFeatures = Config::g_nORBnFeatures;
    float fScaleFactor = Config::g_dORBscaleFactor;
    int nLevels = Config::g_nORBnLevels;
    int finiThFAST = Config::g_nORBiniThFAST;
    int fminThFAST = Config::g_nORBminThFAST;

    // 初始化左右图 ORB 提取器
    mpORBextractorLeft = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, finiThFAST, fminThFAST);
    mpORBextractorRight = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, finiThFAST, fminThFAST);
}

Tracker::~Tracker() {}

Eigen::Matrix4f Tracker::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft.clone();

    // 构建内参矩阵与畸变矩阵
    cv::Mat K = (cv::Mat_<float>(3, 3) << Config::g_dFx, 0, Config::g_dCx,
                 0, Config::g_dFy, Config::g_dCy,
                 0, 0, 1);
    cv::Mat DistCoef = (cv::Mat_<float>(4, 1) << Config::g_dK1, Config::g_dK2, Config::g_dP1, Config::g_dP2);

    mCurrentFrame = Frame(imRectLeft.clone(), imRectRight.clone(), timestamp,
                          mpORBextractorLeft.get(), mpORBextractorRight.get(),
                          mpORBVocabulary, K, DistCoef, Config::g_dBf, Config::g_dThDepth);

    // 执行跟踪状态机主逻辑
    Track();

    // 返回当前帧姿态
    return mCurrentFrame.mTcw;
}

void Tracker::Track()
{
    // Step 0: 状态转换
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // 地图全局更新互斥锁
    std::unique_lock<std::mutex> lockMap(mpMap->mMutexMapUpdate);

    // Step 1: 初始化
    if (mState == NOT_INITIALIZED)
    {
        StereoInitialization();

        if (mpFrameDrawer)
            mpFrameDrawer->Update(this);
        
        if (mState != OK)
            return;
    }
    else
    {
        // 系统已初始化，开始跟踪
        bool bOK = false;

        // Step 2: 跟踪上一帧 / 恒速模型 / 参考关键帧 / 重定位 粗位姿估计
        if (mState == OK)
        {
            // Step 2.1: 检查并更新上一帧中被 LocalMapping 替换的地图点
            CheckReplacedInLastFrame();

            // Step 2.2: 判断使用运动模型还是参考关键帧
            // 若速度模型为空或刚完成重定位，跟踪参考关键帧；否则使用恒速模型
            if (mVelocity.isIdentity() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            {
                bOK = TrackReferenceKeyFrame();
            }
            else
            {
                bOK = TrackWithMotionModel();
                if (!bOK)
                {
                    // 恒速模型失败，回退到参考关键帧跟踪
                    bOK = TrackReferenceKeyFrame();
                }
            }
        }
        else
        {
            // 处于 LOST 状态，执行重定位
            bOK = Relocalize();
        }

        // 设置当前帧的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // Step 3: 跟踪局部地图 (TrackLocalMap)
        if (bOK)
        {
            bOK = TrackLocalMap();
        }

        // 根据局部地图跟踪结果决定最终状态
        if (bOK)
            mState = OK;
        else
            mState = LOST;

        // Step 4: 更新显示与绘制线程
        if (mpFrameDrawer)
            mpFrameDrawer->Update(this);

        // Step 5: 跟踪成功后的状态更新与关键帧决策
        if (bOK)
        {
            // Step 5.1: 更新恒速模型速度 mVelocity = Tcl = Tcw * Twl
            if (!mLastFrame.mTcw.isZero())
            {
                Eigen::Matrix4f LastTwc = Eigen::Matrix4f::Identity();
                LastTwc.block<3, 3>(0, 0) = mLastFrame.GetRotationInverse();
                LastTwc.block<3, 1>(0, 3) = mLastFrame.GetCameraCenter();
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
            {
                mVelocity.setIdentity();
            }

            // Step 5.2: 更新 Viewer 相机位姿
            if (mpViewer)
                mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);

            // Step 5.3: 清除当前帧中没有被关键帧有效观测的地图点 (无效立体匹配点)
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (pMP->GetObservations().empty())
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = nullptr;
                    }
                }
            }

            // Step 5.4: 检测并创建新的关键帧
            if (NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }

            // Step 5.5: 剔除当前帧中被判定为 Outlier 的地图点
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
                }
            }
        }

        // Step 6: 跟踪丢失处理 (刚初始化不久若丢失且关键帧 <= 5 则重置系统)
        if (mState == LOST)
        {
            if (mpMap && mpMap->GetKeyFramesInMap() <= 5)
            {
                std::cout << "[Tracking] 初始化不久即跟丢，执行系统重置..." << std::endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // Step 7: 保存上一帧 (当前帧转为上一帧)
        mLastFrame = Frame(mCurrentFrame);
    }

    // Step 8: 记录位姿轨迹信息 (用于最终导出轨迹)
    if (!mCurrentFrame.mTcw.isZero() && !mCurrentFrame.mTcw.hasNaN())
    {
        Eigen::Matrix4f Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        if (!mlRelativeFramePoses.empty())
        {
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(true);
        }
        else
        {
            mlRelativeFramePoses.push_back(Eigen::Matrix4f::Identity());
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(true);
        }
    }
}

bool Tracker::StereoInitialization()
{
    // 1. 特征点数量门槛（ORB-SLAM2 官方标准为 500 点）
    if (mCurrentFrame.N <= 500)
        return false;

    // 2. 设定初始位姿为世界坐标系原点 (单位矩阵)
    mCurrentFrame.SetPose(Eigen::Matrix4f::Identity());

    // 3. 创建初始关键帧并插入全局地图与词袋数据库
    KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKFini);

    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKFini);

    // 4. 为每个具有正深度且在测量范围内的特征点创建 MapPoint 并关联
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0.0f) // 官方只要 z > 0 即生成地图点，近远点判定由后续模块处理
        {
            Eigen::Vector3f x3D = mCurrentFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap.get());

            // 建立特征点与关键帧的双向观测
            pNewMP->AddObservation(pKFini, i);
            pKFini->AddMapPoint(pNewMP, i);

            // 计算代表性描述子、平均观测方向与深度范围
            pNewMP->ComputeDistinctiveDescriptor();
            pNewMP->UpdateNormalAndDepth();

            // 注册入全局地图并关联至当前普通帧
            mpMap->AddMapPoint(pNewMP);
            mCurrentFrame.mvpMapPoints[i] = pNewMP;
        }
    }

    std::cout << "[Initialization] New stereo map created with " 
              << mpMap->GetMapPointsInMap() << " points" << std::endl;

    // 5. 将初始关键帧推送给局部建图线程
    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKFini);
    }

    // 6. 更新追踪上下文（对齐 ORB-SLAM2 官方核心状态）
    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    // 7. 更新局部关键帧与局部地图点
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();

    // 8. 同步局部参考点给 Map 与 Viewer 可视化
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    if (mpViewer)
    {
        mpViewer->UpdateCurrentCameraPose(mCurrentFrame.mTcw);
    }

    // 9. 显式置位状态
    mState = OK;
    return true;
}

bool Tracker::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9f, true);

    // 1. 基于恒速模型预测位姿初值
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

    // 2. 双目基础搜索窗口为 7
    int th = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, false);

    // 3. 若匹配过少，扩大 2 倍窗口重新搜索
    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, false);
    }

    if (nmatches < 20)
        return false;

    // 4. 执行位姿优化 (Motion-only BA，内部已含 4 轮核函数外点剔除)
    int num_inliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 5. 剔除被判定为 Outlier 的地图点
    for (int i = 0; i < mCurrentFrame.N; ++i)
    {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCurrentFrame.mvbOutlier[i] = false;
        }
    }

    // 6. 原版判定标准：内点数 >= 10 即可交付给 TrackLocalMap
    return num_inliers >= 10;
}

bool Tracker::TrackReferenceKeyFrame()
{
    // Step 1: 将当前帧描述子转化为 BoW 向量加速匹配
    mCurrentFrame.ComputeBoW();

    // Step 2: 通过词袋向量加速当前帧与参考关键帧之间的特征匹配
    ORBmatcher matcher(0.7f, true);
    std::vector<MapPoint *> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    // 粗匹配门槛：少于 15 对直接判定失败
    if (nmatches < 15)
        return false;

    // Step 3: 将参考关键帧匹配到的地图点绑定到当前帧，初值继承上一帧位姿加速收敛
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // Step 4: 执行位姿优化 (Motion-Only BA)
    MotionOnlyBA::Optimize(&mCurrentFrame);

    // Step 5: 根据优化结果剔除外点，并严格统计有效地图点的内点数
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mnVisible++; // 或重置观测状态
                nmatches--;
            }
            else if (pMP->GetObservations().size() > 0)
            {
                nmatchesMap++;
            }
        }
    }

    mnMatchesInliers = nmatchesMap;

    // Step 6: 有效内点数 >= 10 判定跟踪成功
    return nmatchesMap >= 10;
}

bool Tracker::Relocalize()
{
    if (mCurrentFrame.mBowVec.empty())
        mCurrentFrame.ComputeBoW();

    if (!mpKeyFrameDB)
    {
        std::cout << "  └─ [Relocalize] KeyFrameDB 为空！" << std::endl;
        return false;
    }

    // 利用 KeyFrameDatabase 寻找候选关键帧
    std::vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (vpCandidateKFs.empty())
    {
        std::cout << "  └─ [Relocalize] 未检索到词袋候选帧" << std::endl;
        return false;
    }

    std::cout << "  └─ [Relocalize] 检索到 " << vpCandidateKFs.size() << " 个候选帧，开始 PnP 匹配..." << std::endl;

    ORBmatcher matcher(0.75, true);

    for (KeyFrame *pKF : vpCandidateKFs)
    {
        if (!pKF || pKF->mbBad)
            continue;

        std::vector<MapPoint *> vpMapPointMatches;
        int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vpMapPointMatches);

        if (nmatches < 8)
            continue;

        std::vector<cv::Point3f> vPts3D;
        std::vector<cv::Point2f> vPts2D;
        std::vector<int> vMapPointIndices;

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            MapPoint *pMP = vpMapPointMatches[i];
            if (pMP && !pMP->isBad())
            {
                Eigen::Vector3f P3D = pMP->GetWorldPos();
                vPts3D.push_back(cv::Point3f(P3D.x(), P3D.y(), P3D.z()));
                vPts2D.push_back(mCurrentFrame.mvKeysUn[i].pt);
                vMapPointIndices.push_back(i);
            }
        }

        if (vPts3D.size() < 8)
            continue;

        cv::Mat rvec, tvec;
        std::vector<int> inliersPnP;

        bool bPnPSuccess = cv::solvePnPRansac(
            vPts3D, vPts2D,
            mCurrentFrame.mK, cv::Mat(),
            rvec, tvec,
            false, 300, 8.0f, 0.99, inliersPnP, cv::SOLVEPNP_EPNP);

        if (!bPnPSuccess || inliersPnP.size() < 6)
            continue;

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix4f Tcw_pnp = Eigen::Matrix4f::Identity();
        for (int r = 0; r < 3; r++)
        {
            Tcw_pnp(r, 3) = static_cast<float>(tvec.at<double>(r));
            for (int c = 0; c < 3; c++)
            {
                Tcw_pnp(r, c) = static_cast<float>(R_cv.at<double>(r, c));
            }
        }

        mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));
        for (int inlierIdx : inliersPnP)
        {
            int frameIdx = vMapPointIndices[inlierIdx];
            mCurrentFrame.mvpMapPoints[frameIdx] = vpMapPointMatches[frameIdx];
        }

        mCurrentFrame.SetPose(Tcw_pnp);

        int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

        if (nInliers >= 8)
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i] = nullptr;
                    mCurrentFrame.mvbOutlier[i] = false;
                }
            }

            mpReferenceKF = pKF;
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mnMatchesInliers = nInliers;
            mnLastRelocFrameId = mCurrentFrame.mnId;
            std::cout << "  └─ [Relocalize SUCCESS] 重定位成功恢复！参考 KF ID: " << pKF->mnId
                      << " | 匹配点数: " << nInliers << std::endl;
            return true;
        }
    }

    std::cout << "  └─ [Relocalize FAILED] PnP / BA 校验均未通过" << std::endl;
    return false;
}

/**
 * @brief 局部地图跟踪顶层控制与成功门槛
 */
bool Tracker::TrackLocalMap()
{
    // 1. 更新局部关键帧与局部地图点
    UpdateLocalMap();

    // 2. 投影匹配局部地图点
    SearchLocalPoints();

    // 3. 位姿优化 (Motion-Only BA)
    int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 4. 更新内点标记并剔除外点
    mnMatchesInliers = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
            }
            else
            {
                mnMatchesInliers++;
            }
        }
    }

    // 5. 将当前局部地图点同步到 Map 供可视化渲染
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // ORB-SLAM2 官方标准门槛：局部地图跟踪成功判定内点数至少为 30 (保底宽松门槛不低于 15~30)
    return mnMatchesInliers >= 30;
}

bool Tracker::NeedNewKeyFrame()
{
    // 1. 如果 LocalMapping 线程被停止（例如闭环时锁定），禁止插入关键帧
    if (mpLocalMapper && (mpLocalMapper->isStopped() || mpLocalMapper->GetStopRequired()))
        return false;

    const int nKFs = mpMap ? mpMap->GetKeyFramesInMap() : 0;

    // 2. 距离上一次重定位太近且关键帧已有一定积累时，不插入关键帧
    // （若工程中暂无重定位帧记录，此项默认通过）
    const int mMaxFrames = static_cast<int>(Config::g_dFps > 0.0 ? Config::g_dFps : 20.0);
    const int mMinFrames = 0;

    // 3. 统计参考关键帧跟踪到的稳定地图点数量 (nRefMatches)
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;

    int nRefMatches = 0;
    if (mpReferenceKF && !mpReferenceKF->mbBad)
    {
        nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    }
    if (nRefMatches <= 0)
        nRefMatches = 1; // 避免除零

    // 4. 查询 LocalMapping 是否处于空闲状态
    bool bLocalMappingIdle = mpLocalMapper ? mpLocalMapper->AcceptKeyFrames() : true;

    // 5. 双目专属逻辑：统计近点（Close Points）跟踪状况
    int nNonTrackedClose = 0; // 当前帧中存在有效深度但尚未绑定地图点的近点
    int nTrackedClose = 0;    // 当前帧中已成功跟踪并内点匹配的近点

    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        // 深度大于0且小于远近点阈值视为近点
        if (mCurrentFrame.mvDepth[i] > 0.0f && mCurrentFrame.mvDepth[i] < mCurrentFrame.mThDepth)
        {
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                nTrackedClose++;
            else
                nNonTrackedClose++;
        }
    }

    // 双目特有决策：已跟踪近点稀疏(<100)且未跟踪近点充足(>70)，必须立即建关键帧补充地图点
    bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

    // 6. 决策阈值设置
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f;

    const int nFramesPassed = mCurrentFrame.mnId - mnLastKeyFrameId;

    // 条件 1a: 距离上一关键帧已超过最大帧数 (如 1 秒未插帧)
    const bool c1a = nFramesPassed >= mMaxFrames;

    // 条件 1b: 经过了最小间隔帧数且 LocalMapping 空闲
    const bool c1b = (nFramesPassed >= mMinFrames && bLocalMappingIdle);

    // 条件 1c: 跟踪显著变弱 (内点少于参考帧的 25%) 或 满足双目急需近点条件
    const bool c1c = (mnMatchesInliers < nRefMatches * 0.25f) || bNeedToInsertClose;

    // 条件 2: 匹配点相比参考帧明显减少(或需要近点)，且当前内点数满足最低跟踪要求 (>15)
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio) || bNeedToInsertClose) && (mnMatchesInliers > 15);

    // 7. 综合决策与 LocalMapping 队列控制
    if ((c1a || c1b || c1c) && c2)
    {
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            if (mpLocalMapper)
            {
                // 仅在队列未满且确定要插帧时打断
                if (mpLocalMapper->KeyframesInQueue() < 3)
                {
                    mpLocalMapper->RequestStopBA();
                    return true;
                }
                else
                    return false;
            }
            return false;
        }
    }

    return false;
}

void Tracker::CreateNewKeyFrame()
{
    // 如果局部建图器处于停止状态且无法阻止其停止，则不插入
    if (mpLocalMapper && !mpLocalMapper->SetNotStop())
        return;

    // Step 1: 创建新关键帧并与当前帧绑定
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap.get());
    mpReferenceKF = pKF;

    // Step 2: 双目/RGB-D 专属建点逻辑（按深度排序，保底 100 个近点）
    std::vector<std::pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0.0f)
        {
            vDepthIdx.push_back(std::make_pair(z, i));
        }
    }

    if (!vDepthIdx.empty())
    {
        // 深度从小到大排序，优先处理近点
        std::sort(vDepthIdx.begin(), vDepthIdx.end());

        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

            if (!pMP)
            {
                bCreateNew = true;
            }
            else if (pMP->GetObservations().size() < 1)
            {
                // 地图点已无有效观测，重置并重新创建
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }

            if (bCreateNew)
            {
                // 反投影生成世界坐标系下的 3D 点
                Eigen::Vector3f x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap.get());

                pNewMP->AddObservation(pKF, i);
                pKF->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptor();
                pNewMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                nPoints++;
            }
            else
            {
                // 若该点已被成功跟踪且不是外点，只需追加本关键帧的观测
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    pMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pMP, i);
                }
                nPoints++;
            }

            // 停止条件：深度超过阈值且已处理近点数达到 100 个以上
            if (vDepthIdx[j].first > mCurrentFrame.mThDepth && nPoints > 100)
                break;
        }
    }

    // Step 3: 将关键帧送入各个后端处理模块
    if (mpKeyFrameDB)
        mpKeyFrameDB->add(pKF);

    if (mpLocalMapper)
    {
        mpLocalMapper->InsertKeyFrame(pKF);
        mpLocalMapper->Release(); // 允许 LocalMapping 恢复正常状态
    }

    mnLastKeyFrameId = mCurrentFrame.mnId;
}

void Tracker::Reset()
{
    mState = NOT_INITIALIZED;
    mVelocity.setIdentity();
    mpReferenceKF = nullptr;
    mnLastRelocFrameId = 0;
    // 清空历史轨迹记录
    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
}

void Tracker::UpdateLastFrame()
{
    if (!mpReferenceKF || mpReferenceKF->mbBad)
        return;

    // 获取上一帧相对于参考关键帧的相对位姿: T_lr = T_lw * T_rw^-1
    if (!mlRelativeFramePoses.empty() && mlpReferences.back() == mpReferenceKF)
    {
        Eigen::Matrix4f Tlr = mlRelativeFramePoses.back();
        // 依据参考关键帧最新可能已被 LocalMapping 优化过的位姿，重新计算上一帧的绝对位姿
        mLastFrame.SetPose(Tlr * mpReferenceKF->GetPose());
    }
}

void Tracker::ResetVelocity()
{
    // 将恒速模型速度矩阵重置为单位矩阵，下一帧将退化为基于上一帧重投影搜索
    mVelocity.setIdentity();
}

/**
 * @brief 调度更新局部地图
 */
void Tracker::UpdateLocalMap()
{
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 搜集局部关键帧 (当前帧观测帧 + 一级共视邻居 + 二级共视邻居)
 */
/**
 * @brief 搜集局部关键帧 (当前帧观测帧 + 一级共视前 K 个邻居 + 二级共视邻居)
 */
void Tracker::UpdateLocalKeyFrames()
{
    mvpLocalKeyFrames.clear();

    // 1. 统计当前帧所有已匹配地图点的所有有效观测关键帧
    std::map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && !pMP->isBad())
        {
            const std::map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (auto mit = observations.begin(); mit != observations.end(); ++mit)
            {
                if (!mit->first->mbBad)
                    keyframeCounter[mit->first]++;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int maxObs = 0;
    KeyFrame *pKFmax = nullptr;
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // 2. 加入当前帧直接观测到的关键帧，并选出共视点最多的作为参考关键帧
    for (auto &mit : keyframeCounter)
    {
        KeyFrame *pKF = mit.first;
        if (!pKF->mbBad)
        {
            mvpLocalKeyFrames.push_back(pKF);
            if (mit.second > maxObs)
            {
                maxObs = mit.second;
                pKFmax = pKF;
            }
        }
    }

    if (pKFmax)
        mpReferenceKF = pKFmax;

    // 3. 扩充一级共视邻居 (ORB-SLAM2 标准：前 10 个最佳共视邻居)、二级共视邻居 (前 5 个)、生成树子/父节点
    std::vector<KeyFrame *> vpLocalKFWithNeighbors = mvpLocalKeyFrames;
    for (KeyFrame *pKF : mvpLocalKeyFrames)
    {
        if (pKF->mbBad)
            continue;

        // (a) 一级最佳共视前 10 个邻居
        const std::vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for (KeyFrame *pN : vNeighs)
        {
            if (pN && !pN->mbBad)
            {
                vpLocalKFWithNeighbors.push_back(pN);
                // (b) 二级最佳共视前 5 个邻居
                const std::vector<KeyFrame *> vSecondNeighs = pN->GetBestCovisibilityKeyFrames(5);
                for (KeyFrame *p2N : vSecondNeighs)
                {
                    if (p2N && !p2N->mbBad)
                        vpLocalKFWithNeighbors.push_back(p2N);
                }
            }
        }

        // (c) 生成树子节点与父节点
        const std::set<KeyFrame *> spChildren = pKF->GetChilds();
        for (KeyFrame *pChild : spChildren)
        {
            if (pChild && !pChild->mbBad)
                vpLocalKFWithNeighbors.push_back(pChild);
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent && !pParent->mbBad)
            vpLocalKFWithNeighbors.push_back(pParent);
    }

    // 4. 排序与去重
    std::sort(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end());
    vpLocalKFWithNeighbors.erase(
        std::unique(vpLocalKFWithNeighbors.begin(), vpLocalKFWithNeighbors.end()),
        vpLocalKFWithNeighbors.end());

    mvpLocalKeyFrames = vpLocalKFWithNeighbors;
}

/**
 * @brief 搜集局部地图点 (从局部关键帧中提取并去重)
 */
void Tracker::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (KeyFrame *pKF : mvpLocalKeyFrames)
    {
        if (!pKF || pKF->mbBad)
            continue;

        std::vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        for (MapPoint *pMP : vpMPs)
        {
            if (!pMP || pMP->isBad())
                continue;

            // 避免重复收集
            if (std::find(mvpLocalMapPoints.begin(), mvpLocalMapPoints.end(), pMP) == mvpLocalMapPoints.end())
            {
                mvpLocalMapPoints.push_back(pMP);
            }
        }
    }
}

/**
 * @brief 投影搜索局部地图点
 */
void Tracker::SearchLocalPoints()
{
    if (mvpLocalMapPoints.empty())
        return;

    // 1. 过滤掉当前帧中已经成功匹配且为有效内点（Inlier）的地图点
    std::vector<MapPoint *> vpCandidateMPs;
    vpCandidateMPs.reserve(mvpLocalMapPoints.size());

    for (MapPoint *pMP : mvpLocalMapPoints)
    {
        if (!pMP || pMP->isBad())
            continue;

        bool bAlreadyTracked = false;
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            // 核心修复：只有当指针匹配且不是 Outlier 时，才视作已跟踪成功
            if (mCurrentFrame.mvpMapPoints[i] == pMP && !mCurrentFrame.mvbOutlier[i])
            {
                bAlreadyTracked = true;
                break;
            }
        }

        if (!bAlreadyTracked)
        {
            vpCandidateMPs.push_back(pMP);
        }
    }

    if (vpCandidateMPs.empty())
        return;

    // 2. 局部地图匹配设置 nnratio = 0.8f，开启旋转一致性校验，搜索半径 th = 5.0f
    ORBmatcher matcher(0.8f, true);
    matcher.SearchByProjection(mCurrentFrame, vpCandidateMPs, 5.0f);
}

void Tracker::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}