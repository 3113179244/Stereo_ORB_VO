#include "Viewer.h"
#include "Tracker.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Config.h"
#include "System.h"

#include <opencv2/highgui/highgui.hpp>
#include <GL/gl.h>
#include <set>
#include <vector>
#include <unistd.h>

Viewer::Viewer(System *pSystem, std::shared_ptr<Map> pMap, std::shared_ptr<FrameDrawer> pFrameDrawer)
    : mpSystem(pSystem),
      mpMap(pMap),
      mpFrameDrawer(pFrameDrawer),
      mpTracker(nullptr),
      mbStopRequested(false),
      mbStopped(false),
      mbFinishRequested(false),
      mbFinished(false)
{
    // 从 Config 读取帧率与图像展示延时
    mFPS = Config::g_dFps > 0.0 ? Config::g_dFps : 30.0;
    mT = 1000.0 / mFPS;
    mCameraPose = Eigen::Matrix4f::Identity();

    // 加载配置文件中定义的 Viewer 渲染尺寸参数
    mKeyFrameSize      = Config::g_dViewerKeyFrameSize > 0.0 ? static_cast<float>(Config::g_dViewerKeyFrameSize) : 0.05f;
    mKeyFrameLineWidth = Config::g_dViewerKeyFrameLineWidth > 0.0 ? static_cast<float>(Config::g_dViewerKeyFrameLineWidth) : 1.0f;
    mGraphLineWidth    = Config::g_dViewerGraphLineWidth > 0.0 ? static_cast<float>(Config::g_dViewerGraphLineWidth) : 0.9f;
    mPointSize         = Config::g_dViewerPointSize > 0.0 ? static_cast<float>(Config::g_dViewerPointSize) : 2.0f;
    mCameraSize        = Config::g_dViewerCameraSize > 0.0 ? static_cast<float>(Config::g_dViewerCameraSize) : 0.08f;
    mCameraLineWidth   = Config::g_dViewerCameraLineWidth > 0.0 ? static_cast<float>(Config::g_dViewerCameraLineWidth) : 3.0f;

    // 观察视点参数
    mViewpointX = static_cast<float>(Config::g_dViewerPointX);
    mViewpointY = static_cast<float>(Config::g_dViewerPointY);
    mViewpointZ = static_cast<float>(Config::g_dViewerPointZ);
    mViewpointF = static_cast<float>(Config::g_dViewerPointF);

    if (mViewpointF < 1.0f)
    {
        mViewpointX = 0.0f;
        mViewpointY = -0.7f;
        mViewpointZ = -1.8f;
        mViewpointF = 500.0f;
    }
}

Viewer::~Viewer() {}

void Viewer::UpdateCurrentCameraPose(const Eigen::Matrix4f &Tcw)
{
    SetCurrentCameraPose(Tcw);
}

void Viewer::Run()
{
    mbFinished = false;
    mbStopped = false;

    // 1. 初始化 Pangolin 窗口
    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 2. 菜单栏面板（对齐 ORB-SLAM2 官方布局）
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
    pangolin::Var<bool> menuReset("menu.Reset", false, false);

    // 3. 观察相机设置
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    if (mpFrameDrawer)
    {
        cv::namedWindow("ORB-SLAM2: Current Frame");
    }

    bool bFollow = true;

    while (1)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 获取当前相机位姿矩阵
        GetCurrentOpenGLCameraMatrix(Twc);

        // 视角跟踪逻辑
        if (menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if (menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if (!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景

        // 渲染 3D 当前相机
        DrawCurrentCamera(Twc);

        // 渲染关键帧与共视图拓扑
        if (menuShowKeyFrames || menuShowGraph)
            DrawKeyFrames(menuShowKeyFrames, menuShowGraph);

        // 渲染所有地图点与局部参考地图点
        if (menuShowPoints)
            DrawMapPoints();

        pangolin::FinishFrame();

        // 渲染 2D 当前帧特征与跟踪状态
        if (mpFrameDrawer)
        {
            cv::Mat im = mpFrameDrawer->DrawFrame();
            if (!im.empty())
            {
                cv::imshow("ORB-SLAM2: Current Frame", im);
                cv::waitKey(mT);
            }
        }

        // Reset 控制
        if (menuReset)
        {
            menuShowGraph = true;
            menuShowKeyFrames = true;
            menuShowPoints = true;
            bFollow = true;
            menuFollowCamera = true;
            if (mpTracker)
                mpTracker->Reset();
            menuReset = false;
        }

        if (Stop())
        {
            while (isStopped())
            {
                usleep(3000);
            }
        }

        if (CheckFinish())
            break;
            
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    if (mpFrameDrawer)
    {
        cv::destroyWindow("ORB-SLAM2: Current Frame");
    }
    
    SetFinish();
}

void Viewer::DrawMapPoints()
{
    if (!mpMap) return;

    const std::vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();
    const std::vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints();

    std::set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if (vpMPs.empty()) return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);

    // 1. 绘制全局普通地图点（黑色）
    glColor3f(0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        MapPoint *pMP = vpMPs[i];
        if (!pMP || pMP->isBad() || spRefMPs.count(pMP))
            continue;

        Eigen::Vector3f pos = pMP->GetWorldPos();
        glVertex3f(pos.x(), pos.y(), pos.z());
    }

    // 2. 绘制当前局部参考地图点（红色高亮）
    glColor3f(1.0f, 0.0f, 0.0f);
    for (MapPoint *pMP : vpRefMPs)
    {
        if (!pMP || pMP->isBad())
            continue;

        Eigen::Vector3f pos = pMP->GetWorldPos();
        glVertex3f(pos.x(), pos.y(), pos.z());
    }

    glEnd();
}

void Viewer::DrawKeyFrames(bool bDrawKF, bool bDrawGraph)
{
    if (!mpMap) return;

    const std::vector<KeyFrame *> &vpKFs = mpMap->GetAllKeyFrames();

    // 1. 绘制关键帧蓝色视锥体
    if (bDrawKF)
    {
        const float w = mKeyFrameSize;
        const float h = w * 0.75f;
        const float z = w * 0.6f;

        glLineWidth(mKeyFrameLineWidth);
        glColor3f(0.0f, 0.0f, 1.0f); // 蓝色视锥

        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (!pKF || pKF->mbBad) continue;

            Eigen::Matrix4f Twc = pKF->GetPoseInverse();

            glPushMatrix();
            glMultMatrixf(Twc.data());

            glBegin(GL_LINES);
            glVertex3f(0, 0, 0); glVertex3f(w, h, z);
            glVertex3f(0, 0, 0); glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0); glVertex3f(-w, h, z);

            glVertex3f(w, h, z);  glVertex3f(w, -h, z);
            glVertex3f(w, -h, z); glVertex3f(-w, -h, z);
            glVertex3f(-w, -h, z);glVertex3f(-w, h, z);
            glVertex3f(-w, h, z); glVertex3f(w, h, z);
            glEnd();

            glPopMatrix();
        }
    }

    // 2. 绘制共视图拓扑
    if (bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);

        // (a) 生成树 Spanning Tree（绿色，父子连接）
        glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
        glBegin(GL_LINES);
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (!pKF || pKF->mbBad) continue;

            KeyFrame *pParent = pKF->GetParent();
            if (pParent && !pParent->mbBad)
            {
                Eigen::Vector3f O1 = pKF->GetCameraCenter();
                Eigen::Vector3f O2 = pParent->GetCameraCenter();
                glVertex3f(O1.x(), O1.y(), O1.z());
                glVertex3f(O2.x(), O2.y(), O2.z());
            }
        }
        glEnd();

        // (b) 高权重共视图边 Covisibility Graph（青色半透明，权重 >= 100）
        glColor4f(0.0f, 0.7f, 0.7f, 0.3f);
        glBegin(GL_LINES);
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (!pKF || pKF->mbBad) continue;

            Eigen::Vector3f Ow = pKF->GetCameraCenter();
            const std::vector<KeyFrame *> vCovKFs = pKF->GetCovisibleByWeight(100);

            for (KeyFrame *pCovKF : vCovKFs)
            {
                if (!pCovKF || pCovKF->mbBad || pCovKF->mnId < pKF->mnId)
                    continue;

                Eigen::Vector3f Ow2 = pCovKF->GetCameraCenter();
                glVertex3f(Ow.x(), Ow.y(), Ow.z());
                glVertex3f(Ow2.x(), Ow2.y(), Ow2.z());
            }
        }
        glEnd();
    }
}

void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &M)
{
    const float w = mCameraSize;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glPushMatrix();
    glMultMatrixd(M.m);

    glLineWidth(mCameraLineWidth);
    glColor3f(1.0f, 0.0f, 0.0f); // 红色当前相机

    glBegin(GL_LINES);
    glVertex3f(0, 0, 0); glVertex3f(w, h, z);
    glVertex3f(0, 0, 0); glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, h, z);

    glVertex3f(w, h, z);  glVertex3f(w, -h, z);
    glVertex3f(w, -h, z); glVertex3f(-w, -h, z);
    glVertex3f(-w, -h, z);glVertex3f(-w, h, z);
    glVertex3f(-w, h, z); glVertex3f(w, h, z);
    glEnd();

    glPopMatrix();
}

void Viewer::SetCurrentCameraPose(const Eigen::Matrix4f &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexCamera);
    mCameraPose = Tcw;
}

void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    Eigen::Matrix4f Twc;
    {
        std::unique_lock<std::mutex> lock(mMutexCamera);
        Twc = mCameraPose.inverse();
    }

    M.m[0]  = Twc(0, 0);
    M.m[4]  = Twc(0, 1);
    M.m[8]  = Twc(0, 2);
    M.m[12] = Twc(0, 3);
    M.m[1]  = Twc(1, 0);
    M.m[5]  = Twc(1, 1);
    M.m[9]  = Twc(1, 2);
    M.m[13] = Twc(1, 3);
    M.m[2]  = Twc(2, 0);
    M.m[6]  = Twc(2, 1);
    M.m[10] = Twc(2, 2);
    M.m[14] = Twc(2, 3);
    M.m[3]  = Twc(3, 0);
    M.m[7]  = Twc(3, 1);
    M.m[11] = Twc(3, 2);
    M.m[15] = Twc(3, 3);
}

void Viewer::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    std::unique_lock<std::mutex> lock2(mMutexFinish);

    if (mbFinishRequested)
        return false;
    else if (mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;
}

void Viewer::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopped = false;
}

void Viewer::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}