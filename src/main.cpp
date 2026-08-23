#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "System.h"
#include "Config.h"
#include "Viewer.h"
int main(int argc, char **argv)
{
    // 设置默认路径
    std::string strVocFile = "/home/wzj/DBow3/orbvoc.dbow3";
    std::string strConfigFile = "/home/wzj/Stereo_ORB_VO/config/KITTI04-12.yaml";
    std::string strSequenceDir = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07";

    // 按照 ./Stereo_ORB_VO <VocFile> <ConfigFile> <SequenceDir> 解析命令行参数
    if (argc >= 2)
    {
        strVocFile = argv[1];
    }
    if (argc >= 3)
    {
        strConfigFile = argv[2];
    }
    if (argc >= 4) 
    {
        strSequenceDir = argv[3];
    }

    // 确保序列路径末尾有 '/'
    if (!strSequenceDir.empty() && strSequenceDir.back() != '/' && strSequenceDir.back() != '\\')
    {
        strSequenceDir += "/";
    }

    std::string strLeftDir = strSequenceDir + "image_0/";
    std::string strRightDir = strSequenceDir + "image_1/";
    std::string strTimesPath = strSequenceDir + "times.txt";

    // 检查路径
    if (!cv::utils::fs::exists(strLeftDir) || !cv::utils::fs::exists(strRightDir))
    {
        std::cerr << "错误: 找不到路径 " << strLeftDir << " 或 " << strRightDir << " ！" << std::endl;
        return -1;
    }

    // 加载时间戳文件
    std::vector<double> vdTimestamps;
    std::ifstream fileTimes(strTimesPath);
    if (!fileTimes.is_open())
    {
        std::cerr << "错误: 无法打开时间戳文件 " << strTimesPath << std::endl;
        return -1;
    }

    double dTimestamp = 0.0;
    while (fileTimes >> dTimestamp)
    {
        vdTimestamps.push_back(dTimestamp);
    }
    fileTimes.close();

    std::cout << "成功加载 " << vdTimestamps.size() << " 个时间戳。" << std::endl;
    std::cout << "  - 空格键 (Space): 暂停/恢复播放" << std::endl;
    std::cout << "  - Q 键           : 恢复播放" << std::endl;
    std::cout << "  - ESC 键         : 退出程序" << std::endl;

    System SLAM(strConfigFile, strVocFile, System::STEREO, true);
    int nFrameId = 0;
    bool bIsPaused = false;

    // 循环处理每一帧
    while (true)
    {
        std::stringstream ssFilename;
        ssFilename << std::setw(6) << std::setfill('0') << nFrameId << ".png";
        std::string strFilename = ssFilename.str();

        std::string strLeftImgPath = strLeftDir + strFilename;
        std::string strRightImgPath = strRightDir + strFilename;

        // 检查文件是否存在以及是否到达末尾
        if (!cv::utils::fs::exists(strLeftImgPath) || !cv::utils::fs::exists(strRightImgPath) || nFrameId >= static_cast<int>(vdTimestamps.size()))
        {
            std::cout << "\n已到达序列末尾，播放结束。共处理 " << nFrameId << " 帧。" << std::endl;
            break;
        }

        if (!bIsPaused)
        {
            cv::Mat image0 = cv::imread(strLeftImgPath, cv::IMREAD_GRAYSCALE);
            cv::Mat image1 = cv::imread(strRightImgPath, cv::IMREAD_GRAYSCALE);

            if (image0.empty() || image1.empty())
            {
                std::cerr << "错误: 无法读取图像: " << strFilename << std::endl;
                break;
            }

            double dCurrentTimestamp = vdTimestamps[nFrameId];

            // 记录当前帧跟踪开始时间
            auto t_start = std::chrono::steady_clock::now();

            Eigen::Matrix4f Tcw = SLAM.TrackStereo(image0, image1, dCurrentTimestamp);

            if (!image0.empty() && !image1.empty())
            {
                cv::Mat imDraw = SLAM.DrawFrame();
                if (!imDraw.empty())
                {
                    cv::imshow("ORB-SLAM2 Frame Drawer", imDraw);
                }
            }

            // 记录当前帧跟踪结束时间并计算处理耗时 (单位: 秒)
            auto t_end = std::chrono::steady_clock::now();
            double dTrackElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

            // 计算与下一帧之间的时间间隔，保持真实时间速率喂图
            double dWaitTimeSec = 0.0;
            if (nFrameId + 1 < static_cast<int>(vdTimestamps.size()))
            {
                double dNextTimestamp = vdTimestamps[nFrameId + 1];
                double dDeltaTime = dNextTimestamp - dCurrentTimestamp;
                dWaitTimeSec = dDeltaTime - dTrackElapsed;
            }

            // 若处理耗时小于两帧真实间隔，则休眠补足差值，为后端 LocalMapping 与 LoopClosing 留出处理时间
            if (dWaitTimeSec > 0.0)
            {
                std::this_thread::sleep_for(std::chrono::duration<double>(dWaitTimeSec));
            }

            nFrameId++;
        }

        int nWaitTime = bIsPaused ? 10 : 1;
        char cKey = static_cast<char>(cv::waitKey(nWaitTime));

        if (cKey == 27) // ESC
        {
            std::cout << "\n按下 ESC，退出程序。" << std::endl;
            break;
        }
        else if (cKey == ' ') // Space
        {
            bIsPaused = !bIsPaused;
            if (bIsPaused)
                std::cout << "\r[状态] 已暂停播放 (按 Space/Q 键继续)... " << std::flush;
            else
                std::cout << "\r[状态] 恢复播放...                      " << std::flush;
        }
        else if (cKey == 'q' || cKey == 'Q')
        {
            if (bIsPaused)
            {
                bIsPaused = false;
                std::cout << "\r[状态] 恢复播放...                      " << std::flush;
            }
        }
    }

    cv::destroyAllWindows();
    std::string strTrajDir = "/home/wzj/output";
    cv::utils::fs::createDirectories(strTrajDir); 
    std::string strTrajFile = strTrajDir + "/CameraTrajectory.txt";
    SLAM.SaveTrajectoryKITTI(strTrajFile);
    if (SLAM.GetViewer())
    {
        while (!SLAM.GetViewer()->isFinished())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    SLAM.Shutdown();
    return 0;
}