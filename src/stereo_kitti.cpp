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
#include "Viewer.h"
#include "Map.h"
#include "LocalMapping.h"
int main(int argc, char **argv)
{
    // 设置默认路径
    std::string strVocFile = "/home/wzj/DBow3/orbvoc.dbow3";
    std::string strConfigFile = "/home/wzj/Stereo_ORB_VO/config/KITTI04-12.yaml";
    std::string strSequenceDir = "/home/wzj/KITTI/data_odometry_gray/dataset/sequences/07";

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
    std::cout << "  - ESC 键: 退出程序" << std::endl;

    System SLAM(strConfigFile, strVocFile, System::STEREO, true);
    int nFrameId = 0;

    // 循环处理每一帧
    while (true)
    {
        // 检查是否到达末尾
        if (nFrameId >= static_cast<int>(vdTimestamps.size()))
        {
            std::cout << "\n已到达序列末尾，播放结束。共处理 " << nFrameId << " 帧。" << std::endl;
            break;
        }

        std::stringstream ssFilename;
        ssFilename << std::setw(6) << std::setfill('0') << nFrameId << ".png";
        std::string strFilename = ssFilename.str();

        std::string strLeftImgPath = strLeftDir + strFilename;
        std::string strRightImgPath = strRightDir + strFilename;

        if (!cv::utils::fs::exists(strLeftImgPath) || !cv::utils::fs::exists(strRightImgPath))
        {
            std::cout << "\n图像文件不存在，播放结束。共处理 " << nFrameId << " 帧。" << std::endl;
            break;
        }

        double dCurrentTimestamp = vdTimestamps[nFrameId];

        // 1. 记录本帧开始处理的实际时间点
        auto t_frame_start = std::chrono::steady_clock::now();

        // 2. 读取图像
        cv::Mat image0 = cv::imread(strLeftImgPath, cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread(strRightImgPath, cv::IMREAD_GRAYSCALE);

        if (image0.empty() || image1.empty())
        {
            std::cerr << "错误: 无法读取图像: " << strFilename << std::endl;
            break;
        }

        // 3. 执行 SLAM 跟踪
        SLAM.TrackStereo(image0, image1, dCurrentTimestamp);

        // 4. 计算当前帧与下一帧理论上的时间差
        double dExpectedInterval = 0.1; // 默认 10Hz
        if (nFrameId + 1 < static_cast<int>(vdTimestamps.size()))
        {
            dExpectedInterval = vdTimestamps[nFrameId + 1] - dCurrentTimestamp;
        }
        else if (nFrameId > 0)
        {
            dExpectedInterval = dCurrentTimestamp - vdTimestamps[nFrameId - 1];
        }

        nFrameId++;

        // 5. 按键检测（仅响应 ESC 退出）
        char cKey = static_cast<char>(cv::waitKey(1));
        if (cKey == 27) // ESC
        {
            std::cout << "\n按下 ESC，退出程序。" << std::endl;
            break;
        }

        // 6. 控制帧率休眠
        auto t_frame_end = std::chrono::steady_clock::now();
        double dActualElapsed = std::chrono::duration<double>(t_frame_end - t_frame_start).count();
        if (dActualElapsed < dExpectedInterval)
        {
            std::this_thread::sleep_for(std::chrono::duration<double>(dExpectedInterval - dActualElapsed));
        }
    }

    cv::destroyAllWindows();

    std::cout << "\n[System] 数据流输入完成，正在等待后台线程收敛..." << std::endl;

    // 1. 等待 LocalMapping 队列中的待处理关键帧清空
    while (SLAM.GetLocalMapper() && SLAM.GetLocalMapper()->KeyframesInQueue() > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // 2. 如果后台有全局 BA (GBA) 正在运行，等待其收敛回写地图
    if (SLAM.GetLoopCloser())
    {
        while (SLAM.GetLoopCloser()->isRunningGBA())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    std::string strTrajDir = "/home/wzj/output";
    cv::utils::fs::createDirectories(strTrajDir);
    std::string strTrajFile = strTrajDir + "/CameraTrajectory.txt";
    SLAM.SaveTrajectoryKITTI(strTrajFile);

    if (SLAM.GetMap())
    {
        unsigned long nKFs = SLAM.GetMap()->GetKeyFramesInMap();
        unsigned long nMPs = SLAM.GetMap()->GetMapPointsInMap();
        std::cout << "\n------- 系统运行统计 -------" << std::endl;
        std::cout << "关键帧数量 (KeyFrames): " << nKFs << std::endl;
        std::cout << "地图点数量 (MapPoints): " << nMPs << std::endl;
        std::cout << "----------------------------\n"
                  << std::endl;
    }

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