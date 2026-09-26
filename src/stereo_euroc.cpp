#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "LocalMapping.h"
#include "System.h"
#include "Viewer.h"
#include "Map.h"

// 加载 EuRoC 图像路径和时间戳 (按 data.csv 读取)
bool LoadImagesEuRoC(const std::string &strSeqDir,
                     std::vector<std::string> &vstrLeft,
                     std::vector<std::string> &vstrRight,
                     std::vector<double> &vTimestamps)
{
    std::string strPathLeft = strSeqDir + "cam0/data/";
    std::string strPathRight = strSeqDir + "cam1/data/";
    std::string strCsvPath = strSeqDir + "cam0/data.csv";

    std::ifstream fileCsv(strCsvPath);
    if (!fileCsv.is_open())
    {
        std::cerr << "错误: 无法打开 EuRoC data.csv 文件: " << strCsvPath << std::endl;
        return false;
    }

    std::string line;
    std::getline(fileCsv, line); // 跳过表头

    while (std::getline(fileCsv, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string sTimestamp, sImgName;
        std::getline(ss, sTimestamp, ',');
        std::getline(ss, sImgName, ',');

        while (!sImgName.empty() && (sImgName.back() == '\r' || sImgName.back() == '\n' || sImgName.back() == ' '))
            sImgName.pop_back();

        vstrLeft.push_back(strPathLeft + sImgName);
        vstrRight.push_back(strPathRight + sImgName);
        // 纳秒 ns 转换为 秒 s
        vTimestamps.push_back(std::stod(sTimestamp) * 1e-9);
    }
    fileCsv.close();
    return !vstrLeft.empty();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "用法: ./stereo_euroc <词袋文件路径> <配置文件路径> <EuRoC序列mav0路径>" << std::endl;
        return -1;
    }

    std::string strVocFile = argv[1];
    std::string strConfigFile = argv[2];
    std::string strSequenceDir = argv[3];

    if (!strSequenceDir.empty() && strSequenceDir.back() != '/' && strSequenceDir.back() != '\\')
    {
        strSequenceDir += "/";
    }

    // 1. 加载图像列表与时间戳
    std::vector<std::string> vstrLeft, vstrRight;
    std::vector<double> vTimestamps;
    if (!LoadImagesEuRoC(strSequenceDir, vstrLeft, vstrRight, vTimestamps))
    {
        std::cerr << "错误: 读取 EuRoC 图像列表失败！请确认数据集路径指向 mav0 目录。" << std::endl;
        return -1;
    }
    const int nImages = vstrLeft.size();
    std::cout << "[EuRoC] 成功加载 " << nImages << " 帧立体图像。" << std::endl;

    // 2. 读取配置文件中的双目立体标定矩阵并生成 Remap 映射表 (与 ORB-SLAM2 官方保持一致)
    cv::FileStorage fs(strConfigFile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "错误: 无法打开配置文件: " << strConfigFile << std::endl;
        return -1;
    }

    cv::Mat K_l, K_r, D_l, D_r, R_l, R_r, P_l, P_r;
    fs["LEFT.K"] >> K_l;
    fs["RIGHT.K"] >> K_r;
    fs["LEFT.D"] >> D_l;
    fs["RIGHT.D"] >> D_r;
    fs["LEFT.R"] >> R_l;
    fs["RIGHT.R"] >> R_r;
    fs["LEFT.P"] >> P_l;
    fs["RIGHT.P"] >> P_r;

    int cols_l = fs["LEFT.width"];
    int rows_l = fs["LEFT.height"];
    int cols_r = fs["RIGHT.width"];
    int rows_r = fs["RIGHT.height"];

    if (cols_l == 0 || rows_l == 0)
    {
        cols_l = fs["Camera.width"];
        rows_l = fs["Camera.height"];
        cols_r = cols_l;
        rows_r = rows_l;
    }

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() ||
        R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty())
    {
        std::cerr << "错误: EuRoC 配置文件中缺少极线校正相关标定矩阵 (LEFT.K/P/R/D 等)！" << std::endl;
        return -1;
    }

    K_l.convertTo(K_l, CV_32F);
    K_r.convertTo(K_r, CV_32F);
    D_l.convertTo(D_l, CV_32F);
    D_r.convertTo(D_r, CV_32F);
    R_l.convertTo(R_l, CV_32F);
    R_r.convertTo(R_r, CV_32F);
    P_l.convertTo(P_l, CV_32F);
    P_r.convertTo(P_r, CV_32F);

    // 初始化极线校正映射表 (CV_32F)
    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);
    std::cout << "[EuRoC] 极线校正与去畸变映射表 (initUndistortRectifyMap) 初始化完成。" << std::endl;
    fs.release();

    System SLAM(strConfigFile, strVocFile, System::STEREO, true);

    std::cout << "  - ESC 键: 退出程序" << std::endl;

    // 循环处理每一帧
    for (int nFrameId = 0; nFrameId < nImages; ++nFrameId)
    {
        double dCurrentTimestamp = vTimestamps[nFrameId];

        // 1. 记录本帧开始处理的实际时间点
        auto t_frame_start = std::chrono::steady_clock::now();

        // 2. 图像读取与极线校正 (Remap)
        cv::Mat image0 = cv::imread(vstrLeft[nFrameId], cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread(vstrRight[nFrameId], cv::IMREAD_GRAYSCALE);
        if (image0.empty() || image1.empty())
        {
            std::cerr << "\n错误: 图像读取失败: " << vstrLeft[nFrameId] << std::endl;
            break;
        }

        cv::Mat imLeftRect, imRightRect;
        cv::remap(image0, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(image1, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        // 3. 送入 SLAM 跟踪
        SLAM.TrackStereo(imLeftRect, imRightRect, dCurrentTimestamp);

        // 4. 计算与下一帧的理论时间差（EuRoC 约 20Hz，即 0.05s）
        double dExpectedInterval = 0.05;
        if (nFrameId + 1 < nImages)
        {
            dExpectedInterval = vTimestamps[nFrameId + 1] - dCurrentTimestamp;
        }
        else if (nFrameId > 0)
        {
            dExpectedInterval = dCurrentTimestamp - vTimestamps[nFrameId - 1];
        }

        // 5. 按键检测（仅响应 ESC 退出）
        char cKey = static_cast<char>(cv::waitKey(1));
        if (cKey == 27)
        {
            std::cout << "\n按下 ESC 键退出..." << std::endl;
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

    // 5. 保存轨迹 (EuRoC 数据集使用 TUM 格式便于与 ground truth 对齐评估)
    std::string strTrajDir = "/home/wzj/output";
    cv::utils::fs::createDirectories(strTrajDir);
    SLAM.SaveTrajectoryTUM(strTrajDir + "/CameraTrajectory.txt");

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