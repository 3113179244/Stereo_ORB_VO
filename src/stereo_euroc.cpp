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
        if (line.empty()) continue;
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

    // 2. 读取配置文件中的双目立体标定矩阵并生成 Remap 映射表
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

    // 补回映射矩阵声明并初始化极线校正映射表 (CV_32F)
    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3),
                                cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);
    std::cout << "[EuRoC] 极线校正与去畸变映射表 (initUndistortRectifyMap) 初始化完成。" << std::endl;

    // 确保转换为 CV_64F (double)，防止 at<double> 产生数据错位
    P_l.convertTo(P_l, CV_64F);
    P_r.convertTo(P_r, CV_64F);

    // 4. 覆盖静态 Config 为校正后的真实内参
    Config::g_dFx = P_l.at<double>(0, 0);
    Config::g_dFy = P_l.at<double>(1, 1);
    Config::g_dCx = P_l.at<double>(0, 2);
    Config::g_dCy = P_l.at<double>(1, 2);
    
    // 标准极线校正模型中：P_r(0, 3) = -fx * baseline
    Config::g_dBf = std::fabs(P_r.at<double>(0, 3));

    // 图像已由 remap 去除畸变，畸变参数必须全部置为 0
    Config::g_dK1 = 0.0;
    Config::g_dK2 = 0.0;
    Config::g_dP1 = 0.0;
    Config::g_dP2 = 0.0;

    std::cout << "[EuRoC] 校正后生效的内参参数: \n"
              << "  fx: " << Config::g_dFx << ", fy: " << Config::g_dFy << "\n"
              << "  cx: " << Config::g_dCx << ", cy: " << Config::g_dCy << "\n"
              << "  bf: " << Config::g_dBf << " (Baseline: " << (Config::g_dBf / Config::g_dFx) << " m)\n"
              << "  k1, k2, p1, p2 均已重置为 0" << std::endl;
              
    System SLAM(strConfigFile, strVocFile, System::STEREO, true);

    bool bIsPaused = false;

    // 5. 循环处理每一帧
    for (int nFrameId = 0; nFrameId < nImages; ++nFrameId)
    {
        while (bIsPaused)
        {
            char cKey = static_cast<char>(cv::waitKey(10));
            if (cKey == ' ' || cKey == 'q' || cKey == 'Q')
            {
                bIsPaused = false;
                std::cout << "\r[状态] 恢复播放...                      " << std::flush;
            }
            else if (cKey == 27) break;
        }

        cv::Mat image0 = cv::imread(vstrLeft[nFrameId], cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread(vstrRight[nFrameId], cv::IMREAD_GRAYSCALE);
        if (image0.empty() || image1.empty())
        {
            std::cerr << "\n错误: 图像读取失败: " << vstrLeft[nFrameId] << std::endl;
            break;
        }

        // 双目极线校正与去畸变 (Rectification)
        cv::Mat imLeftRect, imRightRect;
        cv::remap(image0, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(image1, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        double dCurrentTimestamp = vTimestamps[nFrameId];
        auto t_start = std::chrono::steady_clock::now();

        // 跟踪处理
        SLAM.TrackStereo(imLeftRect, imRightRect, dCurrentTimestamp);

        cv::Mat imDraw = SLAM.DrawFrame();
        if (!imDraw.empty())
        {
            cv::imshow("ORB-SLAM2 Frame Drawer", imDraw);
        }

        auto t_end = std::chrono::steady_clock::now();
        double dTrackElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

        // 真实速率休眠同步
        double dWaitTimeSec = 0.0;
        if (nFrameId + 1 < nImages)
        {
            double dDeltaTime = vTimestamps[nFrameId + 1] - dCurrentTimestamp;
            dWaitTimeSec = dDeltaTime - dTrackElapsed;
        }

        if (dWaitTimeSec > 0.0)
        {
            std::this_thread::sleep_for(std::chrono::duration<double>(dWaitTimeSec));
        }

        char cKey = static_cast<char>(cv::waitKey(1));
        if (cKey == 27)
        {
            std::cout << "\n按下 ESC 键退出..." << std::endl;
            break;
        }
        else if (cKey == ' ')
        {
            bIsPaused = true;
            std::cout << "\r[状态] 已暂停 (按 Space 或 Q 继续)... " << std::flush;
        }
    }

    cv::destroyAllWindows();

    // 6. 保存轨迹 (EuRoC 数据集通常使用 TUM 格式便于与 ground truth 对齐评估)
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