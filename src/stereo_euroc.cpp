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

// 加载 EuRoC 图像路径和时间戳
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
        // 纳秒 ns 转 秒 s
        vTimestamps.push_back(std::stod(sTimestamp) * 1e-9);
    }
    fileCsv.close();
    return !vstrLeft.empty();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "用法: ./stereo_euroc <词袋文件> <配置文件> <EuRoC数据集目录 (如 .../mav0/)>" << std::endl;
        return -1;
    }

    std::string strVocFile = argv[1];
    std::string strConfigFile = argv[2];
    std::string strSequenceDir = argv[3];

    if (!strSequenceDir.empty() && strSequenceDir.back() != '/' && strSequenceDir.back() != '\\')
    {
        strSequenceDir += "/";
    }

    // 1. 加载图像列表
    std::vector<std::string> vstrLeft, vstrRight;
    std::vector<double> vTimestamps;
    if (!LoadImagesEuRoC(strSequenceDir, vstrLeft, vstrRight, vTimestamps))
    {
        std::cerr << "错误: 读取 EuRoC 图像列表失败！" << std::endl;
        return -1;
    }
    const int nImages = vstrLeft.size();
    std::cout << "[EuRoC] 成功加载 " << nImages << " 帧图像。" << std::endl;

    // 2. 读取标定参数并构建去畸变/极线校正映射表 (Remap)
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

    int width = fs["Camera.width"];
    int height = fs["Camera.height"];
    cv::Size imgSize(width, height);

    if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty())
    {
        std::cerr << "错误: EuRoC 配置文件中缺少标定矩阵参数！" << std::endl;
        return -1;
    }

    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), imgSize, CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), imgSize, CV_32F, M1r, M2r);
    std::cout << "[EuRoC] 极线校正与去畸变映射表初始化完成。" << std::endl;

    // 3. 初始化 SLAM 系统
    System SLAM(strConfigFile, strVocFile, System::STEREO, true);
    bool bIsPaused = false;

    // 4. 逐帧处理
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

        // 双目极线校正与去畸变
        cv::Mat imLeftRect, imRightRect;
        cv::remap(image0, imLeftRect, M1l, M2l, cv::INTER_LINEAR);
        cv::remap(image1, imRightRect, M1r, M2r, cv::INTER_LINEAR);

        double dCurrentTimestamp = vTimestamps[nFrameId];
        auto t_start = std::chrono::steady_clock::now();

        SLAM.TrackStereo(imLeftRect, imRightRect, dCurrentTimestamp);

        cv::Mat imDraw = SLAM.DrawFrame();
        if (!imDraw.empty())
        {
            cv::imshow("ORB-SLAM2 Frame Drawer", imDraw);
        }

        auto t_end = std::chrono::steady_clock::now();
        double dTrackElapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

        // 真实速率同步
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
        if (cKey == 27) break;
        else if (cKey == ' ')
        {
            bIsPaused = true;
            std::cout << "\r[状态] 已暂停 (按 Space 继续)... " << std::flush;
        }
    }

    cv::destroyAllWindows();

    // 5. 保存 TUM 格式轨迹 (方便与 EuRoC groundtruth 评估)
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