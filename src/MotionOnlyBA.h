// Header File: MotionOnlyBA.h
#ifndef MOTIONONLYBA_H
#define MOTIONONLYBA_H

// 前向声明 Frame 类
class Frame;
class ORBextractor;
/**
 * @brief 位姿优化类（仅优化相机运动/位姿，不优化地图点）
 */
class MotionOnlyBA
{
public:
    /**
     * @brief 仅优化当前帧的位姿 (Pose-Only BA)
     * @param pFrame 当前帧指针
     * @return 优化后的内点 (Inlier) 数量
     */
    static int Optimize(Frame *pFrame);
};

#endif // MOTIONONLYBA_H