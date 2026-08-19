#pragma once

#include <vector>
#include <list>
#include <set>
#include <mutex>
#include "DBoW3/DBoW3.h"

typedef DBoW3::Vocabulary ORBVocabulary;
class KeyFrame;
class Frame;

class KeyFrameDatabase
{
public:
    // 构造函数：传入 ORB 词典指针
    KeyFrameDatabase(ORBVocabulary* pVoc);
    ~KeyFrameDatabase();

    // 将关键帧加入数据库（倒排索引）
    void add(KeyFrame* pKF);

    // 从数据库中移除关键帧
    void erase(KeyFrame* pKF);

    // 清空数据库
    void clear();

    // 关键接口：检测用于重定位的候选关键帧集合
    std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* pF);
    std::vector<KeyFrame*> DetectLoopCandidates(KeyFrame* pKF, float minScore);
protected:
    // ORB 词典指针
    ORBVocabulary* mpVoc;

    // 倒排索引 (Inverted Index)：
    // 数组索引表示词典中的 Word ID，对应的 vector 存储包含该 Word 的 KeyFrame 列表
    std::vector<std::list<KeyFrame*>> mvInvertedFile;

    // 互斥锁保护倒排索引的并发读写
    std::mutex mMutex;
};