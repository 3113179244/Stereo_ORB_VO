#include "KeyFrameDatabase.h"
#include "KeyFrame.h"
#include "Frame.h"
#include <algorithm>

KeyFrameDatabase::KeyFrameDatabase(ORBVocabulary *pVoc) : mpVoc(pVoc)
{
    if (mpVoc)
    {
        // 根据词典的大小预分配倒排索引数组
        mvInvertedFile.resize(mpVoc->size());
    }
}

KeyFrameDatabase::~KeyFrameDatabase()
{
}

void KeyFrameDatabase::add(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);

    // 确保关键帧已经计算了 BoW 向量
    pKF->ComputeBoW();

    // 遍历关键帧的所有 Word，将当前关键帧加入对应 Word 的倒排索引列表中
    for (auto vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++)
    {
        mvInvertedFile[vit->first].push_back(pKF);
    }
}

void KeyFrameDatabase::erase(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);

    // 遍历该关键帧包含的所有 Word，从相应的倒排索引中删除自身
    for (auto vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++)
    {
        mvInvertedFile[vit->first].remove(pKF);
    }
}

void KeyFrameDatabase::clear()
{
    std::unique_lock<std::mutex> lock(mMutex);
    for (size_t i = 0; i < mvInvertedFile.size(); i++)
    {
        mvInvertedFile[i].clear();
    }
}

std::vector<KeyFrame *> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *pF)
{
    std::list<KeyFrame *> lKFsSharingWords;

    // 1. 搜集所有与当前帧共享 BoW Word 的关键帧
    {
        std::unique_lock<std::mutex> lock(mMutex);

        for (auto vit = pF->mBowVec.begin(); vit != pF->mBowVec.end(); vit++)
        {
            const std::list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

            for (KeyFrame *pKFi : lKFs)
            {
                // 用 mnRelocQuery 标记防止在同一查询中重复处理同一个关键帧
                if (pKFi->mnRelocQuery != pF->mnId)
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mnRelocQuery = pF->mnId;
                    lKFsSharingWords.push_back(pKFi);
                }
                pKFi->mnRelocWords++; // 统计共享词汇数量
            }
        }
    }

    if (lKFsSharingWords.empty())
        return std::vector<KeyFrame *>();

    // 2. 筛选共享词汇数量最多的门槛 (至少需要达到最大值的 80%)
    int maxCommonWords = 0;
    for (KeyFrame *pKFi : lKFsSharingWords)
    {
        if (pKFi->mnRelocWords > maxCommonWords)
            maxCommonWords = pKFi->mnRelocWords;
    }

    int minCommonWords = static_cast<int>(maxCommonWords * 0.8f);

    std::list<std::pair<float, KeyFrame *>> lScoreAndMatch;
    int nscores = 0;

    // 3. 计算 BoW 相似度得分 (Score)
    for (KeyFrame *pKFi : lKFsSharingWords)
    {
        if (pKFi->mnRelocWords > minCommonWords)
        {
            nscores++;
            // 计算当前帧与候选关键帧之间的得分
            float score = mpVoc->score(pF->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = score;
            lScoreAndMatch.push_back(std::make_pair(score, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return std::vector<KeyFrame *>();

    // 4. 统计候选关键帧与其共视关键帧（Covisibility Group）的累计最高分
    std::list<std::pair<float, KeyFrame *>> lAccScoreAndMatch;
    float bestAccScore = 0.0f;

    for (auto it = lScoreAndMatch.begin(); it != lScoreAndMatch.end(); it++)
    {
        KeyFrame *pKFi = it->second;
        std::vector<KeyFrame *> vpNeighKFs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame *pBestKF = pKFi;

        for (KeyFrame *pNeighKF : vpNeighKFs)
        {
            if (pNeighKF->mnRelocQuery != pF->mnId)
                continue;

            accScore += pNeighKF->mRelocScore;
            if (pNeighKF->mRelocScore > bestScore)
            {
                pBestKF = pNeighKF;
                bestScore = pNeighKF->mRelocScore;
            }
        }

        lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // 5. 按照累计得分门槛（不低于最高分的 75%）挑选最终的重定位候选关键帧
    float minAccScore = 0.75f * bestAccScore;

    std::set<KeyFrame *> spAlreadyAddedKF;
    std::vector<KeyFrame *> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());

    for (auto it = lAccScoreAndMatch.begin(); it != lAccScoreAndMatch.end(); it++)
    {
        if (it->first > minAccScore)
        {
            KeyFrame *pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

std::vector<KeyFrame *> KeyFrameDatabase::DetectLoopCandidates(KeyFrame *pKF, float minScore)
{
    // 1. 搜集当前关键帧自身、直接相连关键帧及各自的一级共视邻居
    std::set<KeyFrame *> spConnectedKeyFrames;
    const std::vector<KeyFrame *> vpConn = pKF->GetConnectedKeyFrames();
    spConnectedKeyFrames.insert(pKF);

    for (size_t i = 0; i < vpConn.size(); i++)
    {
        KeyFrame *pKFi = vpConn[i];
        if (!pKFi || pKFi->mbBad)
            continue;

        spConnectedKeyFrames.insert(pKFi);
        const std::vector<KeyFrame *> vpConn2 = pKFi->GetConnectedKeyFrames();
        for (size_t j = 0; j < vpConn2.size(); j++)
        {
            if (vpConn2[j] && !vpConn2[j]->mbBad)
                spConnectedKeyFrames.insert(vpConn2[j]);
        }
    }

    // 2. 统计与当前关键帧共享 BoW Word 的历史关键帧及词数
    std::map<KeyFrame *, int> KFsharingWords;
    {
        std::unique_lock<std::mutex> lock(mMutex);

        for (auto vit = pKF->mBowVec.begin(); vit != pKF->mBowVec.end(); vit++)
        {
            if (vit->first >= mvInvertedFile.size())
                continue;

            const std::list<KeyFrame *> &lKFs = mvInvertedFile[vit->first];

            for (KeyFrame *pKFi : lKFs)
            {
                if (spConnectedKeyFrames.count(pKFi))
                    continue; // 严格根据拓扑排除共视邻域
                if (std::abs(static_cast<long int>(pKFi->mnId) - static_cast<long int>(pKF->mnId)) < 30)
                    continue;
                KFsharingWords[pKFi]++;
            }
        }
    }

    if (KFsharingWords.empty())
        return std::vector<KeyFrame *>();

    // 3. 计算最大共享单词数并进行初筛（达到最大值的 80%）
    int maxCommonWords = 0;
    for (auto &mit : KFsharingWords)
    {
        if (mit.second > maxCommonWords)
            maxCommonWords = mit.second;
    }

    int minCommonWords = static_cast<int>(maxCommonWords * 0.8f);

    std::map<KeyFrame *, float> KFScoreMap;
    std::list<std::pair<float, KeyFrame *>> lScoreAndMatch;

    // 4. 计算 BoW 相似度得分并按 minScore 过滤
    for (auto &mit : KFsharingWords)
    {
        KeyFrame *pKFi = mit.first;
        if (mit.second > minCommonWords)
        {
            float score = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
            KFScoreMap[pKFi] = score;
            if (score >= minScore)
            {
                lScoreAndMatch.push_back(std::make_pair(score, pKFi));
            }
        }
    }

    if (lScoreAndMatch.empty())
        return std::vector<KeyFrame *>();

    // 5. 累加候选关键帧及其前 10 个共视邻居的总得分
    std::list<std::pair<float, KeyFrame *>> lAccScoreAndMatch;
    float bestAccScore = minScore;

    for (auto it = lScoreAndMatch.begin(); it != lScoreAndMatch.end(); it++)
    {
        KeyFrame *pKFi = it->second;
        std::vector<KeyFrame *> vpNeighKFs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame *pBestKF = pKFi;

        for (KeyFrame *pNeighKF : vpNeighKFs)
        {
            auto scoreIt = KFScoreMap.find(pNeighKF);
            if (scoreIt != KFScoreMap.end() && scoreIt->second > 0.0f)
            {
                accScore += scoreIt->second;
                if (scoreIt->second > bestScore)
                {
                    pBestKF = pNeighKF;
                    bestScore = scoreIt->second;
                }
            }
        }

        lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // 6. 筛选出累加得分高于最高分 75% 的候选帧
    float minAccScore = 0.75f * bestAccScore;

    std::set<KeyFrame *> spAlreadyAddedKF;
    std::vector<KeyFrame *> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (auto it = lAccScoreAndMatch.begin(); it != lAccScoreAndMatch.end(); it++)
    {
        if (it->first > minAccScore)
        {
            KeyFrame *pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}