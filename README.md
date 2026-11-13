# ALS 电影协同过滤推荐

基于 Apache Spark MLlib 交替最小二乘（ALS）的电影推荐实验代码。在 MovieLens 1M 数据集上训练隐因子模型，预测用户对未看影片的评分并生成 Top-N 推荐。

## 实验环境

| 组件 | 版本 |
|---|---|
| Ubuntu Server | 24.04.5 LTS |
| Docker Engine | 29.8.1 |
| Hadoop | 3.3.6 |
| Spark | 3.5.5（Standalone） |
| Scala | 2.12.18 |

集群由 Docker Compose 编排为三个节点：`master`、`worker1`、`worker2`，每个 Worker 提供 1 核 512 MiB。Spark 以 Standalone 模式运行，Master 地址 `spark://master:7077`。

## 数据集

[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)，字段以 `::` 分隔：

| 文件 | 规模 | 字段 |
|---|---|---|
| ratings.dat | 1,000,209 行 | userId::movieId::rating::timestamp |
| movies.dat | 3,883 行 | movieId::title::genres |
| users.dat | 6,040 行 | userId::gender::age::occupation |

数据下载后放入 `~/ml-1m/`，由 `scripts/upload_data.sh` 上传到 HDFS 的 `/user/root/movielens`。

## 运行步骤

```bash
# 1. 启动 Hadoop 与 Spark 集群
cd ~/lab04 && bash start-spark.sh

# 2. 上传数据集到 HDFS
bash scripts/upload_data.sh

# 3. 训练基准模型并评估 RMSE
docker cp src/train_als.scala master:/root/
docker exec -it master /usr/local/spark/bin/spark-shell \
  --master spark://master:7077 -i /root/train_als.scala

# 4. 超参数对比
docker cp src/tune_params.scala master:/root/
docker exec -it master /usr/local/spark/bin/spark-shell \
  --master spark://master:7077 -i /root/tune_params.scala

# 5. 排序指标评估
docker cp src/evaluate_ranking.scala master:/root/
docker exec -it master /usr/local/spark/bin/spark-shell \
  --master spark://master:7077 -i /root/evaluate_ranking.scala
```

## 实验结果

### 模型精度

| 配置 | RMSE | 说明 |
|---|---|---|
| 全局均值基线 | 1.1171 | 所有样本预测为 3.58 |
| rank=5, λ=0.01 | 0.8766 | |
| rank=10, λ=0.01 | 0.8911 | 基准模型 |
| rank=20, λ=0.01 | 0.9396 | |
| rank=50, λ=0.01 | 1.0165 | 过拟合 |
| rank=20, λ=0.1 | **0.8672** | 最优 |

### 正则化系数

| λ | RMSE（rank=20） |
|---|---|
| 0.001 | 1.1002 |
| 0.01 | 0.9396 |
| 0.1 | **0.8672** |
| 1.0 | 1.3538 |

### 迭代次数

用 5 个随机种子重复切分，检验测试集波动：

| seed | iter=10 | iter=20 |
|---|---|---|
| 42 | 0.8894 | 0.8900 |
| 7 | 0.8919 | 0.8898 |
| 123 | 0.8924 | 0.8906 |
| 2024 | 0.8930 | 0.8924 |
| 999 | 0.8924 | 0.8901 |
| **均值** | **0.8918** | **0.8906** |
| **标准差** | 0.0013 | 0.0010 |

两者均值相差 0.0012，小于一个标准差。**迭代 10 次已基本收敛**，增加到 20 次对精度没有实质影响。单次切分（seed=42）曾显示 iter=20 更差，换一种划分结论即不成立，说明单次评估的波动不可忽略。

### 推荐策略：候选池过滤至 100 人评分

无约束排序时，冷门影片的预测分会严重膨胀——用户 1 的 Top-1 是一部仅 11 人评分的电影，预测分 7.52，而评分上限只有 5。原因是样本过少使 λ=0.01 的正则化压不住隐因子，模型为个别记录单独调高因子。

**本项目最终采用"评分人数 ≥ 100"作为默认过滤条件**，只在该候选池内排序：

| 过滤条件 | 可选影片 | 用户 1 的 Top-3 | 预测分 |
|---|---|---|---|
| 无过滤 | 3,674 | Land and Freedom [11]、Bewegte Mann [7]、Nil By Mouth [14] | 6.90 ~ 7.52 |
| ≥ 50 人 | 2,361 | American Pop [81]、Searchers [188]、Trick [75] | 5.02 ~ 5.55 |
| **≥ 100 人（默认）** | **1,849** | Searchers [188]、Murphy's Romance [213]、No Way Out [296] | **4.92 ~ 5.27** |

方括号内为评分人数。阈值选择的依据：取 50 时最高分仍有 5.55，依旧越界；取 100 压到 5.27，最贴近上限；再收紧到 200 则候选池只剩 1,216 部。

代价是候选池从 3,674 部缩到 1,849 部，冷门佳作失去曝光机会，需要在覆盖度与可信度之间取舍。

### 排序指标

| 指标 | 数值 |
|---|---|
| Precision@5 | 0.0006 |
| Precision@10 | 0.0009 |
| NDCG@10 | 0.0008 |

数值偏低源于数据集只记录用户看过的影片：模型推荐的冷门片大概率从未被该用户接触，既不算命中也不算错误。严谨的 Top-N 评估需要负样本采样或留一法协议。

### 推荐结果示例

| 用户 | Top-5 推荐 | 预测得分 |
|---|---|---|
| 1 | Smashing Time (1967)、Saltmen of Tibet (1997)、Ulysses (1954)、Schindler's List (1993)、Shawshank Redemption (1994) | 4.59 ~ 4.76 |
| 100 | Mamma Roma (1962)、Foreign Student (1994)、Schlafes Bruder (1995)、Ulysses (1954)、Schindler's List (1993) | 3.66 ~ 4.30 |
| 6040 | Mamma Roma (1962)、Inheritors (1998)、Ed's Next Move (1996)、Lamerica (1994)、Mirror (1975) | 4.57 ~ 5.06 |

多名用户共享的影片分两类：`Ulysses`（全库仅 1 人评分）、`Mamma Roma`（2 人评分）属于样本过少导致的分数膨胀，并非真实的口味趋同；`Schindler's List`（2304 人评分、均分 4.51）属于群体共识，共享合理。

## 目录结构

```
.
├── README.md
├── src/
│   ├── train_als.scala           # 基准训练与 RMSE 评估
│   ├── tune_params.scala         # rank 与正则化对比
│   ├── check_seed_variance.scala # 多随机种子的测试集波动
│   ├── popularity_filter.scala   # 候选池过滤（默认 ≥100 人）
│   └── evaluate_ranking.scala    # Precision@K / NDCG
└── scripts/
    └── upload_data.sh          # 数据集上传 HDFS
```

## 说明

本仓库只包含实验代码与配置，不包含数据集与训练产物。数据集请从 GroupLens 官方地址获取。
