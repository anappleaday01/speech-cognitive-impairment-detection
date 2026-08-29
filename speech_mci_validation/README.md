# Speech-MCI 验证包（C1 数据集 + 联调脚本）

> 本目录与部署包 `speech_mci_detection/` 分开单独交付，供协作者**验证 API 输入输出是否正确**。
> 生产运行不需要本目录；只有「用 C1 已知样核对评分」时才用。

## 内容

```
speech_mci_validation/
├── c1_data/
│   ├── egemaps_final.csv            # 323/401 人 eGeMAPS 声学特征（88 维，含入口学来源对齐）
│   ├── 2_final_list_train.csv       # 标注（uuid / label=CTRL|MCI|AD / age / education / sex）
│   ├── linguistic_features_full.csv # 70 列语言学特征（训练时全量特征，供溯源）
│   └── transcripts_full/tsv2/*.tsv  # 图片描述转写（speaker=<A> 为被试）
├── test_c1.py                       # 联调脚本：抽样调 /score 接口，核对预测分 vs 真实标注
└── README.md
```

## 用法（先起部署服务）

```bash
# 终端 1：起部署 API
cd speech_mci_detection
python serve.py --port 8000

# 终端 2：用 C1 已知样抽 8 条调 /score，核对
cd speech_mci_validation
python test_c1.py --n 8 --port 8000
```

## 评分是否与标签对应（已在完整标注集实测）

`severity_0_100` 是 0–100 连续排序分，越高=认知障碍越重。120 个标注样本分布：

| 组 | 人数 | 均值 | 中位数 |
|---|---|---|---|
| CTRL（健康） | 36 | 31.5 | 28.6 |
| MCI（轻度） | 49 | 46.6 | 48.1 |
| AD（痴呆） | 35 | 58.5 | 58.2 |

- CTRL vs 认知障碍（MCI+AD）AUC = **0.766**
- CTRL vs AD = 0.86，CTRL vs MCI = 0.70，MCI vs AD = 0.67

单条样本可能落在组内任何位置（个体差异）；分数用于**风险排序/初筛**，非临床确诊。

> 评分分数是**排序序分数**，个体多在中间段；不是「健康就接近 0、痴呆就接近 100」的绝对刻度，而是相对 C1 训练集 5%/95% 分位归一后的相对位置。