# 中文语音认知障碍检测（Speech-MCI）

> **输入音频 → 输出 `severity_0_100`（0–100，越高=认知障碍越重）+ `risk_band`**。
> 本项目基于 C1 中文画述数据集（iFLYTEK 2019 AD，`lzy1012/Alzheimer-s-disease-datasets`，CTRL / MCI / AD，训练样本 323 人）训练，用于「认知健康 vs 认知障碍（MCI+AD 合并）」筛查。

**建模口径**：阴性 = CTRL（健康）；阳性 = 认知障碍 = **MCI + AD 合并**（MCI 是 AD 前驱谱系）。`severity_0_100` 越高越接近认知障碍（实测组均值 CTRL 32 < MCI 55 < AD 70）。**筛查模型，非临床确诊**；主指标为 AUC（阈值无关的排序能力）。

**风险带（方案A：三级不确定带，边界 35/50）**：

| 区间 | severity | 判定 | 建议 |
|---|---|---|---|
| 低风险 | `< 35` | `CTRL-like` | 大概率健康 |
| 灰色带 | `35 ≤ x < 50` | `borderline` | 建议复测/随访 |
| 高风险 | `≥ 50` | `MCI-like` | 疑似认知障碍（含 AD），建议转诊 |

---

## ⚠️ Git LFS 提醒

本仓库的大文件（Whisper 模型 `speech_mci_detection/whisper_models/.../model.bin` 约 461MB，以及评分模型 `my_severity_combined.pkl`）由 **Git LFS** 托管。**clone 前必须先装 Git LFS**，否则只会拉到几 KB 的文件指针，无法得到真实模型内容：

```bash
git lfs install
git clone https://github.com/anappleaday01/speech-cognitive-impairment-detection.git
# 若已用普通 git clone 拉过，可后补大文件：
cd speech-cognitive-impairment-detection && git lfs pull
```

---

## 目录结构

```
speech_mci_detection/    部署版：可独立运行的评分 API 服务
speech_mci_validation/   验证版：C1 数据集 + 联调脚本，用于核对输入输出
```

### speech_mci_detection/（部署版，供服务器/AI Agent 调用）

- **`serve.py`** — HTTP 服务，提供两个评分入口：
  - `POST /score`：传**特征 + 转录文本**（C1 格式）即可评分，零重依赖（仅 stdlib）。
  - `POST /score_audio`：传**音频原始二进制**，自动 openSMILE 抽 88 维 eGeMAPS + Whisper ASR 转写 → 组 96 维特征 → 评分。需安装 opensmile + faster-whisper。
  - `GET /health`：健康检查。
- **`audio_to_score.py`** — 音频端到端底层能力（openSMILE + Whisper，懒加载）。
- **`cn_scorer.py`** — 评分核心：`assemble_combined_X`（声学+人口学+语言学 70 列组装）、分位定标归一化、三级风险带。
- **`my_severity_combined.pkl`** — 96 维 fast 预训练模型，启动即自动加载。
- `extract_linguistic.py` / `extract_syntax_proposition.py` / `extract_tongji_27d_syntax.py` — 语言学特征抽取器（jieba 后端，离线可用）。
- `whisper_models/` — Faster-Whisper small 模型（LFS）。
- README.md — 部署版完整文档（接口用法、依赖安装、启动命令）。

启动：

```bash
cd speech_mci_detection
pip install -r requirements.txt   # 见该目录下 README
python serve.py                    # http://127.0.0.1:8000
```

### speech_mci_validation/（验证版）

- **`c1_data/`** — C1 核心数据：`egemaps_final.csv`（88 维声学特征）、`2_final_list_train.csv`（标签+人口学）、`linguistic_features_full.csv`（语言学特征）、`transcripts_full/tsv2/*.tsv`（转写）。
- **`test_c1.py`** — 联调脚本：从 C1 抽样，组装特征+转写调 `/score`，输出 `severity_0_100 + risk_band` 并对照真实标注，核对输出是否符合「CTRL 低分、障碍高分」。
- **`input_output.xlsx`** — 输入输出总览（字段、口径、示例）供人工核对。
- **`export_io_xlsx.py`** — 生成上述 Excel 的脚本。
- README.md — 验证版使用说明。

联调：

```bash
cd speech_mci_validation
python test_c1.py --n 3 --port 8000   # 需先启动 speech_mci_detection/serve.py
```

---

## 专项结果（10 折训练 CV / AUC，`severity_0_100` 越高认知障碍越重）

| 口径 | AUC |
|---|---|
| CTRL vs MCI（10 折训练 CV，out-of-fold） | 0.708 |
| CTRL vs 认知障碍—MCI+AD（10 折训练 CV，323 人） | 0.744 |
| CTRL vs MCI（holdout 30%，76 人，供参考） | 0.777 |
| CTRL vs 认知障碍—MCI+AD（holdout 30%，97 人，供参考） | 0.821 |

> `severity_0_100` 为**认知障碍风险序分数**（分位数归一化，非 MMSE），用于风险排序/初筛；边界 35/50 取自全量 OOF 分布。

## 部署与交付

- 部署版仅包含运行所需文件（模型、特征抽取器、词典常模），**不含训练数据**；`cn_scorer.build_cn_severity_combined_model` 为 stub，训练在源仓库完成，部署包直接加载 `my_severity_combined.pkl`。
- 详细接口说明见 `speech_mci_detection/README.md`。
- 部署/升级流程、服务端配置与各版本变更记录见 `speech_mci_detection/DEPLOY_GUIDE.md`。