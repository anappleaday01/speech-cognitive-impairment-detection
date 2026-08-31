# Speech-MCI 语音认知障碍检测

> **输入音频 → 输出** **`severity_0_100`（0–100，越高=认知障碍越重）+** **`risk_band`**。
> 音频进来后自动走：openSMILE 抽 88 维 eGeMAPS + Whisper ASR 转写 → 组特征 → 评分。不需要再下载任何模型。

> **⚠️ Git LFS 提醒**：本仓库的大文件（Whisper 模型 `speech_mci_detection/whisper_models/.../model.bin` 约 461MB，以及评分模型 `my_severity_combined.pkl`）由 **Git LFS** 托管。clone 前请先装 Git LFS，否则只会拉到文件指针（几 KB），无法得到真实模型内容：
>
> ```bash
> # 安装后
> git lfs install
> git clone https://github.com/anappleaday01/speech-cognitive-impairment-detection.git
> # 若已用普通 git clone，补拉大文件：
> cd speech-cognitive-impairment-detection && git lfs pull
> ```

## 训练数据与模型

**数据（C1 中文画述）**：iFLYTEK 2019 AD 语音数据集（`lzy1012/Alzheimer-s-disease-datasets`），中文母语者做「图片描述」任务，含 **CTRL（健康）/ MCI（轻度）/ AD（痴呆）** 三组，训练样本 **323 人**。

> **建模口径（重点）**：这是一个「认知健康 vs 认知障碍」**筛查**模型，而非「三分类/二选一组」。
>
> * **阴性 = CTRL（健康）**
>
> * **阳性 = 认知障碍 = MCI + AD 合并**（MCI 是 AD 的前驱谱系，临床上同属认知障碍）
>
> * 模型内部用 `b_impaired`（CTRL vs {MCI,AD}）作为核心判别，所以 `severity_0_100` 越高越接近认知障碍；AD 组均值也更高（实测 CTRL 32 < MCI 55 < AD 70）。
>
> * **不要在训练脚本传** **`--binary`**：`--binary` 会**剔除 AD、只留 CTRL+MCI**（255 人），那是「只做 MCI 早期识别」的另一种口径，与本文交付的「认知障碍筛查」不同。本包已默认「AD+MCI 合并为障」口径，无需任何额外参数。

**特征（96 维 fast）**：声学 23 维核心韵律（F0+Loudness+SpectralFlux+MFCC1，从 eGeMAPS 精简，省 73% 计算量且更准）+ 人口学 3 维（sex/age/education，可选）+ 语言学 70 维（jieba 分词 + 词频/句法常模）。`severity_0_100` 是**认知障碍风险序分数**：模型把 CTRL（健康）当作低分侧、把\*\*认知障碍（MCI 与 AD 合并，MCI 是 AD 的前驱谱系）\*\*当作高分侧，用分位数归一化到 0-100。**CTRL≈低分，认知障碍≈高分**（延续 MCI 居中设计，组均值实测 CTRL 32 < MCI 55 < AD 70）。

**风险带（方案A：三级不确定带，边界 35/50，取自全量 OOF 分布）**：

| 区间  | severity      | 判定           | 建议                |
| --- | ------------- | ------------ | ----------------- |
| 低风险 | `< 35`        | `CTRL-like`  | 大概率健康             |
| 灰色带 | `35 ≤ x < 50` | `borderline` | 建议复测/随访           |
| 高风险 | `≥ 50`        | `MCI-like`   | 疑似认知障碍（含 AD），建议转诊 |

**排序/判别能力（AUC；`severity_0_100`** **越高认知障碍越重）**：

| 口径                                   | AUC   |
| ------------------------------------ | ----- |
| CTRL vs MCI（holdout 30% 留出，76 人）     | 0.777 |
| CTRL vs 认知障碍—MCI+AD（holdout，97 人）    | 0.821 |
| CTRL vs MCI（10 折训练 CV，out-of-fold）   | 0.708 |
| CTRL vs 认知障碍—MCI+AD（10 折训练 CV，323 人） | 0.744 |

> **方案A 说明**：不做「一刀切阳/阴」，而是**连续分 + 三级区间**，把中间重叠区（灰色带 24% 样本）留给复测兜底——这是针对健康/障碍两组分严重重叠（OOF 健康中位 34.7 vs 障碍中位 54.7）的稳妥做法。
> **OOF 分布**：低 32% / 灰 24% / 高 44%；障碍进高风险 58%、健康进低风险 51%。**AUC 是阈值无关的排序能力（主指标）**；主口径为 **CTRL vs 认知障碍（MCI+AD）**。分值用于**风险排序/初筛**，非临床确诊。

## 目录

```
speech_mci_detection/
├── serve.py                  # HTTP 服务入口（唯一要起的程序）
├── cn_scorer.py              # 评分器/反序列化所需（必须与 serve.py 同目录）
├── audio_to_score.py         # /score_audio 底层：openSMILE+Whisper（必须同目录）
├── extract_linguistic.py     # 转写→基础语言学特征（22 列，必须同目录）
├── extract_syntax_proposition.py  # 转写→句法/命题特征（20 列，必须同目录，jieba fallback）
├── extract_tongji_27d_syntax.py   # 转写→依存/分句复杂度特征（28 列，必须同目录，jieba fallback）
├── my_severity_combined.pkl  # ✅ 训练好的模型（96 维 fast，C1 中文 323 人）
├── feature_cols.json         # 特征列清单
├── SUBTLEX-CH-WF.xlsx        # 中文词频常模（词频特征用）
├── chinese_aoa.csv           # 习得年龄常模
├── chinese_concreteness.csv  # 具体性常模
├── chinese_familiarity.csv   # 熟悉度常模
├── requirements.txt          # ✅ 依赖清单（协作者 pip install -r requirements.txt）
└── whisper_models/           # Whisper-small 模型（离线打包，无需联网下载）
```

> 语言学特征共 70 列 = 基础 22 + 句法/命题 20 + 依存/分句 28，由上述三个 `extract_*.py` 在线抽取并合并。
> **三者缺一不可**：只留 `extract_linguistic.py` 会让其余 48 列在推理时填 0，导致对所有人输出同一分数。
> 三个抽取器均有 jieba 降级（`jieba` 已在 requirements），无需强制装 stanza；装 stanza + 捆绑中文模型可补齐
> 短语比例类 16 列，但非必需。

## 一、装依赖（一次性）

需要 Python ≥3.11。装（分词条的依赖见同目录 `requirements.txt`，一条命令全装）：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 等价于：pip install numpy pandas scikit-learn jieba openpyxl \
#             opensmile faster-whisper soundfile
```

> * Whisper 模型已在 `whisper_models/`，`faster-whisper` 会自动从本地加载，**不联网**。
>
> * 只想开 `/score`（不跑音频）可只装「A 核心」部分，`/score_audio` 才需要 opensmile + faster-whisper。
>
> * **平台提示（华为云务必先自测）**：`opensmile` 对 Linux x86\_64 提供官方支持；但**鲲鹏 aarch64 / 罕见发行版不一定有预编译包**。在服务器上先跑
>
>   ```bash
>   uname -m          # 看架构：x86_64 安全；aarch64 是鲲鹏，opensmile 可能有坑
>   python -c "import opensmile; print(opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals))"
>   ```
>
>   import 不报错才说明声学抽取能跑。若鲲鹏上装不上 opensmile，需换 x86\_64 实例，或用已有预抽取好的声学特征走 `/score`（见 README 顶部目录项的 alternatives）。

## 二、启动

```bash
cd speech_mci_detection
python serve.py --port 8000
```

* **默认自动加载同目录 96 维 fast 模型**（`my_severity_combined.pkl`），无需任何额外参数，启动即用。

* 如需换模型可用 `--model path.pkl`（如 MMSE 回归模型）；不传则默认 fast。

* 生产用 `nohup` / `systemd` / supervisor 托管，监听 `0.0.0.0:8000`（外网再加 Nginx 反代）。

## 三、接口约定（App 侧对接这份）

**本包只暴露一个核心接口，输入输出完全固定：**

> **输入 = 一个音频文件（body 二进制）+ 可选人口学（URL 参数）**
> **输出 =** **`severity_0_100`（0–100，越高=认知障碍越重）及全部辅助字段**

音频进来后由服务端自动处理 openSMILE 抽特征 + Whisper 转写 + 评分，客户端**只负责上传音频，不做任何特征计算**。

### POST /score\_audio

```
POST /score_audio?sex=F&age=72&education=6
Content-Type: audio/wav          # 或 audio/mpeg（.mp3）
body = 音频二进制流
```

| 参数          | 位置        | 必填 | 说明                                         |
| ----------- | --------- | -- | ------------------------------------------ |
| `body`      | 请求体       | ✅  | 音频二进制（`.wav` / `.mp3`，按 `Content-Type` 识别） |
| `sex`       | URL query | 可选 | `F` / `M`，缺省空白，影响极小                        |
| `age`       | URL query | 可选 | 数值，缺省按 NaN 处理                              |
| `education` | URL query | 可选 | 受教育年限数值，缺省按 NaN 处理                         |

### 输出字段（全部字段，App 可直接用）

```json
{
  "uuid": "recording",
  "severity_0_100": 49.05,
  "risk_band": "MCI-like",
  "evidence": "sufficient",
  "ood_z": 0.8,
  "saturated": false,
  "mode": "combined",
  "asr_backend": "faster-whisper"
}
```

| 字段                   | 类型          | 含义                                                                                                               |
| -------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `uuid`               | string      | 无后缀文件名（App 侧可忽略，需关联样本可自行覆盖）                                                                                      |
| **`severity_0_100`** | float 0–100 | ⭐ 核心输出。越高=认知障碍越重。健康(CTRL)≈低分，认知障碍（MCI/AD）≈高分                                                                     |
| `risk_band`          | string      | 风险带判定：`CTRL-like`（`<35`）/ `borderline`（`35–50`）/ `MCI-like`（`≥50`）                                               |
| `evidence`           | string      | **证据充分性（新字段，App 建议展示）**：`sufficient`（分布内，分数可靠）/ `low_confidence`（轻度域外，分数保留但低置信，仅供排序）/ `无法判定`（极端域外，分数无信息量，已拉向中性带） |
| `ood_z`              | float       | 离训练分布的平均 \|z\| 距离，越大越不可信（仅供排查）                                                                                   |
| `saturated`          | bool        | 决策值是否钉在 0/100 端点（仅供排查）                                                                                           |
| `mode`               | string      | 恒为 `combined`（声学+语言学都用上了）；若转写为空降级为 `combined_imputed`                                                            |
| `asr_backend`        | string      | 实际转写后端：`faster-whisper` 或 `transformers`，仅供排查                                                                    |

> **App 主读** **`severity_0_100`（连续分，直接用于排序/初筛）；`risk_band`** **只是该分值按切点（35/50）的便捷分档：`<35`→CTRL-like、`35–50`→borderline、`≥50`→MCI-like。**
> **建议 App 语义**：`CTRL-like` 直接低风险；`borderline` 提示"需复测/随访"；`MCI-like` 提示"疑似认知障碍，建议转诊"。`borderline` 与 `MCI-like` 均建议引起关注/进一步检查。
> **evidence 处理建议**：`sufficient` 正常使用分数；`low_confidence` 分数仅作参考、建议复测标准画述录音；`acoustic_only` 分数仅反映声学信号、低置信，不建议对外输出精确分数，提示"录音不符合标准（太短/噪声/非画述任务），请按规范重录"。

### 错误码

| HTTP  | 场景                                  |
| ----- | ----------------------------------- |
| `200` | 正常，返回上表 JSON                        |
| `503` | 未装 Whisper（`faster-whisper`），音频无法转写 |
| `500` | 其他处理异常，`{"error": "..."}`           |

### 调用示例（App / curl 同一契约）

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 音频端到端评分（输入=音频文件，输出=severity_0_100）
curl -X POST "http://127.0.0.1:8000/score_audio?sex=F&age=72&education=6" \
     -H 'Content-Type: audio/wav' \
     --data-binary @recording.wav
```

```bash
# Python 客户端示例（App 后端/脚本）
import requests
with open("recording.wav", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/score_audio",
        params={"sex": "F", "age": 72, "education": 6},
        data=f,
        headers={"Content-Type": "audio/wav"},
    )
print(r.json()["severity_0_100"])
```

## 四、试跑

> demo 演示音频已从部署包移除（是 TTS 合成、会偏到均值附近，不代表真人）。服务起来后拿任意一段真实 `.wav` 试：

```bash
curl -s -X POST "http://127.0.0.1:8000/score_audio" \
     -H 'Content-Type: audio/wav' --data-binary @recording.wav
```

## 已知边界（如实告知，别当诊断工具）

1. 模型是**筛查级**风险初筛/排序工具（CTRL vs 认知障碍 AUC≈0.82、CTRL vs MCI AUC≈0.78，样本仅 323 人），**不是临床确诊**。
2. 模型按 C1 中文画述任务训练；对英文、自由对话等场景的分数仅供管线验证，临床需在对应数据上重标定。
3. 传的音频越清晰、说话越完整，转写越好、分数越准。

## 五、C1 数据集与验证脚本（单独包）

C1 原始数据与联调脚本**不放进部署包**，另行交付 `speech_mci_validation/`（含 `c1_data/` 与 `test_c1.py`），
供协作者用已知标注核对 API 输出。生产只需要本部署包，不需要验证包。

验证包用法：另开终端起本服务后再跑

```bash
cd speech_mci_validation
python test_c1.py --n 8 --port 8000   # 抽 8 条调 /score，核对预测 vs 真实标注
```

`severity_0_100` 与标注的对应关系（120 个标注样本实测）：CTRL 中位 28.6、MCI 48.1、AD 58.2；
CTRL vs 认知障碍（MCI+AD）AUC ≈ 0.77。详见验证包内 README。

***

## 六、部署与升级、版本记录

部署流程、升级动作与各版本变更记录，见 [DEPLOY\_GUIDE.md](./DEPLOY_GUIDE.md)。
