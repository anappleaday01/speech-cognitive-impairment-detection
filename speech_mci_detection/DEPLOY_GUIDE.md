# Speech-MCI 部署与升级指南

> 本文档面向部署/维护人员，说明如何安装、升级本模型服务，以及各版本的变更记录。
> 接口说明、启动方式等使用文档见同目录 [README.md](./README.md)。

## 一、版本管理约定

* 代码、配置、模型、词典、离线模型全部走 Git + Git LFS（`.pkl` / `.bin` / `stanza_models/**/*.pt` 已在根 `.gitattributes` 登记）。

* 克隆前先 `git lfs install`，否则大文件只会拉到指针。

* 每个稳定版打 tag：`deploy-vYYYYMMDD`（例 `deploy-v20260831`）。生产环境拉 tag 而非 `main` HEAD，避免被实验 commit 干扰。

* 服务器部署目录固定为 git worktree 或直接 git clone，**升级不需要手动替换文件**。

## 二、首次部署

```bash
git lfs install
git clone https://github.com/anappleaday01/speech-cognitive-impairment-detection.git
cd speech-cognitive-impairment-detection
git checkout deploy-v20260831          # 切换到交付的稳定版
git lfs pull                            # 确保大文件(pkl/bin/stanza模型)真实落地

python3 -m venv .venv && source .venv/bin/activate
pip install -r speech_mci_detection/requirements.txt
cd speech_mci_detection && python serve.py --port 8000
```

## 三、升级流程

```bash
cd <部署目录>
git fetch --tags origin
git checkout <新tag>                    # 例如 deploy-v20260901
git lfs pull
# 若该版本新增了 Python 依赖（版本记录里会注明）：
pip install -r speech_mci_detection/requirements.txt
# 重启服务
```

升级后建议按「六、升级后验证」核对一次输出。

## 四、按修改类型选择升级动作

| 修改类型                                    | 示例                                                   | 代码侧改动                                                         | 部署侧动作                                                                             |
| --------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **A. 换模型权重**                            | 新训练一版 `my_severity_combined.pkl`                     | 覆盖 pkl，commit → 打 tag → push                                  | `git checkout <新tag>` + `git lfs pull` → 重启服务                                     |
| **B. 改推理代码**（特征组装 / ASR / 词典常模）         | 新增 stanza 离线模型、修复 assemble\_combined\_X 列、加 jieba 降级 | 修改对应 `.py` / `*.csv` / `stanza_models/`，commit → 打 tag → push | `git checkout <新tag>` + `git lfs pull` → `pip install -r requirements.txt` → 重启服务 |
| **C. 改服务配置**（Nginx 上传上限 / systemd / 端口） | 音频 >1MB 返回 HTTP 413、超时                               | 改服务器本机 Nginx/systemd（不进 git，见第五节）                             | 本机 `nginx -t && systemctl reload nginx`                                           |

## 五、服务端配置

### 场景①：Nginx 反代 serve.py:8000（最常见）

打开 Nginx 反代配置（如 `/etc/nginx/conf.d/speech_mci.conf`），在反代该服务的 `server`/`location` 里加：

```nginx
server {
    listen 80;
    server_name 60.204.175.59 your-domain.com;

    # 1) 客户端上传体积：放开到 50MB（单条录音 10 分钟 ~10MB，留足余量）
    client_max_body_size 50m;
    # 2) 反代 buffer 放开一点，长音频不被拆
    client_body_buffer_size 16m;
    # 3) 反代超时：openSMILE + Whisper 一段 10MB wav 约 10~30s，给 120s 兜底
    proxy_read_timeout 120s;
    proxy_send_timeout 120s;

    location /modelserver/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_request_buffering off;   # 音频一到就写后端 tmp，不塞 Nginx 内存
    }
}
```

改完：

```bash
sudo nginx -t                          # 先 dry-run，不要跳过
sudo systemctl reload nginx            # 热加载，不中断已连接
curl -F "file=@big_sample.wav" \
  "http://60.204.175.59/modelserver/v1/models/speech-cognitive-score-v1:upload-and-infer"
# HTTP 200 + JSON，说明 413 已修复
```

> 若前面还有 WAF / API 网关（APIG / ELB 等），Nginx 改完仍 413 时，需在网关控制台把「客户端请求体最大长度」也调到 50MB（默认常为 1–2MB）。

### 场景②：直接裸跑 python serve.py

不经过 Nginx 时不会遇到 413。serve.py 未配置请求体上限（默认足够大）。

## 六、升级后验证

用同一段已知音频本地 curl 打一次，与版本记录中的参考分对照：

```bash
curl -s -X POST "http://127.0.0.1:8000/score_audio?sex=F&age=72&education=9" \
     -H "Content-Type: audio/wav" --data-binary @sample.wav | python -m json.tool
```

* 对照 C1 标准画述音频（`speech_mci_validation/c1_data/` 内任意一段，如 `P0001_0017.tsv` 对应录音）：应返回 `evidence=sufficient` 且有真实梯度分数。

* 短音频 / 噪声 / 非画述（自由对话、英文）录音：会返回 `low_confidence` 或 `acoustic_only`。

* 若 `severity_0_100` 恒为同一值（如 42.9376）且无 `evidence` 字段：说明在用旧版 `my_severity_combined.pkl` / 旧 `cn_scorer.py`。

  * 先 `git lfs pull`，再 `ls -lh speech_mci_detection/my_severity_combined.pkl`（应为 500+KB 实际文件，而非 100 字节指针）。

  * 再确认 `speech_mci_detection/stanza_models/zh-hans/` 下至少有 `tokenize / pos / lemma / depparse / pretrain / backward_charlm / forward_charlm` 7 个目录，否则 stanza 降级会让 16 列依存回退到训练均值、分数偏均值。

  * 若返回 HTTP 413：按第五节改 Nginx（与模型升级是两件事，需分别处理）。

## 七、版本记录

| Tag                | 日期         | 关键变更                                                                                                                                                                                                                                                        | 升级动作                                                                                      |
| ------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `deploy-v20260901` | 2026-08-31 | **`无法判定`（极端域外）改为声学回退打分**：不再笼统拉回中性带 42.5。把语言特征置缺失（填训练均值）后仅用声学+人口学信号重打分，`severity_0_100` 恢复真实梯度（实测 475→34.89 / 365→29.14 / 新录音92→1.03），`evidence=acoustic_only`、`risk_band` 正常三档；`low_confidence`/`sufficient` 行为不变，分布内分数与 AUC 不受影响（C1 323 人 AUC 0.82）。       | `git checkout deploy-v20260901 && git lfs pull` + 重启服务                                    |
| `deploy-v20260831` | 2026-08-31 | **模型换逻辑回归（RBF→LR）**：分布外输入不再塌缩成常数，任意输入都有梯度分数；**新增证据分级（evidence 字段）**：`sufficient`（分布内，分数可靠）/ `low_confidence`（轻度域外，保留分数+风险带但低置信）/ `无法判定`（极端域外，分数拉回中性带 42.5、风险带=无法判定）——解决域外短音频输出 0.0/100.0 被误读为确定健康/障碍的问题。响应新增 `evidence`/`ood_z`/`saturated` 字段。             | `git checkout deploy-v20260831 && git lfs pull` + 重启服务                                    |
| `deploy-v20260830` | 2026-08-30 | **修复 severity 恒=42.9376 的双重根因**：① 打包 stanza 离线模型（`stanza_models/zh-hans`，544MB），句法 16 列真正启用；② 重训 96 维 combined 模型，两边界 SimpleImputer.statistics\_ 不再有 80-95 全 NaN，不同音频分数开始分化。10 折 CV AUC（重训版）：HC-MCI 0.735 / MCI-AD 0.605 / HC-AD 0.832。新增 `stanza>=1.8` 依赖。 | `git checkout deploy-v20260830 && git lfs pull && pip install -r requirements.txt` + 重启服务 |

