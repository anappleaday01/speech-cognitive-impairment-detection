# -*- coding: utf-8 -*-
"""
中文认知风险 评分接口 (CognitiveScorer)
================================================================
基于 C1 (lzy1012/Alzheimer-s-disease-datasets) 数据集训练的中文 AD/MCI/CTRL
风险评分模型，复用 Heitz 2026 演示版 SVC 配置 (C=0.1, balanced)。

核心能力：
  - fit(X, y)              训练 (X=特征矩阵, y=类别标签)
  - predict_proba(X)       返回每类概率 (列序 = classes_)
  - risk_score(X)          返回 0-100 认知障碍风险评分
                              = 100 * (P(AD) + 0.5 * P(MCI))   (ordinal 启发式)
  - score(X, uuids)        批量推理 -> DataFrame(概率 + 风险评分)
  - score_dataset_cv(...)  对已有标签数据集做 leakage-free 交叉验证打分
                              (每条样本用"未见过的折"的模型打分，无泄漏)
  - save/load              序列化，供 serve.py 现场演示加载

演示定位：这是中文演示主线 (非英文 Heitz)。英文 Heitz 仅作背景参照。
局限：C1 无连续 MMSE 金标准，risk_score 是 ordinal 风险指数(0-100)，
      非 MMSE；要 0-100 连续 MMSE 需 TAUKADIAL(申请) 重标定。
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "cn_scorer.pkl")

CLASSES = ["CTRL", "MCI", "AD"]   # ordinal: 0=健康,1=轻度,2=痴呆


class CognitiveScorer:
    """中文认知障碍风险 评分接口。
    风险排序量 = SVC decision_function (弱信号上比校准概率更可靠, 见 smoke test:
    acoustic-only AUC 0.667 / combined 0.697)。0-100 风险评分由 decision_function
    经训练集分位线性映射得到, 单调 = 判别排序。"""
    def __init__(self, classes=None, C=0.1, random_state=1):
        self.classes = list(classes) if classes else list(CLASSES)
        self.C = C
        self.random_state = random_state
        self.pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler()),
            # 不使用 probability=True: 弱信号上 Platt 校准会把排名校准反 (实测 AUC 0.32)。
            # 逻辑回归（linear logit）: 线性组合决策，对分布外输入不饱和，
            # 任意输入都能给出有梯度的 0-100 分（原 RBF SVC 对 OOD 输入会塌缩到常数）。
            ("svc", LogisticRegression(C=C, class_weight="balanced",
                                       random_state=random_state, max_iter=20000)),
        ])
        self.feature_cols_ = None
        self.d_lo_ = None   # 训练集 decision_function 下界 -> 风险 0
        self.d_hi_ = None   # 上界 -> 风险 100

    # ---- 训练 ----
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.feature_cols_ = getattr(X, "columns", None)
        self.pipe.fit(X, y)
        d = np.asarray(self.pipe.decision_function(X)).ravel()
        # 方向校准: 让"风险越高 = AD 越重"。弱信号上 SVC 的 decision_function
        # 可能指向相反类(实测 CTRL 的 d 更高), 故按训练集类均值翻转。
        self.flip_ = 1.0
        if len(self.classes) == 2:
            yarr = np.asarray(y)
            if yarr.dtype.kind in "iufc":
                ad_mask = yarr == 1
            else:
                ad_mask = yarr == self.classes[1]
            mean_ad = d[ad_mask].mean() if ad_mask.any() else 0.0
            mean_ctrl = d[~ad_mask].mean() if (~ad_mask).any() else 0.0
            self.flip_ = 1.0 if mean_ad >= mean_ctrl else -1.0
        d = self.flip_ * d
        # 分位定标: 用训练集 5%/95% 分位做 [0,100] 锚点, 取代 min/max。
        # 原因: 决策值分布有重尾, min/max 会被离群样本把 0/100 拉得很开,
        # 正常人挤在低分区、刻度失真; 分位把锚点收敛到主体范围, 0-100 更有梯度、
        # 且 serve 对外服务与 report 口径用同一套边界, 不再"两把尺子"。
        lo, hi = np.percentile(d, 5), np.percentile(d, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(d.min()), float(d.max())
        self.d_lo_, self.d_hi_ = float(lo), float(hi)
        # ---- OOD / 证据检测基线（供 serve 标注"证据不足"）----
        # 用 StandardScaler 把训练特征转到 z 空间，记每样本平均 |z|，取 98% 分位为阈值。
        # 推理时输入若离训练分布太远（不同任务/短音频/异常声学），mean|z| 显著偏大
        # -> evidence=insufficient，分数为低置信域外输出，不作为精确诊断。
        try:
            Z = self.pipe.named_steps["sc"].transform(
                self.pipe.named_steps["imp"].transform(np.asarray(X, dtype=float)))
            mz = np.nanmean(np.abs(Z), axis=1)
            self.ood_mean_ = float(np.nanmean(mz))
            self.ood_thr_ = float(np.nanpercentile(mz, 98))
        except Exception:
            self.ood_mean_ = 0.0
            self.ood_thr_ = float("inf")
        return self

    # ---- 推理 ----
    def decision(self, X):
        X = np.asarray(X, dtype=float)
        d = np.asarray(self.pipe.decision_function(X))
        d = d.ravel() if d.ndim > 1 else d
        return self.flip_ * d

    def risk_score(self, X):
        """0-100 认知障碍风险评分 (decision_function 经训练集分位映射到 [0,100])。"""
        d = self.decision(X)
        lo, hi = self.d_lo_, self.d_hi_
        if hi is None or lo is None or hi == lo:
            return np.full_like(d, 50.0)
        return np.clip(100.0 * (d - lo) / (hi - lo), 0.0, 100.0)

    def ood_z(self, X):
        """每样本平均 |z|（scaled 空间）：离训练分布越远越大。"""
        X = np.asarray(X, dtype=float)
        imp = self.pipe.named_steps["imp"]
        sc = self.pipe.named_steps["sc"]
        Z = sc.transform(imp.transform(X))
        return np.nanmean(np.abs(Z), axis=1).ravel()

    def evidence(self, X):
        """证据充分性检测。
        ood_z：平均 |z|（离训练分布距离）；saturated：决策值是否钉在 0/100 端点；
        evidence：'sufficient'（分布内，分数可靠）/ 'insufficient'（分布外，低置信）。"""
        X = np.asarray(X, dtype=float)
        z = self.ood_z(X)
        d = self.decision(X)
        lo, hi = self.d_lo_, self.d_hi_
        risk = np.clip(100.0 * (d - lo) / (hi - lo), 0.0, 100.0)
        saturated = (risk <= 0.0) | (risk >= 100.0)
        thr = getattr(self, "ood_thr_", float("inf"))
        insufficient = z > thr
        return z, saturated, insufficient

    def score(self, X, uuids=None):
        """批量推理 -> DataFrame(decision + 0-100 风险评分 + 预测类别)。"""
        X = np.asarray(X, dtype=float)
        d = self.decision(X)
        if len(self.classes) == 2:
            pred = np.where(d > 0, self.classes[1], self.classes[0])
        else:
            pred = self.pipe.predict(X)
        out = pd.DataFrame({
            "decision_function": d,
            "risk_score_0_100": self.risk_score(X),
            "predicted_class": pred,
        })
        if uuids is not None:
            out.insert(0, "uuid", list(uuids))
        return out

    # ---- 对已有标签数据集做 leakage-free 打分 ----
    def score_dataset_cv(self, X, y, uuids=None, n_splits=10):
        """每条样本用交叉验证(未见它的折)的模型打分，避免泄漏。
        逐折重训并按该折训练集类均值定向 decision_function (弱信号上各折 margin
        符号可能翻转, 直接 cross_val_predict 会失真, 故手动处理)。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        ad_pos = 1 if (y.dtype.kind in "iufc") else self.classes[1]
        d = np.empty(X.shape[0], dtype=float)
        pred = np.empty(X.shape[0], dtype=object)
        for tr_i, te_i in skf.split(X, y):
            self.pipe.fit(X[tr_i], y[tr_i])
            dt = np.asarray(self.pipe.decision_function(X[tr_i])).ravel()
            ytr = y[tr_i]
            ma = dt[ytr == ad_pos].mean() if (ytr == ad_pos).any() else 0.0
            mc = dt[ytr != ad_pos].mean() if (ytr != ad_pos).any() else 0.0
            fold_flip = 1.0 if ma >= mc else -1.0
            d_te = np.asarray(self.pipe.decision_function(X[te_i])).ravel() * fold_flip
            d[te_i] = d_te
            if len(self.classes) == 2:
                pred[te_i] = np.where(d_te > 0, self.classes[1], self.classes[0])
            else:
                pred[te_i] = self.pipe.predict(X[te_i])
        out = pd.DataFrame({
            "decision_function": d,
            "risk_score_0_100": self.risk_score(X),
            "predicted_class": pred,
            "true_label": y,
        })
        if uuids is not None:
            out.insert(0, "uuid", list(uuids))
        return out

    # ---- 序列化 ----
    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


# =====================================================================
# 部署版说明：以下不再含训练/数据加载函数（训练数据不随部署包提供）。
# 模型的训练与打包在源仓库完成，部署时直接加载同目录 my_severity_combined.pkl
# （见 serve.py 顶部 _auto_pkl 自动加载；缺失时 build_cn_severity_combined_model
# 直接抛 FileNotFoundError，由调用方捕获回退）。此类保留供 pkl 反序列化使用。
# =====================================================================
SEVERITY_MODEL_PATH = os.path.join(HERE, "cn_severity_scorer.pkl")

# 灰色带（35–50）中心值：域外(evidence=insufficient)样本把 severity 回缩到这里，
# 避免"溢出端点 0.0/100.0"被误读成确定的健康/障碍判定。
NEUTRAL_SEVERITY = 42.5


class CognitiveSeverityScorer:
    """MCI-centered 0-100 severity score.

    两个边界二分类 (复用 CognitiveScorer 的 decision_function + flip_ 方向校准,
    避开弱信号上 Platt 概率反序的坑):
      - b_impaired : CTRL(0) vs {MCI, AD}(1)   -> impairment 边界
      - b_dementia : {HC, MCI}(0) vs AD(1)      -> dementia 边界
    severity = 0.5 * risk_impaired + 0.5 * risk_dementia
      => CTRL ~ 0, MCI ~ 50 (中段), AD ~ 100

    相比单一 AD/CTRL 二分类: MCI 同时参与两个边界的训练(分别作 impaired / non-AD),
    模型对 MCI 区间有显式建模, 输出天然把 MCI 锚定在中点, 而非"顺带"落在中间。
    """
    def __init__(self, C=0.1, random_state=1):
        self.C = C
        self.random_state = random_state
        self.b_impaired = CognitiveScorer(classes=["CTRL", "IMPAIRED"], C=C, random_state=random_state)
        self.b_dementia = CognitiveScorer(classes=["NONAD", "AD"], C=C, random_state=random_state)
        self.classes = list(CLASSES)
        self.feature_cols_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.feature_cols_ = getattr(X, "columns", None)
        yarr = np.asarray(y)
        y_imp = np.isin(yarr, ["MCI", "AD"]).astype(int)   # 1 = impaired
        y_dem = (yarr == "AD").astype(int)                  # 1 = AD
        self.b_impaired.fit(X, y_imp)
        self.b_dementia.fit(X, y_dem)
        return self

    def severity(self, X):
        r1 = self.b_impaired.risk_score(X)
        r2 = self.b_dementia.risk_score(X)
        return 0.5 * r1 + 0.5 * r2

    def evidence(self, X):
        """组合两个边界的证据检测（任一边界域外即 insufficient）。
        返回 (ood_z, saturated, evidence)。"""
        X = np.asarray(X, dtype=float)
        z1, sat1, ins1 = self.b_impaired.evidence(X)
        z2, sat2, ins2 = self.b_dementia.evidence(X)
        return (np.maximum(z1, z2), (sat1 | sat2),
                np.where(ins1 | ins2, "insufficient", "sufficient"))

    def _ling_mask(self):
        """bool 掩码: combined_feature_cols_ 中属于语言学(ling) 的列位置。
        假定 score 的 X 列顺序 == combined_feature_cols_ 顺序（serve.py assemble_combined_X 保证）。"""
        cf = getattr(self, "combined_feature_cols_", None)
        lc = getattr(self, "ling_cols_", None)
        if cf is None or lc is None:
            return None
        lc = set(lc)
        return np.array([c in lc for c in cf])

    def _acoustic_fallback(self, X):
        """声学回退分：语言特征置 NaN(→填训练均值)后重打分，分数仅反映声学/人口学信号。
        用于"无法判定"样本——其语言特征已失真（英文/非画述文本把完整模型分数饱和压到
        0/100 端点），完整模型分数无信息量。返回与 X 等长的 severity 数组；
        无法计算列掩码时退回中性带中心（原回缩行为）。"""
        X = np.asarray(X, dtype=float)
        mask = self._ling_mask()
        if mask is None or mask.sum() == 0:
            return np.full(X.shape[0], NEUTRAL_SEVERITY, dtype=float)
        Xa = X.copy()
        Xa[:, mask] = np.nan
        return np.asarray(self.severity(Xa), dtype=float).ravel()

    def score(self, X, uuids=None):
        X = np.asarray(X, dtype=float)
        sev = self.severity(X)
        z1, sat1, ins1 = self.b_impaired.evidence(X)
        z2, sat2, ins2 = self.b_dementia.evidence(X)
        z = np.maximum(z1, z2)
        sat = sat1 | sat2
        # 分级证据: sufficient(分布内, 分数可靠) / low_confidence(轻度域外,
        # 保留分数+风险带但低置信) / 无法判定(极端域外, 语言内容不适用)。
        # 判据: exc=(z−thr)/thr(取两边界较大者), exc≥2 即"无法判定"。
        thr1, thr2 = self.b_impaired.ood_thr_, self.b_dementia.ood_thr_
        ev = np.full(len(z), "sufficient", dtype=object)
        for i in range(len(z)):
            if not (ins1[i] or ins2[i]):
                continue
            exc = 0.0
            if np.isfinite(thr1) and thr1 > 0:
                exc = max(exc, (z1[i] - thr1) / thr1)
            if np.isfinite(thr2) and thr2 > 0:
                exc = max(exc, (z2[i] - thr2) / thr2)
            ev[i] = "无法判定" if exc >= 2.0 else "low_confidence"
        # 触发声学回退的两类样本（完整模型分均无信息量）：
        #  1) 极端域外 (ev=无法判定, exc≥2)：语言内容失真，完整分被压到端点。
        #  2) low_confidence 且完整分饱和在端点 (saturated)：分被 clip 钉在
        #     0.0/100.0，会被误读成"确定健康/障碍"，同样无信息量。
        # 均改用声学回退分（语言特征视为缺失→填训练均值），evidence 标注 acoustic_only。
        # 回退分仍饱和端点(0.0/100.0)的极端样本：声学信号本身也极端偏离训练分布，
        # 拉向中性带中心，避免 0.0/100.0 端点被误读为确定性判定。
        ev = np.asarray(ev)
        undet = ev == "无法判定"
        sat_low = (ev == "low_confidence") & sat
        to_fb = undet | sat_low
        if to_fb.any():
            try:
                fb = self._acoustic_fallback(X)
                # 回退分是两边界 risk 的平均，可能落在 (0,1] 这种"≈0 而非恰好 0"的
                # 极小值（如 0.1153/0.0646），显示仍是 0.0、会被误读为"确定健康"。
                # 故用 ≤1.0 而非 ==0.0 判定低端饱和；高端饱和仍为 >=100.0。
                edge = (fb <= 1.0) | (fb >= 100.0)
                fb = np.where(edge, NEUTRAL_SEVERITY, fb)
                sev[to_fb] = fb[to_fb]
                ev[undet] = "acoustic_only"
                ev[sat_low] = "acoustic_only"
            except Exception:
                sev[to_fb] = NEUTRAL_SEVERITY
        band = np.where(sev < 35.0, "CTRL-like",
                        np.where(sev < 50.0, "borderline", "MCI-like"))
        out = pd.DataFrame({"severity_0_100": sev, "risk_band": band})
        out["ood_z"] = z
        out["saturated"] = sat
        out["evidence"] = ev
        if uuids is not None:
            out.insert(0, "uuid", list(uuids))
        return out

    def score_dataset_cv(self, X, y, uuids=None, n_splits=10):
        """leakage-free: 每条样本用未见它的折训练的两个边界模型打分。
        注意: 先取两边界的 raw decision (已按折内类均值定向, 保序), 组合后再做
        全局归一化到 [0,100]; 不要先各自归一化再平均(跨折 pooling 会破坏排序)。
        关键: 用散点赋值 sev[te_i]=... 而非 concatenate, 否则折序 != 原始序,
        severity 与 true_label 错位, AUC 失真。"""
        X = np.asarray(X, dtype=float)
        yarr = np.asarray(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        sev = np.empty(X.shape[0], dtype=float)
        for tr_i, te_i in skf.split(X, yarr):
            b1 = CognitiveScorer(classes=["CTRL", "IMPAIRED"], C=self.C, random_state=1)
            b2 = CognitiveScorer(classes=["NONAD", "AD"], C=self.C, random_state=1)
            y1 = np.isin(yarr[tr_i], ["MCI", "AD"]).astype(int)
            y2 = (yarr[tr_i] == "AD").astype(int)
            b1.fit(X[tr_i], y1)
            b2.fit(X[tr_i], y2)
            d1 = b1.decision(X[te_i])   # raw flipped decision (保序)
            d2 = b2.decision(X[te_i])
            sev[te_i] = 0.5 * d1 + 0.5 * d2
        lo, hi = float(sev.min()), float(sev.max())
        if hi > lo:
            sev = 100.0 * (sev - lo) / (hi - lo)
        out = pd.DataFrame({"severity_0_100": sev, "true_label": yarr})
        if uuids is not None:
            out.insert(0, "uuid", list(uuids))
        return out

    def save(self, path=SEVERITY_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path=SEVERITY_MODEL_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


def build_cn_severity_model():
    """部署版占位：训练数据不随包提供，禁止调用。源仓库训练用，见 ROUTE_A_swap_your_data.py。"""
    raise FileNotFoundError("训练数据(egemaps_final.csv 等)不随部署包提供；请加载 my_severity_combined.pkl。")


# =====================================================================
# MCI 居中 严重程度评分 —— combined (声学 eGeMAPS + 人口学 + 中文语言学)
# =====================================================================
COMBINED_MODEL_PATH = os.path.join(HERE, "cn_severity_combined.pkl")
DEMO_COLS = ["sex", "age", "education"]


def _load_linguistic_full():
    """部署版占位：训练用语言学特征来自源仓库，不随部署包提供。"""
    raise FileNotFoundError("训练用语言学特征(linguistic_features_full.csv)不随部署包提供。")


def build_cn_severity_combined_model():
    """部署版占位：在线推理不训练。serve.py/audio_to_score.py 直接加载同目录
    my_severity_combined.pkl; 缺失时调用本函数会抛 FileNotFoundError 由调用方捕获回退。"""
    raise FileNotFoundError("训练数据(egemaps_final.csv / 2_final_list_train.csv / transcripts_full)不随部署包提供；请加载 my_severity_combined.pkl。")


def assemble_combined_X(scorer, rows):
    """rows: list of {uuid, features:{...}, transcript:str} -> (X, uuids, modes)。
    features 需含 88 eGeMAPS + sex/age/education; 若提供 transcript 则在线抽语言学走
    combined, 否则 linguistic 用训练集均值填充 (mode='combined_imputed', 等同忽略语言学)。"""
    # 语言学特征 = 基础语言学 22 列 + 句法/命题 20 列 + 依存/分句复杂度 28 列 = 70 列。
    # 三个抽取器缺一不可：只抽前 22 列会让其余 48 列填 0 → 模型对所有人输出同一个值
    # （实测 /score 全同 42.94 的根因）。三者均有 jieba fallback，未装 stanza/缺模型不崩。
    from extract_linguistic import extract_text
    import extract_syntax_proposition as esp
    import extract_tongji_27d_syntax as etj
    cols = scorer.combined_feature_cols_
    ling_cols = scorer.ling_cols_
    acoustic_demo_cols = [c for c in cols if c not in ling_cols]
    ling_mean = scorer.ling_mean_
    uuids, mats, modes = [], [], []
    for r in rows:
        f = r.get("features", {})
        vec = []
        for c in acoustic_demo_cols:
            if c == "sex":   # 类别字段: 与训练一致 M->1 其他->0
                vec.append(1.0 if str(f.get("sex", "")).upper() == "M" else 0.0)
            else:
                vec.append(float(f.get(c, 0.0)))
        transcript = r.get("transcript")
        if transcript:
            lv = extract_text(transcript, uuid=r.get("uuid", ""))
            # 合并句法/命题(20) + 依存/分句复杂度(28)，补全 70 列语言学特征。
            lv.update(esp.extract_from_tsv_text(transcript, uuid=r.get("uuid", "")))
            lv.update(etj.extract_from_tsv_text(transcript, uuid=r.get("uuid", "")))
            ling_vec = [float(lv.get(c, 0.0)) for c in ling_cols]
            mode = "combined"
        else:
            ling_vec = [float(v) for v in ling_mean]
            mode = "combined_imputed"
        mats.append(vec + ling_vec)
        uuids.append(r.get("uuid", ""))
        modes.append(mode)
    return np.array(mats, dtype=float), uuids, modes


# =====================================================================
# 连续临床量表（MMSE/MoCA/ACE）回归 —— 真连续 0-100 严重度
# 放在本模块以保证 pickle 跨进程可反序列化（ROUTE_A 与 serve.py 共用）。
# =====================================================================
SCORE_SCALES = {"mmse": 30.0, "moca": 30.0, "ace": 100.0, "ace_r": 100.0, "ace-iii": 100.0}
SCORE_ALIASES = ["mmse", "moca", "ace", "ace_r", "ace-r", "aceiii", "ace-iii", "score"]


def detect_clinical_score(labels_df, score_col=None):
    """在 labels CSV 里探测连续临床量表列（MMSE/MoCA/ACE…）→ 归一化到 0-100。
    返回 (col_name, score_0_100 Series) 或 (None, None)。
    没有量表列 → (None, None)，上层回退 decision_function ordinal 代理分。"""
    lo2col = {str(c).lower(): c for c in labels_df.columns}
    if score_col:                       # 显式指定列名
        pick = lo2col.get(str(score_col).lower())
    else:                               # 自动：扫已知量表别名，避免误抓 uuid 等首列
        pick = next((lo2col[a] for a in SCORE_ALIASES if a in lo2col), None)
    if pick is None:
        return None, None
    s = pd.to_numeric(labels_df[pick], errors="coerce")
    s = s.dropna()
    if len(s) < 20:
        return None, None
    scale = SCORE_SCALES.get(str(pick).lower(), float(s.max()) or 100.0)
    s01 = np.clip(s / scale, 0.0, 1.0) * 100.0
    print(f"[INFO] 检测到连续临床量表列 '{pick}'（满分 {scale:g}）→ 归一化到 0-100 "
          f"（n={len(s)}，真实 mean={s.mean():.1f}）")
    return pick, s01


def clinical_band_thresholds(score_col):
    """按认知量表满分推 severity 临床切点（severity=100−量表%）。
    通用近似取满分 80%→低/中界、65%→中/高界（MMSE≥24 正常 / 20 以下重，符合 0.8/0.65）。
    未知名量表 → 回默认 (35, 50)。"""
    full = SCORE_SCALES.get(str(score_col).lower()) if score_col else None
    if not full:
        return (35.0, 50.0)
    return (100.0 * (1 - 0.80), 100.0 * (1 - 0.65))   # (20, 35)


def risk_band(s, binary=False, th=(35.0, 50.0)):
    """把 0-100 severity 划成三级风险带（方案A：不确定带）：
    th=(低,高)，默认 (35,50)：
       <低界  → CTRL-like（低风险）；低界≤x<高界 → borderline（灰色带，建议复测）；
       ≥高界  → MCI-like（高风险，疑似，含 AD）。
    边界 35/50 取全量 OOF 分布（健康中位34.7 / 障碍中位54.7，两组重叠→灰色带缓冲）。
    clinical_band_thresholds 推临床量表界；binary → 单一 MCI-high 界。"""
    if binary:
        return "MCI-high" if s >= 50.0 else "CTRL-like"
    return "CTRL-like" if s < th[0] else ("borderline" if s < th[1] else "MCI-like")


class _RegScorer:
    """连续临床量表回归器（MMSE/MoCA/ACE → 0-100）的薄封装。
    - fit(X, y_score) ：特征 → 连续量表分回归（折内/全量 RidgeCV 选 alpha）。
    - decision(X)     ：预测的 0-100 严重度（高=障碍重，目标已在调用侧做 100−量表）。
    - score(X, uuids) ：serve.py 兼容接口 → DataFrame(severity_0_100, risk_band)。
    单位 = 临床量表尺度（×100/满分后取反），语义与代理分一致：低=健康，高=异常。"""
    def __init__(self, alpha=1.0, random_state=1):
        self.alpha = alpha
        self.random_state = random_state
        self.pipe = None
        self.feature_cols_ = None

    def fit(self, X, y_score):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(X, dtype=float)
        self.pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler()),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 7))),
        ])
        self.pipe.fit(X, np.asarray(y_score, dtype=float))
        return self

    def decision(self, X):
        d = np.asarray(self.pipe.predict(X)).ravel()
        return np.clip(d, 0.0, 100.0)

    def risk_score(self, X):
        return self.decision(X)

    def score(self, X, uuids=None):
        """serve.py 兼容接口：返回 DataFrame(severity_0_100 + risk_band)。
        风险带按 self.score_col_ 走临床量表切点（写输出时烧录）。"""
        d = self.decision(X)
        th = clinical_band_thresholds(getattr(self, "score_col_", None))
        n = len(d)
        return pd.DataFrame({
            "uuid": uuids if uuids is not None else range(n),
            "severity_0_100": np.round(d, 4),
            "risk_band": [risk_band(x, th=th) for x in d],
        })
