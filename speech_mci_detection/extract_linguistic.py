# -*- coding: utf-8 -*-
from __future__ import annotations   # PEP 563：允许在 py3.8/3.9 下使用 `X | Y` / `list[str]` 新式类型注解，不做运行时求值
"""
中文语言学特征抽取器 (C1 数据集, picture description 协议)
================================================================
输入 : transcripts/ 下 9 份 TSV (列 no/start_time/end_time/speaker/value)
       speaker=<A> 为被试(老人)，<B> 为访谈员，sil=静音
       value 中 '&' 为句中停顿/重启分隔，【...】为方言标记，嗯/啊 为填充词
输出 : linguistic_features.csv  (uuid + 中文语言学特征，按 uuid 对齐)

设计原则 (对照 Heitz lit_34 的中文改造, 见 CN_MIGRATION_PLAN.md §2.2.1)：
  - 基础 14 维规则化中文统计（不依赖外部包，可跑）：基础计数 / 词汇丰富度(TTR) / 停顿 / 重复度
  - 填充词率、方言标记为中文特有、直接可解释
  - (Tier 1 增强, 可选) 词频常模 2 维：avg_lg10wf + low_freq_ratio，来自 SUBTLEX-CH
    （免费下载 http://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexch ）
    需要 jieba 分词 + 一个 SUBTLEX-CH 文件（默认路径 $SUBTLEX_CH_PATH 或同目录 SUBTLEXCH.xlsx/.csv）。
    两者任一缺失 → 两列填 NaN，后续 SimpleImputer 自动补训练集均值，不崩。

注：本仓库仅含 9 份转写，故该表只覆盖 9 名说话人；
    若自有 app 数据含全部说话人转写，同一脚本可直接扩展到全量。
"""
import os, re, io
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRANS_DIR = os.path.join(HERE, "transcripts")
OUT = os.path.join(HERE, "linguistic_features.csv")

# 中文填充词 / 停顿标记 (演示用规则集，可随数据扩充)
FILLERS = ["嗯", "啊", "呃", "哎", "哦", "嘛", "呢", "那个", "这个",
           "就是说", "然后", "的话", "其实", "好像", "可能"]
# 去除的噪声符号
NOISE = re.compile(r"[，。？！、；：\"\"''（）()\[\]【】\s&]")

# -------------------- SUBTLEX-CH 词频常模（Tier 1 可选增强）--------------------
# 启发：AD/MCI 倾向于回避低频词（命名困难/词汇枯竭），→ avg_lg10wf 偏高 / low_freq_ratio 偏低
# 是跨语言稳定信号（Heitz sung_5 包含 word frequency，对应词常模维度）。
_SUBTLEX_DF_CACHE = {"df": None, "tried": False, "msg": ""}


# ---------------- Tier 3 增强：中文心理语言学词常模（AoA/具体性/熟悉度）---------------
# 启发：对应 Heitz sung_5 后 3 项（age of acquisition / concreteness / familiarity）。
# 命名性退化时 AD/MCI 更依赖习得早、具体、熟悉的词（语义存储损伤 → 回避抽象/低频/晚习得词）。
# 数据源：公开中文词常模（如 BLP-Chinese / Liu 等 — 待确认下载地址，见 CN_MIGRATION_PLAN §2.3）。
# 每个维度独立可控、缺失自动 NaN 零退化，与 Tier 1 SUBTLEX 完全同款 graceful 机制：
#   - env {NORM}_PATH 指定文件（.csv/.tsv/.xlsx 自动嗅探）
#   - env {NORM}_WORD_COL / {NORM}_VALUE_COL 覆盖列名
#   其中 {NORM} ∈ {AOA, CONC, FAMI}。
_NORM_CACHE: dict[str, dict] = {
    "aoa": {"df": None, "tried": False, "msg": ""},   # Age of Acquisition 习得年龄（越低越早习得）
    "conc": {"df": None, "tried": False, "msg": ""},  # Concreteness 具体性（越具体越 1-5 高分）
    "fami": {"df": None, "tried": False, "msg": ""},  # Familiarity 熟悉度（越常见越高分）
}

# 各维度默认列名（pipeline 到数据源凝视后按需覆盖，宽松大小写识别）
_NORM_WORD_KEYS = {"aoa": ("word",), "conc": ("word", "word_cn"), "fami": ("word",)}
# 值列逻辑名 → 可能物理列名
_NORM_VAL_HINTS = {
    "aoa": ("aoa", "age_acqu", "aoa_sub"),
    "conc": ("conc", "concreteness", "conc_sub"),
    "fami": ("fami", "familiarity", "familiar"),
}


def _norm_default_fname(norm: str) -> list[str]:
    """该维度的候选文件名（宽松，将放 extract_linguistic.py 同目录）。"""
    prefixes = ["chinese_", "cn_", "zh_", ""]
    tags = {
        "aoa": ["AoA", "ageofacquisition", "aoa", "AGEACQ"],
        "conc": ["Concreteness", "concreteness", "conc"],
        "fami": ["Familiarity", "familiarity", "fami"],
    }
    exts = [".csv", ".tsv"]
    out = []
    for p in prefixes:
        for t in tags[norm]:
            for e in exts:
                out.append(f"{p}{t}{e}")
    return out


def _find_norm_default_path(norm: str):
    env = os.environ.get(f"{norm.upper()}_PATH")
    if env and os.path.isfile(env):
        return env
    for cand in _norm_default_fname(norm):
        p = os.path.join(HERE, cand)
        if os.path.isfile(p):
            return p
    return None


def _read_norm_df(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".xlsx"):
        return _read_xlsx_with_header(path)
    sep = "\t" if low.endswith(".tsv") else None
    return pd.read_csv(path, sep=sep, engine="python",
                       encoding="utf-8-sig", on_bad_lines="skip")


def load_word_norm(norm: str) -> pd.DataFrame | None:
    """加载某一中文词常模维度（aoa/conc/fami）并缓存；缺失 → None 不抛异常。
    与 load_subtlex_ch 同款：env 指定路径 → 同目录候选文件 → 自动嗅探 xlsx/csv/tsv。
    值列宽松识别（大小写不敏感 + _NORM_VAL_HINTS 别名），同词取均值。
    """
    key = norm
    if _NORM_CACHE[key]["tried"]:
        return _NORM_CACHE[key]["df"]
    _NORM_CACHE[key]["tried"] = True
    path = _find_norm_default_path(key)
    if path is None:
        _NORM_CACHE[key]["msg"] = (
            f"未找到中文{('AoA' if key=='aoa' else '具体性' if key=='conc' else '熟悉度')}常模文件"
            f"（可用 env {key.upper()}_PATH=/path/… 指定，或把候选文件放到 extract_linguistic.py 同目录）。"
            f"该列填 NaN，不影响其余特征。")
        return None
    try:
        df = _read_norm_df(path)
    except Exception as exc:
        _NORM_CACHE[key]["msg"] = f"常模文件读错 ({os.path.basename(path)}): {exc!r}。该列填 NaN。"
        return None
    lower_to_col = {str(c).strip().lower(): str(c) for c in df.columns}
    wcol = os.environ.get(f"{key.upper()}_WORD_COL") or next(
        (lower_to_col[k] for k in _NORM_WORD_KEYS[key] if k in lower_to_col), None)
    vcol = os.environ.get(f"{key.upper()}_VALUE_COL") or next(
        (lower_to_col[k] for k in _NORM_VAL_HINTS[key] if k in lower_to_col), None)
    if not wcol or not vcol:
        _NORM_CACHE[key]["msg"] = (
            f"常模({key})缺列：需词形列 + 值列，实际列={list(df.columns)[:10]}…。该列填 NaN；"
            f"列名不同请设 env {key.upper()}_WORD_COL / {key.upper()}_VALUE_COL。")
        return None
    df = df[[wcol, vcol]].copy()
    df.columns = ["word", "value"]
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"] != ""]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    # 同词取均值（不同于 SUBTLEX 的 max，因为常模值无单调方向）
    df = df.groupby("word", as_index=False)["value"].mean()
    _NORM_CACHE[key]["df"] = df
    label = {"aoa": "AoA", "conc": "Concreteness", "fami": "Familiarity"}[key]
    _NORM_CACHE[key]["msg"] = (f"中文{label}常模加载 OK：{os.path.basename(path)} → {len(df):,} 词。"
                               f"avg_{key} 已启用。")
    return df


def norm_status_msg() -> str:
    """返回 Tier 3 三组常模加载状态汇总（serve 健康检查打印）。"""
    lines = []
    labels = {"aoa": "AoA", "conc": "Concreteness", "fami": "Familiarity"}
    for key in ("aoa", "conc", "fami"):
        if not _NORM_CACHE[key]["tried"]:
            load_word_norm(key)
        lines.append(f"  - {labels[key]}: {_NORM_CACHE[key]['msg']}")
    return "Tier 3 中文词常模（AoA/具体性/熟悉度）状态：\n" + "\n".join(lines)


def _word_norm_features(tokens: list[str] | None, norms: dict[str, pd.DataFrame | None]) -> dict:
    """基于三组中文词常模，每个维度输出 2 个通道：词频加权均值 + 命中占比。
    共 6 维：avg_aoa/hit_aoa, avg_conc/hit_conc, avg_fami/hit_fami。
    - avg_* = 转写内词频(TF) 加权的常模值均值：一 份转写里高频出现的词权重更大，
      抑制偶发罕见词的噪声。A/B 实测 TFW 在 HC-vs-MCI 提升最均衡（+0.0076）。
    - hit_*  = 命中的去重词数 / 去重词总数，放大概率信号，弥补均值覆盖稀疏。
    前置缺失（jieba / 任一 norm / 空 tokens）→ 缺维 NaN，其余照算。
    """
    out = {"avg_aoa": np.nan, "avg_conc": np.nan, "avg_fami": np.nan,
           "hit_aoa": np.nan, "hit_conc": np.nan, "hit_fami": np.nan}
    if not tokens:
        return out
    from collections import Counter
    tf = Counter(tokens)
    n_uniq = len(tf)
    if n_uniq == 0:
        return out
    keys = {"avg_aoa": "aoa", "avg_conc": "conc", "avg_fami": "fami"}
    for out_col, key in keys.items():
        df = norms.get(key)
        if df is None or len(df) == 0:
            continue
        w2v = dict(zip(df["word"].values, df["value"].values))
        wsum = np.sum([c for t, c in tf.items() if t in w2v])
        wval = np.sum([c * w2v[t] for t, c in tf.items() if t in w2v])
        out[out_col.replace("avg_", "hit_")] = float(np.sum([1 for t in tf if t in w2v]) / n_uniq)
        if wsum > 0:
            out[out_col] = float(wval / wsum)
    return out

def _find_subtlex_default_path():
    """按优先级找 SUBTLEX-CH 文件：env SUBTLEX_CH_PATH → 同目录 SUBTLEXCH.xlsx → SUBTLEXCH.csv。"""
    env = os.environ.get("SUBTLEX_CH_PATH")
    if env and os.path.isfile(env):
        return env
    for cand in ["SUBTLEXCH.xlsx", "SUBTLEXCH.CSV", "SUBTLEXCH.csv",
                 "SUBTLEX-CH.xlsx", "SUBTLEX-CH.csv",
                 "SUBTLEX-CH-WF.txt", "SUBTLEX-CH-WF.csv", "SUBTLEX-CH-WF.xlsx"]:
        p = os.path.join(HERE, cand)
        if os.path.isfile(p):
            return p
    return None


def _read_xlsx_with_header(path: str) -> pd.DataFrame:
    """读官方 SUBTLEX-CH-WF.xlsx：其首 1–2 行是说明（如 'Total word count: ...'），
    真正的表头是 'Word' 所在的那一行。自动定位并返回以该行为表头的 DataFrame。"""
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    header_idx = None
    for i in range(min(5, len(raw))):
        first = str(raw.iloc[i, 0]).strip().lower()
        if first in ("word", "一个词一"):
            header_idx = i
            break
    if header_idx is None:
        # 回退：假定表头就在第 1 行（0-based 0）
        header_idx = 0
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = [str(c) if pd.notna(c) else f"col{i}" for i, c in enumerate(raw.iloc[header_idx])]
    return df

def load_subtlex_ch() -> pd.DataFrame | None:
    """加载 SUBTLEX-CH 词频常模并缓存（单例）；缺失则返回 None，不抛异常。

    加载行为：
      - 优先 env SUBTLEX_CH_PATH，否则在 extract_linguistic.py 同目录下搜 SUBTLEXCH.*
      - .xlsx 需要 openpyxl（`python3 -m pip install openpyxl --user`）
      - 默认 Word 列 = 「Word」、log10 词频列 = 「lg10WF」（与 SUBTLEX-CH 官方发布版列名一致）；
        可通过 env SUBTLEX_CH_WORD_COL / SUBTLEX_CH_LG10WF_COL 覆盖。
      - 任何异常（找不到 / 读错 / 缺列）→ 返回 None，并将原因记录在 _SUBTLEX_DF_CACHE["msg"]。
    """
    if _SUBTLEX_DF_CACHE["tried"]:
        return _SUBTLEX_DF_CACHE["df"]
    _SUBTLEX_DF_CACHE["tried"] = True
    path = _find_subtlex_default_path()
    if path is None:
        _SUBTLEX_DF_CACHE["msg"] = ("未找到 SUBTLEX-CH 文件（可用 env SUBTLEX_CH_PATH=/path/to/SUBTLEXCH.xlsx 指定，"
                                    "或把 SUBTLEXCH.xlsx/.csv 放到 extract_linguistic.py 同目录）。"
                                    "avg_lg10wf/low_freq_ratio 两列将填 NaN，不影响其余 14 维。")
        return None
    try:
        if path.lower().endswith(".xlsx"):
            df = _read_xlsx_with_header(path)
        else:
            # 官方 SUBTLEX-CH-WF 解压出来是制表符分隔的 .txt；用白话自动嗅探分隔符
            df = pd.read_csv(path, sep=None, engine="python",
                             encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as exc:
        _SUBTLEX_DF_CACHE["msg"] = f"SUBTLEX-CH 文件读错 ({os.path.basename(path)}): {exc!r}。两列填 NaN。"
        return None
    wcol = os.environ.get("SUBTLEX_CH_WORD_COL")
    fcol = os.environ.get("SUBTLEX_CH_LG10WF_COL")
    # 官方 SUBTLEX-CH-WF.xlsx 的列名是「Word」和「logW」（Cai & Brysbaert 2010）；
    # SUBTLEX-CH 另一版本用「Lg10WF」。宽松识别：列名大小写不敏感，并兼容 logW/lg10wf。
    lower_to_col = {str(c).strip().lower(): str(c) for c in df.columns}
    wcol = wcol or next((lower_to_col[k] for k in ("word",) if k in lower_to_col), None)
    fcol = fcol or next((lower_to_col[k] for k in ("logw", "lg10wf", "lgwf") if k in lower_to_col), None)
    if not wcol or not fcol:
        _SUBTLEX_DF_CACHE["msg"] = (f"SUBTLEX-CH 缺列：需要 'Word'(词形) 和 'Lg10WF'(log10 词频)，"
                                    f"实际列={list(df.columns)[:10]}…。两列填 NaN；若列名不同请设置 "
                                    "env SUBTLEX_CH_WORD_COL / SUBTLEX_CH_LG10WF_COL 指定。")
        return None
    df = df[[wcol, fcol]].copy()
    df.columns = ["word", "lg10wf"]
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"] != ""]
    df["lg10wf"] = pd.to_numeric(df["lg10wf"], errors="coerce")
    df = df.dropna(subset=["lg10wf"])
    # 同词去重：取最大 lg10wf
    df = df.groupby("word", as_index=False)["lg10wf"].max()
    _SUBTLEX_DF_CACHE["df"] = df
    _SUBTLEX_DF_CACHE["msg"] = (f"SUBTLEX-CH 加载 OK：{os.path.basename(path)} → {len(df):,} 词。"
                                f"两列 avg_lg10wf / low_freq_ratio 已启用。")
    return df

def subtlex_status_msg() -> str:
    """返回 SUBTLEX-CH 加载状态的诊断字符串（供脚本 / serve.py 健康检查打印）。"""
    if not _SUBTLEX_DF_CACHE["tried"]:
        load_subtlex_ch()   # 触发一次加载尝试
    return _SUBTLEX_DF_CACHE["msg"]

def _jieba_tokenize(text_chars: str):
    """jieba 分词（懒加载）；缺失 jieba 返回 None。不把 jieba 放进顶层依赖以免纯规则 14 维受影响。"""
    try:
        import jieba  # 首次 import 会加载 dict，约 50–150ms；之后缓存
    except Exception:
        return None
    tokens = [t for t in jieba.lcut(text_chars) if t and not NOISE.fullmatch(t)]
    return tokens

def _word_freq_features(tokens: list[str] | None, subtlex_df: pd.DataFrame | None) -> dict:
    """基于 SUBTLEX-CH 词频常模输出 2 维：avg_lg10wf, low_freq_ratio(lg10WF < 3.0 的词占比)。

    前置缺失（jieba / subtlex_df / 空 tokens）→ 返回 {NaN, NaN}；scorer 内 SimpleImputer 会补训练均值。
    low_freq 阈值 3.0 = 词频 < 1000 / 百万（≈ 次常用词以下，命名困难时会回避）。可通过 env
    SUBTLEX_LOW_FREQ_THRESHOLD 自定义阈值。
    """
    if subtlex_df is None or not tokens:
        return {"avg_lg10wf": np.nan, "low_freq_ratio": np.nan}
    # 建词→频 dict
    w2f = dict(zip(subtlex_df["word"].values, subtlex_df["lg10wf"].values))  # ~8MB，可接受
    hits = [w2f[t] for t in tokens if t in w2f]
    if not hits:
        return {"avg_lg10wf": np.nan, "low_freq_ratio": np.nan}
    thr = float(os.environ.get("SUBTLEX_LOW_FREQ_THRESHOLD", "3.0"))
    arr = np.asarray(hits, dtype=float)
    return {
        "avg_lg10wf": float(arr.mean()),
        "low_freq_ratio": float((arr < thr).mean()),
    }

def clean_value(v: str) -> str:
    """去掉方言标记、标点、&，保留中文与ascii字母数字用于统计。"""
    v = re.sub(r"【.*?】", "", v)          # 方言标记
    v = re.sub(r"<[^>]+>", "", v)          # 任何 <...> 标记
    v = NOISE.sub("", v)
    return v

def count_fillers(text: str) -> int:
    c = 0
    for f in FILLERS:
        c += text.count(f)
    return c

def extract_text(tsv_text: str, uuid: str = "") -> dict:
    """从 TSV 文本(而非文件)抽取一名说话人的语言学特征, 供 serve.py 在线推理。
    tsv_text 为图片描述任务转写, 列 no/start_time/end_time/speaker/value,
    speaker=<A> 为被试, <B> 为访谈员, sil=静音, value 中 & 为停顿/重启、
    【...】为方言标记、嗯/啊 为填充词。"""
    df = pd.read_csv(io.StringIO(tsv_text), sep="\t", dtype=str, keep_default_na=False)
    return _stats(df, uuid)


def extract_one(path: str) -> dict:
    """从 TSV 文件抽取 (保留旧接口, 9 份 transcripts/ 与全量 transcripts_full/ 通用)。"""
    uuid = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return _stats(df, uuid)


def _stats(df: pd.DataFrame, uuid: str) -> dict:
    a = df[df["speaker"] == "<A>"].copy()          # 被试话轮
    sil = df[df["speaker"] == "sil"].copy()         # 静音段
    def dur(sub):
        s = pd.to_numeric(sub["start_time"], errors="coerce")
        e = pd.to_numeric(sub["end_time"], errors="coerce")
        d = (e - s)
        return d.fillna(0).sum()
    speech_dur = dur(a)
    # 文本
    raw_vals = a["value"].tolist()
    cleaned = [clean_value(v) for v in raw_vals]
    all_chars = "".join(cleaned)
    n_chars = len(all_chars)
    n_utt = len(raw_vals)
    # 填充词
    filler = sum(count_fillers(v) for v in raw_vals)
    # 重复：相邻话轮完全相同的"重启/重复" (清理后)
    reps = sum(1 for i in range(1, len(cleaned)) if cleaned[i] and cleaned[i] == cleaned[i-1])
    # TTR (字符级，中文无空格分词，用字级近似)
    uniq = len(set(all_chars)) if all_chars else 0
    ttr = uniq / n_chars if n_chars else 0.0
    # 方言标记
    dialect = sum(v.count("【") for v in raw_vals)
    # 停顿
    s = pd.to_numeric(sil["start_time"], errors="coerce")
    e = pd.to_numeric(sil["end_time"], errors="coerce")
    sil_lens = (e - s).fillna(0).values
    long_pause = int((sil_lens > 1.0).sum())
    mean_pause = float(sil_lens.mean()) if len(sil_lens) else 0.0
    # 语速
    rate = n_chars / speech_dur if speech_dur > 0 else 0.0
    # (Tier 1 可选) 词频常模 2 维：SUBTLEX-CH + jieba 分词
    #   - 先触发一次 SUBTLEX 加载尝试（缓存结果），serve 健康检查会打印诊断
    subtlex_df = load_subtlex_ch()
    tokens = _jieba_tokenize(all_chars) if n_chars else None
    freq_feats = _word_freq_features(tokens, subtlex_df)
    # (Tier 3 可选) 中国词常模 3 维：AoA/具体性/熟悉度（缺失自动 NaN）
    tg3_norms = {k: load_word_norm(k) for k in ("aoa", "conc", "fami")}
    tg3_feats = _word_norm_features(tokens, tg3_norms)
    return dict(
        uuid=uuid,
        speech_duration=speech_dur,
        n_utterances=n_utt,
        n_chars=n_chars,
        chars_per_utterance=n_chars / n_utt if n_utt else 0.0,
        speech_rate_char_per_s=rate,
        filler_count=filler,
        filler_rate_per_100char=(filler / n_chars * 100) if n_chars else 0.0,
        repetition_count=reps,
        repetition_rate=reps / n_utt if n_utt else 0.0,
        ttr_char=ttr,
        dialect_marker_count=dialect,
        total_pause_duration=float(sil_lens.sum()),
        mean_pause_duration=mean_pause,
        long_pause_count=long_pause,
        # --- Tier 1：SUBTLEX-CH 词频常模（缺失自动 NaN，不影响上游）---
        avg_lg10wf=freq_feats["avg_lg10wf"],
        low_freq_ratio=freq_feats["low_freq_ratio"],
        # --- Tier 3：中文词常模（AoA/具体性/熟悉度，缺失自动 NaN）---
        avg_aoa=tg3_feats["avg_aoa"],
        avg_conc=tg3_feats["avg_conc"],
        avg_fami=tg3_feats["avg_fami"],
        hit_aoa=tg3_feats["hit_aoa"],
        hit_conc=tg3_feats["hit_conc"],
        hit_fami=tg3_feats["hit_fami"],
    )

def main():
    files = [os.path.join(TRANS_DIR, f) for f in os.listdir(TRANS_DIR) if f.endswith(".tsv")]
    # 打印一次 SUBTLEX 状态 + Tier3 常模状态，用户一眼看出是否启用增强
    print(f"[extract_linguistic] SUBTLEX-CH 状态：{subtlex_status_msg()}")
    print(norm_status_msg())
    rows = [extract_one(p) for p in sorted(files)]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"抽取 {len(out)} 名说话人的中文语言学特征 ({out.shape[1]-1} 维) -> {OUT}")
    # 打印 14 原 + 2 新 列覆盖率（NaN 情况）
    cols = [c for c in out.columns if c != "uuid"]
    cov = (out[cols].notna().mean() * 100).round(1)
    print("[coverage %] " + "  ".join(f"{c}={cov[c]}%" for c in cols))
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
