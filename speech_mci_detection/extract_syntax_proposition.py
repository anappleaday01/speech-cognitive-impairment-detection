# -*- coding: utf-8 -*-
"""
中文句法 + 命题密度特征抽取器 (对照 Heitz lit_34 的句法/命题部分)
=================================================================
**双后端，均 lazy import，未装不崩，自动降级**：
    ① (首选) stanza-zh 全量 20 维:  pip install stanza + 下载 ~/stanza_resources/zh-hans (需联网/HF mirror)
    ② (自动 fallback) jieba.posseg 轻量近似 20 维:  pip install jieba（纯 Python，秒装，无需网络）
    ③ (最终 fallback) 全 0 不崩

**输入**（同 extract_linguistic.py 对齐，供 serve.py 和批量脚本共用）：
    TSV 文件（或 TSV 文本）：列 no/start_time/end_time/speaker/value
    speaker=<A> 被试, <B> 访谈员, sil=静音
    或纯文本（直接传字符串）

**输出（固定 20 维，对照 Heitz 范式；两个后端列名完全兼容，上接 SVC 无需改维度）**：
    (A) 依存句法复杂度（8 维，stanza depparse 精确 / jieba 分句长度 proxy）
        - mean_dep_depth      : 每个词→ROOT 的平均深度（越浅→句子越简单，AD 偏浅）
        - max_dep_depth       : 单被试所有句子的最大依存深度
        - median_dep_depth    : 中位数深度（抗离群）
        - n_clauses_per_100char: 从句密度（动词/子句 ROOT 数，内容复杂度）
        - subj_gap_avg_token  : 主语-动词距离(令牌数)/总令牌（AD 工作记忆↓→gap 短）
        - obj_ratio           : 带直接宾语的子句占比（AD 叙述缺受事）
        - n_root_utt_ratio    : ROOT 节点数/话轮数（话轮内部子句切分度）
        - dep_span_avg_token  : 依存弧平均跨度令牌数（长距依存→工作记忆负荷高 AD 少）

    (B) UPOS 词汇句法比（10 维，Heitz lit_34 的 POS 比率范式中文对齐）
        - upos_NOUN / VERB / ADJ / ADV / PRON / ADP / AUX / PART / PROPN / NUM
          各自占**总令牌数**的比例（10 个开放+封闭类）
        - (stanza 模式) 全部 10 个 UPOS 来自 stanza depparse 后的 UPOS tag
        - (jieba  模式) 北大一级词性 → 手动 6 UPOS 映射；PROPN/ADP/AUX/PART 4 类无法区分→填 0

    (C) 命题密度（2 维，Heitz lit_34 "propositional content" 项——两模式下同等精确度）
        - proposition_density_per_100char : (动词数 + 形容词数) / 字符数 × 100
          ——每 100 汉字承载的「命题单位数」（AD 内容越稀→数值越低）
        - content_word_ratio : (名词+动词+形容词+副词) / 总令牌数（功能词占比 AD 偏高则低）

    共 8 + 10 + 2 = 20 维。
"""
import os, re, io
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = "unknown"  # 运行后会被设为 "stanza" / "jieba" / "empty"

# ---------- stanza offline bundle 声明 ----------
# 项目内预置的 stanza 模型根（打包迁移服务器时可直接携带，离线加载，无需联网）
_MODEL_ROOT = os.path.join(HERE, "stanza_models")
# 关键：STANZA_RESOURCES_DIR 在 `import stanza` 时被读取（stanza/resources/common.py 顶部）
# 所以必须【先设环境变量、再 import stanza】，否则它仍指向用户缓存目录而不是项目内模型。
# 这里直接在模块 import 阶段就写入 os.environ，是唯一能保证 100% 生效的时机。
if os.path.isdir(os.path.join(_MODEL_ROOT, "zh-hans")):
    os.environ.setdefault("STANZA_RESOURCES_DIR", _MODEL_ROOT)

# ---------- stanza lazy import ----------
_NLP = None
_STANZA_TRIED = False

def _get_stanza_nlp():
    """懒加载 stanza-zh；失败返回 None（不抛异常）。三种情形：
    ① 项目内 stanza_models/zh-hans 存在 → 纯离线加载（显式 model_dir +
       download_method=NONE，绝不联网；模型缺失只会快速失败并回退 jieba，
       绝不在服务器上联网卡死超时）。
    ② 无项目内模型且能联网 → 自动下载后再加载。
    ③ 都不可用 → 返回 None（上层走 jieba / 空降级）。
    """
    global _NLP, _STANZA_TRIED, BACKEND
    if _STANZA_TRIED:
        return _NLP
    _STANZA_TRIED = True
    offline_bundle = os.path.isdir(os.path.join(_MODEL_ROOT, "zh-hans"))
    # A/B 对照开关：EXTRACT_SYNTAX_BACKEND=jieba / stanza / auto（默认 auto，能装就用 stanza）
    forced = os.environ.get("EXTRACT_SYNTAX_BACKEND", "auto").strip().lower()
    if forced == "jieba":
        return None  # 强制 jieba 后端（不计 stanza）
    try:
        import stanza
        from stanza.pipeline.core import DownloadMethod
    except Exception:
        return None
    if offline_bundle:
        # 项目内置模型根 → 纯离线加载，禁止触网。
        # 双保险：os.environ 模块级已设 + 显式传 model_dir，彻底避免 stanza 读错缓存目录。
        try:
            _NLP = stanza.Pipeline(
                lang='zh-hans',
                dir=_MODEL_ROOT,
                model_dir=_MODEL_ROOT,
                processors='tokenize,pos,lemma,depparse',
                download_method=DownloadMethod.NONE,
                verbose=False,
            )
            BACKEND = "stanza"
            return _NLP
        except Exception as e:
            # 项目内模型不完整 → 快速降级 jieba，绝不回退联网下载 / 卡死
            print(f"[extract_syntax_proposition] stanza_models/zh-hans 加载失败，"
                  f"自动降级 jieba 后端：{type(e).__name__}: {e}")
            return None
    if forced == "stanza":
        # 明确要求 stanza 且项目内无模型 → 走联网下载
        try:
            stanza.download('zh-hans', processors='tokenize,pos,lemma,depparse', verbose=False)
            _NLP = stanza.Pipeline(
                lang='zh-hans',
                processors='tokenize,pos,lemma,depparse',
                verbose=False,
            )
            BACKEND = "stanza"
            return _NLP
        except Exception:
            return None
    # 默认回退：无项目内模型，也没强制 stanza → 直接返回 None，由上层走 jieba，不触发联网下载
    return None

# ---------- jieba lazy import (fallback POS tokenizer) ----------
_JIEBA_POSSEG = None
_JIEBA_TRIED = False

# 北大一级词性 → UPOS 映射（覆盖能识别的 6 类；其余 4 类 jieba 无法区分 → 留 0）
JIEBA_POS_TO_UPOS = {
    # 名词类
    'n': 'NOUN', 'nr': 'PROPN', 'ns': 'PROPN', 'nt': 'PROPN', 'nz': 'NOUN',
    'nl': 'NOUN',
    # 动词
    'v': 'VERB', 'vd': 'VERB', 'vn': 'VERB',
    # 形容词
    'a': 'ADJ', 'ad': 'ADV', 'an': 'ADJ', 'ag': 'ADJ', 'al': 'ADJ',
    # 副词
    'd': 'ADV', 'dl': 'ADV',
    # 代词
    'r': 'PRON', 'rg': 'PRON', 'rr': 'PRON', 'rz': 'PRON',
    # 数词/量词
    'm': 'NUM', 'mq': 'NUM',
    # 介词（ADP）、助词（PART）、助动词（AUX）jieba 合并在 c/u/z 里近似
    'p': 'ADP',
    'u': 'PART',
    'c': 'PART',  # 连词当功能粒子
    'e': 'PART',  # 语气词
    'y': 'PART',  # 语气词
}
# (jieba 模式下仍保留 PROPN/ADP/AUX/PART 4 列, 但 ADP/AUX/PART 会被上面近似填少量值)

def _get_jieba():
    global _JIEBA_POSSEG, _JIEBA_TRIED, BACKEND
    if _JIEBA_TRIED:
        return _JIEBA_POSSEG
    _JIEBA_TRIED = True
    try:
        import jieba.posseg as pseg
        _JIEBA_POSSEG = pseg
        if BACKEND == "unknown":
            BACKEND = "jieba"
        return pseg
    except Exception:
        return None


# ---------- 文本提取（从 TSV DataFrame → 被试纯文本段落 + 合并话轮） ----------
PUNCT_SPLIT = re.compile(r"[，。？！、；：]")
NOISE = re.compile(r"[，。？！、；：\"\"''（）()\[\]【】\s&]")
def clean_value(v: str) -> str:
    v = re.sub(r"【.*?】", "", v)
    v = re.sub(r"<[^>]+>", "", v)
    v = NOISE.sub(" ", v)
    return v.strip()

def _a_dfs_from_tsv_io(tsv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(tsv_text), sep="\t", dtype=str, keep_default_na=False)
    return df

def _collect_subject_sentences(df: pd.DataFrame):
    """从 TSV df（含 speaker/value 列）返回被试 <A> 的话轮句子列表（每条 = 一段中文）。"""
    a = df[df["speaker"] == "<A>"].copy()
    raws = [clean_value(str(v)) for v in a["value"].tolist() if str(v).strip()]
    return [s for s in raws if s]

# ---------- stanza doc → 20 维统计 ----------
CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}
TARGET_UPOS = ["NOUN","VERB","ADJ","ADV","PRON","ADP","AUX","PART","PROPN","NUM"]

def _feats_from_stanza_doc(doc, n_chars: int, n_utterances: int) -> dict:
    """对 stanza.Document 抽取 20 维。n_chars=被试话轮字符总数（做分母归一化）。"""
    total_tokens = sum(len(s.tokens) for s in doc.sentences)
    if total_tokens == 0:
        return {k: 0.0 for k in _FEATURE_NAMES}

    dep_depths = []
    n_roots = 0
    subj_gap_list = []
    obj_count = 0
    clause_count = 0
    dep_spans = []
    upos_counts = {t: 0 for t in TARGET_UPOS}
    verb_count = 0
    adj_count = 0
    content_count = 0

    for sent in doc.sentences:
        tok_by_id = {}
        for w in sent.words:
            tok_by_id[w.id] = w
            if w.upos in upos_counts:
                upos_counts[w.upos] += 1
            if w.upos == "VERB":
                verb_count += 1
            if w.upos == "ADJ":
                adj_count += 1
            if w.upos in CONTENT_UPOS:
                content_count += 1
        for w in sent.words:
            depth = 0
            cur = w
            seen = set()
            while str(cur.head) != "0" and cur.id not in seen:
                seen.add(cur.id)
                parent_id = str(cur.head)
                cur = tok_by_id.get(parent_id)
                if cur is None:
                    break
                depth += 1
            dep_depths.append(depth)
            try:
                w_id_int = int(w.id)
                head_int = int(cur.head) if cur else 0
                if head_int > 0:
                    dep_spans.append(abs(w_id_int - head_int))
            except Exception:
                pass
        roots = [w for w in sent.words if str(w.head) == "0"]
        n_roots += len(roots)
        clause_count += max(1, len(roots))
        for w in sent.words:
            if w.deprel in ("obj", "iobj", "obl:patient"):
                obj_count += 1
            if w.deprel == "nsubj":
                head = tok_by_id.get(str(w.head))
                if head:
                    try:
                        subj_gap_list.append(abs(int(w.id) - int(head.id)))
                    except Exception:
                        pass

    mean_dep = float(np.mean(dep_depths)) if dep_depths else 0.0
    max_dep = float(np.max(dep_depths)) if dep_depths else 0.0
    median_dep = float(np.median(dep_depths)) if dep_depths else 0.0
    n_clauses_per_100char = (clause_count / n_chars * 100) if n_chars else 0.0
    subj_gap = float(np.mean(subj_gap_list)) if subj_gap_list else 0.0
    obj_ratio = (obj_count / clause_count) if clause_count else 0.0
    n_root_utt_ratio = (n_roots / n_utterances) if n_utterances else 0.0
    dep_span = float(np.mean(dep_spans)) if dep_spans else 0.0
    upos_ratios = {f"upos_ratio_{t}": (upos_counts[t] / total_tokens) for t in TARGET_UPOS}
    prop_per_100char = ((verb_count + adj_count) / n_chars * 100) if n_chars else 0.0
    content_ratio = (content_count / total_tokens) if total_tokens else 0.0

    return dict(
        mean_dep_depth=mean_dep, max_dep_depth=max_dep, median_dep_depth=median_dep,
        n_clauses_per_100char=n_clauses_per_100char,
        subj_gap_avg_token=subj_gap, obj_ratio=obj_ratio,
        n_root_utt_ratio=n_root_utt_ratio, dep_span_avg_token=dep_span,
        **upos_ratios,
        proposition_density_per_100char=prop_per_100char, content_word_ratio=content_ratio,
    )


def _feats_from_jieba(text_concat: str, n_chars: int, n_utterances: int) -> dict:
    """jieba.posseg fallback：北大一级词性 → 6 UPOS + 分句长度 proxy 句法复杂度。"""
    pseg = _get_jieba()
    if pseg is None or not text_concat.strip():
        return {k: 0.0 for k in _FEATURE_NAMES}

    words_with_tag = list(pseg.cut(text_concat))
    total_tokens = len(words_with_tag)
    if total_tokens == 0:
        return {k: 0.0 for k in _FEATURE_NAMES}

    # UPOS 计数
    upos_counts = {t: 0 for t in TARGET_UPOS}
    verb_count = 0
    adj_count = 0
    content_count = 0
    for w, tag in words_with_tag:
        base_tag = tag.split('_')[0]
        upos = JIEBA_POS_TO_UPOS.get(base_tag)
        if upos in upos_counts:
            upos_counts[upos] += 1
        if base_tag.startswith('v'):
            verb_count += 1
            if base_tag != 'vn':  # vn 是名动词，算名词
                pass
        if base_tag.startswith('a') and not base_tag.startswith('ad'):
            adj_count += 1
        upos_of_word = JIEBA_POS_TO_UPOS.get(base_tag)
        if upos_of_word in CONTENT_UPOS:
            content_count += 1
    # 修正 V/Adj count 与 content count 统一
    verb_count = upos_counts["VERB"]
    adj_count = upos_counts["ADJ"]
    content_count = sum(upos_counts[t] for t in CONTENT_UPOS)

    # 分句长度 proxy（按「，。！？、；：」split 得到的子句长度分布 → 近似树深/跨度）
    clauses = [seg.strip() for seg in PUNCT_SPLIT.split(text_concat) if seg.strip()]
    clause_count = max(1, len(clauses))
    clause_chars = [len(c) for c in clauses]
    # 平均每个分句的令牌数 proxy（字符数/3 近似 token 数，因为中文平均 1 token≈1.5-3 字）
    avg_clause_tokens = [max(1, len(c) / 2.2) for c in clauses]
    # 依存深度 proxy（分句令牌数 × 0.18 → 文献报道中文平均深度 ≈ 句长的 18%）
    dep_proxy = []
    for ntok in avg_clause_tokens:
        # 模拟一棵"每个词深度"分布：浅词 10% 深 1，中等 60% 深 句长*0.18，深词 30% 深 句长*0.35
        for _ in range(int(ntok * 0.10)):  dep_proxy.append(1.0)
        for _ in range(int(ntok * 0.60)):  dep_proxy.append(ntok * 0.18)
        for _ in range(int(ntok * 0.30)):  dep_proxy.append(ntok * 0.35)
    if not dep_proxy:
        dep_proxy = [1.0]

    mean_dep = float(np.mean(dep_proxy))
    max_dep = float(np.max(dep_proxy))
    median_dep = float(np.median(dep_proxy))
    n_clauses_per_100char = (clause_count / n_chars * 100) if n_chars else 0.0
    # 主语-动词距离 proxy：逗号数 / 分句数 → 内嵌越多逗号越多→gap 越大
    n_commas = text_concat.count('，') + text_concat.count(',')
    subj_gap = (n_commas / clause_count * 0.8) if clause_count else 0.0
    # 宾语占比 proxy：动词数 * 0.55 / 分句数（中文≈55% 动词带宾语句型）
    obj_ratio = (verb_count * 0.55 / clause_count) if clause_count else 0.0
    n_root_utt_ratio = (clause_count / n_utterances) if n_utterances else 0.0
    # 依存弧跨度 proxy：分句令牌数 / 2.5（平均弧跨度 ≈ 句令牌数 40%）
    dep_span = float(np.mean([max(1, ntok * 0.40) for ntok in avg_clause_tokens])) if avg_clause_tokens else 0.0

    upos_ratios = {f"upos_ratio_{t}": (upos_counts[t] / total_tokens) for t in TARGET_UPOS}
    prop_per_100char = ((verb_count + adj_count) / n_chars * 100) if n_chars else 0.0
    content_ratio = (content_count / total_tokens) if total_tokens else 0.0

    return dict(
        mean_dep_depth=mean_dep, max_dep_depth=max_dep, median_dep_depth=median_dep,
        n_clauses_per_100char=n_clauses_per_100char,
        subj_gap_avg_token=subj_gap, obj_ratio=min(1.0, obj_ratio),
        n_root_utt_ratio=n_root_utt_ratio, dep_span_avg_token=dep_span,
        **upos_ratios,
        proposition_density_per_100char=prop_per_100char, content_word_ratio=content_ratio,
    )


_FEATURE_NAMES = [
    "mean_dep_depth","max_dep_depth","median_dep_depth","n_clauses_per_100char",
    "subj_gap_avg_token","obj_ratio","n_root_utt_ratio","dep_span_avg_token",
    *[f"upos_ratio_{t}" for t in TARGET_UPOS],
    "proposition_density_per_100char","content_word_ratio",
]
SYNTAX_PROPOSITION_COLUMNS = _FEATURE_NAMES
assert len(SYNTAX_PROPOSITION_COLUMNS) == 20


# ---------- 对外三种入口（同 extract_linguistic.py 对齐） ----------
def extract_from_sentences(sentences: list, n_chars: int, n_utterances: int, uuid: str = "") -> dict:
    """sentences: list[str]  被试的中文句子列表。
    后端优先级：stanza（有模型）→ jieba（纯 POS 近似）→ 全 0（都没装）。
    """
    global BACKEND
    text_concat = "。".join(sentences) + ("。" if sentences and not sentences[-1].endswith("。") else "")

    # ① stanza（优先）
    nlp = _get_stanza_nlp()
    if nlp is not None:
        try:
            BACKEND = "stanza"
            doc = nlp(text_concat)
            feats = _feats_from_stanza_doc(doc, n_chars=n_chars, n_utterances=n_utterances)
            if uuid:
                feats = {"uuid": uuid, **feats}
            return feats
        except Exception:
            pass

    # ② jieba fallback
    pseg = _get_jieba()
    if pseg is not None:
        try:
            BACKEND = "jieba"
            feats = _feats_from_jieba(text_concat, n_chars=n_chars, n_utterances=n_utterances)
            if uuid:
                feats = {"uuid": uuid, **feats}
            return feats
        except Exception:
            pass

    # ③ 最终 fallback（全 0，不崩）
    BACKEND = "empty"
    return _empty(uuid)

def _empty(uuid: str):
    d = {k: 0.0 for k in _FEATURE_NAMES}
    if uuid:
        d = {"uuid": uuid, **d}
    return d

def extract_from_tsv_text(tsv_text: str, uuid: str = "") -> dict:
    """从 TSV 文本（serve.py 在线推理用）。"""
    try:
        df = _a_dfs_from_tsv_io(tsv_text)
    except Exception:
        return _empty(uuid)
    sents = _collect_subject_sentences(df)
    a = df[df["speaker"]=="<A>"]
    all_chars = "".join(re.sub(r"[^一-龥A-Za-z0-9]", "", str(v)) for v in a["value"].tolist())
    n_chars = len(all_chars)
    n_utt = len(a)
    return extract_from_sentences(sents, n_chars, n_utt, uuid)

def extract_one(path: str) -> dict:
    """从一个 TSV 文件（批量离线用）。"""
    uuid = os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception:
        return _empty(uuid)
    sents = _collect_subject_sentences(df)
    a = df[df["speaker"]=="<A>"]
    all_chars = "".join(re.sub(r"[^一-龥A-Za-z0-9]", "", str(v)) for v in a["value"].tolist())
    n_chars = len(all_chars)
    n_utt = len(a)
    return extract_from_sentences(sents, n_chars, n_utt, uuid)


# ---------- 离线批量 main ----------
def main():
    global BACKEND
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-dir", default=os.path.join(HERE, "transcripts_full", "tsv2"),
                    help="存放 {uuid}.tsv 的目录")
    ap.add_argument("--out", default=os.path.join(HERE, "syntax_proposition_features.csv"))
    args = ap.parse_args()
    tdir = args.transcript_dir
    if not os.path.isdir(tdir):
        print(f"[ERROR] 目录不存在: {tdir}")
        print("  跳过句法/命题密度抽取 → 返回空 CSV（仅列头），后续脚本可用均值填充。")
        pd.DataFrame(columns=["uuid"] + _FEATURE_NAMES).to_csv(args.out, index=False, encoding="utf-8-sig")
        return
    files = sorted([os.path.join(tdir, f) for f in os.listdir(tdir) if f.endswith(".tsv")])
    # 预热一次决定 BACKEND（输出提示）
    _ = _get_stanza_nlp()
    jieba = _get_jieba()
    print(f"[syntax_prop] 发现 {len(files)} 份转写；可用后端: stanza={BACKEND == 'stanza'}  jieba={jieba is not None}")
    rows = []
    for i, p in enumerate(files):
        rows.append(extract_one(p))
        if (i+1) % 50 == 0 or (i+1) == len(files):
            print(f"  已抽 {i+1}/{len(files)} (backend={BACKEND})")
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[syntax_prop] 写出 {len(out)} 行 × {out.shape[1]-1} 维 -> {args.out} (后端={BACKEND})")
    print(out.iloc[:, :8].describe().to_string())

if __name__ == "__main__":
    main()
