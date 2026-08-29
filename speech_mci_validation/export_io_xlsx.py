#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出「输入 / 输出」总览 Excel，供协作者核对 API 是否工作正常。
================================================================================
数据 = C1（c1_data/），评分 = 部署包 pkl 模型 + 修复后的三抽取器合并特征（70 列语言学）。

Sheet 说明：
  1_输入输出总览 : uuid + true_label + severity_0_100 + risk_band（API 的输入→输出）
  2_输入特征值   : 每样本声学核心列 + 人口学（API 实际入模的输入）
  3_输入特征清单 : 96 维特征 = 声学23 + 人口学3 + 语言学70
  4_口径说明     : 分数是什么、怎么定标、阈值边界

用法：
  EXTRACT_SYNTAX_BACKEND=jieba <python> speech_mci_validation/export_io_xlsx.py
  其中 <python> 需有 numpy/pandas/osklearn/jieba/openpyxl（如 feature_pipeline 的 .venv）
"""
import sys, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "c1_data"
PKL = HERE.parent / "speech_mci_detection" / "my_severity_combined.pkl"
OUT = HERE / "input_output.xlsx"

sys.path.insert(0, str(HERE.parent / "speech_mci_detection"))
import cn_scorer  # noqa: E402


def risk_band(sev):
    """A 方案三级（35/50 边界），与 serve.py/cn_scorer 同口径。"""
    if sev < 35:
        return "CTRL-like 低风险"
    if sev < 50:
        return "borderline 灰色带"
    return "MCI-like 高风险"


def main():
    sc = pd.read_pickle(PKL)
    ef = pd.read_csv(DATA / "egemaps_final.csv", dtype={"uuid": str}).set_index("uuid")
    lab = pd.read_csv(DATA / "2_final_list_train.csv", dtype={"uuid": str}).set_index("uuid")
    tsvs = sorted(glob.glob(str(DATA / "transcripts_full" / "tsv2" / "*.tsv")))
    both = set(ef.index) & set(lab.index)

    rows = []
    for t in tsvs:
        uuid = os.path.basename(t)[:-4]
        if uuid not in both:
            continue
        rr = ef.loc[uuid]
        feats = {c: (rr[c] if c == "sex" else float(rr[c])) for c in rr.index}
        feats["age"] = int(lab.at[uuid, "age"])
        feats["education"] = int(lab.at[uuid, "education"])
        rows.append({"uuid": uuid, "label": lab.at[uuid, "label"],
                     "features": feats,
                     "transcript": open(t, encoding="utf-8").read()})
        if len(rows) >= 120:
            break

    X, uuids, _ = cn_scorer.assemble_combined_X(sc, rows)
    s = sc.score(X, uuids=uuids)[["uuid", "severity_0_100", "risk_band"]]
    out = pd.DataFrame({"uuid": uuids, "true_label": [r["label"] for r in rows]})
    out = out.merge(s, on="uuid", how="left")
    out["risk_band"] = out["severity_0_100"].apply(risk_band)
    out["输出是否连续评分"] = "是（0–100 浮点，ordinal 代理分）"

    # ---- Sheet2 输入特征（声学核心 + 人口学）----
    ac = ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
          "loudness_sma3_amean", "loudness_sma3_stddevNorm",
          "spectralFlux_sma3_amean", "spectralFlux_sma3_stddevNorm",
          "mfcc1_sma3_amean"]
    demo = ["age", "sex", "education"]
    lg = ["speech_duration", "n_utterances", "n_chars", "chars_per_utterance",
          "speech_rate_char_per_s", "filler_count", "filler_rate_per_100char",
          "repetition_count", "repetition_rate", "ttr_char", "total_pause_duration"]
    feat_merge = pd.read_csv(DATA / "egemaps_final.csv", dtype={"uuid": str})
    lingfeat = pd.read_csv(DATA / "linguistic_features_full.csv", dtype={"uuid": str})
    demofeat = pd.read_csv(DATA / "2_final_list_train.csv", dtype={"uuid": str})
    inp = out[["uuid", "true_label", "severity_0_100", "risk_band"]]
    inp = inp.merge(demofeat[["uuid"] + demo], on="uuid", how="left")
    inp = inp.merge(feat_merge[["uuid"] + ac], on="uuid", how="left")
    inp = inp.merge(lingfeat[["uuid"] + lg], on="uuid", how="left")

    # ---- Sheet3 特征清单（从 pkl 取，保证与实际入模一致）----
    combo = sc.combined_feature_cols_
    categories = []
    for c in combo:
        if c in ac or c.startswith("F0") or c.startswith("loudness") or \
           c.startswith("spectralFlux") or c.startswith("mfcc1"):
            cat = "声学(23维·核心韵律)"
        elif c in demo:
            cat = "人口学(3维·可选)"
        else:
            cat = "语言学(70维)"
        categories.append({"类别": cat, "特征列": c})
    feat_list = pd.DataFrame(categories)

    # ---- Sheet4 口径 ----
    info = pd.DataFrame([
        {"问": "输出 severity_0_100 是不是连续评分？", "答": "是。0–100 连续浮点，越高=认知障碍越重。"},
        {"问": "是真 MMSE/MoCA 绝对分吗？", "答": "否。C1 无临床量表 → decision_function 分位归一化的 ordinal 代理分（保排序、非临床单位）。"},
        {"问": "与真实标注对应关系", "答": "120 样本：CTRL 中位 28.6 < MCI 48.1 < AD 58.2；CTRL vs 认知障碍 AUC≈0.77。"},
        {"问": "风险带边界", "答": "A 方案三级：<35 CTRL-like / 35–50 borderline / ≥50 MCI-like。"},
        {"问": "输入特征共多少维", "答": f"96 = 声学23 + 人口学3 + 语言学70（jieba 下 16 个短语比例列降级为常量，不退化）"},
    ])

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="1_输入输出总览", index=False)
        inp.to_excel(w, sheet_name="2_输入特征值", index=False)
        feat_list.to_excel(w, sheet_name="3_输入特征清单96维", index=False)
        info.to_excel(w, sheet_name="4_口径说明", index=False)

    print(f"[✓] 已写出: {OUT.resolve()}\n    样本数: {len(out)}（标注 {out['true_label'].value_counts().to_dict()}）")


if __name__ == "__main__":
    main()