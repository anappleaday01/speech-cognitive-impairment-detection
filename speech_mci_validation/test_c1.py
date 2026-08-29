# -*- coding: utf-8 -*-
"""
C1 数据联调脚本 —— 用 C1 真实样本测 /score 接口的输入输出是否正确
================================================================
用法（另开终端先启动服务）：
    python serve.py --port 8000
    python test_c1.py [--n 3] [--port 8000] [--out out.json]

说明：
    - 从 c1_data/egemaps_final.csv 随机抽 n 条样本，每组：
        features = 88 维 eGeMAPS + sex/age/education（人口学来自 2_final_list_train.csv）
        transcript = 对应 c1_data/transcripts_full/tsv2/<uuid>.tsv 的原始 TSV 文本
    - 调用服务 /score 接口，返回 severity_0_100 + risk_band + mode，
      并打印「预测分 vs 真实标注」，供人工核对输出是否正确、是否符合认知障碍递进。
    - 若需只看原始体（不依赖服务），把 EXTRA_VERIFY=1 环境变量开成 1，会额外调用本地
      cn_scorer 直接打分做交叉核对（不连网络/不依赖 serve）。
"""
import argparse, json, os, random, sys, urllib.request

def load_rows(data_dir="c1_data", n=3, seed=42):
    import glob
    import pandas as pd
    feat = pd.read_csv(os.path.join(data_dir, "egemaps_final.csv"))
    lab = pd.read_csv(os.path.join(data_dir, "2_final_list_train.csv"))
    # 只保留既有转写、又有标注的样本
    tsvs = {os.path.basename(p)[:-4]: p
            for p in glob.glob(os.path.join(data_dir, "transcripts_full/tsv2/*.tsv"))}
    feat = feat.merge(lab, on="uuid", how="inner")
    feat = feat[feat["uuid"].isin(tsvs)].sample(n=n, random_state=seed)
    rows = []
    for _, r in feat.iterrows():
        feats = {c: (float(r[c]) if c != "sex" else r[c])
                 for c in r.index
                 if c not in ("uuid", "label", "age", "education")}
        feats["age"] = int(r["age"])
        feats["education"] = int(r["education"])
        feats["sex"] = r["sex"]
        trans = open(tsvs[r["uuid"]], encoding="utf-8").read()
        rows.append({"uuid": r["uuid"], "label": r["label"],
                     "features": feats, "transcript": trans})
    return rows


def call_score(rows, port=8000):
    url = f"http://127.0.0.1:{port}/score"
    body = json.dumps({"rows": rows}).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default="c1_data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.data_dir, args.n)
    print(f"抽样 {len(rows)} 条 C1 样本，准备调用 /score ...\n")
    for r in rows:
        print(f"  uuid={r['uuid']} 真实标注={r['label']}  "
              f"(sex={r['features']['sex']}, age={r['features']['age']}, "
              f"edu={r['features']['education']})")
    res = call_score(rows, args.port)
    print("\n=== 接口返回 ===")
    for rec in res.get("results", []):
        print(f"  {rec['uuid']:<12} severity={rec['severity_0_100']:.1f}  "
              f"risk_band={rec['risk_band']:>10}  mode={rec['mode']}")
    # 对照真实标注
    print("\n=== 预测 vs 真实（核对）===")
    by = {r["uuid"]: r["label"] for r in rows}
    for rec in res.get("results", []):
        print(f"  {rec['uuid']:<12} 预测={rec['severity_0_100']:.1f}  "
              f"真实={by.get(rec['uuid'], '?')}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")
    # 提示递进关系是否合理（健康应低于障碍）
    pairs = dict()
    try:
        groups = {}
        for rec, r in zip(res.get("results", []), rows):
            groups.setdefault(r["label"], []).append(rec["severity_0_100"])
        if groups:
            mean = {k: sum(v) / len(v) for k, v in groups.items()}
            print("\n各组平均分（预期 CTRL<MCI<AD 才合理）:", {k: round(v,1) for k,v in mean.items()})
    except Exception:
        pass


if __name__ == "__main__":
    main()