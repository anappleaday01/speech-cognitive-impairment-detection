# -*- coding: utf-8 -*-
"""
中文认知风险 评分结果接口 (serve.py) —— combined (声学+语言学) 版
================================================================
零依赖 (仅 stdlib http.server) 的 HTTP 评分服务, 现场演示用。

启动:
    python serve.py            # 监听 http://127.0.0.1:8000
    python serve.py --port 9000

接口:
    GET  /health
        -> {"status":"ok","model":"C1 severity (combined)",
            "n_train":323,"feature_dim":96,
            "bands":{"CTRL":"<35","borderline":"35–50","MCI":">=50"}}

    POST /score   body={"uuid":"P0001_0017",
                        "features":{88 eGeMAPS..., "sex":"F","age":52,"education":9},
                        "transcript":"no\\tstart...\\t<A>\\t图片上有哪些人..."}
        -> {"uuid","severity_0_100","risk_band","mode"}
           mode = "combined" (给了转写) | "combined_imputed" (无转写, 语言学填训练均值)

    POST /score   body={"rows":[{uuid,features,transcript?}, ...]}   # 批量(数据集)
        -> {"results":[...], "n":N}

    POST /score_audio   音频端到端：POST 原始音频二进制作请求体
        ?sex=F&age=72&education=6       可选人口学
        -> {"severity_0_100","risk_band","mode":"combined","asr_backend":...}
        内部 = openSMILE 88 eGeMAPS + Whisper ASR 转写 -> assemble_combined_X -> 评分。
        需装有 opensmile + faster-whisper(或 transformers)；否则返回 503。
        注意：评分模型按中文 C1 图片描述任务训练，英文/自由对话音频仅作管道验证，
            分数对中文画述场景才有意义。

说明:
    - 模型在启动时训练 (C1 MCI 居中 combined 严重程度评分, 323人 CTRL/MCI/AD,
      105 维 = 88 eGeMAPS + 3 人口学 + 14 中文语言学, 复用 Heitz SVC C=0.1 balanced)
    - features 需含 88 维 eGeMAPS + sex/age/education; 缺列用 0 填充
    - transcript 为图片描述任务 TSV 文本 (列 no/start_time/end_time/speaker/value,
      speaker=<A> 为被试); 提供则走 combined (MCI 可分离性更高), 不提供则降级
    - 严重程度 0-100: CTRL~0 / MCI~50(中段) / AD~100; risk_band 为三级判定 (CTRL-like / borderline / MCI-like)
"""
import os, sys, json, argparse, pickle, tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cn_scorer import (build_cn_severity_combined_model, assemble_combined_X)

# 默认模型：优先加载同目录 96 维 fast 模型（my_severity_combined.pkl）；缺失或反序列化
# 失败时退回"启动即训练内置 105 维"。独立部署包无训练 csv 时 build 会抛 FileNotFoundError，
# 以 None 兜底后靠 --model 或下方自动加载 fast pkl。
SCORER, _DF, _META = None, None, {"n": 0, "feature_dim": 0, "mode": "pending"}
try:
    SCORER, _DF, _META = build_cn_severity_combined_model()
except FileNotFoundError:
    SCORER, _META = None, {"n": 0, "feature_dim": 0, "mode": "no-builtin"}
except Exception as e:
    # build stub 之外任何异常：同样回退到 pkl，避免启动崩溃
    SCORER, _META = None, {"n": 0, "feature_dim": 0, "mode": f"no-builtin({e})"}


def _band_desc(scorer):
    """0-100 风险带切点（方案A：三级不确定带，无 AD-like）：
    ordinal 模型用 <35 CTRL-like / 35–50 borderline / ≥50 MCI-like；临床量表模型用 <20 /20–35 /≥35。"""
    if getattr(scorer, "is_clinical_", False):
        return {"CTRL": "<20", "borderline": "20–35", "MCI": ">=35"}
    return {"CTRL": "<35", "borderline": "35–50", "MCI": ">=50"}


def _load_model(pkl_path):
    """加载 ROUTE_A 输出的 my_severity_combined.pkl，替代内置模型。
    支持 ordinal 代理分模型与 --score-col MMSE/MoCA 回归模型两种。
    （序列化类别 CognitiveSeverityScorer / _RegScorer 均在 cn_scorer.py，顶部已 import，
    无需再依赖 ROUTE_A_swap_your_data，降低部署文件集。）"""
    with open(pkl_path, "rb") as f:
        scorer = pickle.load(f)
    cols = getattr(scorer, "combined_feature_cols_", None) or []
    meta = dict(
        n=getattr(scorer, "n_train_", 0),
        label_counts=getattr(scorer, "label_counts_", {}),
        feature_dim=len(cols),
        mode="loaded:" + os.path.basename(pkl_path),
    )
    return scorer, None, meta


# 允许调用方在 import 后覆盖：serve.py --model path.pkl（见 main）
def set_model(scorer, meta):
    global SCORER, _META
    SCORER = scorer
    _META = meta


# 若同目录存在 96 维 fast pkl，则始终优先加载作为默认（App 部署即用 fast，免训练）。
_auto_pkl = os.path.join(HERE, "my_severity_combined.pkl")
if os.path.exists(_auto_pkl):
    try:
        SCORER, _DF, _META = _load_model(_auto_pkl)
        _META["mode"] = "default-fast(96d):" + os.path.basename(_auto_pkl)
        print(f"[INFO] 已默认加载 fast 96 维模型: {_auto_pkl}")
    except Exception as e:
        print(f"[warn] 加载默认 fast 模型失败，沿用内置/已加载模型: {e}")


def score_rows(rows):
    """rows: list of {uuid, features, transcript?} -> list of result dicts。"""
    import pandas as pd
    X, uuids, modes = assemble_combined_X(SCORER, rows)
    s = SCORER.score(X, uuids=uuids)
    recs = s.to_dict(orient="records")
    for rec, m in zip(recs, modes):
        rec["mode"] = m
    return recs


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok",
                             "model": "C1 severity (combined)"
                                      + (" · clinical-reg" if getattr(SCORER, "is_clinical_", False) else ""),
                             "n_train": _META["n"], "feature_dim": _META["feature_dim"],
                             "bands": _band_desc(SCORER)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path not in ("/score", "/score_audio"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        if path == "/score":
            self._handle_score(raw)
        else:
            self._handle_score_audio(raw, parsed)

    def _handle_score(self, raw):
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return
        try:
            if "rows" in data:
                results = score_rows(data["rows"])
                self._send(200, {"n": len(results), "results": results})
            else:
                results = score_rows([data])
                self._send(200, results[0] if results else {"error": "empty"})
        except Exception as e:
            self._send(500, {"error": f"score failed: {e}"})

    def _handle_score_audio(self, raw, parsed):
        q = parse_qs(parsed.query)
        def _one(k, default=None):
            v = q.get(k)
            return v[0] if v else default
        sex = _one("sex", "")
        age = _one("age")
        edu = _one("education")
        # 由 Content-Type 推断后缀，供 openSMILE/ffmpeg 正确解码
        ctype = (self.headers.get("Content-Type", "") or "").lower()
        ext = ".mp3" if "mpeg" in ctype else (".wav" if "wav" in ctype else ".wav")
        import audio_to_score
        if not audio_to_score.asr_backend_available():
            self._send(503, {"error": "Whisper 不可用（需装 faster-whisper 或 transformers）"})
            return
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(raw)
            f.flush()
            tmp = f.name
        try:
            rec = audio_to_score.score_audio(
                SCORER, tmp,
                sex=sex, age=float(age) if age else None,
                education=float(edu) if edu else None,
            )
            self._send(200, rec)
        except Exception as e:
            self._send(500, {"error": f"audio score failed: {e}"})
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def log_message(self, *a):
        pass


def main():
    global SCORER, _META
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None,
                    help="可选：ROUTE_A 输出的 my_severity_combined.pkl（proxy 或 MMSE 回归模型），"
                         "缺省用内置 C1 combined 模型")
    args = ap.parse_args()
    if args.model:
        scorer, _, meta = _load_model(args.model)
        SCORER = scorer
        _META = meta
        print(f"[INFO] 已加载外部模型: {args.model}（{_META['feature_dim']} 维，"
              f"{'临床量表回归' if getattr(scorer,'is_clinical_',False) else 'ordinal 代理分'}）")
    print(f"中文认知严重程度评分接口 已启动: http://127.0.0.1:{args.port}")
    print(f"  模型: {os.path.basename(args.model) if args.model else '内置 C1 combined'}  "
          f"特征维: {_META['feature_dim']}")
    print("  输出: severity_0_100 (0-100) + risk_band + mode")
    print("  示例(combined, 带转写):")
    print('   curl -X POST http://127.0.0.1:%d/score -H \'Content-Type: application/json\' \\' % args.port)
    print('        -d \'{"uuid":"P0001_0017","features":{"sex":"F","age":52,"education":9,"transcript":"..."}\'')
    print("  (transcript 为图片描述 TSV 文本; 缺省则 mode=combined_imputed)")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
