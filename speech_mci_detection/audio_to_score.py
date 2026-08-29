# -*- coding: utf-8 -*-
"""
音频端到端评分：audio file -> openSMILE 88 维 eGeMAPS + Whisper ASR 转写
                -> 组装特征 -> 加载中文认知障碍评分器 -> 0-100 严重度。
================================================================================
为 serve.py 的 /score_audio 端点提供底层能力，也可被 ROUTE_A / 独立脚本复用。

设计要点：
1. 全部重依赖（opensmile / faster-whisper / transformers）**懒加载**：
   - 未安装 opensmile、Whisper 时，本模块依旧可 import，serve.py 断言后返回 503。
   - 不影响原 /score（纯 stdlib）路径。
2. 声学：抽**全量 88 维 eGeMAPSv02 functionals**（列名与 C1 egemaps_final.csv 一致）。
   现场组装走 cn_scorer.assemble_combined_X，按加载模型的 combined_feature_cols_
   按名取列 —— 因此同一份 88 维既兼容 105 维内置模型，也兼容 96 维 --fast 模型。
3. 转写：Whisper（优先 faster-whisper 轻量，回退 transformers pipeline）把音频
   转成文本，再构成 C1 兼容 TSV（列 no/start_time/end_time/speaker/value，
   speaker=<A> 为被试），交给 extract_text 抽中文语言学特征。
4. 中文路径安全：openSMILE C++ 后端对路径做 ascii 编码，含中文的 venv/项目路径
   会崩溃 —— 复用 audition audio.SmileExtractor 的补丁思路（拷到临时 ascii 路径）。
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

__all__ = [
    "extract_egemaps", "asr_text", "make_tsv",
    "audio_file_to_features", "score_audio", "asr_backend_available",
]

_SMILE = None            # openSMILE 88 维 eGeMAPS 抽取器（懒加载）
_ASR = None              # Whisper 实例（faster-whisper 或 transformers）
_ASR_BACKEND = None      # "faster-whisper" | "transformers" | None


# ---------------------------------------------------------------------------
# openSMILE
# ---------------------------------------------------------------------------
def _patch_opensmile_ascii():
    """openSMILE C++ 后端对 config_file/CLI options 做 ascii 编码。当 venv / 项目
    路径含非 ASCII(中文) 时其内置 config 根目录非 ASCII → UnicodeEncodeError。
    把整个 config 树拷到临时 ascii 目录并覆写 default_config_root，仅打一次补丁。"""
    try:
        import opensmile.core.smile as _smile_mod
    except Exception:
        return
    OpenSMILE = _smile_mod.Smile
    if getattr(OpenSMILE, "_audio_to_score_ascii_patched", False):
        return
    src_root = os.path.join(os.path.dirname(os.path.realpath(_smile_mod.__file__)), "config")
    if src_root.isascii():
        OpenSMILE._audio_to_score_ascii_patched = True
        return
    ascii_root = tempfile.mkdtemp(prefix="osmile_cfg_")
    shutil.copytree(src_root, ascii_root, dirs_exist_ok=True)

    def _ascii_root(self):            # noqa: ANN001
        return ascii_root

    OpenSMILE.default_config_root = property(_ascii_root)
    try:
        OpenSMILE._audio_to_score_ascii_patched = True
    except Exception:
        pass


def _get_smile():
    global _SMILE
    if _SMILE is None:
        import opensmile
        _patch_opensmile_ascii()
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def _ascii_temp_copy(audio_path: str) -> str:
    """openSMILE 也不能处理中文/非 ascii 输入路径，拷到临时 ascii 路径再处理。"""
    suffix = os.path.splitext(audio_path)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp = f.name
    try:
        shutil.copyfile(audio_path, tmp)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return tmp


def extract_egemaps(audio_path: str) -> dict:
    """从音频抽取全量 88 维 eGeMAPSv02 functionals，返回 {列名: float}。"""
    smile = _get_smile()
    tmp = _ascii_temp_copy(audio_path)
    try:
        processed = smile.process_file(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    vals = np.asarray(processed.iloc[0]).ravel()
    return {c: float(vals[i]) for i, c in enumerate(processed.columns)}


# ---------------------------------------------------------------------------
# Whisper ASR
# ---------------------------------------------------------------------------
def _local_whisper_dir(fw_model: str):
    """本地离线目录：whisper_models/<repo_path 的 -- 化>。若打包后存在则直接用，
    不联网。打包内容用 `cp -RLf` 把 HUB snapshot 平铺成真实文件即可（见
    assets/whisper_small.tgz）。"""
    flat = fw_model.replace("/", "--")
    cand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "whisper_models", flat)
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "model.bin")):
        return cand
    return None


def _get_asr(device: str = "cpu", model: str = "whisper-small") -> tuple:
    """读取或构建 Whisper 实例。优先 faster-whisper（快、无 torch 深度依赖），
    回退到 transformers pipeline（需 torch）。返回 (asr, backend)。"""
    global _ASR, _ASR_BACKEND
    if _ASR is None:
        try:
            from faster_whisper import WhisperModel
            # faster-whisper 的仓库名是 "Systran/faster-whisper-<size>"（size∈base/small/
            # medium/large-v3…）。用户传 "whisper-small" 这种短名时要去掉 "whisper-" 前缀
            # 再拼接，否则会拼成 faster-whisper-whisper-small 导致找不到仓库。
            if "/" in model:
                fw_model = model
            else:
                size = model[8:] if model.startswith("whisper-") else model
                fw_model = f"Systran/faster-whisper-{size}"
            # 优先本地离线目录（打包的解压即用，不联网）；否则回退 HF hub 拉取。
            local_dir = _local_whisper_dir(fw_model)
            if local_dir:
                _ASR = WhisperModel(local_dir, device=device, compute_type="int8")
            else:
                _ASR = WhisperModel(fw_model, device=device, compute_type="int8")
            _ASR_BACKEND = "faster-whisper"
        except Exception as e:
            _fw_err = f"{type(e).__name__}: {e}"
            try:
                from transformers import pipeline
                _ASR = pipeline("automatic-speech-recognition",
                                model=f"openai/{model}")
                _ASR_BACKEND = "transformers"
            except Exception as e2:
                raise RuntimeError(
                    f"Whisper 不可用（faster-whisper: {_fw_err}; transformers: {e2}）"
                ) from e2
    return _ASR, _ASR_BACKEND


def asr_backend_available() -> bool:
    for name in ("faster_whisper", "transformers"):
        try:
            __import__(name)
            return True
        except Exception:
            continue
    return False


def asr_text(audio_path: str, device: str = "cpu", model: str = "whisper-small") -> str:
    """Whisper 转写音频为纯文本。faster-whisper 拼 segments；transformers 取 text。"""
    asr, backend = _get_asr(device=device, model=model)
    if backend == "faster-whisper":
        segments, _info = asr.transcribe(audio_path)
        return " ".join(s.text.strip() for s in segments).strip()
    out = asr(audio_path)
    return (out.get("text") or "").strip() if isinstance(out, dict) else str(out).strip()


# ---------------------------------------------------------------------------
# 组装与评分
# ---------------------------------------------------------------------------
def make_tsv(text: str) -> str:
    """把纯转写文本构造成 C1 兼容 TSV（列 no/start_time/end_time/speaker/value，
    speaker=<A> 为被试）。空文本 → 仅表头（extract_text 会得到空语言学 → 近似静音）。"""
    lines = ["no\tstart_time\tend_time\tspeaker\tvalue"]
    if text and str(text).strip():
        t = str(text).strip()
        lines.append(f"1\t0\t{len(t)}\t<A>\t{t}")
    return "\n".join(lines)


def audio_file_to_features(audio_path: str, sex: str = "", age=None, education=None):
    """音频 + 可选人口学 → (features dict, transcript TSV str)。"""
    feats = extract_egemaps(audio_path)
    feats["sex"] = sex
    if age is not None:
        feats["age"] = float(age)
    if education is not None:
        feats["education"] = float(education)
    tsv = make_tsv(asr_text(audio_path))
    return feats, tsv


def score_audio(scorer, audio_path: str, sex: str = "", age=None, education=None,
                device: str = "cpu", model: str = "whisper-small") -> dict:
    """一键：音频 → 严重度。scorer 为已加载的 CognitiveSeverityScorer / _RegScorer。
    返回 dict(uuid, severity_0_100, risk_band, mode, asr_backend)。"""
    from cn_scorer import assemble_combined_X

    feats, tsv = audio_file_to_features(audio_path, sex=sex, age=age, education=education)
    uuid = os.path.splitext(os.path.basename(audio_path))[0]
    X, uuids, modes = assemble_combined_X(
        scorer, [{"uuid": uuid, "features": feats, "transcript": tsv}]
    )
    s = scorer.score(X, uuids=uuids)
    rec = s.to_dict(orient="records")[0]
    rec["mode"] = modes[0]
    rec["asr_backend"] = _ASR_BACKEND
    return rec


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python audio_to_score.py <audio.wav> [sex] [age] [education]")
        sys.exit(2)
    _model = os.environ.get("AUDIO_SCORE_PKL", "")
    scorer_cls = None
    if _model:
        import pickle
        import cn_scorer  # noqa: F401  确保反序列化类可用
        scorer_cls = pickle.load(open(_model, "rb"))
        print(f"[INFO] loaded model: {_model} (type={type(scorer_cls).__name__})")
    else:
        # 无 AUDIO_SCORE_PKL 时：优先加载同目录 my_severity_combined.pkl（独立部署包自带，
        # 免训练）；否则退回 ROUTE_A 内置 build（需 egemaps_final.csv 训练文件，独立包没有）。
        _auto_pkl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "my_severity_combined.pkl")
        if os.path.exists(_auto_pkl):
            import pickle
            import cn_scorer  # noqa: F401
            scorer_cls = pickle.load(open(_auto_pkl, "rb"))
            print(f"[INFO] loaded model: {_auto_pkl} (type={type(scorer_cls).__name__})")
        else:
            # 无同目录 pkl：退回构建内置模型——部署版 build 为 stub，会抛 FileNotFoundError，
            # 说明缺少模型文件，给出清晰提示而非 Traceback。
            try:
                from cn_scorer import build_cn_severity_combined_model
                scorer_cls, _df, _meta = build_cn_severity_combined_model()
                print(f"[INFO] loaded builtin combined model ({_meta['feature_dim']} dim)")
            except FileNotFoundError:
                sys.exit("[ERROR] 未找到模型文件 my_severity_combined.pkl。请将其放在本脚本同目录，或用 "
                         "环境变量 AUDIO_SCORE_PKL 指定 pkl 路径。")
    kwargs = {}
    if len(sys.argv) >= 3:
        kwargs["sex"] = sys.argv[2]
    if len(sys.argv) >= 4:
        kwargs["age"] = sys.argv[3]
    if len(sys.argv) >= 5:
        kwargs["education"] = sys.argv[4]
    res = score_audio(scorer_cls, sys.argv[1], **kwargs)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=2))