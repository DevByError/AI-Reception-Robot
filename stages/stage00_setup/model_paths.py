"""
Stage 00 - where the optional model files live.

Every model added on 2026-10-27 (Kokoro voice, smart-turn end-of-turn model, ONNX MiniLM
embeddings, streaming Zipformer speech recognizer, openWakeWord) lives under ONE folder,
`models/` at the project root, so there is a single place to download into and a single
folder to .gitignore. The older models (YuNet/SFace, Piper, YOLO) stay where they always were.

Run `python stages/stage00_setup/download_models.py` to fetch everything listed here.
Standard library only, like the rest of stage00.
"""

from __future__ import annotations

import os
from pathlib import Path

from project_env import load_env_file

load_env_file()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# Kokoro-82M (Apache-2.0) - the natural-sounding voice. The fp16 export (~170MB) is the one to
# use on a CPU: measured 3-4x FASTER than the int8 export (RTF ~0.3 vs ~1.1 on 4 cores), which
# spends its time converting back and forth. Override with LUC_KOKORO_MODEL=<path>.
KOKORO_MODEL = Path(os.environ.get("LUC_KOKORO_MODEL")
                    or MODELS_DIR / "kokoro" / "kokoro-v1.0.fp16.onnx")
KOKORO_VOICES = MODELS_DIR / "kokoro" / "voices-v1.0.bin"

# Pipecat smart-turn v3.2 (BSD-2-Clause) - "has this person finished their turn?" from audio.
SMART_TURN_MODEL = MODELS_DIR / "smart-turn" / "smart-turn-v3.2-cpu.onnx"

# all-MiniLM-L6-v2 exported to ONNX - the same embedding model the knowledge base always used,
# without PyTorch. Measured identical to the sentence-transformers vectors (cosine 1.00000 on
# 300 cached chunks), so the existing kb_cache.npz and similarity threshold stay valid.
MINILM_DIR = MODELS_DIR / "all-MiniLM-L6-v2-onnx"

# sherpa-onnx streaming Zipformer (Apache-2.0) - English, int8.
ZIPFORMER_DIR = MODELS_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26"

# openWakeWord feature models (Apache-2.0). The wake-word model itself is whatever OWW_MODEL
# points at - a custom "hey nova" model you train, or a stock one such as hey_jarvis.
OWW_DIR = MODELS_DIR / "openwakeword"


def describe() -> list[tuple[str, Path, bool]]:
    """(label, path, present) for every optional model - used by the startup report."""
    items = [
        ("Kokoro voice", KOKORO_MODEL),
        ("Kokoro voices", KOKORO_VOICES),
        ("smart-turn", SMART_TURN_MODEL),
        ("MiniLM ONNX", MINILM_DIR / "model.onnx"),
        ("Zipformer STT", ZIPFORMER_DIR / "tokens.txt"),
        ("openWakeWord features", OWW_DIR / "melspectrogram.onnx"),
    ]
    return [(label, path, path.exists()) for label, path in items]


if __name__ == "__main__":
    for label, path, present in describe():
        print(f"{'OK     ' if present else 'missing'}  {label:<22} {path}")
