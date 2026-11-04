"""
Stage 04 - an always-on wake word detector (optional).

WAKE_WORD_ENGINE=text (the default) needs nothing extra: the robot's name is looked for in what
Whisper heard (stage08's addressee.py), which works during a conversation and costs nothing.

WAKE_WORD_ENGINE=openwakeword adds a tiny always-listening model (openWakeWord, Apache-2.0,
~2-5% of one CPU core) so somebody can call the robot from across the room or off camera, when
it is not in a conversation. It needs a model for the phrase: train a "hey nova" model with
openWakeWord's notebook (about an hour on a free Colab GPU) and set OWW_MODEL to the .onnx
file, or set OWW_MODEL=hey_jarvis to try it with a stock phrase first.

    pip install openwakeword==0.6.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from model_paths import OWW_DIR  # noqa: E402

CHUNK = 1280  # openWakeWord works on 80ms blocks of 16kHz audio


def _resolve_model(spec: str) -> Path:
    path = Path(spec)
    if path.exists():
        return path
    for candidate in sorted(OWW_DIR.glob(f"{spec}*.onnx")):
        return candidate
    raise FileNotFoundError(
        f"wake word model {spec!r} not found (looked in {OWW_DIR}) - set "
        f"OWW_MODEL to a .onnx file"
    )


class OpenWakeWordDetector:
    def __init__(self, model: str, threshold: float = 0.5):
        from openwakeword.model import Model

        if not model:
            raise ValueError("OWW_MODEL is not set")
        path = _resolve_model(model)
        self._model = Model(
            wakeword_models=[str(path)],
            inference_framework="onnx",
            melspec_model_path=str(OWW_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(OWW_DIR / "embedding_model.onnx"),
        )
        self.name = path.stem
        self.threshold = threshold
        self._pending = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    def process(self, frame: bytes) -> bool:
        """Feed any amount of 16kHz int16 audio; True when the wake phrase was just heard."""
        self._pending = np.concatenate((self._pending, np.frombuffer(frame, dtype=np.int16)))
        heard = False
        while self._pending.size >= CHUNK:
            block, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            scores = self._model.predict(block)
            self.last_score = max(scores.values()) if scores else 0.0
            heard = heard or self.last_score >= self.threshold
        return heard

    def reset(self) -> None:
        self._model.reset()
        self._pending = np.zeros(0, dtype=np.int16)


def create_detector(engine: str, model: str, threshold: float) -> OpenWakeWordDetector | None:
    if engine != "openwakeword":
        return None
    try:
        detector = OpenWakeWordDetector(model, threshold)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            f"[wake] openWakeWord unavailable ({exc}) - the name still works when spoken "
            f"during a conversation."
        )
        return None
    log.info(f"[wake] listening for the wake word '{detector.name}' (openWakeWord)")
    return detector
