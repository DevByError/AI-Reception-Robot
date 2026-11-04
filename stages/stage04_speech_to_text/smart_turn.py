"""
Stage 04 - "has this person finished speaking?", answered by a model.

The robot used to decide that from silence alone (plus a pitch heuristic): wait 550-1500ms of
quiet, then answer. Silence is a poor signal - people pause mid-sentence ("where is the... um...
library") and finish without a long pause ("where's the library?"). Pipecat's smart-turn v3
(BSD-2-Clause, 8MB ONNX, ~20-60ms on a laptop CPU) listens to the last few seconds of audio -
intonation, rhythm, the word that was cut off - and outputs the probability that the turn is
complete. Used by stt_listener_live._endpoint_decision():

    ~250ms of silence + model says "complete"   -> answer now
    model says "incomplete"                      -> keep listening (up to the patient limit)

so a clear question gets its answer about half a second sooner, and a thinking pause is not
cut off. The log-mel feature code (_whisper_features.py) is vendored from Pipecat so neither
pipecat nor transformers is needed - only onnxruntime and numpy.

Model:  python stages/stage00_setup/download_models.py    (models/smart-turn/)
Run:    python smart_turn.py file.wav                     (prints the probability)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from model_paths import SMART_TURN_MODEL  # noqa: E402

from _whisper_features import compute_whisper_log_mel_features  # noqa: E402

SAMPLE_RATE = 16000
WINDOW_S = 8


class SmartTurn:
    def __init__(self, model_path: Path = SMART_TURN_MODEL, threads: int = 1):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} is missing - run "
                                    f"python stages/stage00_setup/download_models.py")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(model_path), sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()
        self.last_ms = 0.0

    def probability(self, pcm16: bytes | np.ndarray) -> float:
        """P(the speaker has finished their turn), from 16kHz int16 mono audio that ENDS at the
        current moment (i.e. includes the short silence after the last word)."""
        started = time.monotonic()
        if isinstance(pcm16, (bytes, bytearray)):
            audio = np.frombuffer(pcm16, dtype=np.int16)
        else:
            audio = np.asarray(pcm16)
        audio = audio.astype(np.float32) / 32768.0
        n = WINDOW_S * SAMPLE_RATE
        audio = audio[-n:] if audio.size > n else np.pad(audio, (n - audio.size, 0))
        features = compute_whisper_log_mel_features(audio, do_normalize=True)[None, ...]
        with self._lock:
            out = self._session.run(None, {"input_features": features.astype(np.float32)})
        self.last_ms = (time.monotonic() - started) * 1000
        return float(np.asarray(out[0]).reshape(-1)[0])


_shared: SmartTurn | None = None
_shared_failed = False


def shared_model() -> SmartTurn | None:
    """One instance per process, loaded on first use; None (said once) if unavailable."""
    global _shared, _shared_failed
    if _shared is None and not _shared_failed:
        try:
            _shared = SmartTurn()
        except Exception as exc:  # noqa: BLE001
            _shared_failed = True
            print(f"[turn] smart-turn model unavailable ({exc}) - using the pitch/silence "
                  f"endpointing instead.")
    return _shared


if __name__ == "__main__":
    import soundfile as sf

    model = SmartTurn()
    for path in sys.argv[1:]:
        data, rate = sf.read(path, dtype="int16")
        if data.ndim > 1:
            data = data[:, 0]
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{path}: needs 16kHz audio, got {rate}")
        p = model.probability(data)
        print(f"{p:.3f}  {'complete' if p >= 0.5 else 'incomplete'}  ({model.last_ms:.0f}ms)  {path}")
