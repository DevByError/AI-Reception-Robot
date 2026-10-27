"""
Stage 06 - Kokoro-82M, the natural-sounding voice. Added 2026-10-27.

Why: Piper's en_US-amy-medium is fast but flat - it is the main reason the robot "sounds like a
robot". Kokoro-82M (Apache-2.0 weights) is one of the most natural open voices that still runs on
a laptop CPU; the int8 ONNX export is ~88MB.

Why not the kokoro-onnx package: every release that reads the current voices file
(voices-v1.0.bin) requires numpy 2, and this project is pinned to numpy 1.26 by mediapipe and
opencv. The part of it this robot needs is small - phonemize with espeak-ng, map phonemes to
token ids, pick the style vector for that length, run one ONNX call - so it lives here instead,
following kokoro-onnx (MIT, github.com/thewh1teagle/kokoro-onnx) closely.

Needs:   pip install phonemizer-fork==3.3.1 espeakng-loader==0.2.4   (espeak-ng is bundled)
Models:  python stages/stage00_setup/download_models.py               (models/kokoro/)
Run:     python kokoro_voice.py      (speaks one line and prints the speed)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from model_paths import KOKORO_MODEL, KOKORO_VOICES  # noqa: E402

SAMPLE_RATE = 24000
MAX_PHONEMES = 510          # the model's context, minus the two pad tokens

# Kokoro v1.0's phoneme vocabulary (the "vocab" table of its config.json).
_VOCAB_SYMBOLS = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, '"': 11, "(": 12, ")": 13,
    "“": 14, "”": 15, " ": 16, "̃": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23,
    "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42,
    "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53,
    "l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63,
    "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75,
    "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87,
    "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112,
    "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125,
    "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136,
    "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157,
    "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177,
}


def _load_espeak() -> None:
    """Points phonemizer at the espeak-ng that espeakng-loader ships, so nothing has to be
    installed system-wide (the usual reason espeak-based TTS fails on Windows)."""
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    import espeakng_loader

    lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY") or espeakng_loader.get_library_path()
    try:
        ctypes.cdll.LoadLibrary(lib)
    except OSError:
        lib = ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak")
        if not lib:
            raise RuntimeError("espeak-ng could not be loaded - pip install espeakng-loader")
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    EspeakWrapper.set_library(lib)


def _trim(audio: np.ndarray, top_db: float = 60.0, pad: int = 480) -> np.ndarray:
    """Cuts the leading/trailing near-silence Kokoro renders (what librosa.effects.trim does in
    kokoro-onnx, without pulling librosa and numba in for one function)."""
    if audio.size == 0:
        return audio
    frame = 512
    n = audio.size // frame
    if n < 2:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms / rms.max()) > -top_db)
    if loud.size == 0:
        return audio
    start = max(int(loud[0]) * frame - pad, 0)
    end = min((int(loud[-1]) + 1) * frame + pad, audio.size)
    return audio[start:end]


class KokoroVoice:
    """One loaded Kokoro model + voice. create(text) -> float32 mono audio at 24kHz."""

    name = "kokoro"
    sample_rate = SAMPLE_RATE

    def __init__(self, voice: str = "af_heart", model_path: Path = KOKORO_MODEL,
                 voices_path: Path = KOKORO_VOICES, threads: int | None = None):
        import onnxruntime as ort
        import phonemizer  # noqa: F401 - fail here, at load time, not mid-sentence

        for path in (model_path, voices_path):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} is missing - run python stages/stage00_setup/download_models.py")
        _load_espeak()
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(model_path), options,
                                             providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        voices = np.load(str(voices_path))
        if voice not in voices.files:
            raise ValueError(f"unknown Kokoro voice {voice!r}; try one of "
                             f"{', '.join(sorted(voices.files)[:12])}, ...")
        self._style = np.asarray(voices[voice], dtype=np.float32)   # (510, 1, 256)
        self.voice = voice
        self._lock = threading.Lock()   # espeak-ng is not re-entrant

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = re.sub(r"\bDr\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\bMr\.(?= [A-Z])", "Mister", text)
        text = re.sub(r"\bMrs\.(?= [A-Z])", "Missus", text)
        text = re.sub(r"\bMs\.(?= [A-Z])", "Miss", text)
        text = re.sub(r"(?<=\d),(?=\d)", "", text)                  # 12,000 -> 12000
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)              # 9-5 -> 9 to 5
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def phonemize(self, text: str) -> str:
        import phonemizer

        with self._lock:
            phonemes = phonemizer.phonemize(self._normalize(text), "en-us",
                                            preserve_punctuation=True, with_stress=True)
        phonemes = phonemes.replace("ʲ", "j").replace("r", "ɹ").replace("x", "k")
        return "".join(p for p in phonemes if p in _VOCAB_SYMBOLS).strip()

    def _run(self, phonemes: str, speed: float) -> np.ndarray:
        tokens = [_VOCAB_SYMBOLS[p] for p in phonemes][:MAX_PHONEMES]
        style = self._style[len(tokens)]
        feed = {"style": style}
        if "input_ids" in self._inputs:
            feed["input_ids"] = np.array([[0, *tokens, 0]], dtype=np.int64)
            feed["speed"] = np.array([speed], dtype=np.int32 if speed == int(speed)
                                     else np.float32)
        else:
            feed["tokens"] = np.array([[0, *tokens, 0]], dtype=np.int64)
            feed["speed"] = np.array([speed], dtype=np.float32)
        return self._session.run(None, feed)[0].reshape(-1)

    def create(self, text: str, speed: float = 1.0) -> np.ndarray:
        phonemes = self.phonemize(text)
        if not phonemes:
            return np.zeros(0, dtype=np.float32)
        parts = []
        # Long text is split at spaces/punctuation into model-sized batches.
        while phonemes:
            if len(phonemes) <= MAX_PHONEMES:
                chunk, phonemes = phonemes, ""
            else:
                cut = max(phonemes.rfind(c, 0, MAX_PHONEMES) for c in ".,;:!? ")
                cut = cut if cut > 0 else MAX_PHONEMES
                chunk, phonemes = phonemes[:cut + 1], phonemes[cut + 1:].lstrip()
            parts.append(_trim(self._run(chunk, max(0.5, min(2.0, speed)))))
        return np.concatenate(parts).astype(np.float32)


if __name__ == "__main__":
    t0 = time.monotonic()
    v = KokoroVoice()
    print(f"loaded in {time.monotonic() - t0:.1f}s")
    v.create("Warm up.")
    for line in ("Hi! Welcome to Lincoln University College.",
                 "The library is on the third floor of Block A, right next to the lifts."):
        t0 = time.monotonic()
        audio = v.create(line)
        took = time.monotonic() - t0
        print(f"{took * 1000:.0f}ms for {audio.size / SAMPLE_RATE:.2f}s of speech "
              f"(real-time factor {took / max(audio.size / SAMPLE_RATE, 1e-6):.2f}): {line}")
    try:
        import sounddevice as sd

        sd.play(audio, SAMPLE_RATE)
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        print(f"(could not play it: {exc!r})")
