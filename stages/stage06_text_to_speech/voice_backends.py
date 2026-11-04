"""
Stage 06 - which voice speaks: Kokoro (natural) or Piper (fast).

Both backends have the same shape - `sample_rate`, and `synthesize(text, rate)` yielding int16
numpy chunks - so LiveTTSSpeaker does not care which one it has.

TTS_BACKEND=auto picks Kokoro when its model files are present AND this computer can run it
comfortably faster than real time, which it checks once at startup by timing one sentence.
On a slow CPU Kokoro would make the robot pause between sentences, and a fast flat voice is a
better experience than a natural voice that stalls - so auto falls back to Piper and says so.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

# Kokoro must synthesize at least this much faster than real time to be chosen automatically.
# 0.5 leaves room for the camera and speech recognition sharing the same CPU.
KOKORO_MAX_RTF = 0.5
_BENCH_LINE = "Hello, and welcome to Lincoln University College. How can I help you today?"


class PiperBackend:
    name = "piper"

    def __init__(self, model_path: Path, use_cuda: bool = False):
        from piper import PiperVoice

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Voice model not found at {model_path} - run "
                f"'python -m piper.download_voices {Path(model_path).stem}' in "
                f"stages/stage06_text_to_speech first (see README.md)."
            )
        self.voice = PiperVoice.load(str(model_path), use_cuda=use_cuda)
        self.sample_rate = int(self.voice.config.sample_rate)
        self._base_length = float(getattr(self.voice.config, "length_scale", 1.0) or 1.0)

    def synthesize(self, text: str, rate: float = 1.0) -> Iterator[np.ndarray]:
        config = None
        if abs(rate - 1.0) > 1e-3:
            from piper import SynthesisConfig

            config = SynthesisConfig(length_scale=self._base_length / rate)
        for chunk in self.voice.synthesize(text, syn_config=config):
            yield chunk.audio_int16_array


class KokoroBackend:
    name = "kokoro"

    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        from kokoro_voice import KokoroVoice

        self.kokoro = KokoroVoice(voice=voice)
        self.sample_rate = self.kokoro.sample_rate
        self.speed = speed
        self.voice = None  # (LiveTTSSpeaker.voice is the Piper object, for older callers)

    def synthesize(self, text: str, rate: float = 1.0) -> Iterator[np.ndarray]:
        audio = self.kokoro.create(text, speed=self.speed * rate)
        if audio.size:
            yield (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    def benchmark(self) -> float:
        """Real-time factor on this machine (seconds of compute per second of speech)."""
        self.kokoro.create("Warm up.")
        started = time.monotonic()
        audio = self.kokoro.create(_BENCH_LINE)
        took = time.monotonic() - started
        return took / max(audio.size / self.sample_rate, 1e-6)


def load_backend(
    choice: str,
    piper_model: Path,
    *,
    use_cuda: bool = False,
    kokoro_voice: str = "af_heart",
    kokoro_speed: float = 1.0,
    max_rtf: float = KOKORO_MAX_RTF,
):
    """The backend TTS_BACKEND asks for, falling back to Piper with a printed reason."""
    if choice in ("auto", "kokoro"):
        try:
            backend = KokoroBackend(voice=kokoro_voice, speed=kokoro_speed)
        except Exception as exc:  # noqa: BLE001 - missing files/packages: fall back
            reason = f"{exc}" if isinstance(exc, FileNotFoundError) else repr(exc)
            log.info(f"[tts] Kokoro voice not available ({reason}) - using Piper.")
        else:
            rtf = backend.benchmark()
            # One timing taken while the rest of the robot is still loading can be unlucky (a
            # real run on 2026-10-27 measured 1.6x, just under the 2x bar). Only a near miss gets
            # two more tries and keeps the best, so a computer that passes first time pays
            # nothing extra and one that is clearly too slow is not tested again.
            if choice != "kokoro" and max_rtf < rtf <= max_rtf * 1.6:
                for _ in range(2):
                    time.sleep(0.5)
                    rtf = min(rtf, backend.benchmark())
                    if rtf <= max_rtf:
                        break
            if choice == "kokoro" or rtf <= max_rtf:
                log.info(
                    f"[tts] voice: Kokoro '{kokoro_voice}' (synthesizes {1 / rtf:.1f}x faster "
                    f"than real time on this computer)"
                )
                return backend
            log.info(
                f"[tts] Kokoro is too slow on this computer ({1 / rtf:.1f}x real time, needs "
                f"{1 / max_rtf:.0f}x) - using Piper. Set TTS_BACKEND=kokoro to force it."
            )
    backend = PiperBackend(piper_model, use_cuda=use_cuda)
    log.info(f"[tts] voice: Piper '{Path(piper_model).stem}'")
    return backend
