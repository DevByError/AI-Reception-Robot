"""
Stage 06 - Speaking (Text-to-Speech)

Goal: convert the answer text to natural speech with the lowest possible latency (streaming
TTS, not waiting for the whole sentence before audio starts). There is no text screen - everything
must be understandable purely by voice.

Success criterion (STAGES.md): Time-To-First-Audio < 1-2 seconds for a medium-length sentence.

TODO (real implementation later):
  - replace MockTTSEngine with a real engine (e.g. local Piper or a cloud streaming TTS
    service).
  - split long sentences into sub-sentences so speech can start before the full LLM answer has
    finished generating.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class AudioChunk:
    text_segment: str
    duration_ms: int  # estimated audio duration for this segment


class TTSEngine(Protocol):
    def synthesize_stream(self, text: str, locale: str) -> list[AudioChunk]:
        ...


class MockTTSEngine:
    """Splits the text into sentences and simulates a synthesis time for each, without real
    audio."""

    def synthesize_stream(self, text: str, locale: str) -> list[AudioChunk]:
        segments = [s.strip() for s in text.split(".") if s.strip()]
        return [
            AudioChunk(text_segment=seg, duration_ms=len(seg) * 40)  # simple fake estimate
            for seg in segments
        ]


class TTSSpeaker:
    def __init__(self, engine: TTSEngine, locale: str = "en-US"):
        self.engine = engine
        self.locale = locale

    def speak(self, text: str) -> dict:
        """
        Returns measurement info: time to the first audio segment (Time-To-First-Audio) and the
        number of segments. In the real implementation: each chunk is actually played through
        the speaker as soon as it's ready (streaming), not waiting for everything.
        """
        start = time.monotonic()
        chunks = self.engine.synthesize_stream(text, self.locale)
        ttfa_ms = (time.monotonic() - start) * 1000  # near-zero with the mock - real measurement
                                                       # comes later

        for chunk in chunks:
            print(f'[TTS/{self.locale}] "{chunk.text_segment}" (~{chunk.duration_ms}ms)')

        return {"time_to_first_audio_ms": ttfa_ms, "num_segments": len(chunks)}


if __name__ == "__main__":
    speaker = TTSSpeaker(MockTTSEngine())

    result = speaker.speak(
        "Take the elevator to the 4th floor. "
        "Then it's the second door on your right. "
        "The Computer Science department office is right there."
    )

    print(f"\nTime-To-First-Audio: {result['time_to_first_audio_ms']:.1f}ms, "
          f"segments={result['num_segments']}")

    assert result["num_segments"] == 3
    print("\nOK: splitting text into segments + measuring time-to-first-audio works - ready to "
          "wire in a real TTS engine.")
