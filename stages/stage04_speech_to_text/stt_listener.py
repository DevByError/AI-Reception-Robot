"""
Stage 04 - Listening (Speech-to-Text)

Goal: after the greeting, record audio and convert it to text quickly, with Voice Activity
Detection (VAD) to know when the person finished speaking. Language for this first stage:
English only (locale is extensible for later).

TODO (real implementation later):
  - replace MockSTTEngine with Whisper (local, streaming) or a cloud STT streaming service.
  - add real VAD (e.g. webrtcvad or silero-vad) instead of the fake threshold below.
  - locale: passed on every call so we can add Arabic/Chinese later without changing the
    interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TranscriptChunk:
    text: str
    is_final: bool   # False = a partial result while still speaking (streaming), True = the
                      # sentence is complete


class STTEngine(Protocol):
    def transcribe_stream(self, audio_chunks: list[bytes], locale: str) -> list[TranscriptChunk]:
        ...


class MockSTTEngine:
    """
    A fake engine for testing: each "audio_chunk" here is actually a ready-made word (for
    simplicity), instead of real audio waveforms. Simulates text arriving incrementally
    (streaming), then a final end-of-sentence marker.
    """

    def transcribe_stream(self, audio_chunks: list[bytes], locale: str) -> list[TranscriptChunk]:
        results = []
        partial = ""
        for chunk in audio_chunks:
            word = chunk.decode("utf-8")
            partial = f"{partial} {word}".strip()
            results.append(TranscriptChunk(text=partial, is_final=False))
        if results:
            results[-1] = TranscriptChunk(text=partial, is_final=True)
        return results


class VoiceActivityDetector:
    """Placeholder VAD: the real implementation analyzes the audio itself; here we rely on an
    explicit signal for testing."""

    def is_speech_ended(self, silence_duration_ms: int, threshold_ms: int = 700) -> bool:
        return silence_duration_ms >= threshold_ms


class STTListener:
    def __init__(self, engine: STTEngine, vad: VoiceActivityDetector, locale: str = "en-US"):
        self.engine = engine
        self.vad = vad
        self.locale = locale

    def listen(self, audio_chunks: list[bytes], trailing_silence_ms: int) -> str | None:
        """
        Returns the final text if speech has ended (per VAD), or None if still speaking / no
        audio yet. In the real implementation: audio_chunks would come live from the
        microphone (streaming), not a ready-made list.
        """
        if not audio_chunks:
            return None

        if not self.vad.is_speech_ended(trailing_silence_ms):
            return None  # still speaking - keep waiting

        chunks = self.engine.transcribe_stream(audio_chunks, self.locale)
        final = [c for c in chunks if c.is_final]
        return final[-1].text if final else None


if __name__ == "__main__":
    listener = STTListener(MockSTTEngine(), VoiceActivityDetector(), locale="en-US")

    # Simulate: the person is still talking (short pause) -> don't return text yet
    words = [b"where", b"is", b"the", b"computer", b"science", b"department"]
    result_mid_speech = listener.listen(words, trailing_silence_ms=200)
    print(f"Mid-speech (short pause): {result_mid_speech!r}")
    assert result_mid_speech is None

    # Simulate: enough silence -> consider it done and return the full text
    result_done = listener.listen(words, trailing_silence_ms=800)
    print(f"After enough silence (speech ended): {result_done!r}")
    assert result_done == "where is the computer science department"

    print("\nOK: VAD logic + assembling the partial text into the final text works correctly.")
