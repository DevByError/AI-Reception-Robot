"""
Stage 06 (live) - real text-to-speech via Piper (local neural TTS, embeds espeak-ng for
phonemization), with genuine incremental streaming playback so speech actually starts as soon as
the first synthesized audio is ready - matching STAGES.md's own success criterion:
Time-To-First-Audio < 1-2 seconds for a medium-length sentence.

Why this isn't just "swap MockTTSEngine for a real TTSEngine" like every other live file reuses
its mock's core class unchanged: TTSEngine.synthesize_stream() (tts_speaker.py) returns a fully
materialized list of AudioChunks before TTSSpeaker.speak() plays anything at all - fine for the
mock (everything's instant/fake), but structurally wrong for measuring or achieving *real*
streaming latency, since "return the whole list" means every chunk, including the last, has to
exist before the first one can be played. Same kind of deliberate interface evolution as Stage
05's new GENERAL_CHAT intent - the old shape genuinely can't do the job, so this stage gets its
own LiveTTSSpeaker instead of forcing real streaming through a batch-shaped protocol.
tts_speaker.py itself is untouched and still passes its own self-test.

LICENSING NOTE: Piper (OHF-Voice/piper1-gpl) is GPL-3.0-or-later - stronger copyleft than
ultralytics' AGPL-3.0 (Stage 01's own licensing note). Fine for prototyping/internal use, but
revisit before shipping this closed-source to the university - see README.md.

Install first: see README.md (includes the one-time voice model download).
Run: python tts_speaker_live.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import sounddevice as sd
from piper import PiperVoice

VOICE_NAME = "en_US-amy-medium"  # real Piper voice (rhasspy/piper-voices), female, US English -
                                  # swapped from the earlier default "en_US-lessac-medium" (male)
                                  # on 2026-08-19 per explicit request. Same "medium" quality tier
                                  # as before, so no CPU-speed/latency change from this swap alone
                                  # - see README.md for how to try a different voice (male or
                                  # female) and for the one-time model download step (this file
                                  # will raise FileNotFoundError with the exact download command
                                  # if en_US-amy-medium.onnx isn't downloaded yet).
VOICE_MODEL = Path(__file__).resolve().parent / f"{VOICE_NAME}.onnx"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sanitize_for_speech(text: str) -> str:
    """Defensive cleanup in case any stray markdown leaks through from an LLM answer - Stage
    05's own prompts already say "no markdown, no asterisks", but this is a cheap
    belt-and-suspenders pass so a stray "*" or "#" is never read aloud literally."""
    text = re.sub(r"[*_#`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    """Same reasoning as the mock's sentence splitting (tts_speaker.py) - synthesizing sentence
    by sentence, rather than the whole answer in one call, is what lets playback start early."""
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


class LiveTTSSpeaker:
    """Loads the Piper voice once, then synthesizes and plays sentence by sentence, starting
    real audio playback as soon as the very first chunk of the very first sentence is ready -
    not waiting for the whole answer to finish synthesizing first."""

    def __init__(self, model_path: Path = VOICE_MODEL, use_cuda: bool = False):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Voice model not found at {model_path} - run "
                f"'python -m piper.download_voices {VOICE_NAME}' in this folder first (see "
                f"README.md)."
            )
        self.voice = PiperVoice.load(str(model_path), use_cuda=use_cuda)

    def speak(self, text: str) -> dict:
        """Same measurement shape as the mock's TTSSpeaker.speak() - time_to_first_audio_ms and
        num_segments - but time_to_first_audio_ms here is measured against REAL audio actually
        reaching the speaker, not against a fully-materialized (and, for the mock, fake) list."""
        text = _sanitize_for_speech(text)
        sentences = _split_sentences(text)
        if not sentences:
            return {"time_to_first_audio_ms": 0.0, "num_segments": 0}

        start = time.monotonic()
        ttfa_ms: float | None = None
        stream: sd.RawOutputStream | None = None

        try:
            for sentence in sentences:
                for chunk in self.voice.synthesize(sentence):
                    if stream is None:
                        stream = sd.RawOutputStream(
                            samplerate=chunk.sample_rate, channels=chunk.sample_channels,
                            dtype="int16",
                        )
                        stream.start()
                    if ttfa_ms is None:
                        ttfa_ms = (time.monotonic() - start) * 1000
                    stream.write(chunk.audio_int16_bytes)
        finally:
            if stream is not None:
                stream.stop()
                stream.close()

        return {"time_to_first_audio_ms": ttfa_ms or 0.0, "num_segments": len(sentences)}


if __name__ == "__main__":
    print(f"Loading Piper voice '{VOICE_NAME}'...")
    speaker = LiveTTSSpeaker()

    result = speaker.speak(
        "Take the elevator to the 4th floor. "
        "Then it's the second door on your right. "
        "The Computer Science department office is right there."
    )

    print(f"\nTime-To-First-Audio: {result['time_to_first_audio_ms']:.1f}ms, "
          f"segments={result['num_segments']}")

    if result["time_to_first_audio_ms"] <= 2000:
        print("OK: within STAGES.md's <1-2s Time-To-First-Audio target.")
    else:
        print("WARNING: slower than STAGES.md's <1-2s Time-To-First-Audio target - try a "
              "smaller/faster voice model (see README.md), or check CPU load.")
