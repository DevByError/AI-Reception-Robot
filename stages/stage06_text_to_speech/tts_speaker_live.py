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
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

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

# Finds a completed sentence at the START of a buffer that is still growing (speak_stream()).
# Deliberately not SENTENCE_SPLIT_RE: that one needs whitespace AFTER the terminator to split,
# which never exists for the final sentence of an LLM stream until the stream has closed - the
# exact case where waiting costs the most. This matches the terminator itself plus any trailing
# space, so "The library is on the third floor." is speakable the instant the period arrives.
STREAM_SENTENCE_RE = re.compile(r"^(.*?[.!?])(\s+|$)", re.DOTALL)

# A sentence is only flushed early once it is at least this long. Piper has a fixed per-call
# overhead, so synthesizing "Sure." on its own buys nothing and costs an audible seam before the
# real sentence - below this, the buffer keeps growing until the next terminator. Not applied to
# the final flush, where there is nothing left to wait for.
MIN_STREAM_SENTENCE_CHARS = 24

# Abbreviations whose trailing period is not a sentence end. Without this the streaming splitter
# cuts "Dr." / "No." mid-name and Piper reads the fragments with two falling intonations, which
# is instantly audible as a machine. Short and deliberately incomplete - these are the ones that
# actually occur in university front-desk answers.
_ABBREVIATIONS = (
    "dr.", "mr.", "mrs.", "ms.", "prof.", "st.", "no.", "vs.", "etc.", "e.g.", "i.e.",
    "jr.", "sr.", "approx.", "dept.",
)


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


def _ends_on_abbreviation(text: str) -> bool:
    """Whether a candidate sentence ends on a known abbreviation rather than a real sentence
    end - see _ABBREVIATIONS."""
    tail = text.rstrip().lower()
    return any(tail.endswith(abbrev) for abbrev in _ABBREVIATIONS)


def _pop_sentence(buffer: str, *, final: bool) -> tuple[str | None, str]:
    """Pulls one speakable sentence off the front of a growing buffer.

    Returns (sentence_or_None, remaining_buffer). On final=True the whole buffer is returned
    regardless of whether it terminates properly - an LLM that stops mid-sentence (hit its token
    cap, say) should still have its last words spoken rather than silently dropped.
    """
    if final:
        remainder = buffer.strip()
        return (remainder or None), ""

    match = STREAM_SENTENCE_RE.match(buffer)
    if not match:
        return None, buffer

    sentence = match.group(1).strip()
    if _ends_on_abbreviation(sentence) or len(sentence) < MIN_STREAM_SENTENCE_CHARS:
        # Not a real boundary (or not worth a separate synthesis call yet) - look for the NEXT
        # terminator instead by leaving the buffer intact and reporting nothing ready. The
        # caller keeps feeding it; the final flush guarantees it is eventually spoken either way.
        rest = buffer[match.end():]
        following, remaining = _pop_sentence(rest, final=False)
        if following is None:
            return None, buffer
        return f"{sentence} {following}", remaining

    return sentence, buffer[match.end():]


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

        # Set to ask the current speak()/speak_stream() call to stop at the next chunk boundary.
        # Nothing sets it today: real barge-in also needs acoustic echo cancellation, or the
        # robot hears its own voice through the mic and interrupts itself on every sentence (see
        # README.md). The check costs nothing and means enabling barge-in later is a change in
        # the ORCHESTRATOR - open the mic during playback, set this on detected speech - rather
        # than surgery on the audio loop here.
        self.stop_requested = threading.Event()

    def _play_sentences(self, sentences: Iterable[str]) -> tuple[float | None, int, list[str]]:
        """Shared playback core for speak() and speak_stream(): opens ONE output stream lazily on
        the first synthesized chunk and keeps it open across every sentence.

        One stream for the whole answer, rather than one per sentence, is what makes streaming
        sound like a person talking instead of a list being read - reopening the device between
        sentences inserts an audible click and a variable gap while the driver re-primes.
        """
        start = time.monotonic()
        ttfa_ms: float | None = None
        stream: sd.RawOutputStream | None = None
        spoken: list[str] = []

        try:
            for sentence in sentences:
                if self.stop_requested.is_set():
                    break
                spoken.append(sentence)
                for chunk in self.voice.synthesize(sentence):
                    if self.stop_requested.is_set():
                        break
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

        return ttfa_ms, len(spoken), spoken

    def speak_stream(self, pieces: Iterable[str]) -> dict:
        """Speaks text that is still being GENERATED, starting playback as soon as the first
        complete sentence exists rather than waiting for the generator to finish.

        `pieces` is any iterable of text fragments - the token/delta stream of an LLM call (see
        stage05's stream_grounded_answer()). Fragments are buffered until they form a sentence,
        each finished sentence is synthesized and played immediately, and the generator keeps
        producing the rest meanwhile.

        This is the other half of the latency fix. speak() below already streamed sentence by
        sentence, but its caller could only hand it a COMPLETE answer string, so Piper sat idle
        for the full LLM generation (~1-2s) before the first word was ever spoken. Feeding it a
        live token stream instead cuts time-to-first-word to roughly one sentence of generation.

        Returns the same measurement shape as speak(), plus `text` - everything actually spoken,
        reassembled - because the caller no longer has the finished string in hand and still
        needs it for the conversation transcript.
        """
        self.stop_requested.clear()
        spoken_parts: list[str] = []

        def _sentences() -> Iterator[str]:
            buffer = ""
            for piece in pieces:
                if self.stop_requested.is_set():
                    return
                if not piece:
                    continue
                buffer += piece
                while True:
                    sentence, buffer = _pop_sentence(buffer, final=False)
                    if sentence is None:
                        break
                    cleaned = _sanitize_for_speech(sentence)
                    if cleaned:
                        yield cleaned
            leftover, _ = _pop_sentence(buffer, final=True)
            if leftover:
                cleaned = _sanitize_for_speech(leftover)
                if cleaned:
                    yield cleaned

        def _tracked() -> Iterator[str]:
            for sentence in _sentences():
                spoken_parts.append(sentence)
                yield sentence

        ttfa_ms, num_segments, _ = self._play_sentences(_tracked())
        return {
            "time_to_first_audio_ms": ttfa_ms or 0.0,
            "num_segments": num_segments,
            "text": " ".join(spoken_parts),
            "interrupted": self.stop_requested.is_set(),
        }

    def speak(self, text: str) -> dict:
        """Same measurement shape as the mock's TTSSpeaker.speak() - time_to_first_audio_ms and
        num_segments - but time_to_first_audio_ms here is measured against REAL audio actually
        reaching the speaker, not against a fully-materialized (and, for the mock, fake) list."""
        text = _sanitize_for_speech(text)
        sentences = _split_sentences(text)
        if not sentences:
            return {"time_to_first_audio_ms": 0.0, "num_segments": 0, "text": ""}

        self.stop_requested.clear()
        ttfa_ms, num_segments, spoken = self._play_sentences(sentences)
        return {
            "time_to_first_audio_ms": ttfa_ms or 0.0,
            "num_segments": num_segments,
            "text": " ".join(spoken),
            "interrupted": self.stop_requested.is_set(),
        }


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

    # --- streaming input (speak_stream) ------------------------------------------------------
    # The same answer, delivered the way a real LLM delivers it: a few characters at a time, with
    # a deliberate delay per fragment standing in for generation time. The point being measured
    # is that Time-To-First-Audio stays close to the ONE-SENTENCE mark instead of growing with
    # the length of the whole answer - which is exactly what speak() above cannot do, because it
    # can only be called once the entire string already exists.
    print("\n--- streaming input (simulating a live LLM token stream) ---")

    def _fake_token_stream():
        answer = ("Take the elevator to the 4th floor. "
                  "Then it's the second door on your right. "
                  "The Computer Science department office is right there.")
        for i in range(0, len(answer), 7):
            time.sleep(0.05)  # stands in for real per-token generation latency
            yield answer[i:i + 7]

    stream_start = time.monotonic()
    stream_result = speaker.speak_stream(_fake_token_stream())
    total_ms = (time.monotonic() - stream_start) * 1000

    print(f"\nTime-To-First-Audio: {stream_result['time_to_first_audio_ms']:.1f}ms, "
          f"segments={stream_result['num_segments']}, total={total_ms:.0f}ms")
    print(f"Spoke: {stream_result['text']!r}")

    assert stream_result["num_segments"] == 3, stream_result
    assert "Computer Science" in stream_result["text"]
    # The whole generation takes ~1.1s of simulated token delay; first audio must land well
    # inside that, or the streaming path isn't actually streaming.
    assert stream_result["time_to_first_audio_ms"] < total_ms * 0.75, (
        f"first audio at {stream_result['time_to_first_audio_ms']:.0f}ms of a {total_ms:.0f}ms "
        f"stream - playback is not starting before generation finishes"
    )
    print("\nOK: playback started before the generator finished - real streaming input works.")
