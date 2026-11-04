"""
Stage 06 (live) - real text-to-speech with genuine incremental streaming playback, so speech
starts as soon as the first synthesized audio is ready - matching STAGES.md's own success
criterion: Time-To-First-Audio < 1-2 seconds for a medium-length sentence.

Two voices since 2026-10-27 (voice_backends.py): Kokoro-82M, which sounds like a person, and
Piper, which is very fast. TTS_BACKEND=auto uses Kokoro when this computer can run it well and
Piper otherwise. Synthesis runs ahead of playback on its own thread, sentences are cached, and
all audio goes out through audio_player.AudioPlayer (device choice, echo-canceller reference,
fast barge-in stop, mouth movement for the face).

Why this isn't just "swap MockTTSEngine for a real TTSEngine" like every other live file reuses
its mock's core class unchanged: TTSEngine.synthesize_stream() (tts_speaker.py) returns a fully
materialized list of AudioChunks before TTSSpeaker.speak() plays anything at all - fine for the
mock (everything's instant/fake), but structurally wrong for measuring or achieving *real*
streaming latency. tts_speaker.py itself is untouched and still passes its own self-test.

LICENSING NOTE: Piper (OHF-Voice/piper1-gpl) is GPL-3.0-or-later - stronger copyleft than
ultralytics' AGPL-3.0 (Stage 01's own licensing note). Fine for prototyping/internal use, but
revisit before shipping this closed-source to the university - see README.md.

Install first: see README.md (includes the one-time voice model download).
Run: python tts_speaker_live.py
"""

from __future__ import annotations

import logging
import queue
import random
import re
import sys

# Same arrangement as stage04: the numbers below keep their names and their comments, and take
# their defaults from the one config module so a deployment can tune them without a code edit.
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

log = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage04_speech_to_text"))
from audio_player import AudioPlayer  # noqa: E402
from realtime_config import RT  # noqa: E402
from voice_backends import load_backend  # noqa: E402


def rt_log(tag: str, message: str) -> None:
    """One tagged line of the real-time trace - see stage04's identical helper."""
    if RT.REALTIME_DEBUG:
        log.info(f"[{tag}] {message}")


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
MIN_STREAM_SENTENCE_CHARS = RT.TTS_MIN_SENTENCE_CHARS  # default 24

# --- getting the FIRST word out sooner ---------------------------------------------------------
# The rule above waits for a full stop, which is right for every sentence except the first one.
# The first is the only one with nothing already playing in front of it, so every millisecond it
# waits is silence the visitor actually experiences - and a grounded answer's opening sentence is
# routinely long ("The Master of Computer Science programme is offered by the Faculty of
# Computer Science and Multimedia, and it is on the third floor."), so waiting for its terminator
# can mean waiting for most of the answer to generate.
#
# So the FIRST fragment may also break at a clause boundary - a comma, semicolon, colon or dash -
# once it is at least FIRST_CLAUSE_MIN_CHARS long. Piper renders a clause with a continuing
# intonation rather than a falling one, so "The Master of Computer Science programme is offered
# by the Faculty of Computer Science and Multimedia," sounds like someone mid-sentence, which is
# exactly what it is. Every LATER fragment goes back to whole sentences: they have audio playing
# ahead of them, so an early break buys no perceptible latency and costs intonation.
#
# Set TTS_FIRST_CLAUSE_FLUSH=false to go back to whole sentences throughout.
FIRST_CLAUSE_RE = re.compile(
    r"^(.{%d,}?[,;:](?=\s)|.{%d,}?\s+[-–—]\s+)"
    % (RT.TTS_FIRST_CLAUSE_MIN_CHARS, RT.TTS_FIRST_CLAUSE_MIN_CHARS),
    re.DOTALL,
)

# Abbreviations whose trailing period is not a sentence end. Without this the streaming splitter
# cuts "Dr." / "No." mid-name and Piper reads the fragments with two falling intonations, which
# is instantly audible as a machine. Short and deliberately incomplete - these are the ones that
# actually occur in university front-desk answers.
_ABBREVIATIONS = (
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "st.",
    "no.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "jr.",
    "sr.",
    "approx.",
    "dept.",
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


def _pop_sentence(buffer: str, *, final: bool, first: bool = False) -> tuple[str | None, str]:
    """Pulls one speakable sentence off the front of a growing buffer.

    Returns (sentence_or_None, remaining_buffer). On final=True the whole buffer is returned
    regardless of whether it terminates properly - an LLM that stops mid-sentence (hit its token
    cap, say) should still have its last words spoken rather than silently dropped.

    first=True additionally allows a clause break, for the one fragment where waiting costs
    audible silence - see FIRST_CLAUSE_RE above.
    """
    if final:
        remainder = buffer.strip()
        return (remainder or None), ""

    match = STREAM_SENTENCE_RE.match(buffer)
    if not match:
        if first and RT.TTS_FIRST_CLAUSE_FLUSH:
            clause = FIRST_CLAUSE_RE.match(buffer)
            if clause:
                head = clause.group(1).strip()
                # An abbreviation's period is not why we are here (this matched a comma, not a
                # full stop), but "No." style fragments can still end the head - reuse the same
                # guard rather than inventing a second one.
                if head and not _ends_on_abbreviation(head):
                    return head, buffer[clause.end() :]
        return None, buffer

    sentence = match.group(1).strip()
    if _ends_on_abbreviation(sentence) or len(sentence) < MIN_STREAM_SENTENCE_CHARS:
        # Not a real boundary (or not worth a separate synthesis call yet) - look for the NEXT
        # terminator instead by leaving the buffer intact and reporting nothing ready. The
        # caller keeps feeding it; the final flush guarantees it is eventually spoken either way.
        rest = buffer[match.end() :]
        following, remaining = _pop_sentence(rest, final=False)
        if following is None:
            return None, buffer
        return f"{sentence} {following}", remaining

    return sentence, buffer[match.end() :]


class LiveTTSSpeaker:
    """Loads the voice once, then synthesizes and plays sentence by sentence, starting real
    audio playback as soon as the very first chunk of the very first sentence is ready - not
    waiting for the whole answer to finish synthesizing first.

    Since 2026-10-27:
      - the voice is a backend (voice_backends.py): Kokoro when this computer can run it
        comfortably, else Piper - TTS_BACKEND;
      - synthesis runs on its own thread, AHEAD of playback, so the next sentence is ready
        when the current one ends (before, the robot went quiet between sentences for as long
        as each one took to synthesize);
      - playback goes through AudioPlayer: the chosen SPEAKER_DEVICE, the echo canceller's
        reference, small blocks for a fast barge-in stop, and the loudness the face uses;
      - every sentence up to CACHE_MAX_CHARS is kept (TTS_AUDIO_CACHE), so fixed lines and
        repeated answers start instantly the second time; prewarm() fills it at startup;
      - each utterance gets a slightly different speaking rate (TTS_RATE_VARIATION).
    """

    CACHE_MAX_CHARS = 300
    CACHE_MAX_BYTES = 48 * 1024 * 1024

    def __init__(
        self,
        model_path: Path = VOICE_MODEL,
        use_cuda: bool = False,
        backend=None,
        player: AudioPlayer | None = None,
    ):
        if backend is None:
            backend = load_backend(
                RT.TTS_BACKEND,
                model_path,
                use_cuda=use_cuda,
                kokoro_voice=RT.KOKORO_VOICE,
                kokoro_speed=RT.KOKORO_SPEED,
            )
        self.backend = backend
        # The Piper voice object, kept for older callers; None when Kokoro is speaking.
        self.voice = getattr(backend, "voice", None)
        if player is None:
            device = None
            if RT.SPEAKER_DEVICE:
                from audio_io import device_name, resolve_device

                device = resolve_device(RT.SPEAKER_DEVICE, "output")
                log.info(f"[audio] speaker: {device_name(device, 'output')}")
            player = AudioPlayer(device=device)
        self.player = player
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._cache_lock = threading.Lock()

        # Set to ask the current speak()/speak_stream() call to stop at the next block (~40ms).
        # The orchestrator's barge-in monitor sets it when the visitor talks over the robot.
        self.stop_requested = threading.Event()

    @property
    def sample_rate(self) -> int:
        return int(self.backend.sample_rate)

    def level_at(self, now: float | None = None) -> float:
        return self.player.level_at(now)

    # -- the cache -------------------------------------------------------------------------------
    @staticmethod
    def _cache_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _cache_get(self, text: str) -> np.ndarray | None:
        if not RT.TTS_AUDIO_CACHE:
            return None
        key = self._cache_key(text)
        with self._cache_lock:
            pcm = self._cache.get(key)
            if pcm is not None:
                self._cache.move_to_end(key)
            return pcm

    def _cache_put(self, text: str, pcm: np.ndarray) -> None:
        if not RT.TTS_AUDIO_CACHE or len(text) > self.CACHE_MAX_CHARS or pcm.size == 0:
            return
        key = self._cache_key(text)
        with self._cache_lock:
            old = self._cache.pop(key, None)
            if old is not None:
                self._cache_bytes -= old.nbytes
            self._cache[key] = pcm
            self._cache_bytes += pcm.nbytes
            while self._cache_bytes > self.CACHE_MAX_BYTES and self._cache:
                _, dropped = self._cache.popitem(last=False)
                self._cache_bytes -= dropped.nbytes

    def _rate(self) -> float:
        spread = RT.TTS_RATE_VARIATION
        return 1.0 + random.uniform(-spread, spread) if spread > 0 else 1.0

    def _synth_sentence(self, sentence: str, rate: float) -> Iterator[np.ndarray]:
        cached = self._cache_get(sentence)
        if cached is not None:
            yield cached
            return
        parts = []
        for chunk in self.backend.synthesize(sentence, rate):
            if self.stop_requested.is_set():
                return
            parts.append(chunk)
            yield chunk
        if parts:
            self._cache_put(sentence, np.concatenate(parts))

    def render(self, text: str, rate: float = 1.0) -> np.ndarray:
        """The whole clip for `text`, from the cache when it is there (fillers, prewarm)."""
        text = _sanitize_for_speech(text)
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        parts = list(self.backend.synthesize(text, rate))
        pcm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
        self._cache_put(text, pcm)
        return pcm

    def prewarm(
        self, lines: Iterable[str], pause_while=None, stop: threading.Event | None = None
    ) -> threading.Thread:
        """Synthesizes fixed lines into the cache in the background, so the first time the robot
        says them they start instantly too. Sentence by sentence, the way speak() looks them up.

        pause_while: a callable; while it returns True (somebody is being served) the prewarm
        waits, so it never takes CPU from a live conversation. stop: abandon it (shutdown)."""

        def _run() -> None:
            started, count = time.monotonic(), 0
            for line in lines:
                for sentence in _split_sentences(_sanitize_for_speech(line)):
                    while pause_while is not None and pause_while():
                        if stop is not None and stop.is_set():
                            return
                        time.sleep(0.2)
                    if stop is not None and stop.is_set():
                        return
                    if self._cache_get(sentence) is None:
                        try:
                            self._cache_put(
                                sentence,
                                np.concatenate(list(self.backend.synthesize(sentence, 1.0))),
                            )
                            count += 1
                        except Exception as exc:  # noqa: BLE001
                            log.warning(f"[tts] prewarm failed on {sentence!r} ({exc!r})")
                            return
            rt_log("TTS", f"prewarmed {count} line(s) in {time.monotonic() - started:.1f}s")

        thread = threading.Thread(target=_run, name="tts-prewarm", daemon=True)
        thread.start()
        return thread

    # -- playing -----------------------------------------------------------------------------------
    def play_clip(self, pcm: np.ndarray, on_first_audio=None) -> dict:
        """Plays already-rendered audio (blocking), honouring stop_requested."""
        return self.player.play(
            [pcm], self.sample_rate, stop=self.stop_requested, on_first_audio=on_first_audio
        )

    def play_clip_async(self, pcm: np.ndarray) -> threading.Thread:
        return self.player.play_async(pcm, self.sample_rate)

    def _play_sentences(
        self, sentences: Iterable[str], on_first_audio=None
    ) -> tuple[float | None, int, list[str]]:
        """Shared playback core for speak() and speak_stream().

        A producer thread pulls sentences (for speak_stream that means pulling the LLM's token
        stream too) and synthesizes them into a queue; this thread plays from the queue. So
        generation, synthesis and playback all overlap, and the next sentence is normally ready
        before the current one has finished playing.
        """
        work: queue.Queue[tuple[str, object] | None] = queue.Queue()
        failure: list[BaseException] = []
        rate = self._rate()

        def _produce() -> None:
            try:
                for sentence in sentences:
                    if self.stop_requested.is_set():
                        break
                    work.put(("sentence", sentence))
                    for chunk in self._synth_sentence(sentence, rate):
                        if self.stop_requested.is_set():
                            break
                        work.put(("audio", chunk))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                failure.append(exc)
            finally:
                work.put(None)

        spoken: list[str] = []

        def _chunks() -> Iterator[np.ndarray]:
            while True:
                item = work.get()
                if item is None:
                    return
                kind, value = item
                if kind == "sentence":
                    spoken.append(value)
                else:
                    yield value

        producer = threading.Thread(target=_produce, name="tts-synth", daemon=True)
        producer.start()

        def _first_audio() -> None:
            rt_log("TTS", "playing audio - first sound")
            if on_first_audio is not None:
                on_first_audio()

        result = self.player.play(
            _chunks(), self.sample_rate, stop=self.stop_requested, on_first_audio=_first_audio
        )
        if result["first_audio_ms"] is not None:
            rt_log("TTS", f"first sound after {result['first_audio_ms']:.0f}ms")
        if failure and not self.stop_requested.is_set():
            raise failure[0]
        return result["first_audio_ms"], len(spoken), spoken

    def speak_stream(self, pieces: Iterable[str], on_first_audio=None) -> dict:
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
        rt_log("TTS", "started - streaming the answer as it generates")

        if not RT.STREAMING_TTS:
            # Opt-out for a setup where streaming misbehaves (a fussy audio device that clicks
            # between synthesis calls, say). Costs exactly what streaming bought: the first word
            # now waits for the LAST token instead of the first sentence. Nothing else changes -
            # same return shape, same transcript, same stop_requested handling - so the caller
            # cannot tell the difference except in the latency number it prints.
            whole = "".join(p for p in pieces if p)
            rt_log("TTS", "STREAMING_TTS=false - waiting for the whole answer before speaking")
            result = self.speak(whole, on_first_audio=on_first_audio)
            result["text"] = _sanitize_for_speech(whole)
            return result

        def _sentences() -> Iterator[str]:
            buffer = ""
            emitted = 0
            for piece in pieces:
                if self.stop_requested.is_set():
                    return
                if not piece:
                    continue
                buffer += piece
                while True:
                    sentence, buffer = _pop_sentence(buffer, final=False, first=(emitted == 0))
                    if sentence is None:
                        break
                    cleaned = _sanitize_for_speech(sentence)
                    if cleaned:
                        emitted += 1
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

        ttfa_ms, num_segments, started = self._play_sentences(_tracked(), on_first_audio)
        interrupted = self.stop_requested.is_set()
        if interrupted:
            rt_log("TTS", f"stopped early after {num_segments} fragment(s) - interrupted")
        # Only what was actually SAID (sentences that started playing) - not everything the
        # model generated ahead of the voice. The transcript and the conversation history both
        # use this, and after an interruption the visitor never heard the rest (2026-10-27,
        # found by the end-to-end test).
        text = " ".join(started if interrupted else spoken_parts)
        if interrupted and text:
            text += " ..."
        return {
            "time_to_first_audio_ms": ttfa_ms or 0.0,
            "num_segments": num_segments,
            "text": text,
            "interrupted": interrupted,
        }

    def speak(self, text: str, on_first_audio=None) -> dict:
        """Same measurement shape as the mock's TTSSpeaker.speak() - time_to_first_audio_ms and
        num_segments - but time_to_first_audio_ms here is measured against REAL audio actually
        reaching the speaker, not against a fully-materialized (and, for the mock, fake) list."""
        text = _sanitize_for_speech(text)
        sentences = _split_sentences(text)
        if not sentences:
            return {"time_to_first_audio_ms": 0.0, "num_segments": 0, "text": ""}

        self.stop_requested.clear()
        ttfa_ms, num_segments, spoken = self._play_sentences(sentences, on_first_audio)
        return {
            "time_to_first_audio_ms": ttfa_ms or 0.0,
            "num_segments": num_segments,
            "text": " ".join(spoken),
            "interrupted": self.stop_requested.is_set(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    print(f"Loading Piper voice '{VOICE_NAME}'...")
    speaker = LiveTTSSpeaker()

    result = speaker.speak(
        "Take the elevator to the 4th floor. "
        "Then it's the second door on your right. "
        "The Computer Science department office is right there."
    )

    print(
        f"\nTime-To-First-Audio: {result['time_to_first_audio_ms']:.1f}ms, "
        f"segments={result['num_segments']}"
    )

    if result["time_to_first_audio_ms"] <= 2000:
        print("OK: within STAGES.md's <1-2s Time-To-First-Audio target.")
    else:
        print(
            "WARNING: slower than STAGES.md's <1-2s Time-To-First-Audio target - try a "
            "smaller/faster voice model (see README.md), or check CPU load."
        )

    # --- streaming input (speak_stream) ------------------------------------------------------
    # The same answer, delivered the way a real LLM delivers it: a few characters at a time, with
    # a deliberate delay per fragment standing in for generation time. The point being measured
    # is that Time-To-First-Audio stays close to the ONE-SENTENCE mark instead of growing with
    # the length of the whole answer - which is exactly what speak() above cannot do, because it
    # can only be called once the entire string already exists.
    print("\n--- streaming input (simulating a live LLM token stream) ---")

    def _fake_token_stream():
        answer = (
            "Take the elevator to the 4th floor. "
            "Then it's the second door on your right. "
            "The Computer Science department office is right there."
        )
        for i in range(0, len(answer), 7):
            time.sleep(0.05)  # stands in for real per-token generation latency
            yield answer[i : i + 7]

    stream_start = time.monotonic()
    stream_result = speaker.speak_stream(_fake_token_stream())
    total_ms = (time.monotonic() - stream_start) * 1000

    print(
        f"\nTime-To-First-Audio: {stream_result['time_to_first_audio_ms']:.1f}ms, "
        f"segments={stream_result['num_segments']}, total={total_ms:.0f}ms"
    )
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
