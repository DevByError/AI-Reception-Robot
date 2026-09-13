"""
Stage 06 (live, part 2) - pre-rendered acknowledgment fillers: the short "Let me see..." the
robot says the instant you stop talking, while the real answer is still being generated.

WHY THIS EXISTS, given that Stage 05 now streams the LLM straight into Piper (see
tts_speaker_live.py's speak_stream): streaming cuts time-to-first-word to roughly one sentence of
generation, but that is still several hundred milliseconds of DEAD SILENCE after the visitor
stops speaking, plus knowledge-base retrieval before it. Silence is what reads as "the machine is
processing" - a person fills it involuntarily ("hmm", "let me see", "sure -") and the listener
stops experiencing it as a wait at all. Measured latency and PERCEIVED latency are different
numbers, and this file only moves the second one.

Two rules make it work rather than annoy:

  1. ZERO synthesis delay. A filler synthesized on demand would add the very latency it exists to
     hide. Every phrase is rendered to raw PCM once at startup (~1-2s total, during model
     loading, where nobody is waiting) and afterwards playing one is a memory write to the audio
     device.

  2. NEVER the same one twice in a row. A robot that says "Let me see" before every single answer
     is more obviously a machine than one that says nothing - the repetition is the tell. See
     _pick_index(): random, but never repeating the phrase it just used.

Fillers are deliberately NOT recorded in the conversation transcript (main_live.py's _record) -
they carry no content, and a transcript alternating "Let me check for you." with real answers is
harder to read back when diagnosing a bad answer weeks later.

Run: python filler_bank.py    (renders the bank and plays every phrase once)
"""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice

# Kept SHORT and content-free on purpose.
#
# SHORT because a filler longer than the generation it covers makes the visit slower, not
# snappier - the real answer cannot start until the acknowledgment finishes. Measured rather
# than guessed: every phrase below renders to under a second of speech through
# en_US-amy-medium, and the self-test at the bottom asserts it stays that way. Two earlier
# candidates were cut for failing exactly that check ("Sure, one moment." at 1.55s, "Right, let
# me check." at 1.47s).
#
# CONTENT-FREE because the filler is chosen before the question has been understood - "Let me
# check the timetable" would be a lie the moment the question turns out to be about something
# else.
DEFAULT_FILLERS = (
    "Let me see.",
    "Good question.",
    "Okay.",
    "One second.",
    "Let me check.",
    "Sure thing.",
    "Right.",
)

# Fillers for the specific case of "the visitor asked something that needs a real lookup" vs. a
# plain acknowledgment. Not used by default (main_live.py passes DEFAULT_FILLERS) - kept here
# because the moment the robot can tell a directory question from open conversation, saying
# "let me look that up" for the first and "hmm" for the second is a noticeably better fit than
# one bank for both.
LOOKUP_FILLERS = (
    "Let me look that up.",
    "Checking that for you.",
    "One moment while I check.",
)


# Everything quieter than this fraction of the clip's own peak counts as silence for trimming.
# Relative rather than absolute so it works whatever level the voice renders at.
# Upper bound on how long any one filler may be, asserted by the self-test. Roughly the time a
# streamed grounded answer takes to produce its first sentence, so the acknowledgment lands and
# clears just as the real answer becomes available.
MAX_FILLER_S = 1.1

# Everything quieter than this fraction of the clip's own peak counts as silence for trimming.
# Relative rather than absolute so it works whatever level the voice renders at.
_SILENCE_FLOOR = 0.02
# A little padding left on each end, so trimming can't clip the attack of the first consonant or
# the decay of the last one - both are audible as a click.
_TRIM_PAD_MS = 20


def _trim_silence(pcm: bytes, sample_rate: int = 22050) -> bytes:
    """Removes the leading and trailing silence Piper renders around a short phrase.

    Measured at ~0.2s per clip, which sounds like nothing until you remember where it lands: the
    trailing silence sits directly between the acknowledgment and the real answer, so it is dead
    air in the exact place this whole file exists to remove. Trimming is a slice of an array that
    is already in memory, done once at startup.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return pcm

    peak = int(np.max(np.abs(samples)))
    if peak == 0:
        return pcm

    loud = np.flatnonzero(np.abs(samples) > peak * _SILENCE_FLOOR)
    if loud.size == 0:
        return pcm

    pad = int(sample_rate * _TRIM_PAD_MS / 1000)
    start = max(int(loud[0]) - pad, 0)
    end = min(int(loud[-1]) + pad + 1, samples.size)
    return samples[start:end].tobytes()


class FillerPlayback:
    """Handle for a filler that is currently playing on a background thread.

    The whole point is that the caller does NOT wait: it starts the filler, spends that time
    doing the retrieval and the LLM call, and only then calls wait() - by which point the filler
    has almost always already finished and wait() returns immediately. That overlap is the
    feature; a blocking play() would add its own duration to every answer.
    """

    def __init__(self, thread: threading.Thread | None, duration_s: float):
        self._thread = thread
        self.duration_s = duration_s

    def wait(self, timeout_s: float | None = None) -> None:
        """Blocks until the filler has finished playing, so the real answer never talks over it.

        timeout_s defaults to a little more than the clip's own length: a filler that somehow
        hangs (a device that stops accepting writes, say) must never be able to hold up the
        answer the visitor is actually waiting for - Stage 09's graceful-degradation rule.
        """
        if self._thread is None:
            return
        self._thread.join(timeout=timeout_s if timeout_s is not None else self.duration_s + 1.0)


_NULL_PLAYBACK = FillerPlayback(None, 0.0)


class FillerBank:
    """Renders a set of short phrases once, then plays them on demand with no synthesis cost."""

    def __init__(self, voice: PiperVoice, phrases: tuple[str, ...] = DEFAULT_FILLERS,
                 verbose: bool = True):
        self.phrases = tuple(phrases)
        self._clips: list[tuple[bytes, int, int]] = []  # (pcm, sample_rate, channels)
        self._last_index: int | None = None
        self._lock = threading.Lock()

        started = time.monotonic()
        for phrase in self.phrases:
            pcm = bytearray()
            sample_rate = 22050
            channels = 1
            for chunk in voice.synthesize(phrase):
                pcm += chunk.audio_int16_bytes
                sample_rate = chunk.sample_rate
                channels = chunk.sample_channels
            self._clips.append((_trim_silence(bytes(pcm)), sample_rate, channels))

        if verbose:
            elapsed_ms = (time.monotonic() - started) * 1000
            total_s = sum(self._clip_duration_s(i) for i in range(len(self._clips)))
            print(f"[filler] pre-rendered {len(self._clips)} acknowledgment phrase(s) in "
                  f"{elapsed_ms:.0f}ms ({total_s:.1f}s of audio, held in memory)")

    def _clip_duration_s(self, index: int) -> float:
        pcm, sample_rate, channels = self._clips[index]
        return len(pcm) / (2 * max(channels, 1) * max(sample_rate, 1))  # 2 bytes per int16 sample

    def _pick_index(self) -> int:
        """Random, but never the phrase used last time - see rule 2 in the module docstring.

        Uses a lock because the pick happens on the caller's thread while a previous filler may
        still be playing on another one, and two visitors in quick succession hearing the same
        phrase is exactly what this is meant to avoid.
        """
        with self._lock:
            if len(self._clips) == 1:
                self._last_index = 0
                return 0
            choices = [i for i in range(len(self._clips)) if i != self._last_index]
            index = random.choice(choices)
            self._last_index = index
            return index

    def play_async(self) -> FillerPlayback:
        """Starts a filler playing immediately and returns without waiting for it.

        Failures are swallowed: an audio-device problem during a filler must never stop the robot
        from giving the real answer, which is the part that actually matters. The caller gets a
        no-op handle whose wait() returns instantly.
        """
        if not self._clips:
            return _NULL_PLAYBACK

        index = self._pick_index()
        pcm, sample_rate, channels = self._clips[index]
        duration_s = self._clip_duration_s(index)

        def _play() -> None:
            stream = None
            try:
                stream = sd.RawOutputStream(
                    samplerate=sample_rate, channels=channels, dtype="int16"
                )
                stream.start()
                stream.write(pcm)
            except Exception as exc:  # noqa: BLE001 - see docstring
                print(f"[filler] playback failed ({exc!r}) - continuing to the real answer.")
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass

        thread = threading.Thread(target=_play, name="filler-playback", daemon=True)
        thread.start()
        return FillerPlayback(thread, duration_s)


if __name__ == "__main__":
    from tts_speaker_live import VOICE_MODEL, VOICE_NAME

    if not Path(VOICE_MODEL).exists():
        raise SystemExit(
            f"Voice model not found at {VOICE_MODEL} - run "
            f"'python -m piper.download_voices {VOICE_NAME}' in this folder first (see "
            f"README.md)."
        )

    print(f"Loading Piper voice '{VOICE_NAME}'...")
    bank = FillerBank(PiperVoice.load(str(VOICE_MODEL)))

    print("\nPhrase lengths after silence-trimming:")
    longest = 0.0
    for index, phrase in enumerate(bank.phrases):
        duration = bank._clip_duration_s(index)
        longest = max(longest, duration)
        print(f'  {duration:.2f}s  "{phrase}"')

    # The property that actually matters, and the one that is easy to break by adding a
    # friendlier-sounding but longer phrase later: a filler must be shorter than the generation
    # it hides, or it stops covering latency and starts adding it.
    assert longest <= MAX_FILLER_S, (
        f'"{bank.phrases[max(range(len(bank.phrases)), key=bank._clip_duration_s)]}" renders to '
        f"{longest:.2f}s, over the {MAX_FILLER_S}s budget - the real answer would be held up "
        f"waiting for the acknowledgment to finish"
    )

    # Play them all, measuring how long it takes to START each one - this is the number that
    # matters. If picking and starting a filler took as long as synthesizing one, the bank would
    # be pointless.
    worst_start_ms = 0.0
    for _ in range(len(bank.phrases)):
        started = time.monotonic()
        playback = bank.play_async()
        start_ms = (time.monotonic() - started) * 1000
        worst_start_ms = max(worst_start_ms, start_ms)
        playback.wait()

    print(f"\nSlowest time to START a filler: {worst_start_ms:.1f}ms")
    assert worst_start_ms < 50, (
        f"{worst_start_ms:.0f}ms to start a pre-rendered clip - that is synthesis-scale latency, "
        f"which defeats the point of pre-rendering"
    )

    # No-repeat rule: 200 picks should never produce the same index twice in a row.
    picks = [bank._pick_index() for _ in range(200)]
    assert all(a != b for a, b in zip(picks, picks[1:])), "a filler repeated back-to-back"

    print("OK: fillers start instantly and never repeat back-to-back.")
