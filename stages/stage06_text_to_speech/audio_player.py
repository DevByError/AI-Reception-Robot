"""
Stage 06 - the one place audio leaves the robot.

Before this, three places opened their own output stream (answers, fillers, the greeting), each
on "the Windows default" device. Everything now plays through AudioPlayer, which:

  - plays on SPEAKER_DEVICE (by name or number - see stage04's resolve_device);
  - writes in small blocks, so a barge-in stops the robot within ~40ms instead of at the end of
    a long buffer;
  - hands every block to the echo canceller (stage04's EchoCanceller) as it is played, which is
    what lets the microphone subtract the robot's own voice;
  - with echo cancellation on, keeps ONE output stream running for the robot's whole life,
    playing silence between sentences (_ContinuousOutput, below);
  - keeps a short history of how loud the robot is at each moment, which the animated face
    (stage08's face_display.py) uses to move its mouth in time with the speech.

Audio is int16 mono numpy arrays everywhere.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

BLOCK_MS = 40
MAX_AHEAD_MS = 200  # continuous mode: how much audio may wait in the queue for the speaker
# A running speaker stream that has not asked for audio for this long is stuck and is reopened;
# a newly opened one gets longer to make its first request.
STALL_S = 1.0
FIRST_CALLBACK_S = 4.0


class _ContinuousOutput:
    """One output stream that never stops: it plays whatever is queued and silence otherwise,
    and hands every block it plays - the silence too - to the echo canceller.

    Why (2026-10-27, found by the end-to-end test): opening a new stream for every sentence
    puts each sentence a few milliseconds earlier or later relative to the microphone than the
    one before - however carefully it is timed, the opening itself varies. The canceller's
    learned echo filter is only valid for one alignment, so every shift made it leak the start
    of the sentence (measured: the cleaned audio was LOUDER than the raw echo for half a
    second, and the robot interrupted itself). A stream that never stops is driven by the
    sound card's own clock, sample by sample, so the alignment never moves. It is also how
    WebRTC does it, and it saves opening a device before every sentence.
    """

    def __init__(
        self,
        sample_rate: int,
        device: int | None,
        block: int,
        tap: Callable[[np.ndarray], None] | None,
        note_level: Callable[[float, np.ndarray], None],
    ):
        import sounddevice as sd

        self.rate = int(sample_rate)
        self.block = block
        self._tap = tap
        self._note_level = note_level
        self._q: deque[np.ndarray] = deque()
        self._queued = 0  # samples waiting
        self._enqueued = 0  # samples ever queued
        self._done = 0  # samples ever played or dropped
        self._cond = threading.Condition()
        self.last_callback = time.monotonic()
        self._calling = False  # has the sound card asked for audio yet?
        self.underflows = 0
        self.broken = False
        self.stream = sd.RawOutputStream(
            samplerate=self.rate,
            channels=1,
            dtype="int16",
            blocksize=block,
            device=device,
            callback=self._callback,
        )
        self.latency = float(getattr(self.stream, "latency", 0.0) or 0.0)
        self.stream.start()

    def _callback(self, outdata, frames, _time_info, status) -> None:
        if status and getattr(status, "output_underflow", False):
            self.underflows += 1
        buf = np.zeros(frames, dtype=np.int16)
        filled = 0
        with self._cond:
            while filled < frames and self._q:
                head = self._q[0]
                take = min(frames - filled, head.size)
                buf[filled : filled + take] = head[:take]
                if take == head.size:
                    self._q.popleft()
                else:
                    self._q[0] = head[take:]
                filled += take
            self._queued -= filled
            self._done += filled
            self.last_callback = time.monotonic()
            self._calling = True
            self._cond.notify_all()
        outdata[:] = buf.tobytes()
        if self._tap is not None:
            try:
                self._tap(buf)
            except Exception:  # noqa: BLE001 - never stop speech over AEC
                self._tap = None
        self._note_level(time.monotonic() + self.latency, buf)

    def enqueue(self, pcm: np.ndarray) -> int:
        """Queues audio; returns the mark wait_done() waits for."""
        with self._cond:
            self._q.append(pcm)
            self._queued += pcm.size
            self._enqueued += pcm.size
            return self._enqueued

    def _alive(self) -> bool:
        # A sound card waking from power saving can take over a second to ask for its first
        # block (seen on a laptop's Realtek speakers, 2026-10-27: 1.2s). Only a stream that HAS
        # been running and then stops is treated as stuck after a second.
        limit = STALL_S if self._calling else FIRST_CALLBACK_S
        if time.monotonic() - self.last_callback > limit:
            if not self.broken:
                log.info(
                    "[audio] the speaker stream stopped asking for audio - it will be reopened."
                )
            self.broken = True
        return not self.broken

    def wait_room(self, max_ahead: int, stop: threading.Event | None) -> None:
        with self._cond:
            while (
                self._queued > max_ahead
                and not (stop is not None and stop.is_set())
                and self._alive()
            ):
                self._cond.wait(0.02)

    def wait_done(self, mark: int, stop: threading.Event | None) -> bool:
        """Blocks until everything up to `mark` has been played. False if stopped first."""
        with self._cond:
            while self._done < mark:
                if stop is not None and stop.is_set():
                    return False
                if not self._alive():
                    return True
                self._cond.wait(0.02)
        return True

    def clear(self) -> None:
        """Drops what is queued (a barge-in). What the sound card already holds still plays -
        a few tens of ms."""
        with self._cond:
            self._done += self._queued
            self._q.clear()
            self._queued = 0
            self._cond.notify_all()

    def close(self) -> None:
        try:
            self.stream.abort()
            self.stream.close()
        except Exception:  # noqa: BLE001
            pass


class AudioPlayer:
    def __init__(
        self,
        device: int | None = None,
        far_tap_factory: Callable[[int], Callable[[np.ndarray], None]] | None = None,
        block_ms: int = BLOCK_MS,
        continuous: bool | None = None,
    ):
        self.device = device
        self.far_tap_factory = far_tap_factory
        self.block_ms = block_ms
        # One never-ending stream whenever there is an echo canceller to keep aligned.
        self.continuous = far_tap_factory is not None if continuous is None else continuous
        self._out: _ContinuousOutput | None = None
        self._levels: deque[tuple[float, float]] = deque(maxlen=400)
        self._levels_lock = threading.Lock()
        self._play_lock = threading.Lock()  # one sound at a time: the AEC reference is one stream
        self.playing = threading.Event()

    # -- loudness for the face -----------------------------------------------------------------
    def level_at(self, now: float | None = None) -> float:
        """How loud the robot is at `now` (0..1), accounting for the device's output delay."""
        now = time.monotonic() if now is None else now
        with self._levels_lock:
            level = 0.0
            for t, lv in reversed(self._levels):
                if t <= now:
                    level = lv if now - t < 0.2 else 0.0
                    break
            return level

    def _note_level(self, when: float, block: np.ndarray) -> None:
        rms = (
            float(np.sqrt(np.mean(block.astype(np.float32) ** 2))) / 32768.0 if block.size else 0.0
        )
        with self._levels_lock:
            self._levels.append((when, min(1.0, rms * 6.0)))

    # -- playback --------------------------------------------------------------------------------
    def play(
        self,
        chunks: Iterable[np.ndarray],
        sample_rate: int,
        stop: threading.Event | None = None,
        on_first_audio: Callable[[], None] | None = None,
    ) -> dict:
        """Plays int16 chunks as they arrive (a generator is fine - that is how streaming works).

        Opens one stream for the whole call, lazily on the first chunk (or, in continuous mode,
        queues onto the stream that is always running). Returns
        {"first_audio_ms": float | None, "seconds": float, "stopped": bool}.
        """
        if self.continuous:
            return self._play_continuous(chunks, sample_rate, stop, on_first_audio)
        import sounddevice as sd

        start = time.monotonic()
        first_audio_ms: float | None = None
        played = 0
        stream = None
        stopped = False
        tap = self.far_tap_factory(sample_rate) if self.far_tap_factory else None
        block = max(1, int(sample_rate * self.block_ms / 1000))
        with self._play_lock:
            self.playing.set()
            try:
                for chunk in chunks:
                    if stop is not None and stop.is_set():
                        stopped = True
                        break
                    pcm = np.asarray(chunk, dtype=np.int16).reshape(-1)
                    for i in range(0, pcm.size, block):
                        if stop is not None and stop.is_set():
                            stopped = True
                            break
                        piece = pcm[i : i + block]
                        if stream is None:
                            stream = sd.RawOutputStream(
                                samplerate=sample_rate,
                                channels=1,
                                dtype="int16",
                                device=self.device,
                            )
                            stream.start()
                        if first_audio_ms is None:
                            first_audio_ms = (time.monotonic() - start) * 1000
                            if on_first_audio is not None:
                                try:
                                    on_first_audio()
                                except Exception as exc:  # noqa: BLE001 - an observer only
                                    log.warning(
                                        f"[tts] first-audio callback failed ({exc!r}) - "
                                        f"continuing to speak."
                                    )
                        if tap is not None:
                            try:
                                tap(piece)
                            except Exception:  # noqa: BLE001 - never stop speech over AEC
                                tap = None
                        latency = getattr(stream, "latency", 0.0) or 0.0
                        self._note_level(time.monotonic() + latency, piece)
                        stream.write(piece.tobytes())
                        played += piece.size
                    if stopped:
                        break
            finally:
                if stream is not None:
                    # abort() on a stop: stop() would first play out whatever is buffered.
                    try:
                        if stopped:
                            stream.abort()
                        else:
                            stream.stop()
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.playing.clear()
                with self._levels_lock:
                    self._levels.append((time.monotonic(), 0.0))
        return {
            "first_audio_ms": first_audio_ms,
            "seconds": played / float(sample_rate),
            "stopped": stopped,
        }

    def _output(self, sample_rate: int) -> _ContinuousOutput:
        out = self._out
        if out is not None and (out.rate != int(sample_rate) or out.broken):
            out.close()
            out = self._out = None
        if out is None:
            tap = self.far_tap_factory(sample_rate) if self.far_tap_factory else None
            block = max(1, int(sample_rate * self.block_ms / 1000))
            out = self._out = _ContinuousOutput(
                sample_rate, self.device, block, tap, self._note_level
            )
        return out

    def _play_continuous(self, chunks, sample_rate, stop, on_first_audio) -> dict:
        start = time.monotonic()
        first_audio_ms: float | None = None
        played = 0
        stopped = False
        with self._play_lock:
            self.playing.set()
            try:
                out = self._output(sample_rate)
                block = out.block
                max_ahead = int(sample_rate * MAX_AHEAD_MS / 1000)
                mark = None
                for chunk in chunks:
                    if stop is not None and stop.is_set():
                        stopped = True
                        break
                    pcm = np.asarray(chunk, dtype=np.int16).reshape(-1)
                    for i in range(0, pcm.size, block):
                        out.wait_room(max_ahead, stop)
                        if stop is not None and stop.is_set():
                            stopped = True
                            break
                        if first_audio_ms is None:
                            first_audio_ms = (time.monotonic() - start) * 1000
                            if on_first_audio is not None:
                                try:
                                    on_first_audio()
                                except Exception as exc:  # noqa: BLE001 - an observer only
                                    log.warning(
                                        f"[tts] first-audio callback failed ({exc!r}) - "
                                        f"continuing to speak."
                                    )
                        piece = pcm[i : i + block].copy()
                        mark = out.enqueue(piece)
                        played += piece.size
                    if stopped:
                        break
                if not stopped and mark is not None and not out.wait_done(mark, stop):
                    stopped = True
                if stopped:
                    out.clear()
            finally:
                self.playing.clear()
        return {
            "first_audio_ms": first_audio_ms,
            "seconds": played / float(sample_rate),
            "stopped": stopped,
        }

    def close(self) -> None:
        """Stops the always-running stream (continuous mode). Safe to call more than once."""
        out, self._out = self._out, None
        if out is not None:
            out.close()

    def play_async(
        self, pcm: np.ndarray, sample_rate: int, stop: threading.Event | None = None
    ) -> threading.Thread:
        """Fire-and-forget playback of one clip (fillers, "mm-hm"). Errors are printed, never
        raised - a missing acknowledgment must not cost the visitor their answer."""

        def _run() -> None:
            try:
                self.play([pcm], sample_rate, stop=stop)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[audio] playback failed ({exc!r}).")

        thread = threading.Thread(target=_run, name="clip-playback", daemon=True)
        thread.start()
        return thread
