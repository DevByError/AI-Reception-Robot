"""A small simulated room for end-to-end tests: a camera, a microphone, a speaker and a visitor.

Everything below the robot's own code is replaced, nothing above it:

  camera      FakeCamera stands in for cv2.VideoCapture - a 1280x720 frame with a real face
              photo pasted in while the visitor is "present", an empty room otherwise.
  speaker     FakeOutputStream stands in for sounddevice.RawOutputStream - like a real sound
              card it plays at real time: in callback mode (the robot's always-running stream)
              it asks for a block every 40ms on its own clock; in blocking mode write() blocks
              while its small buffer is full and abort() drops what is buffered. It records
              what the robot said and sends it into the room as echo.
  microphone  FakeInputStream stands in for sounddevice.RawInputStream - it calls the callback
              in real time with: the visitor's voice + the robot's own echo (delayed and
              quieter, ~60ms late) + a little room noise. So echo cancellation, barge-in and the turn
              detector are all exercised on realistic input.
  lips        FakeMouthDetector stands in for mediapipe: a face's lips "move" exactly while the
              visitor's voice is being played into the microphone with lips=True, and stay
              still for a voice from off camera (lips=False).

The robot code itself - detection, recognition, engagement, turn-taking, the addressee check,
speech recognition (the streaming Zipformer), the turn model, Kokoro/Piper, echo
cancellation, the database (SQLite) - is all real.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import resample_poly

MIC_RATE = 16000
FACE_IMAGE = Path(__file__).resolve().parent / "data" / "face.jpg"


def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32)
    if rate == MIC_RATE:
        return pcm
    from math import gcd

    g = gcd(MIC_RATE, rate)
    return resample_poly(pcm, MIC_RATE // g, rate // g).astype(np.float32)


class World:
    def __init__(
        self,
        echo_gain: float = 0.35,
        noise_rms: float = 0.002,
        fps: float = 15.0,
        echo_delay_s: float = 0.06,
    ):
        self.echo_gain = echo_gain
        self.echo_delay_s = echo_delay_s
        self.noise_rms = noise_rms
        self.fps = fps
        self.stop = threading.Event()
        self.face_present = threading.Event()
        self.lips_moving = threading.Event()
        self._lock = threading.Lock()
        self._visitor = deque()  # float32 samples at 16k, consumed in real time
        # The robot's audio on its way to the microphone, placed by TIME: (first mic sample
        # index, samples). Sample index i of the microphone is time _mic_t0 + i/16000.
        self._echo: list[tuple[int, np.ndarray]] = []
        self._echo_end = 0
        self._mic_t0: float | None = None
        self._mic_index = 0
        self.played: list[tuple[float, float, int, bytes]] = []  # (start, end, rate, pcm)
        self._last_write_end = 0.0  # when the audio written so far finishes playing
        self._play_end = 0.0
        self._writing = 0
        self._run_started = 0.0  # start of the current stretch of continuous output
        self.marks: list[tuple[str, float]] = []  # (what, when) - the visitor's side, for logs
        self.mic_blocks = 0
        self.underflows = 0  # speaker callbacks that came back too late
        self._rng = np.random.default_rng(0)
        self._frame_face = self._make_frame(with_face=True)
        self._frame_empty = self._make_frame(with_face=False)

    # -- the camera ------------------------------------------------------------------------------
    def _make_frame(self, with_face: bool) -> np.ndarray:
        frame = np.full((720, 1280, 3), (70, 80, 90), np.uint8)
        if with_face:
            face = cv2.imread(str(FACE_IMAGE))
            face = cv2.resize(face, (300, 300))
            frame[200:500, 490:790] = face
        return frame

    def camera(self):
        world = self

        class FakeCamera:
            def __init__(self, *_a, **_k):
                self._next = time.monotonic()

            def isOpened(self):
                return True

            def set(self, *_a):
                return True

            def get(self, prop):
                return {cv2.CAP_PROP_FRAME_WIDTH: 1280, cv2.CAP_PROP_FRAME_HEIGHT: 720}.get(prop, 0)

            def read(self):
                if world.stop.is_set():
                    return False, None
                self._next += 1.0 / world.fps
                delay = self._next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self._next = time.monotonic()
                frame = world._frame_face if world.face_present.is_set() else world._frame_empty
                return True, frame.copy()

            def release(self):
                pass

        return FakeCamera

    # -- the speaker -------------------------------------------------------------------------------
    def output_stream(self):
        world = self

        class FakeOutputStream:
            latency = 0.06  # how far ahead of the speaker write() may run

            def __init__(
                self,
                samplerate,
                channels=1,
                dtype="int16",
                device=None,
                blocksize=0,
                callback=None,
                **_k,
            ):
                self.rate = int(samplerate)
                self.callback = callback
                self.blocksize = int(blocksize) or int(self.rate * 0.04)
                self._running = threading.Event()

            def start(self):
                if self.callback is not None:
                    self._running.set()
                    threading.Thread(target=self._run, daemon=True, name="fake-speaker").start()

            def _run(self):
                """The sound card's clock: it asks for each block `latency` before playing it.
                A callback that comes back later than that is an UNDERFLOW, as on a real card:
                it played silence meanwhile, and this block (and everything after it) plays
                late."""
                period = self.blocksize / float(self.rate)
                plays_at = time.monotonic() + self.latency
                buf = bytearray(self.blocksize * 2)
                while self._running.is_set() and not world.stop.is_set():
                    delay = plays_at - self.latency - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    self.callback(memoryview(buf), self.blocksize, None, None)
                    pcm = np.frombuffer(bytes(buf), dtype=np.int16).copy()
                    now = time.monotonic()
                    if now > plays_at:
                        world.underflows += 1
                        plays_at = now
                    world._play_block(plays_at, pcm, self.rate)
                    plays_at += period

            def write(self, data):
                pcm = np.frombuffer(bytes(data), dtype=np.int16)
                seconds = pcm.size / float(self.rate)
                with world._lock:
                    now = time.monotonic()
                    start = max(now, world._play_end)  # plays after what is buffered
                    if start - world._play_end > 0.15:
                        world._run_started = start
                    world._play_end = world._last_write_end = start + seconds
                    world._writing += 1
                    world._add_echo(start, to_16k(pcm, self.rate) / 32768.0 * world.echo_gain)
                    world.played.append((start, start + seconds, self.rate, bytes(data)))
                wait = start + seconds - self.latency - time.monotonic()
                if wait > 0:
                    time.sleep(wait)  # the buffer is full
                with world._lock:
                    world._writing -= 1

            def stop(self):
                self._running.clear()
                # plays out what is buffered
                wait = world._play_end - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

            def abort(self):
                self._running.clear()
                world._abort_playback()

            def close(self):
                self._running.clear()

        return FakeOutputStream

    # -- the microphone ----------------------------------------------------------------------------
    def input_stream(self):
        world = self

        class FakeInputStream:
            def __init__(
                self,
                samplerate,
                blocksize,
                dtype="int16",
                channels=1,
                device=None,
                callback=None,
                **_k,
            ):
                assert int(samplerate) == MIC_RATE
                self.blocksize = int(blocksize)
                self.callback = callback
                self.device = device
                self._running = threading.Event()
                self._thread = None

            def start(self):
                self._running.set()
                self._thread = threading.Thread(target=self._run, daemon=True, name="fake-mic")
                self._thread.start()

            def _run(self):
                period = self.blocksize / float(MIC_RATE)
                next_at = time.monotonic()
                with world._lock:
                    world._mic_t0, world._mic_index = next_at - period, 0
                while self._running.is_set() and not world.stop.is_set():
                    block = world._mic_block(self.blocksize)
                    self.callback(block.tobytes(), self.blocksize, None, None)
                    world.mic_blocks += 1
                    next_at += period
                    delay = next_at - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

            def stop(self):
                self._running.clear()

            def close(self):
                self._running.clear()

            def __enter__(self):
                self.start()
                return self

            def __exit__(self, *exc):
                self.stop()

        return FakeInputStream

    def _play_block(self, at: float, pcm: np.ndarray, rate: int) -> None:
        """One block from the always-running stream, played at `at` - silence included, which
        reaches the room (and the microphone) like everything else."""
        with self._lock:
            self._add_echo(at - 0.0, to_16k(pcm, rate) / 32768.0 * self.echo_gain)
            if not np.any(pcm):
                return
            seconds = pcm.size / float(rate)
            if at - self._play_end > 0.15:
                self._run_started = at
            self._play_end = self._last_write_end = max(self._play_end, at + seconds)
            self.played.append((at, at + seconds, rate, pcm.tobytes()))

    def _abort_playback(self) -> None:
        """What a sound card does on abort(): the buffered audio is never played."""
        with self._lock:
            now = time.monotonic()
            if self._play_end > now:
                self._play_end = self._last_write_end = now
                if self._mic_t0 is not None:
                    cut = int(round((now + self.echo_delay_s - self._mic_t0) * MIC_RATE))
                    self._echo = [
                        (at, seg[: max(0, cut - at)]) for at, seg in self._echo if cut > at
                    ]
                    self._echo_end = min(self._echo_end, cut)

    def _add_echo(self, written_at: float, samples: np.ndarray) -> None:
        """The speaker->room->microphone path is not instant: the output buffer, the DAC and the
        air add tens of ms (echo_delay_s). Called with the lock held."""
        if self._mic_t0 is None:
            return
        at = int(round((written_at + self.echo_delay_s - self._mic_t0) * MIC_RATE))
        if self._echo and abs(at - self._echo_end) < MIC_RATE * 0.05:
            at = self._echo_end  # the same stretch of sound: keep it gapless
        at = max(at, self._mic_index)
        self._echo.append((at, np.asarray(samples, np.float32)))
        self._echo_end = at + len(samples)

    def _mic_block(self, n: int) -> np.ndarray:
        out = self._rng.normal(0.0, self.noise_rms, n).astype(np.float32)
        with self._lock:
            k = min(n, len(self._visitor))
            if k:
                out[:k] += np.array([self._visitor.popleft() for _ in range(k)], np.float32)
            if not self._visitor:
                self.lips_moving.clear()
            a, b = self._mic_index, self._mic_index + n
            keep = []
            for at, seg in self._echo:
                lo, hi = max(a, at), min(b, at + len(seg))
                if lo < hi:
                    out[lo - a : hi - a] += seg[lo - at : hi - at]
                if at + len(seg) > b:
                    keep.append((at, seg))
            self._echo = keep
            self._mic_index = b
        return (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16)

    # -- the visitor -------------------------------------------------------------------------------
    def say(self, pcm16k: np.ndarray, lips: bool = True, wait: bool = True) -> float:
        """Plays a visitor's sentence into the microphone. Returns when it has been heard
        (wait=True). Returns the time it ended."""
        samples = np.asarray(pcm16k, dtype=np.float32)
        if np.abs(samples).max() > 1.5:
            samples = samples / 32768.0
        self.marks.append(
            (f"visitor starts ({samples.size / MIC_RATE:.1f}s, lips={lips})", time.monotonic())
        )
        with self._lock:
            self._visitor.extend(samples)
            if lips:
                self.lips_moving.set()
        if wait:
            time.sleep(samples.size / float(MIC_RATE) + 0.05)
        return time.monotonic()

    # -- what the robot did ------------------------------------------------------------------------
    def robot_speaking(self) -> bool:
        with self._lock:
            return time.monotonic() - self._play_end < 0.15

    def wait_robot_spoke_then_silent(
        self, since: float, silence_s: float = 1.0, timeout_s: float = 60.0
    ) -> float:
        """Waits until the robot has said something after `since` and then been quiet for
        silence_s. Returns when its last sound ended."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                last_end = self._last_write_end
                writing = self._writing
            if writing == 0 and last_end > since and time.monotonic() - last_end >= silence_s:
                return last_end
            time.sleep(0.05)
        raise TimeoutError(f"the robot did not speak (and stop) within {timeout_s}s")

    def wait_robot_starts(self, since: float, timeout_s: float = 30.0) -> float:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                starts = [s for s, _e, _r, _p in self.played if s > since]
            if starts and min(starts) <= time.monotonic():
                return min(starts)
            time.sleep(0.02)
        raise TimeoutError(f"the robot did not start speaking within {timeout_s}s")

    def wait_talking_for(self, since: float, min_s: float, timeout_s: float = 30.0) -> float:
        """Waits until the robot has been talking without a break for min_s, in a stretch that
        began after `since` (skips a short filler). Returns when that stretch began."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                run, busy = self._run_started, time.monotonic() < self._play_end
            if busy and run > since and time.monotonic() - run >= min_s:
                return run
            time.sleep(0.02)
        raise TimeoutError(f"the robot did not talk for {min_s}s straight within {timeout_s}s")

    def first_sound_after(self, t: float) -> float | None:
        with self._lock:
            starts = [s for s, _e, _r, _p in self.played if s >= t]
        return min(starts) if starts else None


def fake_mouth_detector(world: World):
    """A MouthActivityDetector stand-in driven by World.lips_moving."""
    import sys

    sys.path.append(
        str(Path(__file__).resolve().parents[1] / "stages" / "stage02_face_recognition")
    )
    from mouth_activity import MouthActivity

    class FakeMouthDetector:
        def __init__(self, *_a, **_k):
            pass

        def _activity(self, face_id, bbox):
            moving = world.lips_moving.is_set()
            return MouthActivity(
                face_id=face_id,
                bbox=tuple(bbox),
                mouth_open_ratio=0.3,
                activity_score=0.05 if moving else 0.0,
                is_speaking=moving,
                samples=12,
                yaw=0.0,
            )

        def update_faces(self, frame, faces):
            return [self._activity(tid, bbox) for tid, bbox in faces]

        def update(self, frame):
            return []

        def most_likely_speaker(self, activities):
            speaking = [a for a in activities if a.is_speaking]
            return max(speaking, key=lambda a: a.activity_score) if speaking else None

        def close(self):
            pass

    return FakeMouthDetector
