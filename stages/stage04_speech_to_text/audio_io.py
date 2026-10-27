"""
Stage 04 - the microphone: which device, echo cancellation, and an optional capture process.
Added 2026-10-27.

Three problems this file fixes:

1. THE ROBOT HEARS ITSELF. Without echo cancellation the microphone records the robot's own
   voice, so barge-in had to be a loudness threshold ("talk louder than the speaker") that most
   rooms could not support, and the end of every answer leaked into the next listen.
   EchoCanceller runs WebRTC's AEC3 (the canceller in Chrome and Google Meet, via the `livekit`
   package): everything the speaker plays is fed in as a reference and subtracted from the
   microphone. Measured on recorded speech with a simulated room echo (docs/RESULTS.md): the
   robot's voice drops by 50+ dB once converged, delays from 40 to 400ms are handled, and a
   visitor's words after the echo transcribe exactly as they do without any echo.

   It plugs in without touching anything downstream: the microphone frames are cleaned before
   they reach audio_q, so the VAD, the barge-in monitor, calibrate_echo_floor() and Whisper all
   simply see a microphone that no longer hears the speaker. calibrate_echo_floor() then measures
   a tiny echo floor, which is what lets barge-in switch itself on.

2. WRONG DEVICE. The robot always used "the Windows default", which changes when earbuds
   connect. MIC_DEVICE / SPEAKER_DEVICE pick a device by name or number - resolve_device().

3. "INPUT OVERFLOW". The microphone callback runs in the robot's Python process and needs the
   GIL; a heavy camera frame or model load can hold it long enough for PortAudio to throw audio
   away. AUDIO_CAPTURE_PROCESS=true reads the microphone in a tiny separate Python process that
   does nothing else (this file run with --capture), and streams the samples back over a pipe.

Run: python audio_io.py              (lists devices, then shows the live mic level for 5s)
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

AEC_RATE = 16000
AEC_FRAME = 160            # WebRTC processes 10ms frames
FAR_FIFO_MAX_S = 1.5       # never hold more reference audio than this (the mic is not running)


# --- devices ----------------------------------------------------------------------------------

def resolve_device(spec: str | int | None, kind: str) -> int | None:
    """MIC_DEVICE / SPEAKER_DEVICE -> a sounddevice index, or None for the system default.

    Accepts an index ("3") or any part of the device name, case-insensitive ("jabra", "USB
    Audio"). When several devices match a name, the first one that can actually do `kind`
    ("input"/"output") wins - Windows lists the same hardware once per host API (MME,
    DirectSound, WASAPI), and MME (listed first) is the one that resamples for us.
    """
    import sounddevice as sd

    if spec is None or str(spec).strip() == "":
        return None
    spec = str(spec).strip()
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()
    if spec.isdigit():
        index = int(spec)
        if 0 <= index < len(devices) and devices[index][channels_key] > 0:
            return index
        print(f"[audio] {kind} device {index} does not exist or has no {kind} channels - "
              f"using the system default. Run stages/stage00_setup/list_audio_devices.py.")
        return None
    for index, dev in enumerate(devices):
        if spec.lower() in dev["name"].lower() and dev[channels_key] > 0:
            return index
    print(f"[audio] no {kind} device matches {spec!r} - using the system default. Run "
          f"stages/stage00_setup/list_audio_devices.py to see the names.")
    return None


def device_name(index: int | None, kind: str) -> str:
    import sounddevice as sd

    try:
        return str(sd.query_devices(index, kind)["name"])
    except Exception:  # noqa: BLE001 - informational only
        return "unknown device"


# --- resampling (for the echo reference) -------------------------------------------------------

class _ToAecRate:
    """Converts what the speaker plays (22050Hz Piper, 24000Hz Kokoro, ...) to 16kHz mono for
    the canceller, keeping state between blocks so block edges do not click.

    A short low-pass FIR (so content above 8kHz does not fold back into the reference) followed
    by linear interpolation. Deliberately not a library resampler: soxr's streaming mode returns
    its output in uneven bursts (nothing for one block, extra on the next), and with the player
    only a block or two ahead of the speaker, an empty burst starves the reference FIFO and
    shifts the alignment the canceller has learned. Measured: 13 dB of echo removed with soxr at
    a 40ms lead versus 45 dB with this (tests/test_audio_io.py). This one returns exactly the
    right number of samples for every block, with a fixed sub-millisecond delay.
    """

    TAPS = 33

    def __init__(self, in_rate: int):
        self.in_rate = int(in_rate)
        self._pos = 0.0
        self._last = 0.0
        self._fir = None
        self._hist = np.zeros(0, dtype=np.float64)
        if self.in_rate > AEC_RATE:
            cutoff = 7600.0 / self.in_rate
            n = np.arange(self.TAPS) - (self.TAPS - 1) / 2
            h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(self.TAPS)
            self._fir = h / h.sum()
            self._hist = np.zeros(self.TAPS - 1, dtype=np.float64)

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        x = np.asarray(pcm, dtype=np.float64).reshape(-1)
        if self.in_rate == AEC_RATE or x.size == 0:
            return x.astype(np.float32)
        if self._fir is not None:
            joined = np.concatenate((self._hist, x))
            x = np.convolve(joined, self._fir, mode="valid")
            self._hist = joined[-(self.TAPS - 1):]
        step = self.in_rate / AEC_RATE
        padded = np.concatenate(([self._last], x))
        positions = np.arange(self._pos, x.size, step)
        out = np.interp(positions, np.arange(-1, x.size), padded)
        self._pos = positions[-1] + step - x.size if positions.size else self._pos - x.size
        self._last = float(x[-1])
        return out.astype(np.float32)


# --- echo cancellation -------------------------------------------------------------------------

class EchoCanceller:
    """WebRTC AEC3 between what the speaker plays and what the microphone hears.

    Thread model: the speaker's thread calls push_far() (through far_tap()) with every block it
    is about to play; the microphone's thread calls process() on every frame it captured. The
    reference is held in a FIFO and consumed one 10ms frame per 10ms microphone frame, so within
    one stretch of speech the two streams stay aligned by SAMPLE COUNT rather than by wall clock
    - a stalled thread shifts both together. Everything that touches the WebRTC module happens
    on the microphone thread.

    The speaker side should be CONTINUOUS - silence included - which is what stage06's
    AudioPlayer does whenever a canceller is attached (2026-10-27): the alignment then never
    changes, and that matters more than anything else here, because AEC3's learned filter is
    only valid for one alignment. A shift of even a few milliseconds between sentences made it
    leak the start of every sentence (measured in the end-to-end test).

    Where the stream STARTS is set by the clock: the first block after the FIFO was empty (at
    startup, or after the speaker stalled) carries the time it was handed over, and is held
    back until the microphone reaches audio captured at that time.

    Clock drift: a USB microphone and a separate speaker each have their own crystal, a few
    tens of ppm apart, so over an hour the FIFO would slowly grow or shrink by a fraction of a
    second (and a speaker underflow shifts it at once). _track_drift() watches its low-water
    mark and removes or adds 10ms frames of silence when it has moved, which keeps the delay
    the canceller sees where it was.
    """

    def __init__(self, noise_suppression: bool = False):
        from livekit import rtc  # optional dependency - see create()

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True, noise_suppression=noise_suppression,
            high_pass_filter=True, auto_gain_control=False,
        )
        self._far: deque[tuple[np.ndarray, float | None]] = deque()   # (frame, not before)
        self._far_pending = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()
        self._zeros = np.zeros(AEC_FRAME, dtype=np.int16).tobytes()
        self.frames_processed = 0
        self.far_frames_used = 0
        self.far_frames_dropped = 0
        self.drift_corrections = 0
        self._low: int | None = None          # lowest FIFO length in the current window
        self._low_target: int | None = None
        self._windows = 0
        self._window_frames = 0

    @classmethod
    def create(cls, wanted: str = "auto", noise_suppression: bool = False) -> "EchoCanceller | None":
        """The canceller, or None (with one line saying why) when it is off or unavailable."""
        if wanted not in ("auto", "true"):
            return None
        try:
            canceller = cls(noise_suppression=noise_suppression)
        except Exception as exc:  # noqa: BLE001 - ImportError, or a native library problem
            print(f"[aec] echo cancellation unavailable ({exc!r}) - pip install livekit==1.0.0. "
                  f"The robot will still work; barge-in falls back to the loudness check.")
            return None
        print("[aec] software echo cancellation ON (WebRTC AEC3) - the robot will not hear "
              "its own voice.")
        return canceller

    # -- the speaker side --
    def far_tap(self, sample_rate: int):
        """A function the player calls with each int16 block just before writing it."""
        resample = _ToAecRate(sample_rate)

        def _tap(pcm: np.ndarray) -> None:
            self.push_far(resample(pcm))

        return _tap

    def push_far(self, pcm16k, at: float | None = None) -> None:
        """Queues reference audio. `at` is when it is handed to the speaker (now, by default) -
        see the class docstring for what it is used for."""
        x = np.clip(np.asarray(pcm16k, dtype=np.float32), -32768, 32767).astype(np.int16)
        at = time.monotonic() if at is None else at
        with self._lock:
            starts_stretch = not self._far
            if starts_stretch:
                self._far_pending = np.zeros(0, dtype=np.int16)   # leftovers of the last one
            x = np.concatenate((self._far_pending, x)) if self._far_pending.size else x
            whole = x.size // AEC_FRAME * AEC_FRAME
            for start in range(0, whole, AEC_FRAME):
                self._far.append((x[start:start + AEC_FRAME],
                                  at if starts_stretch and start == 0 else None))
            self._far_pending = x[whole:].copy()
            limit = int(FAR_FIFO_MAX_S * AEC_RATE / AEC_FRAME)
            while len(self._far) > limit:
                self._far.popleft()
                self.far_frames_dropped += 1

    def far_backlog_ms(self) -> float:
        with self._lock:
            return len(self._far) * AEC_FRAME * 1000.0 / AEC_RATE

    def _pop_far(self, captured_at: float | None = None) -> bytes:
        """The reference frame for the microphone frame captured at `captured_at` (its start).
        Silence while the next stretch of speech had not been handed to the speaker yet."""
        with self._lock:
            if self._far:
                frame, not_before = self._far[0]
                if (not_before is not None and captured_at is not None
                        and captured_at + AEC_FRAME / AEC_RATE / 2 < not_before):
                    return self._zeros
                self._far.popleft()
                self.far_frames_used += 1
                return frame.tobytes()
        return self._zeros

    DRIFT_WINDOW = 100     # 10ms frames: the low-water mark is taken over one second
    DRIFT_FRAMES = 4       # move it back once it is 40ms away from where it settled

    def _track_drift(self) -> None:
        with self._lock:
            n = len(self._far)
            if n == 0:   # not continuous (startup, a stall): start over
                self._low = self._low_target = None
                self._windows = self._window_frames = 0
                return
            self._low = n if self._low is None else min(self._low, n)
            self._window_frames += 1
            if self._window_frames < self.DRIFT_WINDOW:
                return
            low, self._low, self._window_frames = self._low, None, 0
            self._windows += 1
            if self._windows < 3:          # let the stream settle first
                self._low_target = low
                return
            off = low - self._low_target
            if abs(off) < self.DRIFT_FRAMES:
                return
            # Moved back in one go: drift moves it one frame at a time, but a speaker stall
            # (an output underflow: the card played silence the FIFO never saw) moves it by the
            # length of the stall. Only ever in a pause, never mid-word.
            fixed = 0
            if off > 0:
                while fixed < off and self._far and np.abs(self._far[0][0]).max() <= 64:
                    self._far.popleft()
                    fixed += 1
            elif np.abs(self._far[0][0]).max() <= 64:
                for _ in range(-off):
                    self._far.appendleft((np.zeros(AEC_FRAME, dtype=np.int16), None))
                fixed = -off
            if fixed:
                self.drift_corrections += 1

    # -- the microphone side --
    def process(self, frame: bytes, captured_end: float | None = None) -> bytes:
        """Cleans one microphone frame (any whole number of 10ms frames, 16kHz int16 mono).
        captured_end: when its last sample was captured (time.monotonic()), if known."""
        rtc = self._rtc
        out = bytearray()
        step = AEC_FRAME * 2
        count = len(frame) // step
        for i, start in enumerate(range(0, len(frame) - step + 1, step)):
            at = (None if captured_end is None
                  else captured_end - (count - i) * AEC_FRAME / AEC_RATE)
            self._apm.process_reverse_stream(
                rtc.AudioFrame(self._pop_far(at), AEC_RATE, 1, AEC_FRAME))
            mic = rtc.AudioFrame(frame[start:start + step], AEC_RATE, 1, AEC_FRAME)
            self._apm.process_stream(mic)
            out += bytes(mic.data)
            self.frames_processed += 1
            self._track_drift()
        return bytes(out)


# --- microphone capture ------------------------------------------------------------------------

class MicCapture:
    """Opens the microphone and puts 30ms int16 frames on audio_q - cleaned by the echo
    canceller first when there is one. Used as a context manager exactly like the
    sd.RawInputStream it replaces (`with self.mic_stream:`)."""

    def __init__(self, audio_q: "queue.Queue[bytes]", *, sample_rate: int = 16000,
                 frame_samples: int = 480, block_frames: int = 3, device: int | None = None,
                 canceller: EchoCanceller | None = None, use_process: bool = False):
        self.audio_q = audio_q
        self.sample_rate = sample_rate
        self.frame_bytes = frame_samples * 2
        self.blocksize = frame_samples * block_frames
        self.device = device
        self.canceller = canceller
        self.use_process = use_process
        self.mode = "not started"
        self._stream = None
        self._proc: subprocess.Popen | None = None
        self._raw_q: "queue.Queue[tuple[bytes, float] | None]" = queue.Queue()
        self._clock: float | None = None     # when the last delivered block ended - _stamp()
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._leftover = b""

    # -- delivery: the same for every capture mode --
    def _stamp(self, arrived: float, samples: int) -> float:
        """When a block's last sample was captured, from when it arrived.

        Arrival times jitter (the callback is late whenever another thread holds the GIL), but
        the device delivers audio at a steady rate, so the earliest arrivals are the truth:
        the estimate follows the block period, snaps back to any earlier arrival, and creeps
        forward slowly (0.5ms a block) so that a steady extra delay is absorbed. A jump of more
        than a quarter second (audio was dropped) starts over."""
        period = samples / float(self.sample_rate)
        if self._clock is None or arrived - (self._clock + period) > 0.25:
            self._clock = arrived
        else:
            self._clock = min(arrived, self._clock + period + 0.0005)
        return self._clock

    def _deliver(self, raw: bytes, arrived: float | None = None) -> None:
        data = self._leftover + raw
        whole = len(data) // self.frame_bytes * self.frame_bytes
        self._leftover = data[whole:]
        end = None
        if arrived is not None:
            # the end of the last whole frame, i.e. before any leftover samples
            end = self._stamp(arrived, len(raw) // 2) - len(self._leftover) / 2 / self.sample_rate
        for start in range(0, whole, self.frame_bytes):
            frame = data[start:start + self.frame_bytes]
            if self.canceller is not None:
                captured_end = (None if end is None else
                                end - (whole - start - self.frame_bytes) / 2 / self.sample_rate)
                try:
                    frame = self.canceller.process(frame, captured_end)
                except Exception as exc:  # noqa: BLE001 - never lose the microphone over AEC
                    print(f"[aec] processing failed ({exc!r}) - echo cancellation switched off.")
                    self.canceller = None
            self.audio_q.put(frame)

    def _worker(self) -> None:
        while True:
            item = self._raw_q.get()
            if item is None:
                return
            self._deliver(*item)

    # -- in-process capture --
    def _start_in_process(self) -> None:
        import sounddevice as sd

        def _on_audio(indata, frames, time_info, status):
            if status:
                detail = (" - microphone audio was DROPPED; expect a poor or empty transcript"
                          if status.input_overflow else "")
                print(f"[audio] {status}{detail}", file=sys.stderr)
            if self.canceller is not None:
                # AEC runs on the worker, not in the callback; the arrival time goes with it
                self._raw_q.put((bytes(indata), time.monotonic()))
            else:
                self._deliver(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate, blocksize=self.blocksize, dtype="int16", channels=1,
            device=self.device, callback=_on_audio,
        )
        self._stream.start()
        self.mode = "in-process"

    # -- separate capture process --
    def _capture_command(self) -> list[str]:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--capture",
               "--rate", str(self.sample_rate), "--block", str(self.blocksize)]
        if self.device is not None:
            cmd += ["--device", str(self.device)]
        return cmd

    def _start_process(self) -> bool:
        cmd = self._capture_command()
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                          bufsize=0)
        except Exception as exc:  # noqa: BLE001
            print(f"[audio] could not start the capture process ({exc!r}).")
            return False
        first = threading.Event()

        def _reader() -> None:
            stream = self._proc.stdout
            while self._running.is_set():
                chunk = stream.read(self.blocksize * 2)
                if not chunk:
                    break
                first.set()
                self._raw_q.put((chunk, time.monotonic()))
            if self._running.is_set():
                print("[audio] the capture process stopped - restarting the microphone "
                      "in-process.", file=sys.stderr)
                try:
                    self._start_in_process()
                except Exception as exc:  # noqa: BLE001
                    print(f"[audio] could not reopen the microphone ({exc!r}).", file=sys.stderr)

        thread = threading.Thread(target=_reader, name="mic-pipe-reader", daemon=True)
        thread.start()
        self._threads.append(thread)
        if not first.wait(5.0):
            print("[audio] the capture process sent no audio within 5s - using in-process "
                  "capture instead.")
            self._stop_process()
            return False
        self.mode = "separate process"
        return True

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- lifecycle --
    def start(self) -> None:
        self._running.set()
        worker = threading.Thread(target=self._worker, name="mic-aec-worker", daemon=True)
        worker.start()
        self._threads.append(worker)
        if self.use_process and self._start_process():
            pass
        else:
            self._start_in_process()
        print(f"[audio] microphone: {device_name(self.device, 'input')} "
              f"({self.mode}{', echo-cancelled' if self.canceller else ''})")

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        self._stop_process()
        self._raw_q.put(None)

    def __enter__(self) -> "MicCapture":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _capture_main(rate: int, block: int, device: int | None) -> None:
    """The body of the separate capture process: read the mic, write raw int16 to stdout.

    The PortAudio callback only appends to a queue; a separate thread does the (possibly
    blocking) pipe write, so a busy parent can never make this process drop audio - it just
    buffers until the parent reads again.
    """
    import sounddevice as sd

    out = sys.stdout.buffer
    pending: "queue.Queue[bytes]" = queue.Queue()

    def _on_audio(indata, frames, time_info, status):
        if status:
            print(f"[audio capture] {status}", file=sys.stderr)
        pending.put(bytes(indata))

    with sd.RawInputStream(samplerate=rate, blocksize=block, dtype="int16", channels=1,
                           device=device, callback=_on_audio):
        while True:
            data = pending.get()
            try:
                out.write(data)
                out.flush()
            except (BrokenPipeError, OSError):
                return


def _demo() -> None:
    import sounddevice as sd

    print(sd.query_devices())
    q: "queue.Queue[bytes]" = queue.Queue()
    with MicCapture(q, canceller=EchoCanceller.create("auto")):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            frame = np.frombuffer(q.get(), dtype=np.int16).astype(np.float32)
            level = float(np.sqrt(np.mean(frame ** 2))) / 32768
            print("#" * int(level * 400), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--block", type=int, default=1440)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()
    if args.capture:
        try:
            _capture_main(args.rate, args.block, args.device)
        except KeyboardInterrupt:
            pass
    else:
        _demo()
