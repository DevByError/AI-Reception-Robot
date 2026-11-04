"""Echo cancellation and microphone plumbing (stage04/audio_io.py), without any audio device.

The echo test plays the role of the room: the speaker side feeds the canceller exactly what the
player would (Piper-rate blocks through far_tap), the "microphone" hears that audio delayed and
reverberated plus a visitor's voice, and the test measures how much of the robot's voice is
left and whether the visitor's voice survived.
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path

import numpy as np
import pytest

audio_io = pytest.importorskip("audio_io")
pytest.importorskip("livekit")

# Synthetic speech clips (generated with the Kokoro voice), so the tests need no recordings of
# real people.
SAMPLES = Path(__file__).resolve().parent / "data" / "speech"


def _load(name: str, rate: int = 16000) -> np.ndarray:
    sf = pytest.importorskip("soundfile")
    x, sr = sf.read(str(SAMPLES / name), dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return _resample(x, sr, rate) * 32767.0


def _resample(x, rate_in, rate_out):
    """A proper (polyphase) resampler for making test data - the room's "truth"."""
    if rate_in == rate_out:
        return np.asarray(x, dtype=np.float64)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(rate_in, rate_out)
    return resample_poly(np.asarray(x, dtype=np.float64), rate_out // g, rate_in // g)


def _room(far16: np.ndarray, delay_ms: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = delay_ms * 16
    rir = np.zeros(d + 3200)
    rir[d] = 0.6
    tail = rir.size - d - 1
    rir[d + 1 :] = rng.standard_normal(tail) * np.exp(-np.arange(tail) / 480.0) * 0.05
    return np.convolve(far16, rir)[: far16.size]


def _run(canceller, far_at_voice_rate, voice_rate, mic16, lead_ms=40):
    """Interleaves speaker blocks and microphone frames the way the real threads do.

    lead_ms: the player hands audio to the sound card (and to the canceller) a little before it
    is heard - at least one device buffer. The room delay passed to _room() is measured from
    that hand-over, so it is always larger than the lead, as it is on real hardware.
    """
    tap = canceller.far_tap(voice_rate)
    q: queue.Queue[bytes] = queue.Queue()
    mic = audio_io.MicCapture(q, canceller=canceller)
    out = []
    lead = round(lead_ms * voice_rate / 1000)
    for i in range(0, mic16.size - 479, 480):
        # exact sample positions, so the two "clocks" do not drift apart (real devices drift by
        # a few ppm; rounding to whole blocks here would drift by ~750ppm)
        j0, j1 = round(i * voice_rate / 16000), round((i + 480) * voice_rate / 16000)
        if i == 0:
            j0 = 0
        tap(far_at_voice_rate[j0 + (lead if i else 0) : j1 + lead])
        mic._deliver(np.clip(mic16[i : i + 480], -32768, 32767).astype(np.int16).tobytes())
        while not q.empty():
            out.append(np.frombuffer(q.get(), dtype=np.int16))
    return np.concatenate(out).astype(np.float64)


@pytest.mark.parametrize("delay_ms,lead_ms", [(60, 0), (60, 40), (250, 40), (400, 100)])
def test_echo_of_the_robot_voice_is_removed(delay_ms, lead_ms):
    voice_rate = 22050
    speech = np.concatenate([_load(f"clean_0{i}.wav", voice_rate) for i in (5, 1, 2, 3, 4)])
    speech16 = _resample(speech, voice_rate, 16000)
    mic = _room(speech16, delay_ms)
    canceller = audio_io.EchoCanceller()
    out = _run(canceller, speech, voice_rate, mic, lead_ms=lead_ms)
    n = min(out.size, mic.size)
    later = slice(n // 3, n)  # after the canceller has converged
    erle_db = 10 * np.log10(np.mean(mic[later] ** 2) / (np.mean(out[later] ** 2) + 1e-9))
    assert erle_db > 30, f"only {erle_db:.1f} dB of echo removed"
    assert canceller.far_frames_used > 0


def test_visitor_voice_survives_after_the_robot_stops():
    far = _load("clean_05.wav")
    near = _load("clean_02.wav")
    n = far.size + near.size + 24000
    far_p = np.zeros(n)
    far_p[: far.size] = far
    near_p = np.zeros(n)
    start = far.size + 4000
    near_p[start : start + near.size] = near
    mic = _room(far_p, 60) + near_p
    out = _run(audio_io.EchoCanceller(), far_p, 16000, mic)
    seg = slice(start, start + near.size)
    level_in = np.sqrt(np.mean(near_p[seg] ** 2))
    level_out = np.sqrt(np.mean(out[seg] ** 2))
    assert 0.6 < level_out / level_in < 1.4, (level_in, level_out)


def test_without_canceller_frames_pass_through_unchanged():
    q: queue.Queue[bytes] = queue.Queue()
    mic = audio_io.MicCapture(q)
    data = (np.arange(480 * 3 + 100) % 300).astype(np.int16).tobytes()
    mic._deliver(data)
    frames = [q.get_nowait() for _ in range(q.qsize())]
    assert len(frames) == 3 and all(len(f) == 960 for f in frames)
    assert b"".join(frames) == data[: 480 * 3 * 2]
    mic._deliver(b"\x00" * (960 - 200))  # the 200 leftover bytes complete the next frame
    assert q.qsize() == 1


def test_far_fifo_is_bounded_when_the_mic_is_not_running():
    c = audio_io.EchoCanceller()
    c.push_far(np.ones(16000 * 5))
    assert c.far_backlog_ms() <= audio_io.FAR_FIFO_MAX_S * 1000 + 10


def test_capture_process_pipe(tmp_path):
    """The separate-process path, with a fake child that writes a ramp instead of a mic."""
    fake = tmp_path / "fake_mic.py"
    fake.write_text(
        "import sys, time\n"
        "import numpy as np\n"
        "for i in range(20):\n"
        "    sys.stdout.buffer.write((np.ones(1440) * i).astype(np.int16).tobytes())\n"
        "    sys.stdout.buffer.flush(); time.sleep(0.01)\n"
        "time.sleep(5)\n"
    )
    q: queue.Queue[bytes] = queue.Queue()
    mic = audio_io.MicCapture(q, use_process=True)
    mic._capture_command = lambda: [sys.executable, str(fake)]
    mic._start_in_process = lambda: None  # no sound card here
    mic._running.set()
    import threading

    threading.Thread(target=mic._worker, daemon=True).start()
    assert mic._start_process()
    frames = [q.get(timeout=3) for _ in range(60)]
    mic.stop()
    values = [int(np.frombuffer(f, dtype=np.int16)[0]) for f in frames]
    assert values == sorted(values) and values[0] == 0 and values[-1] == 19


def _stalling_speaker(seed: int, buffer_s: float = 0.06):
    """The robot's voice as a sound card really plays it: sentences with pauses between them,
    one synthesis stall in the middle of each sentence, and a writer that runs up to buffer_s
    ahead of the speaker. Returns (pushes as (handed_over_at, samples), what was played)."""
    rng = np.random.default_rng(seed)
    clips = [_load(f"clean_0{i}.wav") for i in (5, 1, 2, 3, 4)]
    pushes, t, play_end = [], 0.5, 0.0
    played = []
    for clip in clips:
        for s in range(0, clip.size, 640):
            block = clip[s : s + 640]
            ready = t + (rng.uniform(0.1, 0.3) if s == 640 * 30 else 0.0)
            push = max(ready, play_end - buffer_s)
            play = max(push, play_end)
            pushes.append((push, block))
            played.append((play, block))
            play_end, t = play + block.size / 16000, push
        t = play_end + rng.uniform(0.3, 1.2)
    far = np.zeros(int((play_end + 1.0) * 16000))
    for play, block in played:
        i = int(round(play * 16000))
        far[i : i + block.size] += block
    return pushes, far


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_echo_stays_cancelled_across_pauses_and_stalls(seed):
    """Speaker blocks carry the time they were handed over and the microphone blocks the time
    they arrived (late by up to 60ms, at random), so the canceller re-aligns after every gap.
    With sample counting alone, a stall in the middle of a sentence left 5-25x more echo."""
    rng = np.random.default_rng(seed + 100)
    pushes, far = _stalling_speaker(seed)
    mic = _room(far, 60)
    canceller = audio_io.EchoCanceller()
    q: queue.Queue[bytes] = queue.Queue()
    capture = audio_io.MicCapture(q, canceller=canceller)
    events = [(at, 0, "far", block) for at, block in pushes]
    events += [
        ((s + 1440) / 16000 + rng.uniform(0, 0.06), 1, "mic", s)
        for s in range(0, mic.size - 1439, 1440)
    ]
    out = []
    for at, _order, kind, value in sorted(events, key=lambda e: (e[0], e[1])):
        if kind == "far":
            canceller.push_far(value, at=at)
        else:
            capture._deliver(
                np.clip(mic[value : value + 1440], -32768, 32767).astype(np.int16).tobytes(), at
            )
            while not q.empty():
                out.append(np.frombuffer(q.get(), dtype=np.int16))
    out = np.concatenate(out).astype(np.float64)

    def loud_frames(x):  # what barge-in calibration measures: the 95th percentile frame RMS
        frames = x[: x.size // 480 * 480].reshape(-1, 480)
        return np.percentile(np.sqrt(np.mean(frames**2, axis=1)), 95)

    after = slice(3 * 16000, out.size)
    assert loud_frames(out[after]) < loud_frames(mic[after]) / 20, "echo leaked through"
