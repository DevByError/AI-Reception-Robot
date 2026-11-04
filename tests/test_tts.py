"""Speech output (stage06) without a sound card: a fake output stream records what was written
and takes real time to "play" it, so pipelining, barge-in stops, caching and the echo
reference can all be checked."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

sd = pytest.importorskip("sounddevice")
tts_live = pytest.importorskip("tts_speaker_live")
audio_player = pytest.importorskip("audio_player")


class FakeStream:
    instances: list[FakeStream] = []

    def __init__(self, samplerate, channels, dtype, device=None, blocksize=0, callback=None, **_):
        self.samplerate = samplerate
        self.device = device
        self.written = bytearray()
        self.aborted = False
        self.latency = 0.05
        self.callback = callback
        self.blocksize = blocksize or 640
        self._running = threading.Event()
        FakeStream.instances.append(self)

    def start(self):
        if self.callback is not None:  # the always-running stream: pulls at 4x speed
            self._running.set()

            def _run():
                buf = bytearray(self.blocksize * 2)
                while self._running.is_set():
                    self.callback(memoryview(buf), self.blocksize, None, None)
                    self.written += bytes(buf)
                    time.sleep(self.blocksize / self.samplerate * 0.25)

            threading.Thread(target=_run, daemon=True).start()

    def write(self, data):
        self.written += data
        time.sleep(len(data) / 2 / self.samplerate * 0.25)  # plays at 4x speed

    def stop(self):
        self._running.clear()

    def abort(self):
        self.aborted = True
        self._running.clear()

    def close(self):
        self._running.clear()


class SlowBackend:
    """Takes 0.1s per sentence; 0.4s of a 440Hz tone per sentence."""

    name = "fake"
    sample_rate = 16000
    voice = None

    def __init__(self):
        self.calls = []

    def synthesize(self, text, rate=1.0):
        self.calls.append((time.monotonic(), text))
        time.sleep(0.1)
        t = np.arange(int(0.4 * self.sample_rate)) / self.sample_rate
        yield (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)


@pytest.fixture
def speaker(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setattr(sd, "RawOutputStream", FakeStream)
    return tts_live.LiveTTSSpeaker(backend=SlowBackend(), player=audio_player.AudioPlayer())


def test_speaks_every_sentence_and_reports_it(speaker):
    result = speaker.speak("The library is on level three. It opens at eight. Bring your card.")
    assert result["num_segments"] == 3
    assert "Bring your card." in result["text"]
    written = len(FakeStream.instances[0].written) // 2
    assert written == 3 * int(0.4 * 16000)


def test_synthesis_runs_ahead_of_playback(speaker):
    """The next sentence is synthesized while the current one plays, so there is no gap."""
    start = time.monotonic()
    speaker.speak("One sentence here. Two sentences here. Three sentences here.")
    calls = [t - start for t, _ in speaker.backend.calls]
    # sentence 2 starts synthesizing right after sentence 1 (~0.1s), not after it has played
    assert calls[1] < 0.18, calls


def test_second_time_comes_from_the_cache(speaker, monkeypatch):
    monkeypatch.setattr(tts_live.RT, "TTS_AUDIO_CACHE", True)
    speaker.speak("Welcome to Lincoln University College.")
    n = len(speaker.backend.calls)
    result = speaker.speak("Welcome to Lincoln University College.")
    assert len(speaker.backend.calls) == n  # not synthesized again
    assert result["time_to_first_audio_ms"] < 50


def test_stop_cuts_playback_within_a_block(speaker):
    def _interrupt():
        time.sleep(0.15)
        speaker.stop_requested.set()

    threading.Thread(target=_interrupt).start()
    result = speaker.speak("First sentence is long enough. Second one. Third one. Fourth one.")
    assert result["interrupted"]
    stream = FakeStream.instances[0]
    assert stream.aborted
    assert len(stream.written) // 2 < 4 * int(0.4 * 16000)


def test_stream_errors_reach_the_caller(speaker):
    def _pieces():
        yield "Hello there. "
        raise RuntimeError("LLM connection dropped")

    with pytest.raises(RuntimeError, match="dropped"):
        speaker.speak_stream(_pieces())


def test_every_block_goes_to_the_echo_reference(monkeypatch):
    """With an echo canceller attached, ONE stream runs all the time and every block it plays -
    silence included - goes to the canceller, so the two never drift apart."""
    FakeStream.instances.clear()
    monkeypatch.setattr(sd, "RawOutputStream", FakeStream)
    seen = []
    player = audio_player.AudioPlayer(
        far_tap_factory=lambda rate: (lambda pcm: seen.append(pcm.copy()))
    )
    assert player.continuous
    speaker = tts_live.LiveTTSSpeaker(backend=SlowBackend(), player=player)
    speaker.speak("Short line.")
    speaker.speak("Another line.")
    player.close()
    assert len(FakeStream.instances) == 1  # one stream for both sentences
    tapped = np.concatenate(seen)
    played = np.frombuffer(bytes(FakeStream.instances[0].written), dtype=np.int16)
    n = min(tapped.size, played.size)
    assert n and np.array_equal(tapped[:n], played[:n])  # exactly what was played
    assert np.count_nonzero(tapped) >= 2 * int(0.4 * 16000) * 0.9
    assert max(len(x) for x in seen) <= int(16000 * audio_player.BLOCK_MS / 1000)


def test_continuous_stream_stops_within_a_block(monkeypatch):
    FakeStream.instances.clear()
    monkeypatch.setattr(sd, "RawOutputStream", FakeStream)
    player = audio_player.AudioPlayer(far_tap_factory=lambda rate: (lambda pcm: None))
    speaker = tts_live.LiveTTSSpeaker(backend=SlowBackend(), player=player)

    def _interrupt():
        time.sleep(0.15)
        speaker.stop_requested.set()

    threading.Thread(target=_interrupt).start()
    started = time.monotonic()
    result = speaker.speak("First sentence is long enough. Second one. Third one. Fourth one.")
    assert result["interrupted"]
    assert time.monotonic() - started < 0.5
    player.close()


def test_level_follows_the_audio(speaker):
    levels = []

    def _watch():
        for _ in range(20):
            levels.append(speaker.level_at())
            time.sleep(0.02)

    watcher = threading.Thread(target=_watch)
    watcher.start()
    speaker.speak("A sentence to watch.")
    watcher.join()
    assert max(levels) > 0.1
    assert speaker.level_at(time.monotonic() + 1) == 0.0


def test_render_and_prewarm(speaker, monkeypatch):
    monkeypatch.setattr(tts_live.RT, "TTS_AUDIO_CACHE", True)
    speaker.prewarm(["Hello! Welcome.", "Sorry, I didn't catch that."]).join(timeout=5)
    n = len(speaker.backend.calls)
    assert n == 3  # "Hello!", "Welcome.", "Sorry, ..." - one per sentence
    pcm = speaker.render("Welcome.")
    assert pcm.size and len(speaker.backend.calls) == n


def test_kokoro_voice_if_installed():
    """The real Kokoro model, when it has been downloaded: speech comes out, at 24kHz."""
    kv = pytest.importorskip("kokoro_voice")
    pytest.importorskip("phonemizer")
    if not kv.KOKORO_MODEL.exists():
        pytest.skip("Kokoro model not downloaded")
    voice = kv.KokoroVoice()
    audio = voice.create("Hello, welcome to Lincoln University College.")
    seconds = audio.size / kv.SAMPLE_RATE
    assert 1.5 < seconds < 5.0
    assert np.abs(audio).max() > 0.05


class _TimedKokoro:
    """A Kokoro stand-in whose speed test returns the given real-time factors in turn."""

    name = "kokoro"

    def __init__(self, rtfs):
        self.rtfs = list(rtfs)
        self.calls = 0

    def benchmark(self):
        self.calls += 1
        return self.rtfs.pop(0)


@pytest.mark.parametrize(
    "rtfs, chosen, calls",
    [
        ([0.40], "kokoro", 1),  # fast enough first time: measured once
        ([0.62, 0.55, 0.45], "kokoro", 3),  # a near miss during a busy startup, then it passes
        ([0.62, 0.70, 0.60], "piper", 3),  # a near miss three times: really too slow
        ([1.20], "piper", 1),  # clearly too slow: not measured again
    ],
)
def test_kokoro_speed_test_retries_only_a_near_miss(monkeypatch, rtfs, chosen, calls):
    vb = pytest.importorskip("voice_backends")
    kokoro = _TimedKokoro(rtfs)
    monkeypatch.setattr(vb, "KokoroBackend", lambda **_: kokoro)
    monkeypatch.setattr(vb, "PiperBackend", lambda *_a, **_k: type("P", (), {"name": "piper"})())
    monkeypatch.setattr(vb.time, "sleep", lambda _s: None)
    backend = vb.load_backend("auto", "en_US-amy-medium.onnx")
    assert backend.name == chosen and kokoro.calls == calls
