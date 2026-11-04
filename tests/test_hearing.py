"""The listener in a busy room (stage04 stt_listener_live.listen_for_one_utterance).

From a real run on 2026-10-27: the visitor spoke at about -12 dBFS, then the room kept
"speaking" for 5-12 seconds (other people further away), so turns only ended on the
no-new-words rule or the 15-second cap, with the background audio glued onto the question.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

stt = pytest.importorskip("stt_listener_live")

FRAME = stt.FRAME_SAMPLES


def _tone(dbfs: float, frames: int, clip: bool = False) -> list[bytes]:
    t = np.arange(FRAME) / stt.SAMPLE_RATE
    amp = 10 ** (dbfs / 20) * np.sqrt(2) * 32768
    wave = np.sin(2 * np.pi * 220 * t) * amp
    if clip:
        wave = wave * 3
    pcm = np.clip(wave, -32768, 32767).astype(np.int16).tobytes()
    return [pcm] * frames


class _AlwaysSpeech:
    """The VAD as it behaved in that room: everything is speech."""

    def is_speech(self, _frame):
        return True


class _Engine:
    last_confidence = None

    def __init__(self):
        self.calls = 0

    def transcribe_stream(self, audio_chunks, locale):
        self.calls += 1
        return [stt.TranscriptChunk(text="What is your name?", is_final=True)]


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(stt, "PARTIAL_PREVIEW_ENABLED", False)  # only the voice gate can end it
    monkeypatch.setattr(stt, "MIN_UTTERANCE_MS", 0)


def _listen(frames):
    """Runs the listener on a thread: if the turn never ends it blocks on an empty microphone
    queue, which must fail the test, not hang it."""
    q: queue.Queue[bytes] = queue.Queue()
    for f in frames:
        q.put(f)
    engine = _Engine()
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(
            stt.listen_for_one_utterance(
                engine,
                stt.STTListener(engine, stt.PatientVAD(), locale="en-US"),
                _AlwaysSpeech(),
                q,
                timeout_s=5.0,
            )
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=20)
    assert result, "the turn never ended - the background voices held it open"
    return result[0], (len(frames) - q.qsize()) * stt.FRAME_MS / 1000


def test_quieter_background_voices_do_not_hold_the_turn_open(quick, capsys):
    voice = _tone(-12, 50)  # 1.5s of the visitor, close to the mic
    room = _tone(-45, 400)  # 12s of a conversation across the room
    text, used_s = _listen(voice + room)
    assert text == "What is your name?"
    assert used_s < 1.5 + 2.0, f"the turn ran {used_s:.1f}s - background voices kept it open"
    out = capsys.readouterr().out
    assert "quieter background voices" in out


def test_the_visitor_dropping_their_voice_a_little_is_still_them(monkeypatch):
    monkeypatch.setattr(stt, "VOICE_GATE_DB", 25.0)
    assert not stt._gated_by_voice_level(-24.0, [-12.0] * 20)  # 12 dB quieter: still them
    assert stt._gated_by_voice_level(-45.0, [-12.0] * 20)  # 33 dB quieter: someone else
    assert not stt._gated_by_voice_level(-45.0, [-12.0] * 5)  # too little heard to judge yet


def test_gate_off_when_set_to_zero(monkeypatch):
    monkeypatch.setattr(stt, "VOICE_GATE_DB", 0.0)
    assert not stt._gated_by_voice_level(-80.0, [-12.0] * 50)


def test_mic_line_reports_clipping(capsys):
    frames = _tone(-6, 20, clip=True)
    clipped = sum(stt._clipped_samples(f) for f in frames)
    stt._report_mic_levels(
        [stt._frame_dbfs(f) for f in frames], [-46.0] * 20, clipped, len(frames) * FRAME, 0
    )
    out = capsys.readouterr().out
    assert "CLIPPING" in out and "Microphone Boost" in out
