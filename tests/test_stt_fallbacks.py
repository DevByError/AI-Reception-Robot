"""Speech recognition falls back instead of failing: a GPU that cannot actually run Whisper uses
the CPU, and a missing streaming model uses Whisper. No models are loaded - the libraries are
replaced by stand-ins that behave like the failing ones."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

stt = pytest.importorskip("stt_listener_live")


class _Model:
    made: list[tuple[str, str]] = []
    fail_cuda = True

    def __init__(self, size, device, compute_type):
        _Model.made.append((device, compute_type))
        self.device = device

    def transcribe(self, *_a, **_k):
        if self.device == "cuda" and _Model.fail_cuda:
            raise RuntimeError("Library cublas64_12.dll is not found")
        return iter(()), None


@pytest.fixture
def fake_whisper(monkeypatch):
    _Model.made = []
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_Model))
    monkeypatch.setitem(
        sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    )
    monkeypatch.setattr(stt, "_add_nvidia_dll_dirs", lambda: None)
    return _Model


def test_gpu_that_fails_on_first_use_falls_back_to_cpu(fake_whisper, capsys):
    fake_whisper.fail_cuda = True
    model, device = stt._load_whisper("small.en", "auto")
    assert device == "cpu"
    assert fake_whisper.made == [("cuda", "float16"), ("cpu", "int8")]
    assert "requirements-gpu.txt" in capsys.readouterr().out


def test_working_gpu_is_used(fake_whisper):
    fake_whisper.fail_cuda = False
    _model, device = stt._load_whisper("small.en", "auto")
    assert device == "cuda"


def test_cpu_setting_never_touches_the_gpu(fake_whisper):
    _model, device = stt._load_whisper("small.en", "cpu")
    assert device == "cpu" and fake_whisper.made == [("cpu", "int8")]


def test_gpu_failing_mid_conversation_moves_whisper_to_the_cpu(fake_whisper, capsys):
    """Out of GPU memory (or a driver reset) during a visit: the question is still transcribed,
    on the CPU, and the GPU is not tried again."""
    engine = stt.WhisperSTTEngine.__new__(stt.WhisperSTTEngine)
    fake_whisper.fail_cuda = True
    engine.model, engine.device, engine.model_size = (
        _Model("small.en", "cuda", "float16"),
        "cuda",
        "small.en",
    )
    engine.initial_prompt = engine.hotwords = None
    engine.vocabulary, engine.denoise_enabled = [], False
    speech = (np.sin(np.arange(16000) / 5.0) * 8000).astype(np.int16).tobytes()

    chunks = engine.transcribe_stream([speech], "en-US")

    assert chunks[0].text == ""  # the stand-in model hears no words
    assert engine.device == "cpu"
    assert fake_whisper.made[-1] == ("cpu", "int8")
    assert "moving it to the CPU" in capsys.readouterr().out
    engine.transcribe_stream([speech], "en-US")  # stays on the CPU, no new model
    assert fake_whisper.made.count(("cpu", "int8")) == 1


def test_mic_level_line_flags_a_quiet_voice_and_a_loud_room(capsys):
    stt._report_mic_levels([-50.0, -48.0, -52.0], [-70.0] * 10)
    quiet = capsys.readouterr().out
    assert "voice -50 dBFS, room -70 dBFS (20 dB apart)" in quiet and "very quiet" in quiet
    stt._report_mic_levels([-30.0] * 5, [-36.0] * 10)
    assert "nearly as loud as the voice" in capsys.readouterr().out
    stt._report_mic_levels([-28.0] * 5, [-65.0] * 10)
    fine = capsys.readouterr().out
    assert "37 dB apart" in fine and " - " not in fine


def test_missing_streaming_model_uses_whisper(monkeypatch, capsys):
    broken = types.ModuleType("stt_sherpa")

    class _Broken:
        def __init__(self, *a, **k):
            raise FileNotFoundError("models/sherpa-onnx-streaming-zipformer... not found")

    broken.SherpaSTTEngine = _Broken
    monkeypatch.setitem(sys.modules, "stt_sherpa", broken)
    monkeypatch.setattr(stt.RT, "STT_BACKEND", "sherpa")
    monkeypatch.setattr(stt, "WhisperSTTEngine", lambda **k: "whisper-engine")
    assert stt.build_stt_engine() == "whisper-engine"
    assert "using Whisper" in capsys.readouterr().out
