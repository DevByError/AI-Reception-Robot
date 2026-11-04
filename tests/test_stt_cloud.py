"""STT_CLOUD=groq (stage04 stt_cloud.py): the finished question from Groq's Whisper large-v3,
the local Whisper as the backup. Groq is never contacted here - a stand-in answers instead."""

from __future__ import annotations

import io
import queue
import threading
import time
import wave

import numpy as np
import pytest

stt_cloud = pytest.importorskip("stt_cloud")
stt = pytest.importorskip("stt_listener_live")

FAKE_KEY = "gsk_" + "x" * 40


class _Resp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class _Groq:
    """Stands in for requests.Session: records each call and answers with `reply`."""

    def __init__(self, reply=None, delay_s=0.0, error=None, models=("whisper-large-v3",)):
        self.reply = reply or _Resp(
            200,
            {
                "text": " What time does the library close? ",
                "segments": [{"avg_logprob": -0.1, "no_speech_prob": 0.01}],
            },
        )
        self.delay_s = delay_s
        self.error = error
        self.models = models
        self.posts: list[dict] = []

    def post(self, url, headers=None, files=None, data=None, timeout=None):
        self.posts.append(
            {"url": url, "headers": headers, "files": files, "data": data, "timeout": timeout}
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error:
            raise self.error
        return self.reply

    def get(self, url, headers=None, timeout=None):
        return _Resp(200, {"data": [{"id": m} for m in self.models]})


class _Local:
    """The robot's own recognizer."""

    initial_prompt = "Lincoln University College. My name is Nova."
    hotwords = "Lincoln University College, Nova, Tan"

    def __init__(self, text="what time does the library clothes"):
        self.text = text
        self.calls = 0
        self.last_confidence = "local-confidence"
        self.device = "cuda"

    def transcribe_stream(self, audio_chunks, locale):
        self.calls += 1
        return [stt.TranscriptChunk(text=self.text, is_final=True)]


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _engine(http, local=None, clock=None, timeout_s=3.0):
    cloud = stt_cloud.GroqTranscriber(FAKE_KEY, timeout_s=timeout_s, http=http)
    return stt_cloud.CloudFinalEngine(
        local or _Local(),
        cloud,
        stt.ConfidenceReport,
        stt.TranscriptChunk,
        log_transcripts=True,
        clock=clock or time.monotonic,
    )


AUDIO = [b"\x10\x00" * stt.FRAME_SAMPLES] * 10


def test_sends_one_wav_with_the_model_language_and_prompt():
    http = _Groq()
    text, avg_prob, no_speech = stt_cloud.GroqTranscriber(FAKE_KEY, http=http).transcribe(
        b"".join(AUDIO), "en", "Lincoln University College"
    )
    assert text == "What time does the library close?"
    assert avg_prob == pytest.approx(np.exp(-0.1))
    assert no_speech == pytest.approx(0.01)
    sent = http.posts[0]
    assert sent["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert sent["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert sent["data"]["model"] == "whisper-large-v3"
    assert sent["data"]["language"] == "en"
    assert sent["data"]["response_format"] == "verbose_json"
    assert sent["data"]["prompt"] == "Lincoln University College"
    name, wav, mime = sent["files"]["file"]
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() == len(AUDIO) * stt.FRAME_SAMPLES


def test_errors_never_show_the_key():
    http = _Groq(reply=_Resp(401, text=f'{{"error": "Invalid API Key {FAKE_KEY}"}}'))
    with pytest.raises(stt_cloud.CloudSTTError) as err:
        stt_cloud.GroqTranscriber(FAKE_KEY, http=http).transcribe(b"\0\0" * 100)
    assert err.value.status == 401
    assert FAKE_KEY not in str(err.value) and "gsk_..." in str(err.value)


def test_no_key_is_refused_up_front():
    with pytest.raises(stt_cloud.CloudSTTError, match="GROQ_API_KEY"):
        stt_cloud.GroqTranscriber("")


def test_the_finished_question_comes_from_groq(caplog):
    caplog.set_level("INFO")
    local = _Local()
    engine = _engine(_Groq(), local)
    chunks = engine.transcribe_final(AUDIO, "en-US")
    assert chunks[-1].text == "What time does the library close?"
    assert chunks[-1].is_final
    assert engine.last_source == "Groq"
    assert engine.last_confidence.is_reliable
    assert local.calls == 1  # the backup was ready too
    assert 'the local Whisper heard "what time does the library clothes"' in caplog.text


def test_partials_never_leave_the_robot():
    http = _Groq()
    engine = _engine(http)
    assert engine.transcribe_stream(AUDIO, "en-US")[-1].text.startswith("what time")
    assert http.posts == []
    assert engine.last_confidence == "local-confidence"


def test_wrapper_passes_everything_else_to_the_local_engine():
    engine = _engine(_Groq())
    assert engine.device == "cuda"
    assert engine.hotwords.startswith("Lincoln")


def test_bad_key_falls_back_to_local_and_rests(caplog):
    caplog.set_level("INFO")
    clock = _Clock()
    http = _Groq(reply=_Resp(401, text='{"error": "Invalid API Key"}'))
    engine = _engine(http, clock=clock)
    assert engine.transcribe_final(AUDIO, "en-US")[-1].text == "what time does the library clothes"
    out = caplog.text
    assert "check GROQ_API_KEY" in out and FAKE_KEY not in out
    assert engine.last_source == "local"
    clock.now += stt_cloud.REST_AFTER_BAD_KEY_S - 1
    engine.transcribe_final(AUDIO, "en-US")
    assert len(http.posts) == 1  # left alone while resting
    clock.now += 2
    engine.transcribe_final(AUDIO, "en-US")
    assert len(http.posts) == 2  # and tried again after


@pytest.mark.parametrize(
    "status, rest", [(429, stt_cloud.REST_AFTER_QUOTA_S), (503, stt_cloud.REST_AFTER_FAILURE_S)]
)
def test_quota_and_outages_rest_for_less_time(status, rest):
    clock = _Clock()
    engine = _engine(_Groq(reply=_Resp(status, text="busy")), clock=clock)
    engine.transcribe_final(AUDIO, "en-US")
    assert engine._resting_until == pytest.approx(clock.now + rest)


def test_no_network_falls_back_to_local():
    engine = _engine(_Groq(error=ConnectionError("Failed to resolve api.groq.com")))
    assert engine.transcribe_final(AUDIO, "en-US")[-1].text.startswith("what time")
    assert engine.last_source == "local"


def test_a_slow_groq_is_not_waited_for(caplog):
    caplog.set_level("INFO")
    engine = _engine(_Groq(delay_s=2.0), timeout_s=0.5)
    started = time.monotonic()
    text = engine.transcribe_final(AUDIO, "en-US")[-1].text
    took = time.monotonic() - started
    assert text == "what time does the library clothes"
    assert took < 1.5, f"waited {took:.1f}s for a cloud answer capped at 0.5s"
    assert "no answer within 0.5s" in caplog.text


def test_invented_words_on_a_silent_clip_are_dropped():
    reply = _Resp(
        200, {"text": "Thank you.", "segments": [{"avg_logprob": -0.3, "no_speech_prob": 0.7}]}
    )
    engine = _engine(_Groq(reply=reply), _Local(text=""))
    assert engine.transcribe_final(AUDIO, "en-US")[-1].text == ""
    assert engine.last_source == "local"


def test_clear_speech_the_local_model_missed_is_kept():
    reply = _Resp(
        200, {"text": "Hello Nova.", "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.05}]}
    )
    engine = _engine(_Groq(reply=reply), _Local(text="[??]"))
    assert engine.transcribe_final(AUDIO, "en-US")[-1].text == "Hello Nova."


def test_an_unsure_cloud_answer_loses_to_the_local_one():
    reply = _Resp(
        200,
        {
            "text": "Wear is the engine ear in?",
            "segments": [{"avg_logprob": -1.6, "no_speech_prob": 0.05}],
        },
    )
    engine = _engine(_Groq(reply=reply), _Local(text="where is engineering"))
    assert engine.transcribe_final(AUDIO, "en-US")[-1].text == "where is engineering"


def test_prompt_carries_the_names_and_stays_short():
    local = _Local()
    local.hotwords = ", ".join(f"Programme {i}" for i in range(200))
    prompt = _engine(_Groq(), local)._prompt()
    assert prompt.startswith("Lincoln University College. My name is Nova.")
    assert len(prompt) <= stt_cloud.MAX_PROMPT_CHARS
    assert not prompt.endswith(",")


def test_startup_check_reports_a_wrong_model(caplog):
    caplog.set_level("INFO")
    clock = _Clock()
    engine = _engine(_Groq(models=("whisper-large-v3-turbo",)), clock=clock)
    engine.cloud.model = "whisper-large-v9"
    engine.check_in_background()
    for _ in range(100):
        if engine._resting_until:
            break
        time.sleep(0.01)
    assert "GROQ_STT_MODEL" in caplog.text
    assert engine._resting_until > clock.now


# --- the listener uses it for the finished question -----------------------------------------


class _CloudStandIn:
    last_confidence = None

    def __init__(self):
        self.finals = 0

    def transcribe_stream(self, audio_chunks, locale):
        return [stt.TranscriptChunk(text="where is the library", is_final=True)]

    def transcribe_final(self, audio_chunks, locale):
        self.finals += 1
        return [stt.TranscriptChunk(text="Where is the library?", is_final=True)]


class _LoudIsSpeech:
    def is_speech(self, frame):
        return np.abs(np.frombuffer(frame, dtype=np.int16)).max() > 1000


def _frames(amplitude, n):
    t = np.arange(stt.FRAME_SAMPLES) / stt.SAMPLE_RATE
    pcm = (np.sin(2 * np.pi * 220 * t) * amplitude).astype(np.int16).tobytes()
    return [pcm] * n


def _listen(engine, frames):
    q: queue.Queue[bytes] = queue.Queue()
    for f in frames:
        q.put(f)
    result: list = []
    worker = threading.Thread(
        target=lambda: result.append(
            stt.listen_for_one_utterance(
                engine,
                stt.STTListener(engine, stt.PatientVAD(), locale="en-US"),
                _LoudIsSpeech(),
                q,
                timeout_s=5.0,
            )
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=20)
    assert result, "the turn never ended"
    return result[0]


@pytest.mark.parametrize("adaptive", [True, False])
def test_listener_takes_the_finished_words_from_the_cloud(monkeypatch, adaptive):
    monkeypatch.setattr(stt, "PARTIAL_PREVIEW_ENABLED", False)
    monkeypatch.setattr(stt, "MIN_UTTERANCE_MS", 0)
    monkeypatch.setattr(stt, "ADAPTIVE_ENDPOINTING_ENABLED", adaptive)
    engine = _CloudStandIn()
    text = _listen(engine, _frames(8000, 40) + _frames(0, 200))
    assert text == "Where is the library?"
    assert engine.finals >= 1


def test_listener_takes_the_cloud_words_when_the_partials_stop_changing(monkeypatch):
    monkeypatch.setattr(stt, "PARTIAL_PREVIEW_ENABLED", True)
    monkeypatch.setattr(stt, "PARTIAL_INTERVAL_S", 0.0)
    monkeypatch.setattr(stt, "PARTIAL_MIN_DURATION_S", 0.0)
    monkeypatch.setattr(stt, "MIN_UTTERANCE_MS", 0)
    monkeypatch.setattr(stt, "VOICE_GATE_DB", 0.0)
    engine = _CloudStandIn()
    text = _listen(engine, _frames(8000, 400))  # never goes quiet: only the words can end it
    assert text == "Where is the library?"
    assert engine.finals == 1


# --- build_stt_engine -----------------------------------------------------------------------


class _FakeWhisper(_Local):
    def __init__(self, **kwargs):
        super().__init__()


def test_cloud_off_by_default_keeps_the_local_engine(monkeypatch):
    monkeypatch.setattr(stt, "WhisperSTTEngine", _FakeWhisper)
    monkeypatch.setattr(stt.RT, "STT_BACKEND", "whisper")
    monkeypatch.setattr(stt.RT, "STT_CLOUD", "off")
    assert isinstance(stt.build_stt_engine(), _FakeWhisper)


def test_cloud_without_a_key_says_so_and_stays_local(monkeypatch, capsys):
    monkeypatch.setattr(stt, "WhisperSTTEngine", _FakeWhisper)
    monkeypatch.setattr(stt.RT, "STT_BACKEND", "whisper")
    monkeypatch.setattr(stt.RT, "STT_CLOUD", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert isinstance(stt.build_stt_engine(), _FakeWhisper)
    assert "console.groq.com" in capsys.readouterr().out


def test_cloud_with_a_key_wraps_the_local_engine(monkeypatch):
    monkeypatch.setattr(stt, "WhisperSTTEngine", _FakeWhisper)
    monkeypatch.setattr(stt.RT, "STT_BACKEND", "whisper")
    monkeypatch.setattr(stt.RT, "STT_CLOUD", "groq")
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setattr(stt_cloud.CloudFinalEngine, "check_in_background", lambda self: None)
    engine = stt.build_stt_engine()
    assert isinstance(engine, stt_cloud.CloudFinalEngine)
    assert isinstance(engine.local, _FakeWhisper)
    assert engine.cloud.model == stt.RT.GROQ_STT_MODEL
    assert engine.deadline_s == stt.RT.STT_CLOUD_TIMEOUT_S
