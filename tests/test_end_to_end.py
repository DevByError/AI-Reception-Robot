"""The whole robot, end to end, in a simulated room (see sim_world.py).

A visitor walks up, is greeted, asks a question the robot has to "think" about, interrupts the
answer with a different question, a bystander off camera says something that is NOT for the
robot, the visitor asks the robot its name using the wake word, says goodbye and declines to be
remembered. Then, with nobody on camera, somebody calls the robot by its wake word.

Real: face detection + tracking, the engagement gate, face recognition, the conversation loop,
the addressee check, the streaming speech recognizer (Zipformer), the turn model, Kokoro, echo
cancellation, barge-in, fillers, SQLite memory, openWakeWord. Fake: camera, microphone, speaker,
lip movement (no mediapipe on a photo), and the cloud language model (a slow stand-in, so the
"answer is late - say 'mm, okay'" path is exercised).

Slow (about two minutes) - skip with:  python -m pytest tests -m "not e2e"
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

sd = pytest.importorskip("sounddevice")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sherpa_onnx")
pytest.importorskip("livekit")

from sim_world import FACE_IMAGE, World, fake_mouth_detector, to_16k  # noqa: E402

if not FACE_IMAGE.exists():
    pytest.skip(
        "needs a photo of a face at tests/data/face.jpg (see tests/data/README.md)",
        allow_module_level=True,
    )


class SlowLLM:
    """Stands in for Groq/Gemini: streams, but takes FIRST_TOKEN_S before the first word."""

    FIRST_TOKEN_S = 1.4
    supports_streaming = True
    supports_general_streaming = True

    GROUNDED = (
        "The Faculty of Engineering offers diplomas, bachelor's degrees and "
        "postgraduate programmes. These include civil, mechanical and electrical "
        "engineering. Most bachelor's programmes take four years. You can apply online "
        "through the university website, and the admissions office can help you choose."
    )

    def __init__(self):
        self.calls = []

    def stream_grounded_answer(self, question, chunks):
        self.calls.append(("grounded", question))
        time.sleep(self.FIRST_TOKEN_S)
        for word in self.GROUNDED.split(" "):
            yield word + " "
            time.sleep(0.01)

    def generate_grounded_answer(self, question, chunks):
        self.calls.append(("grounded-blocking", question))
        return chunks[0].text if chunks else "I'm not sure."

    def stream_general_answer(self, question):
        self.calls.append(("general", question))
        time.sleep(self.FIRST_TOKEN_S)
        yield "UNIVERSITY_RELATED: no\nANSWER: "
        yield "I'm happy to help with that."

    def classify_and_answer_general(self, question):
        self.calls.append(("general-blocking", question))
        return False, "I'm happy to help with that."


@pytest.fixture(scope="module")
def visitor_voice():
    """The visitor's sentences, spoken by a different Kokoro voice, at 16kHz."""
    kv = pytest.importorskip("kokoro_voice")
    if not kv.KOKORO_MODEL.exists():
        pytest.skip("Kokoro model not downloaded")
    voice = kv.KokoroVoice(voice="am_adam")
    lines = {
        "programmes": "What programmes does the faculty of engineering offer?",
        "library": "Wait, where is the library?",
        "football": "Did you watch the football match last night?",
        "name": "Nova, what is your name?",
        "bye": "No thanks, that's all.",
        "decline": "No thanks.",
        "wake": "Hey Jarvis!",
        "wake_question": "Where is the library?",
    }
    out = {}
    for key, text in lines.items():
        audio = voice.create(text)
        pcm = to_16k(audio * 32767.0, kv.SAMPLE_RATE) / 32768.0
        out[key] = (pcm * 0.6).astype(np.float32)  # a normal speaking level at a metre
    return out


@pytest.fixture
def robot_env(tmp_path, monkeypatch):
    world = World()
    monkeypatch.setenv("LUC_DB_BACKEND", "sqlite")
    monkeypatch.setenv("LUC_DB_SQLITE_PATH", str(tmp_path / "robot.db"))
    monkeypatch.setattr(sd, "RawInputStream", world.input_stream())
    monkeypatch.setattr(sd, "RawOutputStream", world.output_stream())
    monkeypatch.setattr(
        sd, "query_devices", lambda *a, **k: {"name": "simulated", "default_samplerate": 48000.0}
    )
    monkeypatch.setattr(cv2, "VideoCapture", world.camera())
    monkeypatch.setattr(cv2, "waitKey", lambda *_a: -1)

    import identity

    cache = tmp_path / "identity_cache.json"
    shutil.copy(identity.CACHE_FILE, cache)
    monkeypatch.setattr(identity, "CACHE_FILE", cache)

    import main_live as ml

    llm = SlowLLM()
    monkeypatch.setattr(ml, "MouthActivityDetector", fake_mouth_detector(world))
    monkeypatch.setattr(ml, "_build_answer_generator", lambda: llm)
    monkeypatch.setattr(ml.RobotOrchestrator, "_refresh_knowledge_base", lambda self: None)
    monkeypatch.setattr(ml, "CONSENT_LISTEN_TIMEOUT_S", 5.0)
    monkeypatch.setattr(ml, "LISTEN_TIMEOUT_S", 6.0)
    for name, value in {
        "STT_BACKEND": "sherpa",
        "TTS_BACKEND": "kokoro",
        "HEADLESS": True,
        "FACE_DISPLAY": False,
        "CAMERA_PREVIEW": False,
        "TTS_RATE_VARIATION": 0.0,
        "LOG_TRANSCRIPTS": True,
    }.items():
        monkeypatch.setattr(ml.RT, name, value)
    return world, ml, llm


def _start(ml, world):
    robot = ml.RobotOrchestrator()
    seen_states: list[str] = []

    class SpyFace:
        def show(self, state, level=0.0, look_x=0.0):
            if not seen_states or seen_states[-1] != state:
                seen_states.append(state)

        def close(self):
            pass

    robot.face_display = SpyFace()
    _maybe_record_audio(robot, world)
    thread = threading.Thread(target=robot.run, name="robot", daemon=True)
    thread.start()
    # The barge-in calibration sentence comes first; wait for it to finish.
    world.wait_robot_spoke_then_silent(0.0, silence_s=1.0, timeout_s=90)
    return robot, thread, seen_states


def _maybe_record_audio(robot, world):
    """E2E_DUMP=<file.pkl>: keeps every microphone frame (before and after echo cancellation)
    and every block handed to the speaker, with times, for looking at a failure afterwards."""
    path = os.environ.get("E2E_DUMP")
    if not path or robot.echo is None:
        return
    rec = {"near": [], "far": []}
    process, push_far = robot.echo.process, robot.echo.push_far

    def _process(frame, captured_end=None):
        out = process(frame, captured_end)
        rec["near"].append((captured_end or time.monotonic(), frame, out))
        return out

    def _push_far(pcm, at=None):
        rec["far"].append((time.monotonic() if at is None else at, np.asarray(pcm).copy()))
        return push_far(pcm, at)

    robot.echo.process, robot.echo.push_far = _process, _push_far
    world.recording = (path, rec)


def _stop(world, thread):
    world.stop.set()
    thread.join(timeout=20)
    if getattr(world, "recording", None):
        import pickle

        path, rec = world.recording
        rec["played"] = [(a, b) for a, b, _r, _p in world.played]
        rec["marks"] = world.marks
        with open(path, "wb") as fh:
            pickle.dump(rec, fh)
    # A timeline of what the robot played, for reading a failure.
    t0 = world.played[0][0] if world.played else 0.0
    runs = []
    for start, end, _rate, _pcm in world.played:
        if runs and start - runs[-1][1] < 0.15:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    print(f"speaker underflows (late audio callbacks): {world.underflows}")
    print(
        "robot audio (s from first sound):",
        ", ".join(f"{a - t0:.2f}-{b - t0:.2f}" for a, b in runs),
    )
    for label, t in world.marks:
        print(f"  mark {label}: {t - t0:.2f}")
    assert not thread.is_alive(), "the robot did not shut down"


def _transcript(robot):
    db = robot.db
    with db._cursor(dictionary=True) as cur:
        cur.execute("SELECT conversation_id FROM conversations ORDER BY conversation_id")
        ids = [row["conversation_id"] for row in cur.fetchall()]
    return [db.conversation_transcript(cid) for cid in ids]


def _words(text):
    return set(re.findall(r"[a-z']+", (text or "").lower()))


def test_a_whole_visit(robot_env, visitor_voice):
    world, ml, llm = robot_env
    robot, thread, states = _start(ml, world)
    try:
        assert robot.echo is not None, "echo cancellation should be on"
        assert robot._barge_in_threshold is not None, "barge-in should be on with AEC"

        # --- a visitor walks up and is greeted --------------------------------------------
        arrived = time.monotonic()
        world.face_present.set()
        greet_start = world.wait_robot_starts(arrived, timeout_s=20)
        assert (
            greet_start - arrived >= ml.RT.ENGAGE_DWELL_S
        ), "greeted before the engagement dwell time"
        world.wait_robot_spoke_then_silent(greet_start, silence_s=0.8)

        # --- a question the language model has to answer (slow -> a filler) ---------------
        asked = world.say(visitor_voice["programmes"])
        answer_start = world.wait_robot_starts(asked, timeout_s=20)
        first_sound_ms = (answer_start - asked) * 1000
        print(f"first sound {first_sound_ms:.0f}ms after the question ended")
        assert first_sound_ms < 2500

        # --- ... and interrupts the answer with a different question ----------------------
        # Wait until the real answer (not the short filler) has been playing for a while, then
        # talk over it.
        world.wait_talking_for(asked, min_s=1.5)
        interrupted_at = world.say(visitor_voice["library"])
        world.wait_robot_spoke_then_silent(interrupted_at, silence_s=1.0)

        # --- a bystander off camera says something that is not for the robot -------------
        world.say(visitor_voice["football"], lips=False)
        time.sleep(1.5)
        # --- the visitor asks its name, using the wake word ------------------------------
        asked = world.say(visitor_voice["name"])
        world.wait_robot_spoke_then_silent(asked, silence_s=0.8)

        # --- goodbye, and no thanks to being remembered -----------------------------------
        asked = world.say(visitor_voice["bye"])
        consent_end = world.wait_robot_spoke_then_silent(asked, silence_s=1.2)
        world.say(visitor_voice["decline"])
        world.wait_robot_spoke_then_silent(consent_end + 0.5, silence_s=1.0)
        world.face_present.clear()
        time.sleep(3.0)
    finally:
        _stop(world, thread)

    conversations = _transcript(robot)
    assert conversations, "no conversation was recorded"
    turns = conversations[-1]
    for t in turns:
        print(f"  {t['speaker']:>7}: {t['text']}")
    visitor = [t["text"] for t in turns if t["speaker"] == "visitor"]
    robot_lines = [t["text"] for t in turns if t["speaker"] == "robot"]

    # The greeting is the first line.
    assert turns[0]["speaker"] == "robot"
    # Every visitor sentence was heard, except the bystander's.
    assert any(
        {"engineering"} <= _words(v) or {"programmes", "programs"} & _words(v) for v in visitor
    ), visitor
    assert any("library" in _words(v) for v in visitor), visitor
    assert not any(
        {"football", "match"} & _words(v) for v in visitor
    ), f"the bystander's sentence was treated as a question: {visitor}"
    assert any("name" in _words(v) for v in visitor), visitor
    # The robot never transcribed its own voice as the visitor's.
    for v in visitor:
        assert not (
            {"diplomas", "postgraduate", "faculty's"} & _words(v)
        ), f"the robot heard itself: {v!r}"
    # The interrupted answer was cut short, and the library question answered.
    assert any("first floor" in r.lower() for r in robot_lines), robot_lines
    grounded = [r for r in robot_lines if "engineering" in r.lower()]
    assert (
        grounded and "admissions office" not in grounded[0].lower()
    ), "the answer was not cut off by the interruption"
    # Its name, and the wake word stripped off the question.
    assert any("nova" in r.lower() for r in robot_lines[1:]), robot_lines
    # Declined: nothing was registered.
    assert len(robot.face_store) == 0
    assert any("won't remember" in r.lower() for r in robot_lines), robot_lines
    # The face showed the conversation.
    assert "THINKING" in states and "AI_SPEAKING" in states and "USER_SPEAKING" in states
    # Echo cancellation actually ran on the microphone audio.
    assert robot.echo.frames_processed > 1000 and robot.echo.far_frames_used > 100


def test_woken_by_voice_from_off_camera(robot_env, visitor_voice, monkeypatch):
    world, ml, llm = robot_env
    pytest.importorskip("openwakeword")
    import model_paths

    if not (model_paths.OWW_DIR / "hey_jarvis_v0.1.onnx").exists():
        pytest.skip("hey_jarvis wake word model not downloaded")
    monkeypatch.setattr(ml.RT, "WAKE_WORD_ENGINE", "openwakeword")
    monkeypatch.setattr(ml.RT, "OWW_MODEL", "hey_jarvis")
    robot, thread, states = _start(ml, world)
    try:
        assert robot.wake_detector is not None
        time.sleep(1.0)
        # Nobody on camera. "Hey Jarvis!" ... "Where is the library?"
        world.say(visitor_voice["wake"])
        time.sleep(0.3)
        asked = world.say(visitor_voice["wake_question"])
        world.wait_robot_spoke_then_silent(asked, silence_s=1.0, timeout_s=20)
        time.sleep(0.5)
        world.say(visitor_voice["bye"])
        time.sleep(4.0)
    finally:
        _stop(world, thread)

    turns = _transcript(robot)[-1]
    for t in turns:
        print(f"  {t['speaker']:>7}: {t['text']}")
    robot_lines = [t["text"] for t in turns if t["speaker"] == "robot"]
    assert any("first floor" in r.lower() for r in robot_lines), robot_lines
    # No greeting: the visit was started by the visitor, so the first line is the answer.
    assert "first floor" in robot_lines[0].lower(), robot_lines
