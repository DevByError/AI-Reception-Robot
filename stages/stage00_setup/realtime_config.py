"""
Stage 00 - the one place every real-time conversation setting is read from.

Sits beside project_env.py for the same reason that file does: this is "something every stage
needs before it can run", and it must import nothing heavier than the standard library, because
it is read at module-import time by stage04's audio loop, stage06's synthesizer and stage08's
orchestrator - all three of which are imported before numpy/torch/whisper/piper are touched.

WHY THIS EXISTS
---------------
Every value below already existed in this project, as a module-level constant next to the code
that used it, with a comment explaining the measurement behind it. Those comments are the
valuable part and they have deliberately been left exactly where they were. What did not exist
was any way to change one of them without editing source - so tuning the robot for a new room,
a new microphone or a noisier lobby meant a code edit, and the numbers that matter most were
spread across three files in two stages.

So: the constants keep their names and their homes, and each one now takes its DEFAULT from
here. Nothing is hard-coded twice, no comment moved, and a deployment tunes the robot through
.env (or real environment variables) instead of through git.

PRECEDENCE - highest first, same rule as project_env.load_env_file():
  1. a real environment variable        (what a deployed kiosk should use)
  2. the project-root .env file
  3. the default named below            (the value this project measured and shipped)

NAMING
------
Every setting is read under BOTH `LUC_<NAME>` and a bare `<NAME>`. The prefixed form matches
the rest of this project (LUC_DB_HOST, LUC_OLLAMA_CHAT_MODEL...) and is what .env.example
documents; the bare form exists because these names - END_OF_TURN_TIMEOUT, STREAMING_TTS - are
the ones people actually reach for, and silently ignoring a setting somebody clearly meant is a
worse failure than accepting two spellings. Prefixed wins if both are set.

Usage:

    from realtime_config import RT
    if RT.INTERRUPTION_ENABLED:
        ...

Run `python realtime_config.py` to print every setting, its value, and where that value came
from - which is the fastest way to confirm a .env edit actually took effect.
"""

from __future__ import annotations

import os
from pathlib import Path

# project_env lives in this same folder, so no sys.path juggling is needed when this module is
# imported by a stage that already added stage00_setup to the path.
try:
    from project_env import load_env_file
except ImportError:  # imported by full path / from another stage before the path was extended
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from project_env import load_env_file

load_env_file()

PREFIX = "LUC_"

# Filled in by the readers below so `python realtime_config.py` can report where each value
# actually came from. Diagnostics only - nothing reads this to make a decision.
_SOURCES: dict[str, str] = {}


def _raw(name: str) -> tuple[str | None, str]:
    """The raw string for a setting, plus which spelling supplied it."""
    prefixed = os.environ.get(PREFIX + name)
    if prefixed is not None and prefixed.strip() != "":
        return prefixed, PREFIX + name
    bare = os.environ.get(name)
    if bare is not None and bare.strip() != "":
        return bare, name
    return None, "default"


def get_str(name: str, default: str) -> str:
    value, source = _raw(name)
    _SOURCES[name] = source
    return default if value is None else value.strip()


def get_bool(name: str, default: bool) -> bool:
    """Accepts the spellings people actually write in a .env file.

    Anything unrecognised falls back to the default rather than raising: a typo in one optional
    tuning flag must not stop a kiosk from booting, and the warning below makes it findable.
    """
    value, source = _raw(name)
    _SOURCES[name] = source
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on", "y"):
        return True
    if lowered in ("0", "false", "no", "off", "n"):
        return False
    print(f"[config] {name}={value!r} is not a yes/no value - using {default}.")
    _SOURCES[name] = "default (unparseable)"
    return default


def get_int(name: str, default: int, *, minimum: int | None = None,
            maximum: int | None = None) -> int:
    value, source = _raw(name)
    _SOURCES[name] = source
    if value is None:
        return default
    try:
        parsed = int(float(value.strip()))  # float() first so "550.0" is accepted too
    except ValueError:
        print(f"[config] {name}={value!r} is not a number - using {default}.")
        _SOURCES[name] = "default (unparseable)"
        return default
    return _clamp(name, parsed, default, minimum, maximum)


def get_float(name: str, default: float, *, minimum: float | None = None,
              maximum: float | None = None) -> float:
    value, source = _raw(name)
    _SOURCES[name] = source
    if value is None:
        return default
    try:
        parsed = float(value.strip())
    except ValueError:
        print(f"[config] {name}={value!r} is not a number - using {default}.")
        _SOURCES[name] = "default (unparseable)"
        return default
    return _clamp(name, parsed, default, minimum, maximum)


def _clamp(name, parsed, default, minimum, maximum):
    """Range-checks a parsed value, loudly.

    Out-of-range is treated as a typo, not as an instruction. These settings drive a real-time
    audio loop, and a value like VAD_AGGRESSIVENESS=7 or MIN_SILENCE_DURATION=-200 does not
    degrade the robot gracefully - webrtcvad raises, or the endpointer ends every turn
    instantly. Refusing the value and saying so beats either.
    """
    if minimum is not None and parsed < minimum:
        print(f"[config] {name}={parsed} is below the minimum {minimum} - using {default}.")
        _SOURCES[name] = "default (out of range)"
        return default
    if maximum is not None and parsed > maximum:
        print(f"[config] {name}={parsed} is above the maximum {maximum} - using {default}.")
        _SOURCES[name] = "default (out of range)"
        return default
    return parsed


def get_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = get_str(name, default).lower()
    if value not in allowed:
        print(f"[config] {name}={value!r} is not one of {allowed} - using {default!r}.")
        _SOURCES[name] = "default (not a valid choice)"
        return default
    return value


class _RealtimeConfig:
    """Every real-time knob, read once at import.

    Read once rather than per access on purpose: these are consulted inside a 30ms audio loop,
    and os.environ lookups there would be pure waste. A change to .env takes effect on the next
    start, which is the right granularity for a kiosk - nothing here should change underneath a
    conversation that is already in progress.
    """

    def __init__(self) -> None:
        # --- master switches ------------------------------------------------------------------
        # VAD off does NOT mean "transcribe everything": it means fall back to the fixed
        # trailing-silence threshold instead of the adaptive endpointer. The per-frame webrtcvad
        # pass is what decides when a recording starts at all and cannot be disabled without
        # removing the ability to hear anybody.
        self.VAD_ENABLED: bool = get_bool("VAD_ENABLED", True)

        # Partial transcripts while somebody is still talking. Costs real CPU (a second Whisper
        # pass over a growing buffer) on hardware that is also running YOLO and mediapipe - see
        # PARTIAL_PREVIEW_ENABLED in stage04. On by default now because it is the only visible
        # evidence of what the robot is hearing DURING a turn, which is what makes the turn
        # detector debuggable instead of mysterious.
        self.STREAMING_STT: bool = get_bool("STREAMING_STT", True)
        self.STREAMING_TTS: bool = get_bool("STREAMING_TTS", True)

        # "auto" is the honest default and the reason this is not a plain bool. Barge-in without
        # hardware echo cancellation depends entirely on whether THIS room and THIS microphone
        # leave a gap between the robot's own voice and a person talking over it - a question
        # only a measurement can answer (see calibrate_echo_floor in stage04). So: auto measures
        # at startup and switches barge-in on only if the gap is really there; true forces it on
        # regardless; false never calibrates and never interrupts.
        self.INTERRUPTION_ENABLED: str = get_choice(
            "INTERRUPTION_ENABLED", "auto", ("auto", "true", "false")
        )
        self.CONTINUOUS_CONVERSATION: bool = get_bool("CONTINUOUS_CONVERSATION", True)
        self.CONVERSATION_MODE: str = get_choice(
            "CONVERSATION_MODE", "continuous", ("continuous", "push_to_talk")
        )

        # --- turn detection: the four numbers that decide when the robot answers --------------
        # Each maps onto a constant that stage04 already tuned and documents.
        #
        #   MIN_SILENCE_DURATION  -> ENDPOINT_EAGER_MS    earliest a turn may end, and only with
        #                                                 positive evidence (a landing pitch
        #                                                 contour AND a complete-sounding
        #                                                 transcript). Never on silence alone.
        #   MAX_SILENCE_DURATION  -> ENDPOINT_PATIENT_MS  hard stop; the turn ends here whatever
        #                                                 the signals say.
        #   END_OF_TURN_TIMEOUT   -> ENDPOINT_UNKNOWN_MS  used when the pitch contour cannot be
        #                                                 read at all (very common - English
        #                                                 phrases ending on an unvoiced consonant
        #                                                 have no measurable contour). The
        #                                                 absence of evidence, not evidence.
        #   MIN_SPEECH_DURATION   -> MIN_UTTERANCE_MS     below this the clip is a cough or a
        #                                                 click and is dropped WITHOUT ever
        #                                                 reaching Whisper.
        self.MIN_SILENCE_DURATION: int = get_int(
            "MIN_SILENCE_DURATION", 550, minimum=100, maximum=5000
        )
        self.MAX_SILENCE_DURATION: int = get_int(
            "MAX_SILENCE_DURATION", 1500, minimum=200, maximum=10000
        )
        self.END_OF_TURN_TIMEOUT: int = get_int(
            "END_OF_TURN_TIMEOUT", 900, minimum=150, maximum=10000
        )
        self.MIN_SPEECH_DURATION: int = get_int(
            "MIN_SPEECH_DURATION", 400, minimum=0, maximum=5000
        )
        self.MAX_UTTERANCE_S: float = get_float(
            "MAX_UTTERANCE_S", 15.0, minimum=1.0, maximum=120.0
        )

        # --- how sensitive the microphone side is ---------------------------------------------
        self.VAD_AGGRESSIVENESS: int = get_int(
            "VAD_AGGRESSIVENESS", 1, minimum=0, maximum=3
        )
        self.SPEECH_START_FRAMES: int = get_int(
            "SPEECH_START_FRAMES", 4, minimum=1, maximum=50
        )
        self.PREROLL_FRAMES: int = get_int("PREROLL_FRAMES", 20, minimum=0, maximum=100)
        # While somebody is talking to the robot, sound this many dB quieter than their own
        # voice is not them - another conversation across the room, a corridor. It no longer
        # keeps their turn open. 0 switches the filter off. Added 2026-10-27: on a real run the
        # visitor spoke at about -12 dBFS, the room carried on "speaking" for 5-12 seconds after
        # they stopped, and every one of those turns waited for the stability rule or the cap.
        self.VOICE_GATE_DB: float = get_float("VOICE_GATE_DB", 25.0, minimum=0.0, maximum=60.0)

        # --- barge-in -------------------------------------------------------------------------
        self.BARGE_IN_MARGIN: float = get_float(
            "BARGE_IN_MARGIN", 2.5, minimum=1.05, maximum=50.0
        )
        self.BARGE_IN_MIN_SPEECH_MS: int = get_int(
            "BARGE_IN_MIN_SPEECH_MS", 330, minimum=60, maximum=5000
        )
        self.BARGE_IN_WARMUP_MS: int = get_int(
            "BARGE_IN_WARMUP_MS", 700, minimum=0, maximum=5000
        )
        # Above this measured echo level, calibration refuses to enable barge-in: the speaker is
        # simply too loud into this microphone for any threshold to separate the robot's own
        # voice from a person's, and shipping a barge-in that fires on the robot itself is worse
        # than not having one. See _calibrate_barge_in() in stage08.
        self.BARGE_IN_MAX_ECHO_FLOOR: float = get_float(
            "BARGE_IN_MAX_ECHO_FLOOR", 0.08, minimum=0.001, maximum=1.0
        )
        # Whether the words that TRIGGERED an interruption are carried into the next listen
        # instead of discarded. See BargeInMonitor.captured_frames - without this the visitor
        # has to repeat the sentence they just interrupted with, which is precisely the thing
        # barge-in was supposed to avoid.
        self.BARGE_IN_KEEP_AUDIO: bool = get_bool("BARGE_IN_KEEP_AUDIO", True)

        # --- partial transcripts ---------------------------------------------------------------
        self.PARTIAL_MIN_DURATION_S: float = get_float(
            "PARTIAL_MIN_DURATION_S", 1.2, minimum=0.2, maximum=30.0
        )

        # --- ending a turn when the WORDS stop changing ------------------------------------------
        # A third endpointing signal, independent of both the pitch contour and the VAD, and the
        # only one that survives a microphone whose room noise reads as continuous speech.
        #
        # From a real run (2026-09-17, Realtek mic array): the visitor asked "What is your name?"
        # in about two seconds. webrtcvad at aggressiveness 1 then reported speech CONTINUOUSLY
        # for the next nineteen, so silence_ms never accumulated, the endpointer was never given
        # a pause to judge, and the turn only ended by hitting the 15-second safety cap -
        # "finished speaking after 20829ms of audio and a 0ms pause". The visitor waited twenty
        # seconds to be told a name.
        #
        # Whisper knew perfectly well they had stopped: the partial transcript came back as
        # "What is your name?" three times in a row, unchanged. Nobody producing new words for
        # several seconds has finished talking, whatever the VAD thinks of the room tone.
        #
        # So: this many consecutive identical partials, on a transcript that does not end
        # mid-thought, ends the turn. At PARTIAL_INTERVAL_S=1.5 two of them is about three
        # seconds of no new words - slower than a clean pitch-contour endpoint, far faster than
        # the cap, and it cannot be fooled by background noise. One more identical partial ends
        # the turn even on a transcript that DOES end mid-thought (TURN_STALL_PARTIALS in
        # stt_listener_live.py) after "[??] [??] is" held a visitor until
        # the cap.
        #
        # Requires STREAMING_STT (there are no partials otherwise). Set to 0 to switch off.
        self.TURN_STABLE_PARTIALS: int = get_int(
            "TURN_STABLE_PARTIALS", 2, minimum=0, maximum=20
        )
        self.PARTIAL_INTERVAL_S: float = get_float(
            "PARTIAL_INTERVAL_S", 1.5, minimum=0.2, maximum=30.0
        )

        # --- speaking side ----------------------------------------------------------------------
        # How long a fragment has to be before it is flushed to Piper. Piper has a fixed per-call
        # overhead, so a two-word fragment buys nothing and costs an audible seam.
        self.TTS_MIN_SENTENCE_CHARS: int = get_int(
            "TTS_MIN_SENTENCE_CHARS", 24, minimum=1, maximum=400
        )
        # The FIRST fragment of an answer may additionally break at a comma/semicolon/colon once
        # it is this long, instead of waiting for a full stop. Only the first: every later
        # sentence has audio already playing in front of it, so breaking them early buys nothing
        # and costs intonation. See _pop_sentence() in stage06.
        self.TTS_FIRST_CLAUSE_FLUSH: bool = get_bool("TTS_FIRST_CLAUSE_FLUSH", True)
        self.TTS_FIRST_CLAUSE_MIN_CHARS: int = get_int(
            "TTS_FIRST_CLAUSE_MIN_CHARS", 40, minimum=10, maximum=400
        )

        # --- logging ----------------------------------------------------------------------------
        # The tagged real-time trace ([VAD]/[TURN]/[STT]/[LLM]/[TTS]/[STATE]/[INTERRUPTION]).
        # Separate from the existing per-stage prints, which are untouched.
        self.REALTIME_DEBUG: bool = get_bool("REALTIME_DEBUG", True)
        # Whether partial transcripts are printed in full. Off means lengths only. A partial
        # transcript is a visitor's own words, and a kiosk log is not always a private place -
        # PRIVACY.md. OFF by default since 2026-10-27: an unattended kiosk is the normal case,
        # and a visitor's words should not end up in a console log nobody chose to keep. Turn it
        # on while developing or tuning the turn detector.
        self.LOG_TRANSCRIPTS: bool = get_bool("LOG_TRANSCRIPTS", False)

        # =========================================================================================
        # Added 2026-10-27 (review fixes). Every default below is chosen so the robot works out
        # of the box; each optional model falls back to the older behaviour when its file or
        # package is missing, and says so once at startup.
        # =========================================================================================

        # --- answering faster ----------------------------------------------------------------
        # Order the answer chain tries its backends in. Only backends that are actually usable
        # (a key is set / the server answers) are kept. Groq first because it streams its first
        # token in a few hundred milliseconds; Gemini is a "thinking" model that cannot switch
        # thinking off and so costs 1-3s before the first word.
        self.ANSWER_BACKENDS: str = get_str("ANSWER_BACKENDS", "groq,gemini,ollama")
        self.GROQ_MODEL: str = get_str("GROQ_MODEL", "llama-3.3-70b-versatile")
        # Gemini's Google Search grounding on general questions. Adds latency and is metered
        # separately; off by default now that Groq answers first.
        self.GEMINI_SEARCH: bool = get_bool("GEMINI_SEARCH", False)
        # Stream general (non-knowledge-base) answers sentence by sentence too, instead of
        # waiting for the whole reply.
        self.STREAM_GENERAL_ANSWERS: bool = get_bool("STREAM_GENERAL_ANSWERS", True)
        # Short "Mm, okay." acknowledgment. auto = only when the answer is NOT ready after
        # FILLER_DELAY_MS, so a fast answer is never prefaced; always = every question; off.
        self.FILLER_MODE: str = get_choice("FILLER_MODE", "auto", ("auto", "always", "off"))
        self.FILLER_DELAY_MS: int = get_int("FILLER_DELAY_MS", 700, minimum=0, maximum=5000)
        # Keep synthesized audio for fixed lines and direct answers, so a repeat plays instantly.
        self.TTS_AUDIO_CACHE: bool = get_bool("TTS_AUDIO_CACHE", True)

        # --- who to greet --------------------------------------------------------------------
        # face = YuNet face detection + a light tracker (no PyTorch, much less CPU).
        # yolo = the original YOLOv8 person tracker (needs torch + ultralytics, AGPL-3.0).
        self.PRESENCE_BACKEND: str = get_choice("PRESENCE_BACKEND", "face", ("face", "yolo"))
        # Upper bound on how often the heavy vision work runs. The camera is still read every
        # loop so frames never go stale; detection simply skips frames above this rate.
        self.VISION_FPS: float = get_float("VISION_FPS", 10.0, minimum=1.0, maximum=60.0)
        # Only greet people who are near, roughly facing the robot, and have stayed a moment.
        self.ENGAGEMENT_GATE: bool = get_bool("ENGAGEMENT_GATE", True)
        # "Near": face width as a fraction of the frame width. 0.05 is ~64px at 1280 wide,
        # roughly two metres away on a typical 70-degree webcam.
        self.ENGAGE_MIN_FACE_FRACTION: float = get_float(
            "ENGAGE_MIN_FACE_FRACTION", 0.05, minimum=0.005, maximum=0.9
        )
        # "Facing": how far the nose may sit from the midpoint between the eyes, as a fraction
        # of the eye distance. 0 = looking straight at the camera; ~0.5 = head turned ~45deg.
        self.ENGAGE_MAX_YAW: float = get_float("ENGAGE_MAX_YAW", 0.35, minimum=0.05, maximum=2.0)
        # "Staying": how long both have to hold before the robot speaks.
        self.ENGAGE_DWELL_S: float = get_float("ENGAGE_DWELL_S", 1.0, minimum=0.0, maximum=30.0)

        # --- who is being spoken to ------------------------------------------------------------
        # auto   = ignore speech when the camera can see faces but none of them was facing the
        #          robot with moving lips while it was said; if there is no face data at all,
        #          answer (better than going deaf when the camera misses a frame).
        # strict = like auto, but with no face data only the wake word counts.
        # off    = the old behaviour: everything heard during a visit is a question.
        self.ADDRESSEE_CHECK: str = get_choice(
            "ADDRESSEE_CHECK", "auto", ("auto", "strict", "off")
        )
        # Share of the speech window in which the speaker's lips must have been moving.
        self.ADDRESSEE_MIN_LIP_FRACTION: float = get_float(
            "ADDRESSEE_MIN_LIP_FRACTION", 0.25, minimum=0.0, maximum=1.0
        )
        # The wake word. "text" = look for the robot's name in what Whisper heard (no extra
        # model); "openwakeword" = a small always-on detector (needs OWW_MODEL); "off".
        self.WAKE_WORD_ENGINE: str = get_choice(
            "WAKE_WORD_ENGINE", "text", ("text", "openwakeword", "off")
        )
        # Extra phrases that count as the wake word, comma separated. The robot's own name
        # (from robot_identity) and "hey <name>" are always included.
        self.WAKE_WORDS: str = get_str("WAKE_WORDS", "")
        self.OWW_MODEL: str = get_str("OWW_MODEL", "")
        self.OWW_THRESHOLD: float = get_float("OWW_THRESHOLD", 0.5, minimum=0.05, maximum=0.99)
        # Say a short "I'll be with you in a moment" to people who arrive mid-conversation.
        self.ACKNOWLEDGE_NEWCOMERS: bool = get_bool("ACKNOWLEDGE_NEWCOMERS", True)

        # --- audio devices and echo cancellation ----------------------------------------------
        # Pick a device by (part of) its name or its index, e.g. MIC_DEVICE=Jabra. Empty = the
        # system default. Run stages/stage00_setup/list_audio_devices.py to see the names.
        self.MIC_DEVICE: str = get_str("MIC_DEVICE", "")
        self.SPEAKER_DEVICE: str = get_str("SPEAKER_DEVICE", "")
        # Software echo cancellation (WebRTC audio processing, via the `livekit` package): the
        # robot's own voice is subtracted from the microphone signal. auto = on when the package
        # is installed. Turn it off if you use a speakerphone that already cancels echo.
        self.AEC_ENABLED: str = get_choice("AEC_ENABLED", "auto", ("auto", "true", "false"))
        # Read the microphone in its own process, so heavy vision or model work can never
        # starve it ("input overflow"). Off by default; recommended on a small CPU.
        self.AUDIO_CAPTURE_PROCESS: bool = get_bool("AUDIO_CAPTURE_PROCESS", False)

        # --- hearing ---------------------------------------------------------------------------
        # whisper = faster-whisper (most accurate); sherpa = streaming Zipformer (text is ready
        # the moment you stop talking, runs comfortably on a Raspberry Pi 5).
        self.STT_BACKEND: str = get_choice("STT_BACKEND", "whisper", ("whisper", "sherpa"))
        self.WHISPER_MODEL: str = get_str("WHISPER_MODEL", "small.en")
        # auto = use an NVIDIA GPU if faster-whisper can actually run on it, else the CPU.
        self.STT_DEVICE: str = get_choice("STT_DEVICE", "auto", ("auto", "cpu", "cuda"))
        # groq = the finished question is transcribed by Groq's Whisper large-v3 (needs
        # GROQ_API_KEY), with the local Whisper as the backup; partials stay local.
        # off = everything local (stage04's stt_cloud.py).
        self.STT_CLOUD: str = get_choice("STT_CLOUD", "off", ("off", "groq"))
        self.GROQ_STT_MODEL: str = get_str("GROQ_STT_MODEL", "whisper-large-v3")
        # Longest the robot waits for the cloud transcript before using the local one.
        self.STT_CLOUD_TIMEOUT_S: float = get_float(
            "STT_CLOUD_TIMEOUT_S", 3.0, minimum=0.5, maximum=15.0
        )
        # smart_turn = a small model that listens to HOW you finished (Pipecat smart-turn v3,
        # 8MB) and lets the robot answer ~250ms after a clearly finished question.
        # pitch = the older pitch-contour + transcript heuristics only.
        self.TURN_DETECTOR: str = get_choice(
            "TURN_DETECTOR", "smart_turn", ("smart_turn", "pitch")
        )
        self.SMART_TURN_THRESHOLD: float = get_float(
            "SMART_TURN_THRESHOLD", 0.5, minimum=0.05, maximum=0.99
        )
        self.SMART_TURN_MIN_SILENCE_MS: int = get_int(
            "SMART_TURN_MIN_SILENCE_MS", 250, minimum=60, maximum=3000
        )

        # --- speaking ---------------------------------------------------------------------------
        # kokoro = Kokoro-82M (much more natural); piper = the original voice. auto = kokoro when
        # its model files are present, otherwise piper.
        self.TTS_BACKEND: str = get_choice("TTS_BACKEND", "auto", ("auto", "kokoro", "piper"))
        self.KOKORO_VOICE: str = get_str("KOKORO_VOICE", "af_heart")
        self.KOKORO_SPEED: float = get_float("KOKORO_SPEED", 1.0, minimum=0.5, maximum=2.0)
        # Small random change of speaking rate per sentence (0.04 = +/-4%), so long answers do
        # not have the exact same rhythm sentence after sentence.
        self.TTS_RATE_VARIATION: float = get_float(
            "TTS_RATE_VARIATION", 0.04, minimum=0.0, maximum=0.2
        )
        # A quiet "mm-hm" when the visitor pauses mid-sentence. auto = only when echo
        # cancellation is active, because otherwise the microphone records the robot's own
        # "mm-hm" into the visitor's question.
        self.BACKCHANNELS: str = get_choice("BACKCHANNELS", "auto", ("auto", "true", "false"))
        self.BACKCHANNEL_AFTER_MS: int = get_int(
            "BACKCHANNEL_AFTER_MS", 600, minimum=200, maximum=5000
        )

        # --- screen -----------------------------------------------------------------------------
        # No windows at all (for a robot box with no monitor). Stop with Ctrl+C or a service stop.
        self.HEADLESS: bool = get_bool("HEADLESS", False)
        # The animated face (the Expression Layer) in its own window.
        self.FACE_DISPLAY: bool = get_bool("FACE_DISPLAY", True)
        self.FACE_FULLSCREEN: bool = get_bool("FACE_FULLSCREEN", False)
        # The camera preview with boxes. Useful while developing; off on a finished robot.
        self.CAMERA_PREVIEW: bool = get_bool("CAMERA_PREVIEW", True)

    # -- derived helpers -------------------------------------------------------------------------

    def barge_in_wanted(self) -> bool:
        """Whether to CALIBRATE at startup. 'auto' and 'true' both do; only 'false' skips it."""
        return self.INTERRUPTION_ENABLED in ("auto", "true")

    def barge_in_forced(self) -> bool:
        """Whether to run barge-in even if calibration says this room cannot support it."""
        return self.INTERRUPTION_ENABLED == "true"

    def aec_wanted(self) -> bool:
        """Whether to try software echo cancellation at all ('auto' and 'true' do)."""
        return self.AEC_ENABLED in ("auto", "true")

    def backchannels_wanted(self, aec_active: bool) -> bool:
        """'auto' only when echo cancellation is running - see BACKCHANNELS."""
        if self.BACKCHANNELS == "true":
            return True
        if self.BACKCHANNELS == "false":
            return False
        return aec_active

    def answer_backend_order(self) -> list[str]:
        """ANSWER_BACKENDS as a clean list, e.g. ['groq', 'gemini', 'ollama']."""
        known = ("groq", "gemini", "claude", "ollama")
        order = [b.strip().lower() for b in self.ANSWER_BACKENDS.split(",") if b.strip()]
        unknown = [b for b in order if b not in known]
        if unknown:
            print(f"[config] ANSWER_BACKENDS has unknown name(s) {unknown} - ignoring them. "
                  f"Known: {', '.join(known)}.")
        return [b for b in order if b in known] or ["groq", "gemini", "ollama"]

    def push_to_talk(self) -> bool:
        """Both switches point at the same behaviour, so honour either one being set to
        hands-off - CONTINUOUS_CONVERSATION=false is the obvious spelling for somebody who read
        the flag list, CONVERSATION_MODE=push_to_talk for somebody who read the mode list."""
        return self.CONVERSATION_MODE == "push_to_talk" or not self.CONTINUOUS_CONVERSATION

    def validate(self) -> list[str]:
        """Cross-field checks that no single-field range check can catch.

        Returns the problems as strings rather than raising: each one is repaired in place to a
        sane ordering, because a kiosk that boots with slightly-off turn timing is far more
        useful than one that refuses to boot over a .env edit.
        """
        problems: list[str] = []
        if self.MIN_SILENCE_DURATION >= self.MAX_SILENCE_DURATION:
            problems.append(
                f"MIN_SILENCE_DURATION ({self.MIN_SILENCE_DURATION}ms) must be below "
                f"MAX_SILENCE_DURATION ({self.MAX_SILENCE_DURATION}ms) - the eager threshold "
                f"cannot sit at or past the hard stop, or no turn can ever end early. "
                f"Raising MAX to {self.MIN_SILENCE_DURATION * 2}ms."
            )
            self.MAX_SILENCE_DURATION = self.MIN_SILENCE_DURATION * 2
        if not (self.MIN_SILENCE_DURATION <= self.END_OF_TURN_TIMEOUT
                <= self.MAX_SILENCE_DURATION):
            clamped = max(self.MIN_SILENCE_DURATION,
                          min(self.END_OF_TURN_TIMEOUT, self.MAX_SILENCE_DURATION))
            problems.append(
                f"END_OF_TURN_TIMEOUT ({self.END_OF_TURN_TIMEOUT}ms) must sit between "
                f"MIN_SILENCE_DURATION ({self.MIN_SILENCE_DURATION}ms) and "
                f"MAX_SILENCE_DURATION ({self.MAX_SILENCE_DURATION}ms) - it is the fallback "
                f"used when the pitch contour cannot be read, so outside that range it would "
                f"either never or always apply. Using {clamped}ms."
            )
            self.END_OF_TURN_TIMEOUT = clamped
        if self.TTS_FIRST_CLAUSE_MIN_CHARS < self.TTS_MIN_SENTENCE_CHARS:
            problems.append(
                f"TTS_FIRST_CLAUSE_MIN_CHARS ({self.TTS_FIRST_CLAUSE_MIN_CHARS}) is below "
                f"TTS_MIN_SENTENCE_CHARS ({self.TTS_MIN_SENTENCE_CHARS}) - a clause break "
                f"should never fire earlier than a real sentence break would. Using "
                f"{self.TTS_MIN_SENTENCE_CHARS}."
            )
            self.TTS_FIRST_CLAUSE_MIN_CHARS = self.TTS_MIN_SENTENCE_CHARS
        return problems


RT = _RealtimeConfig()

for _problem in RT.validate():
    print(f"[config] {_problem}")


if __name__ == "__main__":
    print("Real-time conversation settings\n")
    print(f"  (.env file: {Path(__file__).resolve().parents[2] / '.env'})\n")
    width = max(len(k) for k in vars(RT))
    for key, value in vars(RT).items():
        origin = _SOURCES.get(key, "default")
        marker = "  " if origin == "default" else "* "
        print(f"{marker}{key:<{width}}  {str(value):<12}  [{origin}]")
    print("\n* = overridden from the environment or .env; everything else is the shipped "
          "default.")
    print(f"\nbarge-in: calibrate at startup={RT.barge_in_wanted()}, "
          f"force on regardless of the measurement={RT.barge_in_forced()}")
    print(f"conversation: {'push-to-talk' if RT.push_to_talk() else 'continuous (hands-free)'}")
