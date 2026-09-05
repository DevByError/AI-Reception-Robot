"""
Stage 08 (live) - the real end-to-end robot: real camera-based presence detection (Stage 01) +
real face recognition (Stage 02) feeding the real GreetingStateMachine (Stage 03), then a real
microphone question (Stage 04) answered by the real knowledge base + dialogue manager (Stage 05)
and spoken back with real streaming TTS (Stage 06). Stage 07's data pipeline runs on its own
background schedule, not as part of this per-visit loop (see stage07_data_pipeline/) - it's what
keeps the knowledge base Stage 05 reads from here up to date.

This is the real counterpart to main.py (kept as-is, still a pure-Mock skeleton, useful for
regression-testing the wiring logic itself without any hardware) - every constructor here follows
the same "swap only what's fake" pattern every other stage's _live file in this project has used.

Reused directly rather than reimplemented, because each of these is already real/tested code:
  - detect_people()/PerTrackPresenceTracker (stage01's presence_detector_live.py)
  - FaceMemoryStore/SFaceEmbedder (stage02's face_memory.py/sface_embedder.py)
  - match_faces_to_people() (stage03's greeting_live.py - links a detected face to a tracked
    person's bounding box)
  - GreetingStateMachine/ExpressionController (stage03's greeting_state_machine.py)
  - listen_for_one_utterance() (stage04's stt_listener_live.py - factored out of that file's own
    __main__ loop specifically so this orchestrator could reuse it instead of duplicating ~80
    lines of subtle real-time VAD/preroll/timeout logic)
  - DialogueManager + _build_answer_generator()/SAMPLE_CHUNKS (stage05's dialogue_manager.py /
    dialogue_manager_live.py - reusing _build_answer_generator() means flipping stage05's own
    MODE constant automatically applies here too, no separate switch to keep in sync)
  - LiveTTSSpeaker (stage06's tts_speaker_live.py - real streaming Piper TTS)
  - UniversityScraper/UniversityDataIndexer (stage07's scraper.py/indexer.py - real
    lincoln.edu.my content, scraped fresh into this run's knowledge base at startup; see the
    comment in __init__ for why this bypasses scraper.py's own "only index what changed"
    snapshot logic)

Startup fetches real content from lincoln.edu.my (Stage 07) before the camera/greeting loop
starts - this adds real, noticeable startup latency (roughly a minute, mostly the ~35 polite,
rate-limited page fetches) on top of model loading. That's a deliberate first-pass tradeoff, not
an oversight - decoupling this into a real background refresh cycle is the same "wire into a
real scheduler" TODO already called out in indexer.py, not something worth solving here just to
get real data flowing end to end.

Threading (added 2026-08-19, replacing the original single-threaded design): the camera and all
perception run continuously on the main thread, while each visit - greeting, listening,
thinking, speaking - runs on its own short-lived thread (_run_visit). Previously all of that ran
inline in the camera loop, so the video feed froze for the entire conversation: the robot was
blind between "hello" and "goodbye", kept talking to people who had already walked away, and the
preview window stopped responding. The split is deliberate about which side gets what: OpenCV's
window and capture stay on the main thread (the portable, supported way to use them), and the
blocking audio/LLM work moves off it. Shared state between the two is small and explicitly
locked - see _state_lock in __init__.

Persistent memory (added 2026-09-04): enrolled faces and full conversation transcripts live in
MySQL (stage02's robot_db.py) instead of only in process memory, so a student enrolled today is
still recognized after the robot is closed and reopened - which was the whole point of asking
for their name. Every spoken line goes in, both sides, with the timestamp and (for answers) the
dialogue route that produced it. If MySQL isn't reachable the robot still starts and runs
normally, it just forgets everything on exit and says so loudly at startup - see
robot_db.connect_or_none(). Connection settings come from the project-root `.env` file (copy
`.env.example`).

The same database also holds who the ROBOT is (table `robot_identity`, read by stage05's
identity.py): its name, its creator, that creator's email and nationality. Renaming the robot is
an UPDATE, not a code edit - and because Whisper's vocabulary bias is built from the same values
(see real_vocabulary()), a rename changes what the robot listens for as well as what it says.

Active speaker detection (same date): when several people are visible, mouth movement decides
which one is actually talking to the robot - see stage02's mouth_activity.py for how, and
_active_speaker_track_id() for how that's joined back to a tracked person. This matters beyond
convenience: without it, the "can I remember you?" flow could attach one person's spoken name to
a different person's face.

What's still a deliberate simplification, not a bug:
  - Consent-to-remember uses simple heuristic yes/no + name parsing on the raw transcript (see
    _parse_yes_no()/_parse_name_from_speech() below), not a real NLU intent classifier - good
    enough for a short, direct answer to a direct question ("yes" / "no thanks" / "my name is
    Ahmad"), but a sarcastic or roundabout answer could be misread. Documented, not hidden - and
    consistent with PRIVACY.md, _parse_yes_no() defaults to False (don't register) on anything
    ambiguous or unheard, since explicit consent must never be assumed.
  - Active speaker detection is visual only - it measures lip motion, and does not correlate it
    against the audio waveform. Someone silently mouthing words (or chewing) near the robot can
    therefore be picked as the speaker. See mouth_activity.py's module docstring for why the
    stronger audio-visual version isn't claimed here.
  - One visit at a time. A second group arriving mid-conversation waits until the current visit
    ends rather than being greeted in parallel - correct for a single mouth and one speaker, but
    it does mean the robot can look like it's ignoring new arrivals while busy.

Install first (every upstream stage this file wires together):
  pip install -r stages/stage01_presence_detection/requirements.txt
  pip install -r stages/stage02_face_recognition/requirements.txt
  pip install -r stages/stage03_proactive_greeting/requirements.txt
  pip install -r stages/stage04_speech_to_text/requirements.txt
  pip install -r stages/stage05_nlu_dialogue/requirements.txt
  pip install -r stages/stage06_text_to_speech/requirements.txt
  (or just: pip install -r stages/stage08_orchestrator/requirements.txt - see that file)
Models: stage02's YuNet/SFace ONNX files and stage06's Piper voice must already be downloaded
(see each stage's own README) before this will run.

Run: python stages/stage08_orchestrator/main_live.py
Press 'q' in the camera preview window to quit (or Ctrl+C in the terminal).
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import sounddevice as sd
from ultralytics import YOLO

STAGES_DIR = Path(__file__).resolve().parent.parent
for stage_folder in [
    "stage01_presence_detection",
    "stage02_face_recognition",
    "stage03_proactive_greeting",
    "stage04_speech_to_text",
    "stage05_nlu_dialogue",
    "stage06_text_to_speech",
    "stage07_data_pipeline",
]:
    sys.path.append(str(STAGES_DIR / stage_folder))

from presence_detector_live import PerTrackPresenceTracker, detect_people  # noqa: E402
from face_memory import RETENTION_DAYS, FaceMemoryStore  # noqa: E402
from robot_db import connect_or_none  # noqa: E402
from sface_embedder import SFaceEmbedder  # noqa: E402
from mouth_activity import MouthActivityDetector  # noqa: E402
from greeting_state_machine import (  # noqa: E402
    ExpressionController,
    GreetingStateMachine,
    Person,
    RobotState,
)
from greeting_live import match_faces_to_people  # noqa: E402
from stt_listener import STTListener  # noqa: E402
from stt_listener_live import (  # noqa: E402
    FRAME_SAMPLES,
    SAMPLE_RATE,
    PatientVAD,
    StreamingVAD,
    WhisperSTTEngine,
    listen_for_one_utterance,
    real_initial_prompt,
    real_vocabulary,
)
from identity import use_database as load_identity  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from dialogue_manager import DialogueManager  # noqa: E402
from dialogue_manager_live import SAMPLE_CHUNKS, _build_answer_generator  # noqa: E402
from tts_speaker_live import LiveTTSSpeaker  # noqa: E402
from scraper import REAL_SEED_URLS, UniversityScraper  # noqa: E402
from indexer import UniversityDataIndexer  # noqa: E402

# Same reasoning as stage02/stage03: face embedding is more expensive than plain person
# tracking, so it only runs a few times per second rather than on every frame.
RECOGNIZE_EVERY_N_FRAMES = 5

# How long to hold off greeting a newly-arrived person while recognition is still checking who
# they are - see greeting_live.py's identical constant for the full reasoning (without this, a
# returning visitor is greeted before recognition has had a fair chance, and gets the generic
# "Hi there!" instead of "Welcome back").
RECOGNITION_WAIT_S = 2.5

# How long to wait for the person to actually ask something (either the first question, or a
# follow-up after "do you have any other questions?") before giving up and moving on - this
# replaces greeting_live.py's fixed LISTENING_TIMEOUT_S stand-in with a real signal (did they
# actually say anything) now that Stage 04 is wired in.
LISTEN_TIMEOUT_S = 8.0
CONSENT_LISTEN_TIMEOUT_S = 6.0

# Brief pause before every _listen() call, right before the stale-audio-queue drain (see
# _listen() below) - gives the room a moment to stop physically ringing with the robot's own
# last words first. The queue drain removes whatever got QUEUED during this pause, so this is
# what actually gets the tail of the robot's own voice/room echo out of the next listen instead
# of just discarding old digital backlog - real acoustic reverb needs a moment to happen and get
# captured before it can be drained away. Added 2026-08-19 alongside the queue-drain fix and the
# hearing-quality pass below (bigger model, always-on normalization) - all aimed at the same
# goal, a visitor's actual words reaching Whisper cleanly instead of noise/echo/backlog.
SETTLE_DELAY_S = 0.25

# Where the scraped-and-embedded knowledge base is cached between runs, and how long a cached
# copy stays usable before the robot re-scrapes lincoln.edu.my for fresh content. 24h is a
# deliberate balance: a university website changes on the order of weeks (programme pages, news),
# so daily is comfortably fresh, while still meaning a normal day's testing pays the ~1 minute
# scrape at most once. Run with --refresh to force a rescrape now regardless of age.
KB_CACHE_PATH = Path(__file__).resolve().parent / "kb_cache.npz"
KB_CACHE_MAX_AGE_S = 24 * 60 * 60

# Safety cap on how many questions one visit will loop through - see _handle_conversation().
# Without this, a visitor who keeps saying "yes" to "any other questions?" (or a misheard "yes"
# on a noisy mic) could loop indefinitely; this bounds one visit to a reasonable real
# conversation length instead.
MAX_QUESTIONS_PER_VISIT = 6

# Checked as substrings against the whole cleaned utterance, not as an exact-match word set -
# found from a real run (2026-08-12) that a plain word-by-word check gets this wrong two
# different ways: (1) Whisper transcribes with punctuation ("Yeah.", "Yes."), and splitting on
# whitespace alone leaves the trailing period attached, so "yeah." never equals "yeah" as a set
# member - real "Yeah."/"Yes." answers were silently read as "no" simply because the words never
# matched anything; (2) common affirmative idioms contain the literal words "no"/"not" ("no
# problem", "why not") or neither yes nor no as a token at all ("of course") - a same-day real
# test with those exact phrases got misread as refusals (or, for "of course", matched nothing
# and defaulted to no). Fixed by stripping punctuation first, then matching short phrases as
# substrings instead of requiring an exact per-word match.
#
# Reused for two different yes/no-shaped questions - "would you like me to remember you?" and
# (see _handle_conversation()) "do you have any other questions?" - so NEGATIVE_PHRASES also
# covers ways of saying "I'm done asking", not just ways of declining to be remembered.
NEGATIVE_PHRASES = (
    "no thanks", "no thank you", "not interested", "no way", "nah", "nope",
    "don't", "dont", "do not", "not really", "rather not",
    "that's all", "thats all", "that is all", "that's it", "thats it", "that is it",
    "i'm good", "im good", "no more", "nothing else", "all good", "i'm done", "im done",
)
AFFIRMATIVE_PHRASES = (
    "no problem", "why not", "of course", "sure", "yeah", "yeh", "yes", "yep", "yup",
    "okay", "ok", "alright", "please", "definitely", "sounds good", "sounds great",
    "go ahead", "absolutely",
)


def _clean_for_phrase_match(text: str) -> str:
    cleaned = re.sub(r"[^a-z' ]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_yes_no(text: str | None) -> bool:
    """Heuristic reading of a short, direct answer to a yes/no question - good enough for "yes"
    / "sure, go ahead" / "no problem" / "of course" / "that's all, thanks", not a real NLU
    classifier (see module docstring and the NEGATIVE_PHRASES/AFFIRMATIVE_PHRASES comment above
    for what this catches and why). Defaults to False on anything ambiguous or unheard - the
    safe default for what this is STILL used for (the remember-me consent ask in
    _handle_consent_if_unknown() - never register a face without clear, unambiguous consent). An
    unambiguous negative phrase always wins over any affirmative phrase found in the same
    utterance (e.g. "no, not interested" shouldn't be flipped true by anything else in it).

    NOT used anymore for "do you have any other questions?" (see _handle_conversation()) - that
    needed a three-way read (clear no / bare yes / an actual next question), not this plain
    binary one; see _is_clear_negative()/_is_bare_affirmation() below."""
    if not text:
        return False
    cleaned = _clean_for_phrase_match(text)
    if any(phrase in cleaned for phrase in NEGATIVE_PHRASES):
        return False
    return any(phrase in cleaned for phrase in AFFIRMATIVE_PHRASES)


def _is_clear_negative(text: str | None) -> bool:
    """True only for an unambiguous "no more questions" - same NEGATIVE_PHRASES list as
    _parse_yes_no(). Unheard/empty counts as a clear negative here (nothing more to say)."""
    if not text:
        return True
    return any(phrase in _clean_for_phrase_match(text) for phrase in NEGATIVE_PHRASES)


def _is_bare_affirmation(text: str) -> bool:
    """True only for a SHORT, content-free "yes" ("yes", "sure", "yeah please") - deliberately
    NOT true for a real follow-up question, even one that happens to start with "yes"/"sure".

    Added 2026-08-19 after a real run: _handle_conversation() used to run the plain _parse_yes_no
    binary check on the response to "do you have any other questions?" - a visitor who just asked
    their next question directly ("Where is the Engineering Department?") instead of first
    saying "yes" got misread as a "no" (nothing in it matched AFFIRMATIVE_PHRASES), and the robot
    said "Alright, glad I could help!" instead of answering - the second question was silently
    dropped. Real visitors skip the "yes" and go straight to the next question far more often
    than they answer the meta-question literally, so the loop now only treats SHORT responses as
    pure confirmation (needing one more listen for the actual question); anything longer is
    assumed to already BE the next question and is answered directly instead of discarded."""
    cleaned = _clean_for_phrase_match(text)
    if len(cleaned.split()) > 3:
        return False  # too long to be just "yes" - almost certainly real question content
    return any(phrase in cleaned for phrase in AFFIRMATIVE_PHRASES)


_NAME_PREFIXES = (
    "my name is ", "i am ", "i'm ", "im ", "it's ", "its ", "it is ", "call me ",
)


def _parse_name_from_speech(text: str | None) -> str:
    """Strips a common spoken preamble ("my name is Ahmad" -> "Ahmad") and title-cases the
    remainder; falls back to "Guest" if nothing usable came through. Heuristic, not real NLU -
    see module docstring."""
    if not text:
        return "Guest"
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in _NAME_PREFIXES:
        if lowered.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    name = stripped.strip().strip(".!?").title()
    return name or "Guest"


class _LiveTTSAdapter:
    """Adapts LiveTTSSpeaker (Stage 06, real Piper streaming TTS) to the simple .speak(text) ->
    None interface GreetingStateMachine expects - same idea as main.py's mock _TTSAdapter, just
    wrapping the real engine instead of the mock one.

    Also prints what's being said - unlike the mock TTSStub (greeting_state_machine.py), real
    Piper audio through LiveTTSSpeaker gives no console trace at all of what the robot actually
    said, which makes it impossible to tell from the log alone whether an answer was wrong, or
    the visitor just didn't hear/like a correct one. Printing here (and via
    RobotOrchestrator._speak() for everything after the greeting) closes that gap.

    on_spoken is the same idea one step further: the greeting is the robot's first line of the
    conversation, and a transcript that starts at the first answer instead of at "Welcome back,
    Ahmad" is missing the part that says who it thought it was talking to. GreetingStateMachine
    calls this adapter directly, so this hook is the only place that line can be captured."""

    def __init__(self, speaker: LiveTTSSpeaker, on_spoken=None):
        self.speaker = speaker
        self.on_spoken = on_spoken

    def speak(self, text: str) -> None:
        print(f'[TTS] "{text}"')
        if self.on_spoken is not None:
            self.on_spoken(text)
        self.speaker.speak(text)


class RobotOrchestrator:
    """The real robot: camera in, real recognition + greeting, real spoken questions answered
    from the real knowledge base, spoken back.

    Two threads (see run() and _run_visit()) - perception runs continuously while conversation
    happens alongside it. See the module docstring for that split, for how active speaker
    detection picks who's talking, and for what's still deliberately simplified."""

    def __init__(self, force_refresh: bool = False):
        print("Loading YOLOv8n + YuNet + SFace models...")
        self.yolo = YOLO(str(STAGES_DIR / "stage01_presence_detection" / "yolov8n.pt"))
        self.embedder = SFaceEmbedder()

        # Persistent memory (Stage 02's robot_db.py). connect_or_none() means a MySQL problem
        # degrades the robot to a single-session memory instead of stopping it from starting -
        # it prints exactly what was lost, see its docstring.
        self.db = connect_or_none()

        # Who the robot is - its name and its creator - also comes out of MySQL now (stage05's
        # identity.py, table `robot_identity`). Loaded HERE, once, on the connection we already
        # have: everything below this line (the Whisper prompt, the greeting, every identity
        # answer) then reads it straight from memory, so no visitor ever waits on a query for
        # "what's your name". Passing self.db=None is the offline case, not a failure - it
        # resolves from identity.py's cache file instead and says so.
        load_identity(self.db)

        self.face_store = FaceMemoryStore(is_match=self.embedder.is_match, db=self.db)
        if self.db is not None:
            print(f"Face memory: {len(self.face_store)} person(s) loaded from MySQL.")
            self._purge_expired_people()

        self.presence = PerTrackPresenceTracker()

        # --- cross-thread state -----------------------------------------------------------
        # The camera/perception loop (main thread, run()) writes these; the conversation thread
        # (_run_visit) reads them, and writes to face_store when registering someone new. Every
        # touch of the group below goes through _state_lock: without it, a visitor being
        # registered mid-conversation could race a face lookup happening on the same dicts one
        # frame later. The lock is only ever held for short dictionary operations, never across
        # a blocking call (speaking, listening, an API request), so it can't stall the camera.
        self._state_lock = threading.Lock()
        self._resolved_names: dict[int, str | None] = {}
        # Kept alongside _resolved_names rather than looked up from the name when needed: names
        # are not unique (two students called Ahmad) and not stable (a re-enrollment), so the
        # person_id is the only safe thing to attach a stored conversation to.
        self._resolved_person_ids: dict[int, str] = {}
        self._last_embedding_by_track: dict[int, object] = {}
        self._face_bbox_by_track: dict[int, tuple[int, int, int, int]] = {}
        self._seen_ids: set[int] = set()
        self._activities: list = []  # latest MouthActivity list, refreshed every frame

        # --- conversation recording -------------------------------------------------------
        # Only ever touched from the visit thread, and only one visit runs at a time (see the
        # module docstring), so these need no lock - unlike the perception state above.
        self._conversation_id: int | None = None
        self._conversation_person_id: str | None = None
        self._turn_index = 0

        print("Loading mediapipe face mesh (active speaker detection)...")
        try:
            self.mouth_detector = MouthActivityDetector()
        except Exception as exc:
            # Optional capability: without it the robot still works exactly as it did before,
            # it just can't tell two simultaneous speakers apart. Not worth refusing to start
            # over - especially since mediapipe is the newest and least-proven dependency here.
            print(f"[mouth] could not start active speaker detection ({exc!r}) - continuing "
                  f"without it. Install it with: pip install -r "
                  f"stages/stage02_face_recognition/requirements.txt")
            self.mouth_detector = None

        print("Loading Piper voice...")
        self.tts = LiveTTSSpeaker()
        self.expression = ExpressionController()
        self.greeting_sm = GreetingStateMachine(
            self.expression,
            _LiveTTSAdapter(self.tts, on_spoken=lambda text: self._record("robot", text)),
        )

        print("Loading faster-whisper...")
        self.stt_engine = WhisperSTTEngine(
            initial_prompt=real_initial_prompt(), vocabulary=real_vocabulary()
        )
        self.stt_listener = STTListener(self.stt_engine, PatientVAD(), locale="en-US")
        self.vad = StreamingVAD()
        self.audio_q: "queue.Queue[bytes]" = queue.Queue()

        def _on_audio(indata, frames, time_info, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            self.audio_q.put(bytes(indata))

        self.mic_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES, dtype="int16", channels=1,
            callback=_on_audio,
        )

        # _build_answer_generator() reads dialogue_manager_live.py's own MODE constant - reusing
        # it (rather than picking a generator here independently) means flipping MODE there
        # automatically applies to the real robot too, with nothing to keep in sync by hand.
        self.kb = KnowledgeBase()

        # Try the on-disk cache before touching the network. Before this existed (added
        # 2026-08-19), every launch re-scraped 87 rate-limited pages and re-embedded ~14,800
        # chunks just to rebuild the same knowledge base as last time - roughly a minute of dead
        # startup time per run, every run.
        #
        # KB_CACHE_MAX_AGE_S is what keeps this honest: the whole point of Stage 07 is that the
        # robot's facts track the real website, so the cache expires and re-scrapes on its own
        # rather than serving last month's programme list forever. Pass --refresh (see __main__)
        # to force a fresh scrape immediately.
        loaded_from_cache = False
        if not force_refresh:
            loaded_from_cache = self.kb.load(KB_CACHE_PATH, max_age_s=KB_CACHE_MAX_AGE_S)

        if not loaded_from_cache:
            # Real Stage 07 content. Deliberately does NOT use scraper.detect_changes() here (the
            # normal "only index what changed" path a standalone/scheduled scraper run would use)
            # - detect_changes() compares against snapshot.json on disk, which may already
            # remember these exact page hashes from an earlier standalone run of
            # scraper.py/indexer.py. This orchestrator builds a fresh KnowledgeBase whenever the
            # cache misses, so "nothing changed since the snapshot" would wrongly mean "index
            # nothing", leaving that fresh KB empty even though it never actually received the
            # real chunks. Indexing every scraped page unconditionally sidesteps that mismatch.
            print(f"Fetching real content from lincoln.edu.my ({len(REAL_SEED_URLS)} school "
                  f"pages + their programme links) - this takes about a minute, and the result "
                  f"is cached so the next launch won't need to...")
            scraper = UniversityScraper(REAL_SEED_URLS)
            indexer = UniversityDataIndexer(self.kb)
            real_urls = scraper.discover_urls()
            real_pages = scraper.scrape_once(real_urls)
            real_count = indexer.index_changed_pages(real_pages)
            print(f"Indexed {real_count} real chunk(s) from {len(real_urls)} "
                  f"lincoln.edu.my page(s).")
            try:
                self.kb.save(KB_CACHE_PATH)
                print(f"Cached the knowledge base to {KB_CACHE_PATH.name} for future launches.")
            except Exception as exc:
                # A cache that can't be written is a lost optimization, not a broken robot -
                # everything needed to run is already in memory at this point.
                print(f"[kb] could not write the cache ({exc!r}) - continuing without it.")

        # Loaded AFTER the scraped content (and after any cache load) so these always overwrite
        # rather than get overwritten: these are the hand-verified facilities directory, the one
        # part of the knowledge base that is manually maintained ground truth. They're also
        # cheap - ten chunks - so there's no reason to cache them, and re-upserting each run
        # means edits to SAMPLE_CHUNKS take effect immediately instead of being masked by a
        # stale cached copy of the old text.
        self.kb.upsert_many(list(SAMPLE_CHUNKS))
        print(f"Knowledge base ready: {len(self.kb)} chunk(s) total.")

        self.dialogue = DialogueManager(self.kb, _build_answer_generator())

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera index=0")

    # --- persistent memory ----------------------------------------------------------------

    def _purge_expired_people(self) -> None:
        """Runs the PRIVACY.md retention policy once at startup. Startup rather than a timer
        because the robot is expected to be restarted regularly and RETENTION_DAYS is measured
        in months - a scheduled sweep would add a thread to do the same job less predictably."""
        try:
            expired = self.face_store.purge_expired()
        except Exception as exc:
            print(f"[db] retention sweep failed ({exc!r}) - nothing was deleted.")
            return
        if expired:
            print(f"[privacy] deleted {len(expired)} person(s) not seen in "
                  f"{RETENTION_DAYS} days (PRIVACY.md retention policy).")

    def _begin_conversation(self, person_id: str | None) -> None:
        """Opens the transcript for one visit. person_id is None for a visitor the robot doesn't
        know - the conversation is still recorded, just not attached to anybody (and gets
        attached retroactively if they enroll before leaving, see _handle_consent_for_speaker)."""
        self._conversation_id = None
        self._conversation_person_id = person_id
        self._turn_index = 0
        if self.db is None:
            return
        try:
            self._conversation_id = self.db.start_conversation(person_id)
        except Exception as exc:
            print(f"[db] could not start a conversation record ({exc!r}) - this visit will not "
                  f"be saved.")

    def _record(self, speaker: str, text: str, intent: str | None = None) -> None:
        """One line of the transcript. Called from _speak()/_listen()/the greeting adapter
        rather than from the handful of places that call them, so a new spoken line anywhere in
        the visit is recorded automatically instead of only if somebody remembered to log it.

        Failures are printed and swallowed: losing a transcript line is not a reason to stop
        talking to the person standing in front of the robot."""
        if self._conversation_id is None or not text:
            return
        try:
            self.db.log_utterance(self._conversation_id, self._turn_index, speaker, text, intent)
            self._turn_index += 1
        except Exception as exc:
            print(f"[db] could not record a {speaker} line ({exc!r}).")

    def _end_conversation(self) -> None:
        if self._conversation_id is None:
            return
        try:
            self.db.end_conversation(self._conversation_id)
            print(f"[db] saved conversation #{self._conversation_id} "
                  f"({self._turn_index} line(s)).")
        except Exception as exc:
            print(f"[db] could not close the conversation record ({exc!r}).")
        finally:
            self._conversation_id = None

    def _listen(self, timeout_s: float) -> str | None:
        time.sleep(SETTLE_DELAY_S)

        # Drain any stale backlog before listening. The mic callback queues real-time audio
        # continuously, including while this thread is blocked doing something else entirely -
        # TTS playback, or a dialogue/API call (real seconds, more with a Gemini 503 retry) - see
        # the module docstring's "single-threaded and blocking" tradeoff. Left undrained, the
        # next listen_for_one_utterance() call starts by racing through however many seconds of
        # QUEUED OLD audio built up during that block (very possibly the robot's own TTS voice
        # bleeding into the mic - there's no echo cancellation) instead of live audio. That
        # backlog drains far faster than real time, so silence_ms (counted per-frame,
        # FRAME_MS-at-a-time) and elapsed_ms (real wall-clock) fall out of lockstep - which is
        # what produced the repeated, rapid-fire "speech detected... too short - ignoring" bursts
        # and garbled transcripts seen in real testing (2026-08-19), not a VAD-tuning problem.
        drained = 0
        while True:
            try:
                self.audio_q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f"[stt] (discarded {drained} stale audio frame(s) queued while busy)")

        heard = listen_for_one_utterance(
            self.stt_engine, self.stt_listener, self.vad, self.audio_q, timeout_s=timeout_s
        )
        # The transcript stores what Whisper HEARD, including its [??] low-confidence markers -
        # not a cleaned-up version. When a stored answer later looks wrong, the recorded
        # question is the only way to tell a bad answer apart from a misheard question.
        self._record("visitor", heard or "")
        return heard

    def _speak(self, text: str, intent: str | None = None) -> None:
        """Every spoken line outside the greeting goes through here instead of calling
        self.tts.speak() directly, so it shows up in the console log - real Piper audio gives no
        trace of what was actually said otherwise, which makes a wrong (or apologetic) answer
        indistinguishable from a correct one when reading the log after the fact. See
        _LiveTTSAdapter's docstring for the greeting-path equivalent."""
        print(f'[TTS] "{text}"')
        self._record("robot", text, intent)
        self.tts.speak(text)

    def _answer_one(self, question: str) -> None:
        result = self.dialogue.handle_question(question)
        # Prints WHY the answer came out the way it did (which route was taken, how many KB
        # chunks backed it) - without this, an apology/fallback answer looks identical in
        # the log to "the KB just didn't have it" vs. "retrieval found nothing because the
        # transcript had a [??] in it" vs. other real failure modes - see the STT confidence
        # markers already printed by listen_for_one_utterance() for that half of the picture.
        print(f"[dialogue] intent={result.intent.name}, {len(result.used_chunks)} chunk(s) "
              f"used, question={question!r}")
        # The intent is stored on the answer line so the saved transcript keeps the same "why
        # did it answer that way" trace the console log has - a row of apologies in the database
        # with no route attached is a lot harder to diagnose weeks later.
        self._speak(result.answer, intent=result.intent.name)

    def _handle_conversation(self) -> None:
        """Answers questions in a loop - "do you have any other questions?" after each answer -
        instead of ending the visit after exactly one question. Found from a real test run
        (2026-08-12): the old code asked to remember the visitor immediately after the first
        answer, cutting them off mid-conversation before they'd actually finished asking
        everything they came in for. Bounded by MAX_QUESTIONS_PER_VISIT so a misheard "yes" (or
        a genuinely chatty visitor) can't loop forever.

        The response to "do you have any other questions?" is read three ways, not as a plain
        yes/no (see _is_bare_affirmation()'s docstring for the real failure this fixes,
        2026-08-19): a clear negative ends the visit; a short bare "yes"/"sure" with no content
        loops back to listen for the actual next question; anything else (the common case - a
        visitor who just asks their next question directly, e.g. "Where is the Engineering
        Department?", without first saying "yes") is treated AS that next question and answered
        immediately instead of being discarded."""
        question = self._listen(LISTEN_TIMEOUT_S)
        questions_answered = 0

        while question and questions_answered < MAX_QUESTIONS_PER_VISIT:
            self._answer_one(question)
            questions_answered += 1

            self._speak("Do you have any other questions?")
            more = self._listen(CONSENT_LISTEN_TIMEOUT_S)

            if _is_clear_negative(more):
                self._speak("Alright, glad I could help!")
                return
            if more and _is_bare_affirmation(more):
                question = self._listen(LISTEN_TIMEOUT_S)  # bare "yes" - listen for what it is
            else:
                question = more  # they skipped the "yes" and asked it directly - answer it as-is

        if questions_answered == 0:
            self._speak("No worries, I'm here if you need anything.")
        elif questions_answered >= MAX_QUESTIONS_PER_VISIT:
            self._speak("I'll let you go for now - come find me again if you think of anything "
                         "else!")
        # else: the loop already ended cleanly via the clear-negative return above, or via a
        # final unheard/empty `question` (falls out of the while condition - already implicitly
        # "no more" the same way the pre-2026-08-19 "not question" timeout case worked).

    def _active_speaker_track_id(self, candidate_ids: list[int]) -> int | None:
        """Which of these tracked people is the one actually talking to the robot, judged by
        whose mouth has been moving (see stage02's mouth_activity.py).

        This is the fix for the real conflict case: two people stand in front of the robot, one
        asks a question, and the robot has to decide whose face to attach that conversation - and
        any resulting name - to. Guessing wrong doesn't just answer the wrong person, it can save
        someone's name against someone else's face, which is a privacy problem, not only a
        usability one (PRIVACY.md).

        Returns None when it can't tell - no mouth detector, nobody visibly speaking, or no
        confident match to a tracked person. Callers must handle that rather than assume an
        answer; "I'm not sure who spoke" is a legitimate and common outcome (one person facing
        away, poor lighting), and the safe response to it is to fall back rather than guess.
        """
        if self.mouth_detector is None or not candidate_ids:
            return None
        with self._state_lock:
            activities = list(self._activities)
            bboxes = {tid: self._face_bbox_by_track.get(tid) for tid in candidate_ids}

        speaker = self.mouth_detector.most_likely_speaker(activities)
        if speaker is None:
            return None

        # Join the mouth-movement face back to a tracked person by geometry - the two detectors
        # number faces independently, so position is the only thing they share.
        best_tid = None
        best_distance = float("inf")
        sx, sy, sw, sh = speaker.bbox
        speaker_center = (sx + sw / 2, sy + sh / 2)
        for tid, bbox in bboxes.items():
            if bbox is None:
                continue
            bx, by, bw, bh = bbox
            center = (bx + bw / 2, by + bh / 2)
            distance = (
                (center[0] - speaker_center[0]) ** 2 + (center[1] - speaker_center[1]) ** 2
            ) ** 0.5
            if distance < best_distance and distance <= max(bw, bh):
                best_tid = tid
                best_distance = distance
        return best_tid

    def _initial_conversation_person(self, arriving: list[Person]) -> str | None:
        """Who to file this visit under at the moment it starts.

        Only answers when exactly one of the arriving people is recognized. Nobody has spoken
        yet at greeting time, so active-speaker detection has nothing to work with here - and
        with two known people present, guessing means filing one student's questions under
        another student's name. The visit is instead left unattributed and re-checked once
        somebody has actually talked (see _attribute_conversation_to_speaker)."""
        with self._state_lock:
            known = [self._resolved_person_ids[p.track_id] for p in arriving
                     if p.track_id in self._resolved_person_ids]
        return known[0] if len(known) == 1 else None

    def _attribute_conversation_to_speaker(self, arriving: list[Person]) -> None:
        """Second attempt at filing an unattributed visit, run after the talking is done - by
        which point mouth-movement detection has had a whole conversation to watch, so it can
        answer the question _initial_conversation_person() couldn't."""
        if (self.db is None or self._conversation_id is None
                or self._conversation_person_id is not None):
            return  # no record to attach, or it is already filed under somebody
        tid = self._active_speaker_track_id([p.track_id for p in arriving])
        if tid is None:
            return
        with self._state_lock:
            person_id = self._resolved_person_ids.get(tid)
        if person_id is None:
            return  # the speaker is someone the robot doesn't know - nothing to attach to
        try:
            self.db.attach_conversation_to_person(self._conversation_id, person_id)
            self._conversation_person_id = person_id
        except Exception as exc:
            print(f"[db] could not attribute this conversation ({exc!r}) - it stays saved "
                  f"without a name on it.")

    def _handle_consent_for_speaker(self, arriving: list[Person]) -> None:
        """Offers to remember the person who was actually talking, rather than asking every
        person in view one after another.

        Changed 2026-08-19 (was: a loop calling this once per arriving track). Asking each
        visitor in turn is wrong in a group: the robot would say "I don't recognize you, would
        you like me to remember you?" two or three times in a row to people who never spoke to
        it, and whoever answered "yes" would have their name attached to whichever face the loop
        happened to be on at the time. Now it asks once, of whoever was speaking."""
        candidate_ids = [p.track_id for p in arriving]
        tid = self._active_speaker_track_id(candidate_ids)

        if tid is None:
            if len(candidate_ids) > 1:
                # Genuinely ambiguous with several people present - saying nothing is better
                # than risking someone's name on the wrong face.
                print(f"[consent] {len(candidate_ids)} people present and no clear speaker - "
                      f"skipping the remember-me offer for this visit.")
                return
            tid = candidate_ids[0]  # only one person here, so there's nothing to confuse
        else:
            print(f"[consent] active speaker is track {tid}")

        with self._state_lock:
            if self._resolved_names.get(tid):
                return  # already known (or registered earlier this visit) - nothing to ask
            embedding = self._last_embedding_by_track.get(tid)
        if embedding is None:
            return  # never got a clear face sample for them - nothing usable to register

        self._speak("I don't recognize you. Would you like me to remember you for next time?")
        answer = self._listen(CONSENT_LISTEN_TIMEOUT_S)
        if not _parse_yes_no(answer):
            self._speak("No problem, I won't remember this visit.")
            return

        self._speak("Great! What's your name?")
        name_answer = self._listen(CONSENT_LISTEN_TIMEOUT_S)
        name = _parse_name_from_speech(name_answer)
        with self._state_lock:
            rec = self.face_store.register(name=name, embeddings=[embedding],
                                            consent_given=True)
            self._resolved_names[tid] = name
            self._resolved_person_ids[tid] = rec.person_id

        # They were anonymous when this conversation started; now they aren't. See
        # RobotDatabase.attach_conversation_to_person() for why this is worth backfilling.
        if self.db is not None and self._conversation_id is not None:
            try:
                self.db.attach_conversation_to_person(self._conversation_id, rec.person_id)
                self._conversation_person_id = rec.person_id
            except Exception as exc:
                print(f"[db] could not attach this conversation to {name!r} ({exc!r}) - it "
                      f"stays saved, just without a name on it.")

        self._speak(f"Nice to meet you, {name}! I'll remember you next time.")

    def _run_visit(self, arriving: list[Person]) -> None:
        """One complete visit - greet, answer questions, offer to remember - start to finish.

        Runs on its OWN thread (see run()). Everything in here blocks for real seconds at a time:
        Piper speaking a sentence, the microphone waiting for someone to finish a question, an
        embedding search, sometimes a Gemini call. Before 2026-08-19 this all ran inline in the
        camera loop, which meant the video feed froze solid for the entire visit - the robot
        couldn't see anything between "hello" and "goodbye", so someone could walk away mid-answer
        and it would keep talking to an empty room, and the preview window went unresponsive.
        Moving it here lets the camera keep running the whole time.
        """
        # Opened before the greeting, not after it, so the greeting itself is the first line of
        # the saved transcript (that's the line naming who the robot thought it was talking to).
        self._begin_conversation(self._initial_conversation_person(arriving))
        try:
            self.greeting_sm.on_people_entered(arriving)
            if self.greeting_sm.state == RobotState.LISTENING:
                # Questions loop until the visitor says they're done (see _handle_conversation())
                # - only THEN does it ask about remembering them, not after the first question.
                self._handle_conversation()
                self._attribute_conversation_to_speaker(arriving)
                self._handle_consent_for_speaker(arriving)
            self.greeting_sm.on_interaction_timeout_or_done()
        except Exception as exc:
            # A crash on this thread would otherwise vanish silently - Python prints the
            # traceback and the thread dies, but the main loop would keep running and simply
            # never start another visit, which looks like "the robot stopped responding to
            # people" rather than like an error. Reset the state machine so the next visitor
            # still gets served.
            print(f"[visit] failed ({exc!r}) - resetting so the next visitor still works.")
            try:
                self.greeting_sm.on_interaction_timeout_or_done()
            except Exception:
                pass
        finally:
            # In `finally` so a crashed visit still closes its transcript instead of leaving a
            # row with a NULL ended_at that reads, forever, like a conversation still going on.
            self._end_conversation()

    def run(self) -> None:
        print("Live robot running. Press 'q' in the preview window to quit.")
        pending_since: dict[int, float] = {}
        ready_to_greet: set[int] = set()
        frame_count = 0
        visit_thread: threading.Thread | None = None

        try:
            with self.mic_stream:
                while True:
                    ok, frame = self.cap.read()
                    if not ok:
                        print("Could not read a frame - stopping.")
                        break
                    frame_count += 1

                    people = detect_people(self.yolo, frame)
                    seen_ids = {p.track_id for p in people}
                    entered_ids, left_ids = self.presence.update(seen_ids)

                    with self._state_lock:
                        self._seen_ids = seen_ids
                        for tid in left_ids:
                            self._resolved_names.pop(tid, None)
                            self._resolved_person_ids.pop(tid, None)
                            self._last_embedding_by_track.pop(tid, None)
                            pending_since.pop(tid, None)
                            ready_to_greet.discard(tid)

                    for tid in entered_ids:
                        pending_since[tid] = time.monotonic()

                    # Mouth movement, every frame - unlike face recognition below, this NEEDS
                    # every frame: the signal is how much the mouth ratio varies over time, so
                    # sampling it every 5th frame would throw away most of the variation it's
                    # trying to measure (and alias fast lip movement into apparent stillness).
                    # It's cheap enough to afford - see mouth_activity.py.
                    if self.mouth_detector is not None:
                        try:
                            activities = self.mouth_detector.update(frame)
                        except Exception as exc:
                            print(f"[mouth] detector failed ({exc!r}) - disabling active speaker "
                                  f"detection for the rest of this run; the robot falls back to "
                                  f"answering whoever it recognized first.")
                            self.mouth_detector = None
                            activities = []
                        with self._state_lock:
                            self._activities = activities

                    if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0 and people:
                        face_detections = self.embedder.detect(frame)
                        matches = match_faces_to_people(people, face_detections)
                        for tid, det in matches.items():
                            with self._state_lock:
                                self._last_embedding_by_track[tid] = det.embedding
                                self._face_bbox_by_track[tid] = det.bbox
                                match = self.face_store.find_match(det.embedding)
                                if match:
                                    self.face_store.reinforce(match, det.embedding)
                                    self._resolved_names[tid] = match.name
                                    self._resolved_person_ids[tid] = match.person_id
                            if match and tid in pending_since:
                                pending_since.pop(tid)
                                ready_to_greet.add(tid)

                    now = time.monotonic()
                    for tid in [
                        t for t, since in pending_since.items() if now - since >= RECOGNITION_WAIT_S
                    ]:
                        pending_since.pop(tid)
                        ready_to_greet.add(tid)

                    # Drop anyone who became "ready" but then walked off before a greeting fired.
                    ready_to_greet &= seen_ids

                    # Reap a finished visit before considering a new one.
                    if visit_thread is not None and not visit_thread.is_alive():
                        visit_thread.join()
                        visit_thread = None

                    # `visit_thread is None` is the real guard here - it's owned entirely by this
                    # thread, so it can't race. The greeting-state check is a secondary sanity
                    # check on shared state (the visit thread is what moves that state machine),
                    # kept because a mismatch between the two would indicate a genuine bug worth
                    # not papering over, not because it's load-bearing on its own.
                    if (
                        ready_to_greet
                        and visit_thread is None
                        and self.greeting_sm.state == RobotState.IDLE
                    ):
                        arriving_ids = sorted(ready_to_greet)
                        with self._state_lock:
                            arriving = [
                                Person(track_id=tid, known_name=self._resolved_names.get(tid))
                                for tid in arriving_ids
                            ]
                        ready_to_greet.clear()
                        visit_thread = threading.Thread(
                            target=self._run_visit, args=(arriving,), daemon=True
                        )
                        visit_thread.start()

                    self._draw_overlay(frame)
                    cv2.imshow("Stage 08 - live robot (press q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\nStopped (Ctrl+C).")
        finally:
            # Give an in-flight visit a moment to finish speaking rather than cutting the audio
            # off mid-word, but don't hang the shutdown on it if it's stuck waiting for a reply.
            if visit_thread is not None and visit_thread.is_alive():
                print("Waiting for the current conversation to wrap up...")
                visit_thread.join(timeout=5.0)
            self.cap.release()
            cv2.destroyAllWindows()
            if self.mouth_detector is not None:
                self.mouth_detector.close()
            if self.db is not None:
                self.db.close()

    def _draw_overlay(self, frame) -> None:
        """Draws who the robot currently thinks is talking to it, so the speaker detection is
        visible while testing instead of being an invisible internal decision. Reads a snapshot
        under the lock and does the drawing outside it - drawing is comparatively slow and must
        not hold up the interaction thread."""
        with self._state_lock:
            activities = list(self._activities)
        if not activities:
            return
        speaker = self.mouth_detector.most_likely_speaker(activities) if self.mouth_detector else None
        for activity in activities:
            x, y, w, h = activity.bbox
            is_speaker = speaker is not None and speaker.face_id == activity.face_id
            color = (0, 255, 0) if is_speaker else (160, 160, 160)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if is_speaker:
                cv2.putText(frame, "talking to me", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


if __name__ == "__main__":
    # --refresh forces a fresh lincoln.edu.my scrape even if the on-disk knowledge-base cache is
    # still within KB_CACHE_MAX_AGE_S - use it after the website has changed (or after editing
    # the scraper) instead of hunting down and deleting the cache file by hand.
    force_refresh = "--refresh" in sys.argv
    if force_refresh:
        print("--refresh given: ignoring the knowledge-base cache and re-scraping now.")
    orchestrator = RobotOrchestrator(force_refresh=force_refresh)
    orchestrator.run()
