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

On the first run (or when the cache has expired) startup fetches content from the university
website (Stage 07) before the camera loop starts, which takes about a minute. Later launches use
the cached knowledge base and refresh it in the background.

Threading: the camera and all
perception run continuously on the main thread, while each visit - greeting, listening,
thinking, speaking - runs on its own short-lived thread (_run_visit). Previously all of that ran
inline in the camera loop, so the video feed froze for the entire conversation: the robot was
blind between "hello" and "goodbye", and the preview window stopped responding. (That rewrite
also claimed to stop the robot "talking to people who had already walked away" - it did not.
It made the camera keep SEEING them leave, but nothing on the conversation side ever read that,
so the robot still finished its answer to an empty room. That half was finished on 2026-09-05 -
see ABSENCE_GRACE_S and _abandon_if_alone.) The split is deliberate about which side gets what:
OpenCV's
window and capture stay on the main thread (the portable, supported way to use them), and the
blocking audio/LLM work moves off it. Shared state between the two is small and explicitly
locked - see _state_lock in __init__.

Persistent memory: enrolled faces and full conversation transcripts live in
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

import logging
import queue
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

STAGES_DIR = Path(__file__).resolve().parent.parent
for stage_folder in [
    "stage00_setup",
    "stage01_presence_detection",
    "stage02_face_recognition",
    "stage03_proactive_greeting",
    "stage04_speech_to_text",
    "stage05_nlu_dialogue",
    "stage06_text_to_speech",
    "stage07_data_pipeline",
    "stage08_orchestrator",
    "stage09_deployment",
]:
    sys.path.append(str(STAGES_DIR / stage_folder))

import visit_memory
from addressee import (
    AddresseeJudge,
    FaceSample,
    LipHistory,
    find_wake_word,
    wake_phrases,
)
from audio_io import EchoCanceller, MicCapture, device_name, resolve_device
from audio_player import AudioPlayer
from conversation_state import ConversationStateMachine, ConvState
from dialogue_manager import DialogueManager
from dialogue_manager_live import SAMPLE_CHUNKS, _build_answer_generator
from engagement import EngagementTracker
from face_display import FaceDisplay, look_x_for
from face_memory import RETENTION_DAYS, FaceMemoryStore
from face_presence import FacePresenceDetector
from filler_bank import BACKCHANNELS, FillerBank
from greeting_live import match_faces_to_people
from greeting_state_machine import (
    ExpressionController,
    GreetingStateMachine,
    Person,
    RobotState,
)
from identity import get_identity
from identity import use_database as load_identity
from indexer import UniversityDataIndexer
from knowledge_base import KnowledgeBase
from logging_config import LatencyTracker
from model_paths import describe as describe_models
from mouth_activity import MouthActivityDetector
from presence_detector_live import PerTrackPresenceTracker, detect_people
from realtime_config import RT
from robot_db import connect_or_none
from scraper import REAL_SEED_URLS, UniversityScraper
from sface_embedder import SFaceEmbedder
from stt_listener import STTListener
from stt_listener_live import (
    BARGE_IN_ENABLED,
    BARGE_IN_MARGIN,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SMART_TURN_ENABLED,
    BargeInMonitor,
    PatientVAD,
    StreamingVAD,
    _smart_turn_model,
    build_stt_engine,
    calibrate_echo_floor,
    listen_for_one_utterance,
    real_hotwords,
    real_initial_prompt,
    real_vocabulary,
)
from tts_speaker_live import LiveTTSSpeaker
from wake_word import create_detector as create_wake_detector


def rt_log(tag: str, message: str) -> None:
    """One tagged line of the real-time trace - see stage04's identical helper. The existing
    per-stage prints ([visit], [dialogue], [stt]...) are untouched; this is the separate,
    switchable trace of what the CONVERSATION is doing."""
    if RT.REALTIME_DEBUG:
        log.info(f"[{tag}] {message}")


# Same reasoning as stage02/stage03: face embedding is more expensive than plain person
# tracking, so it only runs a few times per second rather than on every frame.
#
# A TIME interval rather than "every 5th frame" (changed 2026-09-13, after measuring this
# machine). Every-5-frames silently means something different on every machine, because it is
# really "every 5 x however long a frame takes". Measured here: YOLOv8n tracking on CPU-only
# torch costs ~170ms per frame, so the camera loop runs at about 5fps and every-5-frames put
# recognition a FULL SECOND apart. The vote below then needs IDENTITY_VOTE_REQUIRED=3 separate
# recognitions, i.e. ~3 seconds, while RECOGNITION_WAIT_S gives up after 2.5 - so on this
# hardware a face could match perfectly on every attempt and still never be named in time. The
# robot greeted a known visitor as a stranger, and it looked exactly like the face memory not
# working.
#
# Expressed as an interval, the cadence no longer depends on the frame rate: ~0.2s is four to
# five recognitions inside RECOGNITION_WAIT_S with room to spare for a couple of bad frames,
# and on hardware fast enough for it to matter it caps the work instead of running flat out.
# YuNet+SFace measured ~45ms per pass at 640x480, next to YOLO's ~170ms, so this is a small
# addition to a frame that is already paying for person tracking.
RECOGNIZE_INTERVAL_S = 0.2

# How many recent recognition attempts vote on who a tracked person is, and how many of them have
# to agree before the robot will use a name - see _vote_identity(). At RECOGNIZE_INTERVAL_S=0.2
# a five-deep window fills in about a second, which fits comfortably inside the
# RECOGNITION_WAIT_S pause that already exists before greeting. Keep that relationship in mind
# if you change either number: IDENTITY_VOTE_REQUIRED x RECOGNIZE_INTERVAL_S is the FLOOR on how
# long it can possibly take to name somebody, and if that floor rises above RECOGNITION_WAIT_S
# then no visitor is ever recognized in time, however well their face matches.
#
# 3-of-5 rather than a bare majority of whatever has arrived: it tolerates two bad frames out of
# five, which is about what a head turn or a moment of backlight costs in practice.
IDENTITY_VOTE_WINDOW = 5
IDENTITY_VOTE_REQUIRED = 3

# How long a face box stays drawable after the detection that measured it - see _draw_overlay.
# Comfortably longer than the gap between two recognition passes (so an ordinary box does not
# flicker), and short enough that a box whose face has genuinely gone stops being drawn while
# the person is still in the room rather than lingering as a second box beside them.
FACE_BOX_MAX_AGE_S = 0.8

# How often, at most, to report a face being skipped for poor quality. This is the one message
# that explains a visitor never being recognized, so it has to be visible - but the condition
# lasts as long as somebody stands too far away, so it also has to be rate-limited rather than
# printed on every pass.
QUALITY_LOG_INTERVAL_S = 4.0

# How much two boxes have to overlap before they are taken to be the same face - see
# _boxes_overlap(). Generous on purpose: the two detectors involved (YuNet and mediapipe)
# frame a face differently, YuNet tightly around the features and mediapipe wider, so genuine
# agreement about one face shows up as a partial overlap rather than a close one.
SAME_FACE_IOU = 0.25

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

# How often the background refresher wakes up to rebuild the knowledge base - see
# _refresh_knowledge_base(). Six hours matches the interval STAGES.md's own success criterion
# names ("university data updates reflected in the knowledge base within the scheduled interval,
# e.g. every 6 hours, with no manual intervention"), which until now nothing implemented: the
# only refresh that existed was the cache expiring and forcing a blocking rescrape at the next
# restart, which is not a schedule and needs somebody to restart the robot.
KB_REFRESH_CHECK_INTERVAL_S = 6 * 60 * 60

# How often the refresh thread re-checks whether the robot has finished with a visitor, when it
# is holding off for one - see _wait_until_not_serving_anybody(). Short enough that a refresh
# starts promptly once the lobby empties, long enough to cost nothing while it waits.
KB_REFRESH_YIELD_POLL_S = 2.0

# Safety cap on how many questions one visit will loop through - see _handle_conversation().
# Without this, a visitor who keeps saying "yes" to "any other questions?" (or a misheard "yes"
# on a noisy mic) could loop indefinitely; this bounds one visit to a reasonable real
# conversation length instead.
MAX_QUESTIONS_PER_VISIT = 6

# How many times in one visit the robot will ask somebody to repeat themselves before it stops
# trying and just answers whatever it heard. Two is roughly what a person tolerates: a third
# "sorry, one more time?" stops reading as attentiveness and starts reading as a broken machine,
# and in a genuinely noisy lobby the transcript is not going to improve on attempt four either.
MAX_REPAIRS_PER_VISIT = 2

# How many extra chances a visitor gets when they clearly SPOKE but the transcription came back
# completely empty - see _listen_with_retry(). Separate from MAX_REPAIRS_PER_VISIT, which is
# about a transcript that arrived garbled: this is about one that did not arrive at all, which
# needs a plain "sorry, could you say that again" rather than a question about the part that was
# heard, because no part was heard. One extra chance, deliberately: if two consecutive
# recordings transcribe to nothing the microphone or the load is the problem, and a third
# prompt just makes the robot look broken more slowly.
MAX_EMPTY_LISTEN_RETRIES = 1

# How many 30ms VAD frames the microphone callback collects per call. This is really a buffer
# depth: PortAudio's input buffer on this host API is exactly ONE block, so asking for one frame
# at a time (which is what this did) means any stall longer than 30ms loses microphone audio
# outright - and a real run logged "[audio] input overflow" doing exactly that, mid-question,
# while the knowledge-base refresh was saturating the CPU. Measured here: blocksize 480 gives a
# 30ms cushion, 1440 gives 90ms, 4800 gives 300ms.
#
# Three frames is a 3x cushion for ~90ms of added input latency, which the VAD does not care
# about at the resolution it works at. Passing `latency="high"` instead does nothing at all on
# this host API (measured - the stream still reports 30ms), so the block size is the only lever.
# Raise it if "input overflow" still appears; the cost is latency before the VAD sees speech.
MIC_BLOCK_FRAMES = 3

# How long to wait for an in-flight conversation to stop when the robot is closing - see
# _wind_down(). It only has to cover finishing the current audio chunk and unwinding, because
# _shutting_down already stops it beginning anything new; it does NOT have to cover the rest of
# a sentence or an API call in flight.
SHUTDOWN_JOIN_TIMEOUT_S = 5.0

# Frame size to request from the camera - see where it is applied in run() for the measurements
# behind it. The short version: at 640x480 a visitor sitting at a normal distance produced faces
# too small for the recognizer to accept, so nobody was ever recognized.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Whether to play a short spoken acknowledgment ("Let me see.", "Good question.") the instant a
# question ends, while the real answer is still being retrieved and generated - see stage06's
# filler_bank.py for the full argument for them.
#
# OFF, on explicit instruction after hearing it in use: the robot should give the
# answer, not preface it. Worth recording what this costs, because it is not free and the
# measurements are in this project's own logs - time-to-first-word for a grounded answer runs
# 2.7-4.2s on this hardware, and the filler existed to make that stretch feel like the robot had
# started responding rather than like it had frozen. With it off, that is now plain silence.
#
# Nothing else changes: _answer_one() already treats a missing bank as "no acknowledgment", and
# the background answer priming that used to overlap the filler (see _prime_stream) still runs,
# so this costs perceived snappiness only, never real latency. Set back to True to restore them;
# filler_bank.py is left intact and still has its own self-test.
#
# 2026-10-27: now FILLER_MODE in .env, and the default ("auto") keeps the spirit of that
# instruction - a fast answer is NEVER prefaced. Only when the first words of the answer are
# still not ready FILLER_DELAY_MS (700ms) after the visitor stopped talking does a short
# "Mm, okay." fill the gap, because three seconds of dead air reads as a frozen robot.
# FILLER_MODE=off restores complete silence; "always" is the old behaviour.
ACKNOWLEDGMENT_FILLERS_ENABLED = RT.FILLER_MODE != "off"

# --- added 2026-10-27 -----------------------------------------------------------------------------
# The quiet "mm-hm" while a visitor pauses mid-sentence: only after they have been talking for a
# while (a two-word question needs no encouragement), and never twice in quick succession.
BACKCHANNEL_MIN_SPEECH_S = 2.5
BACKCHANNEL_MIN_GAP_S = 6.0
BACKCHANNEL_VOLUME = 0.6

# How long the robot keeps a visit going for somebody who woke it by voice (openWakeWord) but is
# not visible to the camera - they called it from the side or from behind. Renewed every time
# they say something to it.
VOICE_PRESENCE_S = 15.0

# Barge-in with echo cancellation: the residual echo can be close to nothing, and a threshold of
# "2.5 x nothing" would fire on room noise. This is the floor under the threshold (-42 dBFS,
# well under normal speech at a metre).
MIN_AEC_BARGE_IN_RMS = 0.008
# With echo cancellation, an echo reading above this is a disturbed measurement, not the room -
# _calibrate_barge_in() measures once more (converged echo is ~0.002, the noise floor).
AEC_FLOOR_RECHECK = 0.005

# Lip movement lags the audio by a few camera frames (mouth_activity's rolling window), so the
# window the addressee check reads is stretched by this much at the end.
LIP_WINDOW_TAIL_S = 0.4

# How many dialogue turns of this visit to keep for follow-up questions ("how long is it?" after
# asking about a programme - see dialogue_manager's _contextualize/_history_block). Both sides
# count, so this is roughly three exchanges: enough for a pronoun to still have a referent, short
# enough that an early unrelated question doesn't drag on the retrieval for a later one.
VISIT_HISTORY_TURNS = 6

# How long the robot keeps talking after the last time it saw anybody, before deciding the visit
# is over.
#
# This closes a real gap. The camera loop has tracked who is present on every frame since the
# threading rewrite, and _seen_ids was written under the lock every frame - but the conversation
# thread never read it. main_live's own module docstring claims that rewrite fixed the robot
# "talking to people who had already walked away"; it only fixed the other half of that bug (the
# frozen video feed). The robot would still deliver a full answer, ask "anything else?", and wait
# out the timeout facing an empty room.
#
# The grace period is what makes it safe to act on: face/person detection drops a frame
# constantly - someone turns their head, steps sideways, is briefly occluded - so reacting to the
# first empty frame would abandon people mid-question. Several seconds of continuous absence is a
# person who has actually gone.
ABSENCE_GRACE_S = 4.0

# How much has to be LEFT of an utterance, once the control phrases are stripped out of it, for
# it to count as a real question rather than an interruption - see _detect_control_intent().
# "Wait, how long is the Master of Computer Science" leaves eight words and a "how"; "no no,
# that's not what I meant" leaves "that is".
CONTROL_REMAINDER_MIN_WORDS = 4

# Words that mark the remainder as an actual question rather than a fragment.
_QUESTION_CUES = frozenset(
    {
        "what",
        "where",
        "when",
        "how",
        "who",
        "why",
        "which",
        "whose",
        "can",
        "could",
        "do",
        "does",
        "is",
        "are",
        "tell",
        "show",
        "explain",
        "about",
    }
)

# How many times in one visit the robot will accept "wait"/"never mind" before it stops treating
# them as control and just answers whatever it heard. Without a cap, a visitor who keeps saying
# "no no" (or a mic that keeps mishearing something as it) never gets an answer at all.
MAX_INTERRUPTIONS_PER_VISIT = 4

# STAGES.md's own success criterion: "time from the end of a question to the start of the spoken
# answer: < 2-3 seconds". Written down there since the project started and never measured until
# now - see _report_turn_latency(). Taking the strict end of the stated range deliberately: a
# target you round in your own favour stops being a target.
ANSWER_LATENCY_TARGET_MS = 2000.0

# And the other one: "time from a person entering to the start of the greeting: < 1 second".
GREETING_LATENCY_TARGET_MS = 1000.0

# Checked as substrings against the whole cleaned utterance, not as an exact-match word set -
# found from a real run that a plain word-by-word check gets this wrong two
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
    "no thanks",
    "no thank you",
    "not interested",
    "no way",
    "nah",
    "nope",
    "don't",
    "dont",
    "do not",
    "not really",
    "rather not",
    "that's all",
    "thats all",
    "that is all",
    "that's it",
    "thats it",
    "that is it",
    "i'm good",
    "im good",
    "no more",
    "nothing else",
    "all good",
    "i'm done",
    "im done",
)
AFFIRMATIVE_PHRASES = (
    "no problem",
    "why not",
    "of course",
    "sure",
    "yeah",
    "yeh",
    "yes",
    "yep",
    "yup",
    "okay",
    "ok",
    "alright",
    "please",
    "definitely",
    "sounds good",
    "sounds great",
    "go ahead",
    "absolutely",
)


# --- what the robot says between answers -------------------------------------------------------
# Every one of these used to be a single hard-coded string, said identically every time. Within
# one visit the robot asks "do you have any other questions?" after EVERY answer, so a visitor
# asking three things heard the exact same sentence three times - which is the moment a
# conversation stops sounding like a conversation. Same information, several ways of saying it.
#
# These are the robot's own connective tissue, not answers, so they stay hard-coded rather than
# generated: they must be instant (no model call between an answer and the next question), they
# must never hallucinate, and they must work with no API key and no network at all.
_MORE_QUESTIONS = (
    "Do you have any other questions?",
    "Anything else I can help with?",
    "Was there anything else?",
    "Anything else you wanted to ask?",
    "Can I help with anything else?",
)

_GOODBYE_AFTER_HELP = (
    "Alright, glad I could help!",
    "Happy to help - take care!",
    "No problem at all. Good luck!",
    "Anytime! Have a good one.",
)

_NOTHING_ASKED = (
    "No worries, I'm here if you need anything.",
    "That's alright - I'm here if you think of something.",
    "No problem. Just come back if you need anything.",
)

_OUT_OF_QUESTIONS = (
    "I'll let you go for now - come find me again if you think of anything else!",
    "Let's leave it there for now - come back any time.",
    "I'd better let you get on. Come find me if anything else comes up!",
)

# Said the moment a question is heard but before it can possibly be answered - see
# _handle_repair(). A visitor whose question was misheard gets asked about the part that WAS
# heard, rather than a flat "could you repeat that", which is what a person does and which
# usually gets the answer on the second try instead of the third.
_REPAIR_WITH_CONTEXT = (
    "Sorry, I only caught something about {topic} - what did you want to know?",
    "I think I heard {topic}, but I missed the rest - could you say that again?",
    "Sorry, I got {topic} but not the whole question. What was it?",
)

_REPAIR_NO_CONTEXT = (
    "Sorry, I didn't catch that - could you say it again?",
    "I missed that, sorry. One more time?",
    "Sorry, could you repeat that for me?",
)


# Remembers the last phrasing used from each group above, so _pick() can avoid repeating it.
# Module-level rather than a default argument: a mutable default is the same object for the life
# of the process, which is exactly the behaviour wanted here, but it reads like the classic
# Python bug and would invite somebody to "fix" it.
_LAST_PICKED: dict[tuple[str, ...], str] = {}


def _pick(options: tuple[str, ...]) -> str:
    """Chooses one phrasing at random, never repeating the previous choice from the same group.

    Random alone is not enough: with five options, saying the same one twice in a row happens
    one time in five, and back-to-back repetition is precisely the thing these lists exist to
    remove - a visitor hearing "Anything else?" twice running notices it more than they would
    have noticed one fixed string. Keyed on the group itself, so each list tracks its own last
    choice independently.
    """
    import random

    if len(options) == 1:
        return options[0]
    choices = [o for o in options if o != _LAST_PICKED.get(options)]
    chosen = random.choice(choices)
    _LAST_PICKED[options] = chosen
    return chosen


# --- interrupting the robot / changing course ---------------------------------------------------
# Everything a visitor says was previously treated as a question. So "wait", "no no, that's not
# what I meant" and "actually, let me ask something else" were all sent to the knowledge base as
# if they were questions about the university, retrieved nothing, and came back as the fixed
# apology - which is the exact opposite of what the person asked for. A receptionist you cannot
# interrupt or redirect is not having a conversation with you.
#
# Two distinct things, because they need different responses:
#
#   STOP    - "wait", "stop", "hold on", "one second". They want the robot to stop talking and
#             listen. The robot goes quiet and waits, and does NOT lose the thread of what was
#             being discussed.
#   CHANGE  - "no no", "never mind", "forget it", "that's not what I meant", "let me ask
#             something else". They are abandoning the current topic. The robot acknowledges AND
#             DROPS the conversation history, because the whole point of follow-up context (see
#             dialogue_manager's _contextualize) becomes actively harmful once the subject has
#             been abandoned: the next question would otherwise be searched against the topic
#             they just rejected.
#
# Matched as substrings on the cleaned utterance, same approach and same reasoning as
# NEGATIVE_PHRASES/AFFIRMATIVE_PHRASES above - Whisper punctuates, and these arrive as real
# spoken fragments rather than tidy keywords.
_STOP_PHRASES = (
    "wait",
    "hold on",
    "hang on",
    "stop",
    "one second",
    "one moment",
    "just a second",
    "give me a second",
    "pause",
    "shut up",
    "be quiet",
    "let me finish",
    "let me speak",
)

_CHANGE_PHRASES = (
    "no no",
    "never mind",
    "nevermind",
    "forget it",
    "forget that",
    "scratch that",
    "not what i meant",
    "not what i asked",
    "that's not it",
    "thats not it",
    "let me change",
    "change the question",
    "different question",
    "another question",
    "something else",
    "ask something else",
    "start over",
    "start again",
    "go back",
    "i meant",
    "actually i",
    "instead",
)

# Said back when interrupted. Short by design - the visitor stopped the robot BECAUSE it was
# talking, so the response to that cannot be another speech.
_STOP_ACK = (
    "Sorry - go ahead.",
    "Okay, I'm listening.",
    "Sure, go on.",
    "Of course - what is it?",
)

_CHANGE_ACK = (
    "No problem - what would you like to know?",
    "Sure, let's start again. What can I help with?",
    "Okay, forget that one. What did you want to ask?",
    "My mistake - go ahead.",
)


# Said when somebody says only the robot's name ("Nova?"), or wakes it from across the room.
_WAKE_REPLIES = (
    "Yes?",
    "Yes? How can I help?",
    "I'm here - what can I do for you?",
)

# Said, once, to somebody who walks up while the robot is busy with another visitor - the
# glance-and-nod a receptionist gives so the second person knows they have been seen.
_NEWCOMER_ACK = (
    "Hi! I'll be with you in just a moment.",
    "Hello! I'll be right with you.",
)


def _fixed_lines() -> list[str]:
    """Every line the robot says word for word, for LiveTTSSpeaker.prewarm(). Greetings that
    contain a visitor's name are left out - those can only be synthesized once the name is
    known."""
    from greeting_state_machine import _UNKNOWN_GROUP, _UNKNOWN_SINGLE

    lines: list[str] = []
    for group in (
        _UNKNOWN_SINGLE,
        _UNKNOWN_GROUP,
        _MORE_QUESTIONS,
        _GOODBYE_AFTER_HELP,
        _NOTHING_ASKED,
        _OUT_OF_QUESTIONS,
        _STOP_ACK,
        _CHANGE_ACK,
        _REPAIR_NO_CONTEXT,
        _WAKE_REPLIES,
        _NEWCOMER_ACK,
    ):
        lines.extend(group)
    lines += [
        "I don't recognize you. Would you like me to remember you for next time?",
        "No problem, I won't remember this visit.",
        "Great! What's your name?",
    ]
    return lines


def _detect_control_intent(text: str | None) -> str | None:
    """Reads an utterance as an interruption/redirection rather than a question.

    Returns "stop", "change", or None. CHANGE is checked first: "no no, wait, that's not what I
    meant" contains both, and the stronger signal is that the topic is being abandoned, not
    merely paused.

    The hard part is that a control word can also just be the opening of a real question.
    "Wait, how long is the Master of Computer Science?" must be answered, not acknowledged with
    "okay, I'm listening" - that would throw the question away at the moment it was finally
    asked clearly. A plain word-count cap does not separate these: it is eight words either way,
    and it wrongly rejected "no no, that's not what I meant", which is precisely the phrasing
    this feature exists for.

    So the test is what SURVIVES the control phrases. Strip them out, and if what remains is a
    real question - long enough to carry content, and shaped like one - the utterance is a
    question with a filler on the front. If what remains is a fragment ("that is"), it was an
    interruption.
    """
    if not text:
        return None
    cleaned = _clean_for_phrase_match(text)
    if not cleaned:
        return None

    matched = [p for p in _CHANGE_PHRASES if p in cleaned]
    kind = "change"
    if not matched:
        matched = [p for p in _STOP_PHRASES if p in cleaned]
        kind = "stop"
    if not matched:
        return None

    remainder = cleaned
    for phrase in matched:
        remainder = remainder.replace(phrase, " ")
    words = remainder.split()
    if len(words) >= CONTROL_REMAINDER_MIN_WORDS and any(cue in words for cue in _QUESTION_CUES):
        return None  # a real question wearing a control word as a prefix
    return kind


def _prime_stream(pieces):
    """Starts producing the first piece of an answer stream on a background thread, and returns
    an iterator that replays it followed by the rest.

    Without this the acknowledgment filler covers the wrong thing. `handle_question_stream()`
    returns a LAZY generator: the knowledge-base search has happened by then, but not one token
    of generation - that only starts when something consumes the iterator. So a caller that
    plays a filler, waits for it, and only then starts speaking has overlapped the filler with
    retrieval alone (a few hundred milliseconds) and left the expensive part, generating the
    first sentence, to happen in silence afterwards.

    Pulling one item on a thread makes the model generate its first sentence WHILE the
    acknowledgment is playing, which is what makes the two features add up instead of running
    one after the other.

    Exceptions don't need special handling here: _stream_grounded() already catches its own
    failures and yields the fallback answer, so this thread either gets a piece or gets
    StopIteration.
    """
    return _PrimedStream(pieces)


class _PrimedStream:
    """The iterator _prime_stream() returns. `ready` is set the moment the first piece exists
    (or the stream turned out empty / failed) - that is what lets _answer_one() play an
    acknowledgment ONLY when the answer is slow (FILLER_MODE=auto) instead of every time."""

    def __init__(self, pieces):
        self._pieces = pieces
        self._box: dict = {}
        self.ready = threading.Event()
        self._started = time.monotonic()
        self._iter = None
        self._thread = threading.Thread(target=self._pull, name="answer-priming", daemon=True)
        self._thread.start()

    def _pull(self) -> None:
        try:
            self._box["first"] = next(self._pieces)
            rt_log(
                "LLM",
                f"first token received after " f"{(time.monotonic() - self._started) * 1000:.0f}ms",
            )
        except StopIteration:
            self._box["empty"] = True
        except Exception as exc:  # noqa: BLE001 - handed to the consumer, not lost here
            self._box["error"] = exc
        finally:
            self.ready.set()

    def _replay(self):
        self._thread.join()
        if "error" in self._box:
            raise self._box["error"]
        if "empty" in self._box:
            return
        yield self._box["first"]
        yield from self._pieces

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = self._replay()
        return next(self._iter)


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


_GOODBYE_WORDS = ("bye", "goodbye", "good bye", "thanks", "thank you", "see you", "cheers")
GOODBYE_MAX_WORDS = 6


def _sounds_like_goodbye(text: str | None) -> bool:
    """True for a short "no thanks, that's all" / "ok thank you, bye" said where a question was
    expected - most often right after the visitor interrupted an answer they had heard enough
    of. Stricter than _is_clear_negative(): it has to be short, and once the goodbye words are
    taken out, nothing question-like may be left ("thanks, and where is the library?" is a
    question). Silence is NOT a goodbye here (2026-10-27, found by the end-to-end test: the
    robot answered "No thanks, that's all" as if it were a question)."""
    if not text:
        return False
    cleaned = _clean_for_phrase_match(text)
    if not cleaned or len(cleaned.split()) > GOODBYE_MAX_WORDS:
        return False
    matched = [p for p in NEGATIVE_PHRASES + _GOODBYE_WORDS if p in cleaned]
    if not matched:
        return False
    remainder = f" {cleaned} "
    for phrase in sorted(matched, key=len, reverse=True):
        remainder = remainder.replace(phrase, " ")
    return not any(word in _QUESTION_CUES for word in remainder.split())


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
    "my name is ",
    "i am ",
    "i'm ",
    "im ",
    "it's ",
    "its ",
    "it is ",
    "call me ",
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
            stripped = stripped[len(prefix) :]
            break
    name = stripped.strip().strip(".!?").title()
    return name or "Guest"


class _ShuttingDown(Exception):
    """Raised inside a background refresh to abandon it when the robot is closing down.

    An exception rather than a return value because the abandon point is inside a callback
    handed to the scraper, several frames below the code that can act on it - see _scrape_into().
    """


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], min_iou: float = SAME_FACE_IOU
) -> bool:
    """Whether two (x, y, w, h) boxes overlap enough to be the same face.

    Plain intersection-over-union. Used by _draw_overlay to avoid drawing a second rectangle
    over a face that already has one - see the note there about the two detectors involved.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap = ix * iy
    if overlap == 0:
        return False
    union = aw * ah + bw * bh - overlap
    return union > 0 and overlap / union >= min_iou


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
        log.info(f'[TTS] "{text}"')
        if self.on_spoken is not None:
            self.on_spoken(text)
        self.speaker.speak(text)


class RobotOrchestrator:
    """The real robot: camera in, real recognition + greeting, real spoken questions answered
    from the real knowledge base, spoken back.

    Two threads (see run() and _run_visit()) - perception runs continuously while conversation
    happens alongside it. See the module docstring for that split, for how active speaker
    detection picks who's talking, and for what's still deliberately simplified."""

    def __init__(self, force_refresh: bool = False, skip_knowledge_base: bool = False):
        # Which optional models are on disk - one line each, so a missing download is obvious
        # at startup instead of showing up later as "the voice sounds old" or "it is slow".
        for label, _path, present in describe_models():
            log.warning(
                f"  {'OK     ' if present else 'missing'}  {label:<22} "
                f"{'' if present else '(run stages/stage00_setup/download_models.py)'}"
            )

        # Presence: faces by default - YuNet on a 640px copy of the frame, ~10-20ms,
        # no PyTorch - instead of the YOLOv8 body tracker (~170ms per frame on a laptop CPU, ~2GB
        # of PyTorch, AGPL-3.0). A reception robot only needs the people whose face it can see,
        # and the face landmarks are what the "are they facing me?" gate needs anyway.
        # PRESENCE_BACKEND=yolo brings the old tracker back.
        self.yolo = None
        self.face_presence = None
        if RT.PRESENCE_BACKEND == "face":
            try:
                self.face_presence = FacePresenceDetector()
                log.info("Presence: faces (YuNet) - no PyTorch needed.")
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[presence] face presence unavailable ({exc!r}) - trying YOLO.")
        if self.face_presence is None:
            from ultralytics import YOLO  # optional: pulls in PyTorch

            log.info("Loading YOLOv8n person tracker...")
            self.yolo = YOLO(str(STAGES_DIR / "stage01_presence_detection" / "yolov8n.pt"))
        log.info("Loading YuNet + SFace face recognition...")
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

        # score= is what lets find_match() pick the BEST candidate instead of the first
        # one over the threshold, and refuse to choose between two people it cannot tell
        # apart - see FaceMemoryStore.find_match(). Without it the store silently falls
        # back to the old first-match behaviour.
        self.face_store = FaceMemoryStore(
            is_match=self.embedder.is_match, score=self.embedder.similarity, db=self.db
        )
        if self.db is not None:
            log.info(f"Face memory: {len(self.face_store)} person(s) loaded from MySQL.")
            self._purge_expired_people()

        self.presence = PerTrackPresenceTracker()

        # --- cross-thread state -----------------------------------------------------------
        # The camera/perception loop (main thread, run()) writes these; the conversation thread
        # (_run_visit) reads them, and writes to face_store when registering someone new. Every
        # touch of the group below goes through _state_lock: without it, a visitor being
        # registered mid-conversation could race a face lookup happening on the same dicts one
        # frame later. The lock is only ever held for short dictionary operations, never across
        # a blocking call (speaking, listening, an API request), so it can't stall the camera.
        # Set when the robot is closing down. The conversation and knowledge-base threads are
        # daemons, so without this they keep going right through teardown: in a real run, 'q'
        # was pressed mid-visit, the mic stream closed as the camera loop's `with` block
        # exited, and the visit thread then tried to speak into a closed device -
        # PortAudioError('Unanticipated host error', ... 'Cannot perform this operation while
        # media data is still playing'). It then kept printing while the interpreter was
        # finalizing, which is the "Fatal Python error: _enter_buffered_busy ... possibly due
        # to daemon threads" that ended the run. Both threads check this before starting
        # anything new. See _wind_down().
        self._shutting_down = threading.Event()
        self._state_lock = threading.Lock()
        self._resolved_names: dict[int, str | None] = {}
        # Kept alongside _resolved_names rather than looked up from the name when needed: names
        # are not unique (two students called Ahmad) and not stable (a re-enrollment), so the
        # person_id is the only safe thing to attach a stored conversation to.
        self._resolved_person_ids: dict[int, str] = {}
        self._last_embedding_by_track: dict[int, object] = {}
        self._face_bbox_by_track: dict[int, tuple[int, int, int, int]] = {}
        # When each of those boxes was last measured. Recognition only runs a few times a
        # second, so a box is always a little older than the frame it is drawn on; this is what
        # lets the overlay drop one that has gone properly stale (a head turned away, a face
        # that stopped being detected) instead of leaving it hanging in the air where the person
        # used to be. See _draw_overlay.
        self._face_bbox_at: dict[int, float] = {}
        # Recent recognition answers per track (person_id or None), for _vote_identity().
        self._identity_votes: dict[int, list[str | None]] = {}
        # The most recent FaceDetection per track, kept for the overlay so the operator can see
        # WHY a face is being ignored (too small / too blurry) instead of guessing.
        self._face_quality_by_track: dict[int, object] = {}
        self._seen_ids: set[int] = set()
        self._activities: list = []  # latest MouthActivity list, refreshed every frame

        # --- conversation recording -------------------------------------------------------
        # Only ever touched from the visit thread, and only one visit runs at a time (see the
        # module docstring), so these need no lock - unlike the perception state above.
        self._conversation_id: int | None = None
        self._conversation_person_id: str | None = None
        self._turn_index = 0
        # This visit's dialogue so far, as (speaker, text) - what makes follow-up questions
        # resolvable. Reset per visit in _begin_conversation.
        self._visit_history: list[tuple[str, str]] = []
        # RMS above which a sound during playback counts as somebody talking over the robot.
        # None = barge-in off (also what a failed or unusable calibration falls back to).
        self._barge_in_threshold: float | None = None
        # Words the visitor interrupted the robot with, carried into the next listen so they
        # don't have to say them twice - see BargeInMonitor.captured_frames in stage04. Written
        # by _speak_interruptible on the visit thread and consumed by the next _listen() on the
        # same thread, so no lock.
        self._carried_audio: list[bytes] = []
        # Set by _answer_one for the duration of one answer, so the latency report can time the
        # gap the visitor actually experiences (question ends -> first word out) rather than how
        # long the answer took to say. None whenever no answer is in flight.
        self._first_audio_hook = None
        # The conversation's own state machine (stage08's conversation_state.py). Observational:
        # it records what the audio/dialogue/playback layers report and never gates them, so a
        # fault here can produce a wrong log line and cannot produce a wrong answer. It is what
        # makes "why did it answer me there?" answerable from a log.
        self.conv = ConversationStateMachine()
        # Set while building the knowledge base below; initialised here so the background
        # refresher can read it even in modes that skip the knowledge base entirely.
        self._kb_stale = False
        # Wall-clock of the last frame anybody was visible on. Written by the camera thread,
        # read by the conversation thread - see _visitor_present().
        self._last_seen_at = time.monotonic()

        # --- who is talking to the robot ---------------------------------------
        # Engagement: only greet people who are near, facing the robot and staying (stage08's
        # engagement.py). Lip history: per-frame "whose lips are moving, who is facing me",
        # timestamped, so the conversation thread can ask about the exact seconds a sentence was
        # heard (addressee.py). Both are written by the camera thread only.
        self.engagement = EngagementTracker(
            RT.ENGAGE_MIN_FACE_FRACTION, RT.ENGAGE_MAX_YAW, RT.ENGAGE_DWELL_S
        )
        self.lip_history = LipHistory()
        robot_name = get_identity().robot_name
        self.addressee = AddresseeJudge(
            mode=RT.ADDRESSEE_CHECK,
            min_lip_fraction=RT.ADDRESSEE_MIN_LIP_FRACTION,
            wake=wake_phrases(robot_name, RT.WAKE_WORDS) if RT.WAKE_WORD_ENGINE != "off" else [],
        )
        # When the current utterance started and when its last pause began (monotonic), from
        # stage04's turn events - see _on_turn_event. The speech window for the addressee check.
        self._speech_started_at: float | None = None
        self._speech_paused_at: float | None = None
        self._turn_finished_at: float | None = None
        self._visitor_stopped_at: float | None = None
        self._last_backchannel_at = 0.0
        self._prefetching = threading.Event()
        # A visit is running (or about to): the idle audio consumer stops reading the
        # microphone so the visit thread gets every frame. See _idle_audio_loop.
        self._visit_active = threading.Event()
        # Set by the idle wake-word listener; the camera loop starts a visit from it.
        self._wake_requested_at: float | None = None
        self._woken_visit = False
        self._voice_presence_until = 0.0
        # Skip the stale-audio drain on the next listen, once: after a wake word, the question
        # often follows in the same breath ("hey Nova, where is the library?").
        self._keep_backlog = False
        # People who arrived during somebody else's visit, waiting for a "be right with you".
        self._newcomers_waiting: set[int] = set()
        self._newcomers_acknowledged: set[int] = set()
        self._look_x = 0.0

        log.info("Loading mediapipe face mesh (active speaker detection)...")
        try:
            self.mouth_detector = MouthActivityDetector()
        except Exception as exc:
            # Optional capability: without it the robot still works exactly as it did before,
            # it just can't tell two simultaneous speakers apart. Not worth refusing to start
            # over - especially since mediapipe is the newest and least-proven dependency here.
            log.warning(
                f"[mouth] could not start active speaker detection ({exc!r}) - continuing "
                f"without it. Install it with: pip install -r "
                f"stages/stage02_face_recognition/requirements.txt"
            )
            self.mouth_detector = None

        # --- audio out and in -------------------------------------------------
        # Software echo cancellation sits between the two: every block the speaker plays is
        # handed to the canceller as the reference (AudioPlayer's far tap), and every block the
        # microphone hears is cleaned against it before anything else sees it (MicCapture). That
        # is what lets the robot listen WHILE it talks - barge-in, "mm-hm" - without hearing
        # itself. Off or missing (pip install livekit==1.0.0) means exactly the old behaviour.
        self.echo = EchoCanceller.create(RT.AEC_ENABLED)
        speaker_device = resolve_device(RT.SPEAKER_DEVICE, "output") if RT.SPEAKER_DEVICE else None
        log.info(f"Speaker: {device_name(speaker_device, 'output')}")
        self.player = AudioPlayer(
            device=speaker_device,
            far_tap_factory=self.echo.far_tap if self.echo is not None else None,
        )
        log.info("Loading the voice...")
        self.tts = LiveTTSSpeaker(player=self.player)

        # Pre-rendered "let me see..." acknowledgments, synthesized ONCE here rather than on
        # demand - the whole point is that playing one costs nothing at the moment it is needed
        # (see stage06's filler_bank.py). Optional: a failure here costs a bit of perceived
        # snappiness, not the ability to answer, so it degrades to None rather than refusing to
        # start - same rule as mediapipe above.
        self.fillers = None
        if ACKNOWLEDGMENT_FILLERS_ENABLED:
            try:
                self.fillers = FillerBank(self.tts)
            except Exception as exc:
                log.warning(
                    f"[filler] could not pre-render acknowledgments ({exc!r}) - continuing "
                    f"without them; answers will just start after a short silence."
                )
        # "Mm-hm." while somebody pauses mid-sentence. Only with echo cancellation (by default):
        # without it the microphone records the robot's own "mm-hm" into the visitor's question.
        self.backchannels = None
        if RT.backchannels_wanted(self.echo is not None):
            try:
                self.backchannels = FillerBank(
                    self.tts, phrases=BACKCHANNELS, verbose=False, volume=BACKCHANNEL_VOLUME
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    f"[filler] could not pre-render back-channels ({exc!r}) - skipping them."
                )

        self.expression = ExpressionController()
        self.greeting_sm = GreetingStateMachine(
            self.expression,
            _LiveTTSAdapter(self.tts, on_spoken=lambda text: self._record("robot", text)),
        )

        log.info("Loading speech recognition...")
        # Whisper by default (GPU when faster-whisper can really use one, else CPU int8); the
        # streaming Zipformer with STT_BACKEND=sherpa - see build_stt_engine in stage04.
        self.stt_engine = build_stt_engine(
            initial_prompt=real_initial_prompt(),
            vocabulary=real_vocabulary(),
            hotwords=real_hotwords(),
        )
        self.stt_listener = STTListener(self.stt_engine, PatientVAD(), locale="en-US")
        self.vad = StreamingVAD()
        if SMART_TURN_ENABLED:
            _smart_turn_model()  # load now, not on the first visitor's first pause
        self.audio_q: queue.Queue[bytes] = queue.Queue()

        # The microphone: 30ms frames onto audio_q, echo-cancelled when self.echo exists.
        # MIC_BLOCK_FRAMES is still the PortAudio buffer depth (see its note above). With
        # AUDIO_CAPTURE_PROCESS=true the device is read by a separate process, so heavy vision
        # or model work on this one can never make PortAudio drop microphone audio.
        mic_device = resolve_device(RT.MIC_DEVICE, "input") if RT.MIC_DEVICE else None
        self.mic_stream = MicCapture(
            self.audio_q,
            sample_rate=SAMPLE_RATE,
            frame_samples=FRAME_SAMPLES,
            block_frames=MIC_BLOCK_FRAMES,
            device=mic_device,
            canceller=self.echo,
            use_process=RT.AUDIO_CAPTURE_PROCESS,
        )

        # Say which microphone this is actually using. No device is named above, so it is
        # whatever Windows currently calls the default input - and that changes under you:
        # connecting Bluetooth earbuds switches it to their hands-free profile, which is 8kHz
        # narrowband, and Whisper transcribes that badly enough to look like a broken robot. The
        # device is the first thing to check when recognition is poor and the last thing anybody
        # thinks of, because nothing ever printed it. Purely informational - any failure here
        # must not stop the robot starting.
        try:
            import sounddevice as sd

            device = sd.query_devices(self.mic_stream.device, "input")
            log.info(
                f"Microphone: {device['name']} "
                f"(native {int(device['default_samplerate'])}Hz, capturing at {SAMPLE_RATE}Hz)"
            )
            if device["default_samplerate"] < SAMPLE_RATE:
                log.info(
                    "  ^ this device runs BELOW 16kHz - almost certainly a Bluetooth headset's "
                    "hands-free\n    profile. Expect poor transcription; pick a different "
                    "input in Windows sound settings."
                )
        except Exception:
            pass

        # _build_answer_generator() reads dialogue_manager_live.py's own MODE constant - reusing
        # it (rather than picking a generator here independently) means flipping MODE there
        # automatically applies to the real robot too, with nothing to keep in sync by hand.
        # Skipped by --calibrate-barge-in: that mode only needs the microphone and the voice,
        # and scraping plus embedding the whole site first would mean waiting a minute to
        # measure a number that has nothing to do with the knowledge base.
        if skip_knowledge_base:
            log.warning("Skipping the knowledge base (not needed for this mode).")
            self.kb = KnowledgeBase()
            self.dialogue = None
        else:
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
            #
            # STALE-BUT-USABLE: a cache older than KB_CACHE_MAX_AGE_S used to mean a
            # blocking rescrape before the robot would greet anybody - about a minute of a kiosk
            # standing there refusing to work, every time it was restarted after a day off.
            # Serving day-old programme information for a few minutes is obviously better than
            # serving nothing at all, so an expired cache is now loaded anyway and refreshed on a
            # background thread (see _refresh_knowledge_base). Only a completely ABSENT cache -
            # the genuine first run - still blocks, because there is nothing to serve.
            loaded_from_cache = False
            self._kb_stale = False
            if not force_refresh:
                loaded_from_cache = self.kb.load(KB_CACHE_PATH, max_age_s=KB_CACHE_MAX_AGE_S)
                if not loaded_from_cache and self.kb.load(KB_CACHE_PATH, max_age_s=None):
                    loaded_from_cache = True
                    self._kb_stale = True
                    log.info(
                        "Knowledge base cache is past its expiry - starting with it anyway "
                        "and refreshing in the background."
                    )

            if not loaded_from_cache:
                # Real Stage 07 content. Deliberately does NOT use scraper.detect_changes() here (the
                # normal "only index what changed" path a standalone/scheduled scraper run would use)
                # - detect_changes() compares against snapshot.json on disk, which may already
                # remember these exact page hashes from an earlier standalone run of
                # scraper.py/indexer.py. This orchestrator builds a fresh KnowledgeBase whenever the
                # cache misses, so "nothing changed since the snapshot" would wrongly mean "index
                # nothing", leaving that fresh KB empty even though it never actually received the
                # real chunks. Indexing every scraped page unconditionally sidesteps that mismatch.
                log.info(
                    f"Fetching real content from lincoln.edu.my ({len(REAL_SEED_URLS)} school "
                    f"pages + their programme links) - this takes about a minute, and the result "
                    f"is cached so the next launch won't need to..."
                )
                scraper = UniversityScraper(REAL_SEED_URLS)
                indexer = UniversityDataIndexer(self.kb)
                real_urls = scraper.discover_urls()
                real_pages = scraper.scrape_once(real_urls)
                real_count = indexer.index_changed_pages(real_pages)
                log.info(
                    f"Indexed {real_count} real chunk(s) from {len(real_urls)} "
                    f"lincoln.edu.my page(s)."
                )
                try:
                    self.kb.save(KB_CACHE_PATH)
                    log.info(
                        f"Cached the knowledge base to {KB_CACHE_PATH.name} for future launches."
                    )
                except Exception as exc:
                    # A cache that can't be written is a lost optimization, not a broken robot -
                    # everything needed to run is already in memory at this point.
                    log.warning(
                        f"[kb] could not write the cache ({exc!r}) - continuing without it."
                    )

            # Loaded AFTER the scraped content (and after any cache load) so these always overwrite
            # rather than get overwritten: these are the hand-verified facilities directory, the one
            # part of the knowledge base that is manually maintained ground truth. They're also
            # cheap - ten chunks - so there's no reason to cache them, and re-upserting each run
            # means edits to SAMPLE_CHUNKS take effect immediately instead of being masked by a
            # stale cached copy of the old text.
            self.kb.upsert_many(list(SAMPLE_CHUNKS))
            log.info(f"Knowledge base ready: {len(self.kb)} chunk(s) total.")

            # MEASURED AND REJECTED: feeding the scraped programme names into
            # Whisper's hotword bias.
            #
            # The idea was that the site holds what students actually ask about - "Bachelor of
            # Computer Science", "Diploma in Psychology", "Doctor of Philosophy in Cybersecurity" -
            # none of which were among the 29 hand-typed vocabulary terms, so each was a long proper
            # noun Whisper had to guess at unaided. It sounds obviously right and it is wrong.
            #
            # Tested on seven synthesized questions containing those exact names, at clean / 10dB /
            # 5dB SNR. small.en already transcribes them at 0-3% WER unaided - they are ordinary
            # English words in an ordinary order, so there is nothing to help with. Adding ~25
            # knowledge-base terms then made it WORSE, reproducibly: "is there a Doctor of Philosophy
            # in Cybersecurity" came back as "is there a TO of Philosophy in Cybersecurity" in all
            # three noise conditions with those terms, and never without them. A long bias list is
            # not free - it dilutes, and it can actively pull decoding around.
            #
            # What IS kept is the short real_hotwords() list the engine is constructed with (see its
            # own note in stt_listener_live.py): the institution, the robot's name, its creator's
            # name - genuinely unusual proper nouns rather than ordinary English. Do not re-add the
            # knowledge-base terms here without re-running that measurement.

            self.dialogue = DialogueManager(
                self.kb, _build_answer_generator(), stream_general=RT.STREAM_GENERAL_ANSWERS
            )

        self.cap = cv2.VideoCapture(0)
        # Ask for a bigger frame than the camera's own default. This is not cosmetic - it is
        # what decides whether a visitor can be recognized at all.
        #
        # Measured from a real run's own log: a person sitting normally in front of the robot
        # produced faces 44-51px wide against sface_embedder's MIN_FACE_WIDTH_PX of 60, so
        # EVERY frame was discarded before matching ("too small (46px wide, need 60)") and the
        # robot greeted an enrolled person as a stranger. The camera's default is 640x480; it
        # supports 1280x720 (checked), and the same person at the same distance is then roughly
        # twice as many pixels wide, which clears both the match gate and the stricter
        # MIN_ENROLL_FACE_WIDTH_PX. It also raises the measured sharpness, because the aligned
        # crop is built from real detail instead of being upscaled from 45px of it.
        #
        # The cost is real and worth stating: YuNet measured ~45ms per pass at 640x480 and
        # ~146ms at 1280x720, so the camera loop slows down (YOLO already costs ~170ms/frame on
        # CPU-only torch). Recognition working at all beats a smoother preview. Set these back
        # to 640/480 if you would rather have the frame rate and can stand close enough.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        actual = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        log.warning(
            f"Camera: {actual[0]}x{actual[1]}"
            + (
                ""
                if actual == (CAPTURE_WIDTH, CAPTURE_HEIGHT)
                else f" (asked for {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} - the camera refused; faces "
                f"will be smaller, so stand closer)"
            )
        )
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera index=0")

        # Optional always-on wake word (WAKE_WORD_ENGINE=openwakeword) - lets somebody call the
        # robot from off camera. The robot's name in a sentence works without it.
        self.wake_detector = create_wake_detector(
            RT.WAKE_WORD_ENGINE, RT.OWW_MODEL, RT.OWW_THRESHOLD
        )

        # The face on the screen (the Expression Layer). Drawn by the camera loop.
        self.face_display = None
        if RT.FACE_DISPLAY and not RT.HEADLESS:
            self.face_display = FaceDisplay(fullscreen=RT.FACE_FULLSCREEN)

        # Synthesize the fixed lines now, in the background, so the first time each is said it
        # starts instantly too (TTS_AUDIO_CACHE). Greetings with a name in them cannot be done
        # ahead, the rest can.
        # Paused while anybody is being served, so it never competes with a live conversation.
        # Also paused during the barge-in calibration: synthesis on every core starves the audio
        # threads, and a measurement taken then is of a stalled speaker, not of the room.
        self._calibrating = threading.Event()
        if RT.TTS_AUDIO_CACHE:
            self.tts.prewarm(
                _fixed_lines(),
                pause_while=lambda: self._visit_active.is_set() or self._calibrating.is_set(),
                stop=self._shutting_down,
            )

    # --- persistent memory ----------------------------------------------------------------

    def _purge_expired_people(self) -> None:
        """Runs the PRIVACY.md retention policy once at startup. Startup rather than a timer
        because the robot is expected to be restarted regularly and RETENTION_DAYS is measured
        in months - a scheduled sweep would add a thread to do the same job less predictably."""
        try:
            expired = self.face_store.purge_expired()
        except Exception as exc:
            log.warning(f"[db] retention sweep failed ({exc!r}) - nothing was deleted.")
            return
        if expired:
            log.info(
                f"[privacy] deleted {len(expired)} person(s) not seen in "
                f"{RETENTION_DAYS} days (PRIVACY.md retention policy)."
            )

    def _begin_conversation(self, person_id: str | None) -> None:
        """Opens the transcript for one visit. person_id is None for a visitor the robot doesn't
        know - the conversation is still recorded, just not attached to anybody (and gets
        attached retroactively if they enroll before leaving, see _handle_consent_for_speaker)."""
        self._conversation_id = None
        self._conversation_person_id = person_id
        self._turn_index = 0
        self._visit_history = []
        # A fresh conversation starts from IDLE whatever the last one ended in - somebody can
        # walk away mid-answer, leaving the machine in AI_SPEAKING, and the next visitor must
        # not inherit that.
        self.conv.reset("new visit")
        self._carried_audio = []
        if self.db is None:
            return
        try:
            self._conversation_id = self.db.start_conversation(person_id)
        except Exception as exc:
            log.warning(
                f"[db] could not start a conversation record ({exc!r}) - this visit will not "
                f"be saved."
            )

    def _record(self, speaker: str, text: str, intent: str | None = None) -> None:
        """One line of the transcript. Called from _speak()/_listen()/the greeting adapter
        rather than from the handful of places that call them, so a new spoken line anywhere in
        the visit is recorded automatically instead of only if somebody remembered to log it.

        Failures are printed and swallowed: losing a transcript line is not a reason to stop
        talking to the person standing in front of the robot."""
        if not text:
            return

        # The in-memory dialogue for THIS visit, which is what lets a follow-up question work
        # ("how long is it?" after asking about a programme - see dialogue_manager's
        # _contextualize). Kept before the database guard below on purpose: the robot must still
        # follow a conversation when MySQL is down, and losing that would turn a database outage
        # into a visible drop in how well it converses rather than just a forgotten visitor.
        self._visit_history.append((speaker, text))
        if len(self._visit_history) > VISIT_HISTORY_TURNS:
            del self._visit_history[:-VISIT_HISTORY_TURNS]

        if self._conversation_id is None:
            return
        try:
            self.db.log_utterance(self._conversation_id, self._turn_index, speaker, text, intent)
            self._turn_index += 1
        except Exception as exc:
            log.warning(f"[db] could not record a {speaker} line ({exc!r}).")

    def _end_conversation(self) -> None:
        if self._conversation_id is None:
            return
        try:
            self.db.end_conversation(self._conversation_id)
            log.info(
                f"[db] saved conversation #{self._conversation_id} "
                f"({self._turn_index} line(s))."
            )
        except Exception as exc:
            log.warning(f"[db] could not close the conversation record ({exc!r}).")
        finally:
            self._conversation_id = None
            rt_log(
                "TURN",
                f"visit finished - states this visit went through: " f"{self.conv.recent(10)}",
            )
            self.conv.reset("visit ended")

    def _listen(self, timeout_s: float) -> str | None:
        # Same reason as _speak(): the microphone is closing, so there is nothing to listen to
        # and blocking here for the timeout would just delay the shutdown by that long.
        if self._shutting_down.is_set():
            return None
        interrupting = bool(getattr(self, "_carried_audio", None))
        if not interrupting:
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
        # and garbled transcripts seen in real testing, not a VAD-tuning problem.
        #
        # (2026-10-27: with echo cancellation the backlog is no longer the robot's own voice,
        # but it is still old - and right after a wake word it is the opposite: the start of
        # the visitor's question, which is why _keep_backlog skips the drain once.)
        #
        # And not after an interruption either: the visitor is still MID-SENTENCE
        # then, and everything queued since the barge-in fired is the rest of that sentence.
        # Draining it kept only the first half-second ("Wait, the li-") and threw the question
        # away - found by the end-to-end simulation.
        drained = 0
        if getattr(self, "_keep_backlog", False) or interrupting:
            self._keep_backlog = False
        else:
            while True:
                try:
                    self.audio_q.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
        if drained:
            log.info(f"[stt] (discarded {drained} stale audio frame(s) queued while busy)")

        # ...but NOT the words somebody interrupted the robot with. Those are the one part of
        # the backlog that is real, wanted, live speech, so they are taken before the drain
        # above ever sees them (BargeInMonitor consumed them off the queue itself) and handed
        # to the listener as the start of this utterance.
        carried, self._carried_audio = self._carried_audio, []
        if not carried:
            self.conv.to(ConvState.LISTENING, f"waiting up to {timeout_s:.0f}s")

        # Listen until something is said TO the robot. A sentence the addressee
        # check rejects - two visitors talking to each other, somebody on the phone behind them
        # - is not answered and not recorded; the robot simply keeps listening, for whatever is
        # left of the timeout. See stage08's addressee.py.
        deadline = time.monotonic() + timeout_s
        while True:
            self._speech_started_at = self._speech_paused_at = None
            interrupted_at = getattr(self, "_interrupted_at", None) if carried else None
            remaining = max(0.5, deadline - time.monotonic())
            heard = listen_for_one_utterance(
                self.stt_engine,
                self.stt_listener,
                self.vad,
                self.audio_q,
                timeout_s=remaining,
                prefix_frames=carried or None,
                on_turn_event=self._on_turn_event,
            )
            carried = []
            if interrupted_at is not None:
                # The speech window of an interruption starts at the barge-in (minus the audio
                # carried over), not when this listen got round to starting.
                self._speech_started_at = interrupted_at
                self._interrupted_at = None
            if heard is None and self.conv.is_in(ConvState.LISTENING):
                rt_log("TURN", f"nobody spoke within {timeout_s:.0f}s")
            if not heard:
                break
            decision = self._judge_addressee(heard)
            if decision is None:
                break
            if not decision.addressed:
                log.info(f"[addressee] not answering - {decision.reason}")
                rt_log("TURN", f"ignored an utterance: {decision.reason}")
                self.conv.to(ConvState.LISTENING, "that was not said to the robot")
                if time.monotonic() >= deadline:
                    heard = None
                    break
                continue
            rt_log("TURN", f"addressed to the robot: {decision.reason}")
            if getattr(self, "_woken_visit", False):
                self._voice_presence_until = time.monotonic() + VOICE_PRESENCE_S
            if decision.question.strip(" ,.!?"):
                heard = decision.question
                break
            # Only the robot's name: "Nova?" - answer like a person would, and listen again.
            self._record("visitor", heard)
            self._speak(_pick(_WAKE_REPLIES))
            deadline = time.monotonic() + timeout_s
        # The transcript stores what Whisper HEARD, including its [??] low-confidence markers -
        # not a cleaned-up version. When a stored answer later looks wrong, the recorded
        # question is the only way to tell a bad answer apart from a misheard question.
        self._record("visitor", heard or "")
        return heard

    def _judge_addressee(self, heard: str):
        """Was `heard` said to the robot? None when there is no judge (old behaviour).

        Reads the lip history over the seconds the sentence was being spoken (from the turn
        events), stretched a little at the end because lip movement is measured over a rolling
        window and so lags the audio."""
        judge = getattr(self, "addressee", None)
        if judge is None:
            return None
        history = getattr(self, "lip_history", None)
        frames, faces = 0, []
        start = getattr(self, "_speech_started_at", None)
        if history is not None and start is not None:
            end = getattr(self, "_speech_paused_at", None) or time.monotonic()
            frames, faces = history.summarize(start, end + LIP_WINDOW_TAIL_S)
        try:
            return judge.decide(heard, frames, faces)
        except Exception as exc:  # noqa: BLE001 - never go deaf over a diagnostic
            log.warning(f"[addressee] check failed ({exc!r}) - answering anyway.")
            return None

    def _probe_answer_backends(self, reason: str) -> None:
        """Asks the answer chain to re-check its backends in the background, if it is a chain
        that can (see ChainAnswerGenerator.probe_now).

        Wrapped rather than called directly because the generator is not always a chain: with
        no Ollama and no API key the robot runs on UniversityOnlyAnswerGenerator, and with only
        one backend configured it runs on that backend alone. Neither has probe_now(), and
        neither should have to grow one just so the camera loop can call it unconditionally.

        Never raises. This runs on the camera thread, once per arrival, and a diagnostic that
        could interrupt person detection would be a bad trade for what it buys.
        """
        probe = getattr(getattr(self, "dialogue", None), "answer_generator", None)
        probe = getattr(probe, "probe_now", None)
        if probe is None:
            return
        try:
            probe(reason)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                f"[chain] backend probe could not start ({exc!r}) - carrying on; the answer "
                f"chain still falls back normally, it just finds out the slower way."
            )

    def _on_turn_event(self, event: str, detail: str = "") -> None:
        """Bridges stage04's audio-level turn events onto the conversation state machine.

        This is the only place the two vocabularies meet, deliberately. stage04 reports what the
        MICROPHONE observed ("paused", "resumed") and knows nothing about conversations; the
        state machine models the conversation and knows nothing about frames or pitch contours.
        Keeping the translation in one small function is what lets either side be retuned
        without the other having to care.

        The mapping that matters is "paused" -> USER_PAUSED rather than USER_FINISHED. A pause
        is not an end of turn, and the whole adaptive endpointer exists to tell the two apart -
        so only "finished", which stage04 emits once the endpointer has actually committed, is
        allowed to move the conversation on.
        """
        now = time.monotonic()
        # Anything except the endpointer deciding "mid-sentence, keep waiting" makes a pending
        # "mm-hm" stale - see _maybe_backchannel.
        if event != "holding" and not (event == "resumed" and "mid-thought" in (detail or "")):
            self._backchannel_token = getattr(self, "_backchannel_token", 0) + 1
        if event == "speech_started":
            # The VAD confirms speech SPEECH_START_FRAMES (~120ms) after it began, and the
            # preroll reaches further back; 0.3s puts the window start where the words did.
            carried = re.search(r"carried over (\d+)ms", detail or "")
            self._speech_started_at = now - (int(carried.group(1)) / 1000.0 if carried else 0.3)
            self._speech_paused_at = None
            self.conv.to(ConvState.USER_SPEAKING, detail or "VAD confirmed speech")
        elif event == "paused":
            self._speech_paused_at = now - 0.06  # reported after two silent 30ms frames
            self.conv.to(ConvState.USER_PAUSED, "mid-turn silence - still listening")
        elif event == "holding":
            self._maybe_backchannel()
        elif event == "resumed":
            self._speech_paused_at = None
            self.conv.to(ConvState.USER_SPEAKING, detail or "they carried on")
        elif event == "partial":
            self._prefetch(detail)
        elif event == "finished":
            self._turn_finished_at = now
            # When their voice actually stopped: the start of the final pause, which is the
            # endpointer's silence wait earlier than "finished". Latency is measured from here.
            self._visitor_stopped_at = self._speech_paused_at or now
            self.conv.to(ConvState.USER_FINISHED, "endpointer committed")
        elif event == "discarded":
            # Speech was detected and then rejected as too short. No turn happened, so go back
            # to waiting rather than pretending one ended.
            self.conv.to(ConvState.LISTENING, f"discarded {detail} - not real speech")

    def _prefetch(self, partial: str) -> None:
        """Starts the knowledge-base search for a partial transcript while the visitor is still
        talking (DialogueManager.prefetch), so it is already done when they stop. One at a time,
        on its own thread - it must never slow down the listening loop that calls this."""
        dialogue = getattr(self, "dialogue", None)
        if dialogue is None or not partial or not hasattr(dialogue, "prefetch"):
            return
        if self._prefetching.is_set():
            return
        question = partial
        judge = getattr(self, "addressee", None)
        if judge is not None and judge.wake:
            question = find_wake_word(partial, judge.wake)[1] or partial
        history = list(self._visit_history)
        self._prefetching.set()

        def _run() -> None:
            try:
                dialogue.prefetch(question, history=history)
            finally:
                self._prefetching.clear()

        threading.Thread(target=_run, name="kb-prefetch", daemon=True).start()

    def _maybe_backchannel(self) -> None:
        """A quiet "mm-hm" for a visitor who has paused MID-SENTENCE - only ever called when
        the endpointer has decided the pause is not the end of the turn ("holding"), never on a
        plain pause, which is just as often the end of a question. Also only after they have
        been talking a while, not twice within a few seconds, and only if the pause is still
        going BACKCHANNEL_AFTER_MS after it began (checked on a timer)."""
        bank = getattr(self, "backchannels", None)
        started, paused = self._speech_started_at, self._speech_paused_at
        if bank is None or started is None or paused is None:
            return
        if paused - started < BACKCHANNEL_MIN_SPEECH_S:
            return
        if time.monotonic() - self._last_backchannel_at < BACKCHANNEL_MIN_GAP_S:
            return
        token = getattr(self, "_backchannel_token", 0)

        def _fire() -> None:
            if getattr(self, "_backchannel_token", 0) != token or self._shutting_down.is_set():
                return  # they carried on, or the turn ended - no "mm-hm" now
            self._last_backchannel_at = time.monotonic()
            rt_log("TTS", "back-channel while the visitor pauses mid-sentence")
            bank.play_async()

        delay = max(0.05, RT.BACKCHANNEL_AFTER_MS / 1000.0 - (time.monotonic() - paused))
        timer = threading.Timer(delay, _fire)
        timer.daemon = True
        timer.start()

    def _busy_with_a_visitor(self) -> bool:
        """Whether the robot is currently dealing with somebody.

        Deliberately broader than "a visit thread is running": a person the camera can see but
        has not greeted yet is about to be talked to, and starting a minute of heavy work in
        that gap lands the load squarely on their greeting. Both signals are already maintained
        for other reasons - the state machine by the visit thread, _seen_ids by the camera loop.
        """
        if self.greeting_sm.state != RobotState.IDLE:
            return True
        with self._state_lock:
            return bool(self._seen_ids)

    def _wait_until_not_serving_anybody(self) -> None:
        """Blocks until nobody is in front of the robot. Called from the refresh thread only."""
        waited = 0.0
        while self._busy_with_a_visitor() and not self._shutting_down.is_set():
            if waited == 0.0:
                log.info("[kb] somebody is here - holding the refresh until they are done.")
            time.sleep(KB_REFRESH_YIELD_POLL_S)
            waited += KB_REFRESH_YIELD_POLL_S
        if waited and not self._shutting_down.is_set():
            log.info(f"[kb] clear after {waited:.0f}s - starting the refresh now.")

    def _scrape_into(self, kb: KnowledgeBase, yield_to_visitors: bool = False) -> int:
        """Fetches lincoln.edu.my into `kb` and returns how many chunks were indexed.

        Shared by the blocking first-run path and the background refresh, so the two cannot
        drift - in particular the deliberate choice to index every scraped page rather than use
        scraper.detect_changes(), which is explained where it is first used above.

        yield_to_visitors: pause between stages whenever somebody walks up. Only the background
        refresh passes this; the first-run path is startup, where there is nobody to yield to
        and nothing to serve until it finishes. The checks are between stages rather than inside
        them because that is where the natural boundaries are - the page fetches are already
        rate-limited and interruptible between URLs, while the embedding pass is one bulk call
        into the model and cannot be stopped partway. Holding off STARTING that pass is
        therefore the meaningful protection, and it is also the expensive half.
        """

        def _yield() -> None:
            if yield_to_visitors:
                self._wait_until_not_serving_anybody()
                if self._shutting_down.is_set():
                    raise _ShuttingDown()

        scraper = UniversityScraper(REAL_SEED_URLS)
        indexer = UniversityDataIndexer(kb)
        urls = scraper.discover_urls()
        _yield()
        # The per-page hook matters as much as the checks between stages: this loop is the long
        # one (~150 rate-limited fetches, about a minute), so it is what a visitor arriving
        # mid-refresh would otherwise be competing with for the whole of their conversation.
        pages = scraper.scrape_once(urls, before_each_page=_yield if yield_to_visitors else None)
        _yield()
        count = indexer.index_changed_pages(pages)
        kb.upsert_many(list(SAMPLE_CHUNKS))
        return count

    def _refresh_knowledge_base(self) -> None:
        """Re-scrapes the site into a FRESH knowledge base and swaps it in when it is complete.

        Runs on a daemon thread for the life of the process. Two properties make the swap safe
        without any locking:
          - the new base is built entirely off to the side, so a search running concurrently is
            reading a fully-formed object the whole time;
          - the swap itself is two attribute assignments, which are atomic in CPython. A question
            already in flight finishes against the old base, which is correct - it is a complete,
            consistent knowledge base, just a few minutes out of date.

        This is why startup does not have to choose between fresh data and starting promptly.
        """
        while not self._shutting_down.is_set():
            due_in = KB_REFRESH_CHECK_INTERVAL_S if not self._kb_stale else 0
            time.sleep(due_in)
            if self._shutting_down.is_set():
                return
            # Never start a refresh on top of a live conversation. "Background" was only true of
            # the THREAD, not of the load: a full refresh fetches ~150 pages and re-embeds
            # ~14,800 chunks through a sentence-transformers model, and on a CPU-only machine
            # that is not spare capacity, it is the whole machine.
            #
            # This was not theoretical. In a real run the startup cache was stale, so due_in was
            # 0 and the refresh began the instant the robot came up - which is also the instant
            # the first visitor was being greeted. The audio callback lost its CPU, PortAudio
            # reported "input overflow", the recording reached Whisper with holes in it, and the
            # transcription came back empty: the robot asked what it could help with, heard
            # nothing at all, and ended the visit. Serving the person standing in front of the
            # robot always outranks having fresher data to serve them from.
            self._wait_until_not_serving_anybody()
            try:
                log.info("[kb] refreshing the knowledge base in the background...")
                fresh = KnowledgeBase()
                count = self._scrape_into(fresh, yield_to_visitors=True)
                if len(fresh) == 0:
                    log.info("[kb] refresh produced an empty knowledge base - keeping the old one.")
                else:
                    self.kb = fresh
                    if self.dialogue is not None:
                        self.dialogue.kb = fresh
                    self._kb_stale = False
                    log.info(
                        f"[kb] refreshed: {count} chunk(s) indexed, {len(fresh)} total, now "
                        f"live."
                    )
                    try:
                        fresh.save(KB_CACHE_PATH)
                    except Exception as exc:
                        log.warning(f"[kb] could not write the cache ({exc!r}).")
            except _ShuttingDown:
                # Not a failure - the robot is closing and this work no longer matters. Return
                # quietly: printing anything here risks doing it while the interpreter is
                # already finalizing.
                return
            except Exception as exc:
                # The robot keeps answering from the knowledge base it already has. A scrape
                # failing is a data-freshness problem, never a reason to stop serving visitors.
                log.warning(f"[kb] background refresh failed ({exc!r}) - keeping the current data.")
                self._kb_stale = False  # don't hammer the site in a tight retry loop

    def _calibrate_barge_in(self, manage_stream: bool = False) -> float | None:
        """Measures this room's echo of the robot's own voice and derives a barge-in threshold.

        Run once at startup, and only when barge-in is switched on. Everything the microphone
        hears while the robot is speaking and nobody else is, is echo - so anything sustained
        and clearly above that level is somebody talking over it. See the barge-in note in
        stage04's stt_listener_live.py for why this is a suppressor rather than a canceller, and
        when it is not good enough.
        """
        # The measurement is worthless unless the microphone is actually feeding audio_q.
        # run() already holds the stream open, so it passes manage_stream=False; the standalone
        # --calibrate-barge-in path has to open it itself.
        calibrating = getattr(self, "_calibrating", None)
        if calibrating is not None:
            calibrating.set()
        try:
            # With echo cancellation, the first second is the canceller converging - skip it.
            aec = getattr(self, "echo", None) is not None
            skip = 1.5 if aec else 0.0

            def _measure(text=None) -> float:
                return calibrate_echo_floor(
                    self.audio_q, lambda t: self.tts.speak(t), sample_text=text, skip_first_s=skip
                )

            def _measure_twice_if_odd() -> float:
                floor = _measure()
                # With the canceller converged the echo is near the noise floor. A reading far
                # above that means something disturbed the measurement (a stall at startup),
                # and a threshold set from it would make barge-in need a raised voice for the
                # whole session (2026-10-27, seen in one of the repeated end-to-end runs).
                # One more sentence, now with a converged canceller; the lower reading wins.
                if aec and floor > AEC_FLOOR_RECHECK:
                    again = _measure("And once more, so I can hear how this room sounds.")
                    log.info(
                        f"[barge-in] first echo reading {floor:.4f} looked high - measured "
                        f"again: {again:.4f}."
                    )
                    floor = min(floor, again)
                return floor

            if manage_stream:
                with self.mic_stream:
                    floor = _measure_twice_if_odd()
            else:
                floor = _measure_twice_if_odd()
        except Exception as exc:
            log.warning(f"[barge-in] calibration failed ({exc!r}) - barge-in stays off.")
            return None
        finally:
            if calibrating is not None:
                calibrating.clear()

        if floor < 0.0:
            log.info(
                "[barge-in] the microphone sent no audio while the robot spoke - is it live? "
                "barge-in stays off."
            )
            return None
        aec = getattr(self, "echo", None) is not None
        threshold = floor * BARGE_IN_MARGIN
        if aec:
            # The canceller removes nearly all of the echo, so the measured floor can be tiny -
            # even exactly zero: with a loud speaker right next to the microphone its suppressor
            # silences the microphone completely while the robot talks (2026-10-27, a laptop's
            # own speakers: 0.0000). That is the canceller working, not a dead microphone, and
            # it used to switch barge-in off. A threshold of a few times "nearly nothing" would
            # fire on room noise, hence the floor.
            threshold = max(threshold, MIN_AEC_BARGE_IN_RMS)
        log.info(
            f"[barge-in] this room returns the robot's own voice at RMS {floor:.4f}; "
            f"interrupting needs {threshold:.4f} ({BARGE_IN_MARGIN}x that)."
        )
        if floor <= 0.0 and not aec:
            log.info(
                "[barge-in] the microphone heard pure silence - is it muted? barge-in stays " "off."
            )
            return None
        if floor > RT.BARGE_IN_MAX_ECHO_FLOOR:
            # The speaker is loud into this microphone. A person would have to shout to clear
            # the margin, and the robot would interrupt itself on its own voice long before
            # that. Say so plainly rather than shipping a feature that misfires.
            log.info(
                f"[barge-in] the microphone hears the robot loudly (RMS {floor:.4f} is above "
                f"the {RT.BARGE_IN_MAX_ECHO_FLOOR:.4f} ceiling), so there is no room for a "
                f"reliable threshold. This needs a directional mic or a speakerphone with "
                f"hardware echo cancellation."
            )
            if RT.barge_in_forced():
                # INTERRUPTION_ENABLED=true is an explicit instruction to run it anyway, so it
                # runs - but the measurement said this will misfire, and that has to be on the
                # record rather than discovered later as "the robot keeps cutting itself off".
                log.info(
                    "[barge-in] INTERRUPTION_ENABLED=true - switching it on regardless, as "
                    "asked. Expect the robot to interrupt ITSELF on its own voice; set "
                    "INTERRUPTION_ENABLED=auto to let the measurement decide, or raise "
                    "BARGE_IN_MARGIN."
                )
                return threshold
            log.info(
                "[barge-in] barge-in stays off (INTERRUPTION_ENABLED=auto). Set it to true "
                "to force it on anyway."
            )
            return None
        return threshold

    def _speak_interruptible(self, speak_fn) -> bool:
        """Runs one spoken response with barge-in watching, if it is enabled and calibrated.

        Returns whether the visitor talked over it. The caller uses that to skip straight to
        listening instead of continuing as though the robot had finished its turn.
        """
        if self._barge_in_threshold is None:
            speak_fn()
            return False

        monitor = BargeInMonitor(
            self.audio_q, self.vad, self.tts.stop_requested, self._barge_in_threshold
        )
        monitor.start()
        try:
            speak_fn()
        finally:
            interrupted = monitor.stop()

        if interrupted:
            self.conv.to(ConvState.INTERRUPTED, "visitor talked over the answer")
            # The words they interrupted WITH. Held for the next _listen(), which starts the
            # utterance from them instead of from scratch - so "Wait, I mean the AI department"
            # is answered, not thrown away and asked for again.
            self._carried_audio = monitor.take_captured_audio()
            triggered_at = getattr(monitor, "triggered_at", None) or time.monotonic()
            self._interrupted_at = triggered_at - len(self._carried_audio) * 0.03
        return interrupted

    def _vote_identity(self, track_id: int, match) -> object | None:
        """Decides who a tracked person is from several recognitions, not one.

        Recognition runs several times a second (RECOGNIZE_INTERVAL_S) and a SINGLE pass's
        answer used to be enough to name somebody out loud. Face recognition simply is not that reliable per frame
        - a head turn, a moment of backlight, someone walking behind - so the robot could greet
        the wrong student off one unlucky comparison, and then keep that name for the whole visit
        because _resolved_names was never revisited.

        So: keep the last few answers for each track and require most of them to agree. The cost
        is a fraction of a second before a name is used, which is hidden anyway inside
        RECOGNITION_WAIT_S (the pause that already exists so recognition gets a fair chance
        before the greeting). The benefit is that one bad frame can no longer name somebody.

        Returns the agreed FaceRecord, or None while there is no consensus yet.

        Called with _state_lock already held.
        """
        history = self._identity_votes.setdefault(track_id, [])
        history.append(match.person_id if match else None)
        if len(history) > IDENTITY_VOTE_WINDOW:
            del history[:-IDENTITY_VOTE_WINDOW]

        winner_id, votes = Counter(history).most_common(1)[0]
        if winner_id is None or votes < IDENTITY_VOTE_REQUIRED:
            return None
        return self.face_store.get(winner_id)

    def _visitor_present(self) -> bool:
        """Is anybody still standing in front of the robot?

        Read from the conversation thread, written by the camera thread - the whole reason
        _seen_ids and _last_seen_at are under _state_lock. Tolerant by design (see
        ABSENCE_GRACE_S): a single frame with nobody in it is normal, several seconds of them is
        somebody who left.
        """
        # Somebody who called the robot by voice from off camera is "present" while they keep
        # talking to it - see VOICE_PRESENCE_S.
        if time.monotonic() < getattr(self, "_voice_presence_until", 0.0):
            return True
        with self._state_lock:
            if self._seen_ids:
                return True
            gone_for = time.monotonic() - self._last_seen_at
        return gone_for < ABSENCE_GRACE_S

    def _abandon_if_alone(self) -> bool:
        """True when the visit should stop because the visitor has gone.

        Also cuts off speech already in progress: LiveTTSSpeaker checks stop_requested between
        audio chunks, so a long answer stops within a fraction of a second rather than being
        delivered in full to an empty room.
        """
        if self._visitor_present():
            return False
        log.info(f"[visit] nobody visible for {ABSENCE_GRACE_S:.0f}s - ending this visit.")
        self.tts.stop_requested.set()
        return True

    def _last_heard_reliable(self) -> bool:
        """Whether the last thing Whisper transcribed is worth acting on - see stage04's
        ConfidenceReport.is_reliable.

        Missing/absent confidence counts as reliable, deliberately: the failure mode to avoid is
        a robot that keeps asking a visitor to repeat a question it heard perfectly well.
        """
        report = getattr(self.stt_engine, "last_confidence", None)
        if report is None:
            return True
        try:
            return bool(report.is_reliable)
        except Exception:  # noqa: BLE001 - never let a diagnostic break the conversation
            return True

    def _speak(self, text: str, intent: str | None = None) -> None:
        """Every spoken line outside the greeting goes through here instead of calling
        self.tts.speak() directly, so it shows up in the console log - real Piper audio gives no
        trace of what was actually said otherwise, which makes a wrong (or apologetic) answer
        indistinguishable from a correct one when reading the log after the fact. See
        _LiveTTSAdapter's docstring for the greeting-path equivalent."""
        # Nothing new starts once the robot is closing: the audio device is about to go away
        # (or already has), and speaking into it raises a host error that surfaces as a crashed
        # visit. Silently skipped rather than raised - the visit is being abandoned anyway, and
        # a traceback here would bury the real reason for the shutdown.
        if self._shutting_down.is_set():
            return
        log.info(f'[TTS] "{text}"')
        self._record("robot", text, intent)
        self.tts.speak(
            text,
            on_first_audio=lambda: self.conv.to(ConvState.AI_SPEAKING, "fixed line"),
        )

    def _speak_stream(self, result, intent: str | None = None) -> str:
        """Speaks a StreamingDialogueResult, starting playback as soon as the first sentence
        exists rather than waiting for the whole answer to be generated.

        Returns what was actually said, which is the reason this isn't just tts.speak_stream():
        the caller no longer holds the finished answer string, and the transcript still needs it.
        The line is recorded AFTER speaking for the same reason - there is nothing to record
        until the stream has been consumed.
        """

        # AI_SPEAKING is set from inside the player, at the moment real audio reaches the
        # device - not here, before the call. The difference is the whole answer's synthesis
        # time, and marking it early would make the state machine's "answer began Nms after
        # they stopped" measure the decision to speak rather than the speech.
        def _on_first_audio() -> None:
            self.conv.to(ConvState.AI_SPEAKING, "first audio out")
            # _answer_one sets this for the duration of one answer, to time the gap that the
            # visitor actually experiences - see the note there.
            hook = getattr(self, "_first_audio_hook", None)
            if hook is not None:
                hook()

        spoken = self.tts.speak_stream(result.pieces, on_first_audio=_on_first_audio)
        text = (spoken.get("text") or "").strip()
        log.info(f'[TTS] "{text}"')
        if spoken.get("time_to_first_audio_ms"):
            log.info(
                f"[TTS] (first word after {spoken['time_to_first_audio_ms']:.0f}ms, "
                f"{spoken['num_segments']} sentence(s))"
            )
        # The intent is read AFTER the stream has been spoken: a streamed general answer starts
        # as GENERAL_CHAT and is corrected once the model's label line has been parsed, which
        # happens while it is being consumed.
        if intent is None:
            intent = getattr(getattr(result, "intent", None), "name", None)
        self._record("robot", text, intent)
        return text

    def _report_turn_latency(
        self,
        tracker: LatencyTracker,
        stopped_at: float | None = None,
        filler_at: float | None = None,
        answer_at: float | None = None,
    ) -> None:
        """Prints where the time actually went for one question, against STAGES.md's own target.

        Stage 09 has shipped LatencyTracker since it was written and NOTHING has ever imported
        it - so the project has committed to "under 2-3 seconds from the end of a question to
        the start of the spoken answer" and has never once measured it. Every latency decision
        in this project (greedy decoding, streaming the LLM into Piper, the acknowledgment
        fillers) has been argued from reasoning rather than from a number.
        """
        m = tracker.measurements
        if not m:
            return
        if "first_audio" not in m:
            print(
                f"[latency] retrieve+generate={m.get('retrieve+generate', 0.0):.0f}ms | no "
                f"answer audio (interrupted or stopped before the first word)"
            )
            return
        # Time-to-first-WORD is the number that matters to a person standing there, not the
        # total: the robot is already talking while the rest of the answer generates.
        to_first_word = m.get("retrieve+generate", 0.0) + m.get("first_audio", 0.0)
        parts = " ".join(f"{k}={v:.0f}ms" for k, v in m.items())
        # Measured from when the visitor's voice stopped, which also counts the
        # endpointer's silence wait and the speech recognition - the wait they really have.
        heard = ""
        if stopped_at is not None and answer_at is not None:
            to_first_word = (answer_at - stopped_at) * 1000
            heard = " (from the end of their voice"
            if filler_at is not None:
                heard += f"; a filler covered it after {(filler_at - stopped_at) * 1000:.0f}ms"
            heard += ")"
        verdict = "OK" if to_first_word <= ANSWER_LATENCY_TARGET_MS else "SLOW"
        print(
            f"[latency] {parts} | time-to-first-word {to_first_word:.0f}ms{heard} "
            f"({verdict} vs {ANSWER_LATENCY_TARGET_MS:.0f}ms target)"
        )

    def _answer_one(self, question: str) -> bool:
        # An acknowledgment plays IMMEDIATELY, on its own thread, while retrieval and generation
        # happen behind it - see stage06's filler_bank.py. Started before the dialogue call, not
        # after, which is the whole point: the silence being covered is the silence that starts
        # the instant the visitor stops talking.
        # FILLER_MODE=always: the old behaviour, an acknowledgment for every question. The
        # default "auto" decides below, once it knows whether the answer is slow.
        filler = None
        filler_at = None
        if self.fillers is not None and RT.FILLER_MODE == "always":
            filler = self.fillers.play_async()
            filler_at = time.monotonic()
        tracker = LatencyTracker()
        self.conv.to(ConvState.THINKING, "retrieval + generation")
        rt_log(
            "LLM",
            f"started generation ({len(self._visit_history)} turn(s) of context "
            f'carried, so follow-ups like "how long does it take to walk there?" '
            f"still resolve)",
        )

        # Everything said so far this visit, so a follow-up ("how long is it?") still
        # knows what "it" refers to - see dialogue_manager's _contextualize().
        with tracker.measure("retrieve+generate"):
            result = self.dialogue.handle_question_stream(question, history=self._visit_history)
            # Generation starts NOW, on a background thread, so it happens while the
            # acknowledgment is still playing rather than in the silence after it - see
            # _prime_stream(). Measured inside this block because the wait for the first piece
            # is real time the visitor experiences, even though it overlaps the filler.
            result.pieces = _prime_stream(result.pieces)

        # Prints WHY the answer came out the way it did (which route was taken, how many KB
        # chunks backed it) - without this, an apology/fallback answer looks identical in
        # the log to "the KB just didn't have it" vs. "retrieval found nothing because the
        # transcript had a [??] in it" vs. other real failure modes - see the STT confidence
        # markers already printed by listen_for_one_utterance() for that half of the picture.
        log.info(
            f"[dialogue] intent={result.intent.name}, {len(result.used_chunks)} chunk(s) "
            f"used, streaming={result.is_streaming}, question={question!r}"
        )

        # FILLER_MODE=auto: wait for the first words of the answer until FILLER_DELAY_MS after
        # the visitor stopped talking. Ready by then (identity questions, cached and fast
        # answers): no filler, the answer just starts. Not ready: a short "Mm, okay." covers
        # the rest of the wait, so the visitor never stands in front of a silent robot.
        if filler is None and self.fillers is not None and RT.FILLER_MODE == "auto":
            # Counted from when their voice stopped, not from when the endpointer committed -
            # the silence they hear starts at the former.
            finished_at = (
                getattr(self, "_visitor_stopped_at", None)
                or getattr(self, "_turn_finished_at", None)
                or time.monotonic()
            )
            if time.monotonic() - finished_at > 30.0:
                finished_at = time.monotonic()  # stale: not from this question
            budget = RT.FILLER_DELAY_MS / 1000.0 - (time.monotonic() - finished_at)
            ready = getattr(result.pieces, "ready", None)
            if ready is not None and not ready.wait(max(0.0, budget)):
                rt_log(
                    "TTS",
                    f"answer not ready {RT.FILLER_DELAY_MS}ms after the question - "
                    f"acknowledging while it finishes",
                )
                filler = self.fillers.play_async()
                filler_at = time.monotonic()

        # Wait for the acknowledgment to finish before the real answer starts, so the robot
        # never talks over itself. By now it has almost always finished already - that overlap
        # is exactly what was bought by starting it early.
        if filler is not None:
            filler.wait()

        # The intent is stored on the answer line so the saved transcript keeps the same "why
        # did it answer that way" trace the console log has - a row of apologies in the database
        # with no route attached is a lot harder to diagnose weeks later.
        # MEASURED WITH A TIMESTAMP, NOT A CONTEXT MANAGER. tracker.measure() times the whole
        # block, and this block BLOCKS until the robot has finished speaking - so "first_audio"
        # was really reporting the length of the answer. A real run printed
        # "first_audio=2453ms ... SLOW" for a reply whose first word was actually out in 156ms:
        # the robot was being told off for talking for two seconds. Every latency verdict on
        # the answer path has been wrong in that direction since the tracker was first wired up.
        #
        # The honest number is the gap between starting to speak and the first sound reaching
        # the speaker, which is exactly when on_first_audio fires (see _speak_stream).
        speak_started = time.monotonic()
        first_audio_at: list[float] = []
        self._first_audio_hook = lambda: first_audio_at.append(time.monotonic())
        try:
            interrupted = self._speak_interruptible(lambda: self._speak_stream(result))
        finally:
            self._first_audio_hook = None
        if first_audio_at:
            tracker.measurements["first_audio"] = (first_audio_at[0] - speak_started) * 1000
        stopped_at = getattr(self, "_visitor_stopped_at", None)
        if stopped_at is not None and speak_started - stopped_at > 30.0:
            stopped_at = None  # stale - this question did not come from a fresh turn
        self._report_turn_latency(
            tracker, stopped_at, filler_at, first_audio_at[0] if first_audio_at else None
        )
        return interrupted

    def _listen_with_retry(self, timeout_s: float) -> str | None:
        """Listens for a reply, asking again if speech was heard but nothing came out of it.

        _listen() has two very different failure modes that both come back falsy, and treating
        them the same is what made the robot give up on a visitor instantly:

          None - the timeout expired without anybody speaking. Nothing to retry; they are not
                 talking to the robot.
          ""   - speech WAS detected and recorded, and the transcription came back empty. That
                 is the robot's failure, not the visitor's: they spoke, and got "That's alright,
                 I'm here if you think of something" as though they had stood there silently.
                 stage04 even prints "please try again" on this path - and then nothing ever
                 gave them the chance, because the caller's `while question:` had already
                 dropped out of the loop.

        Empty transcripts are not an edge case on this hardware. The audio callback shares a CPU
        with YOLO, mediapipe, Whisper and Piper; when it gets starved PortAudio drops input
        frames ("[audio] input overflow"), the recording arrives at Whisper full of holes, and
        Whisper returns nothing at all. Asking once more is exactly what a person does, and it
        usually works, because the second attempt is not competing with startup.
        """
        for attempt in range(1 + MAX_EMPTY_LISTEN_RETRIES):
            heard = self._listen(timeout_s)
            if heard:
                return heard
            if heard is None:
                return None  # silence, not a failed transcription - nothing to ask again about
            # Only worth another try if they are still standing there. Otherwise this is room
            # noise being transcribed to nothing, and re-prompting means talking to an empty
            # room - which is the failure ABSENCE_GRACE_S exists to prevent.
            if not self._visitor_present():
                return None
            if attempt == MAX_EMPTY_LISTEN_RETRIES:
                break
            log.info(
                f"[stt] transcription came back empty - asking again "
                f"(attempt {attempt + 2}/{MAX_EMPTY_LISTEN_RETRIES + 1})."
            )
            self._speak(_pick(_REPAIR_NO_CONTEXT))
        return None

    def _handle_repair(self, heard: str) -> str | None:
        """When a question came through too garbled to answer, ask about the part that DID come
        through instead of asking for the whole thing again.

        "Sorry, I only caught something about the engineering department - what did you want to
        know?" is what a person says, and it usually gets a usable answer on the second attempt.
        A flat "could you repeat that?" throws away the words Whisper actually got right and
        invites the visitor to repeat the same sentence at the same volume, which fails the same
        way. Returns the re-heard question, or None if they gave up or wandered off.
        """
        # The words Whisper was CONFIDENT about - everything not flagged with its
        # low-confidence marker (see stage04's WORD_CONFIDENCE_THRESHOLD). Short function words
        # are dropped: "sorry, I only caught something about the and is" helps nobody.
        confident = [w for w in re.sub(r"\[\?\?\]", " ", heard).split() if len(w.strip(".,?!")) > 3]
        topic = " ".join(confident[:4]).strip(".,?! ")

        if topic:
            self._speak(_pick(_REPAIR_WITH_CONTEXT).format(topic=topic))
        else:
            self._speak(_pick(_REPAIR_NO_CONTEXT))
        return self._listen(LISTEN_TIMEOUT_S)

    def _handle_conversation(self, first_question: str | None = None) -> None:
        """Answers questions in a loop - "do you have any other questions?" after each answer -
         instead of ending the visit after exactly one question. Found from a real test run
        : the old code asked to remember the visitor immediately after the first
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
        question = first_question or self._listen_with_retry(LISTEN_TIMEOUT_S)
        questions_answered = 0
        repairs_used = 0
        interruptions_used = 0

        while question and questions_answered < MAX_QUESTIONS_PER_VISIT:
            # Checked before every answer, not just at the start: an answer takes seconds, and
            # somebody who has already walked off should not be talked at for the rest of them.
            if self._abandon_if_alone():
                return

            # "No thanks, that's all" where a question was expected - typically after they cut
            # an answer short - is a goodbye, not something to look up.
            if questions_answered and _sounds_like_goodbye(question):
                self._speak(_pick(_GOODBYE_AFTER_HELP))
                return

            # "wait" / "no no, that's not what I meant" is a command, not a question. Handled
            # BEFORE the confidence check and before retrieval, because sending it to the
            # knowledge base is exactly the old broken behaviour: it matches nothing and comes
            # back as an apology, which reads as the robot ignoring you at the one moment you
            # were trying to take control.
            control = _detect_control_intent(question)
            if control and interruptions_used < MAX_INTERRUPTIONS_PER_VISIT:
                interruptions_used += 1
                log.info(
                    f"[visit] heard a {control!r} command, not a question "
                    f"({interruptions_used}/{MAX_INTERRUPTIONS_PER_VISIT})."
                )
                if control == "change":
                    # Drop the thread deliberately - see the _CHANGE_PHRASES note. Keeping it
                    # would search their NEXT question against the topic they just abandoned.
                    self._visit_history = []
                    self._speak(_pick(_CHANGE_ACK))
                else:
                    self._speak(_pick(_STOP_ACK))
                question = self._listen_with_retry(LISTEN_TIMEOUT_S)
                continue
            # Stage 04 has reported per-transcription confidence since it was built, with a
            # docstring saying it exists so "a caller can ask 'did you say X?' instead of
            # confidently acting on a wrong guess" - and until now no caller ever looked at it.
            # A garbled question went straight into retrieval, matched nothing sensible, and
            # came back as the fixed apology, which reads to the visitor as "the robot doesn't
            # know" when the truth is "the robot didn't hear". Bounded by MAX_REPAIRS_PER_VISIT
            # so a noisy lobby can't trap somebody in a loop of being asked to repeat themselves.
            if not self._last_heard_reliable() and repairs_used < MAX_REPAIRS_PER_VISIT:
                repairs_used += 1
                log.info(
                    f"[stt] transcript looks unreliable - asking about the part I did catch "
                    f"(repair {repairs_used}/{MAX_REPAIRS_PER_VISIT})."
                )
                question = self._handle_repair(question)
                continue

            interrupted = self._answer_one(question)
            questions_answered += 1
            if not interrupted:
                self._acknowledge_newcomers()

            if interrupted:
                # They started talking over the answer. Skip "anything else?" entirely - they
                # are already mid-sentence, and asking a question at them now would talk over
                # THEM. Just listen.
                log.info("[visit] answer was interrupted - listening instead of prompting.")
                question = self._listen_with_retry(LISTEN_TIMEOUT_S)
                continue

            # ...and again before asking "anything else?", which is the single most common
            # moment for the robot to be left talking to nobody - the visitor has their answer
            # and leaves while it is still speaking.
            if self._abandon_if_alone():
                return

            # --- continuous mode: just listen, don't prompt ------------------------------------
            # In a real conversation nobody says "do you have any other questions?" after every
            # sentence - they stop talking, and the other person either speaks or doesn't. Asking
            # first makes every turn wait on a robot prompt, which is the conversational
            # equivalent of pressing a button to talk, and it is what made this feel like
            # submitting separate voice recordings rather than having a conversation.
            #
            # So: listen first. If they ask something, answer it - no prompt, no "yes" needed.
            # Only if they stay silent does the robot fall through to the prompt below, which is
            # exactly where a prompt belongs: the visitor has gone quiet and the robot needs to
            # find out whether they are done or just thinking.
            #
            # Nothing is lost by this. The prompt path, the bare-"yes" handling and the
            # clear-negative exit are all still there and still reachable - they just stop being
            # mandatory. CONVERSATION_MODE=push_to_talk (or CONTINUOUS_CONVERSATION=false)
            # restores the always-prompt behaviour exactly as it was.
            if not RT.push_to_talk():
                rt_log("TURN", "continuous mode - listening for a follow-up without prompting")
                follow_up = self._listen_with_retry(CONSENT_LISTEN_TIMEOUT_S)
                if follow_up:
                    # NOTE the `if follow_up` guard, which is load-bearing.
                    # _is_clear_negative() treats silence as a clear "no more questions", and it
                    # is right to: everywhere else it is called, the robot has just ASKED
                    # "anything else?", and saying nothing in reply to a direct question does
                    # mean no. Here nothing has been asked, so silence means only that they have
                    # not started talking yet - it must fall through to the prompt below, not
                    # end the visit. Calling _is_clear_negative on it unguarded made the robot
                    # say goodbye the moment a visitor paused to think after an answer.
                    if _is_clear_negative(follow_up):
                        self._speak(_pick(_GOODBYE_AFTER_HELP))
                        return
                    if _is_bare_affirmation(follow_up):
                        # "yeah" on its own with no prompt in front of it is somebody gathering
                        # their thoughts, not an answer to anything. Give them the longer window
                        # to actually ask it.
                        question = self._listen_with_retry(LISTEN_TIMEOUT_S)
                    else:
                        question = follow_up
                    continue
                if self._abandon_if_alone():
                    return
                # Silence. NOW the prompt earns its place.

            self._speak(_pick(_MORE_QUESTIONS))
            # Also retried: an empty transcript here used to read as "no more questions" and end
            # the visit, so somebody whose follow-up question was dropped got shown the door
            # rather than asked again.
            more = self._listen_with_retry(CONSENT_LISTEN_TIMEOUT_S)

            if _is_clear_negative(more):
                self._speak(_pick(_GOODBYE_AFTER_HELP))
                return
            if more and _is_bare_affirmation(more):
                # bare "yes" - the real question is still coming
                question = self._listen_with_retry(LISTEN_TIMEOUT_S)
            else:
                question = more  # they skipped the "yes" and asked it directly - answer it as-is

        if not self._visitor_present():
            log.warning("[visit] visitor already gone - skipping the closing line.")
        elif questions_answered == 0:
            self._speak(_pick(_NOTHING_ASKED))
        elif questions_answered >= MAX_QUESTIONS_PER_VISIT:
            self._speak(_pick(_OUT_OF_QUESTIONS))
        # else: the loop already ended cleanly via the clear-negative return above, or via a
        # final unheard/empty `question` (falls out of the while condition - already implicitly
        # "no more" the same way the pre-2026-08-19 "not question" timeout case worked).

    def _acknowledge_newcomers(self) -> None:
        """Says "I'll be with you in a moment" once, between answers, if somebody new walked up
        during this visit (queued by the camera loop). Between answers because that is the one
        moment it interrupts nobody."""
        waiting = getattr(self, "_newcomers_waiting", None)
        if not waiting or not RT.ACKNOWLEDGE_NEWCOMERS:
            return
        with self._state_lock:
            fresh = set(self._newcomers_waiting) - self._newcomers_acknowledged
            self._newcomers_acknowledged |= fresh
            self._newcomers_waiting.clear()
        if fresh:
            log.info(f"[visit] {len(fresh)} more visitor(s) waiting - letting them know.")
            self._speak(_pick(_NEWCOMER_ACK))

    def _speaker_from_lip_history(self, candidate_ids: list[int]) -> int | None:
        """Who did the talking this visit, from the lip history: the candidate whose lips were
        moving most while facing the robot, over the whole visit. Much steadier than "whose
        mouth is moving right now", which at the end of a visit is usually nobody's.

        Only meaningful when lip-history face ids are presence track ids (face presence)."""
        history = getattr(self, "lip_history", None)
        since = getattr(self, "_visit_started_at", None)
        if history is None or since is None or getattr(self, "face_presence", None) is None:
            return None
        _frames, faces = history.summarize(since, time.monotonic())
        scored = [
            (f.speaking_fraction, f.face_id)
            for f in faces
            if f.face_id in candidate_ids and f.facing_fraction >= 0.3
        ]
        if not scored:
            return None
        scored.sort(reverse=True)
        best_fraction, best = scored[0]
        if best_fraction < 0.1:
            return None
        # Two people talking about equally much: do not guess (see PRIVACY.md).
        if len(scored) > 1 and scored[1][0] >= best_fraction * 0.8:
            return None
        return best

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
        if not candidate_ids:
            return None
        from_history = self._speaker_from_lip_history(candidate_ids)
        if from_history is not None:
            return from_history
        if self.mouth_detector is None:
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
            known = [
                self._resolved_person_ids[p.track_id]
                for p in arriving
                if p.track_id in self._resolved_person_ids
            ]
        return known[0] if len(known) == 1 else None

    def _attribute_conversation_to_speaker(self, arriving: list[Person]) -> None:
        """Second attempt at filing an unattributed visit, run after the talking is done - by
        which point mouth-movement detection has had a whole conversation to watch, so it can
        answer the question _initial_conversation_person() couldn't."""
        if (
            self.db is None
            or self._conversation_id is None
            or self._conversation_person_id is not None
        ):
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
            log.warning(
                f"[db] could not attribute this conversation ({exc!r}) - it stays saved "
                f"without a name on it."
            )

    def _handle_consent_for_speaker(self, arriving: list[Person]) -> None:
        """Offers to remember the person who was actually talking, rather than asking every
        person in view one after another.

        Changed 2026-08-19 (was: a loop calling this once per arriving track). Asking each
        visitor in turn is wrong in a group: the robot would say "I don't recognize you, would
        you like me to remember you?" two or three times in a row to people who never spoke to
        it, and whoever answered "yes" would have their name attached to whichever face the loop
        happened to be on at the time. Now it asks once, of whoever was speaking."""
        candidate_ids = [p.track_id for p in arriving]
        if not candidate_ids:
            return  # woken by voice with nobody on camera - no face to remember
        tid = self._active_speaker_track_id(candidate_ids)

        if tid is None:
            if len(candidate_ids) > 1:
                # Genuinely ambiguous with several people present - saying nothing is better
                # than risking someone's name on the wrong face.
                log.warning(
                    f"[consent] {len(candidate_ids)} people present and no clear speaker - "
                    f"skipping the remember-me offer for this visit."
                )
                return
            tid = candidate_ids[0]  # only one person here, so there's nothing to confuse
        else:
            log.info(f"[consent] active speaker is track {tid}")

        with self._state_lock:
            if self._resolved_names.get(tid):
                return  # already known (or registered earlier this visit) - nothing to ask
            embedding = self._last_embedding_by_track.get(tid)
        if embedding is None:
            return  # never got a clear face sample for them - nothing usable to register

        self._speak("I don't recognize you. Would you like me to remember you for next time?")
        # Retried on an empty transcript like every other reply, and that does NOT weaken the
        # consent rule: an answer the robot never managed to hear still defaults to "no" (see
        # _parse_yes_no and PRIVACY.md). The only thing the retry changes is that somebody who
        # actually said "yes" into a dropped recording gets asked once more, instead of being
        # silently recorded as having declined.
        answer = self._listen_with_retry(CONSENT_LISTEN_TIMEOUT_S)
        if not _parse_yes_no(answer):
            self._speak("No problem, I won't remember this visit.")
            return

        self._speak("Great! What's your name?")
        # Worth retrying here more than anywhere else: this name is written to the database and
        # said back to them on every future visit, so hearing nothing and enrolling them under
        # whatever _parse_name_from_speech() falls back to is a mistake that persists.
        name_answer = self._listen_with_retry(CONSENT_LISTEN_TIMEOUT_S)
        name = _parse_name_from_speech(name_answer)
        with self._state_lock:
            rec = self.face_store.register(name=name, embeddings=[embedding], consent_given=True)
            self._resolved_names[tid] = name
            self._resolved_person_ids[tid] = rec.person_id

        # They were anonymous when this conversation started; now they aren't. See
        # RobotDatabase.attach_conversation_to_person() for why this is worth backfilling.
        if self.db is not None and self._conversation_id is not None:
            try:
                self.db.attach_conversation_to_person(self._conversation_id, rec.person_id)
                self._conversation_person_id = rec.person_id
            except Exception as exc:
                log.warning(
                    f"[db] could not attach this conversation to {name!r} ({exc!r}) - it "
                    f"stays saved, just without a name on it."
                )

        self._speak(f"Nice to meet you, {name}! I'll remember you next time.")

    def _run_visit(
        self, arriving: list[Person], detected_at: float | None = None, woken: bool = False
    ) -> None:
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
        person_id = self._initial_conversation_person(arriving)
        self._begin_conversation(person_id)
        self._visit_started_at = time.monotonic()
        self._woken_visit = woken
        if hasattr(self, "addressee"):
            self.addressee.reset()
        try:
            if woken:
                # Called by voice ("hey Nova"): no greeting - they already started the
                # conversation. If the question follows in the same breath it is answered
                # straight away; if they wait, the robot says "Yes?" like a person would.
                self._voice_presence_until = time.monotonic() + VOICE_PRESENCE_S
                self._keep_backlog = True
                self.greeting_sm.state = RobotState.LISTENING
                self.expression.set_state(RobotState.LISTENING)
                first = self._listen(1.5)
                if not first:
                    self._speak(_pick(_WAKE_REPLIES))
                self._handle_conversation(first_question=first or None)
                self._attribute_conversation_to_speaker(arriving)
                self._handle_consent_for_speaker(arriving)
                self.greeting_sm.on_interaction_timeout_or_done()
                return

            # Only for a visitor the robot recognizes AND is alone with. In a group the recall
            # would be spoken over everyone - "did you get anywhere with the Master of Computer
            # Science?" in front of three strangers is a privacy problem, not a nice touch.
            recall = ""
            if person_id is not None and len(arriving) == 1:
                recall = self._recall_line_for(person_id)
                if recall:
                    print(f"[memory] recalling: {recall!r}")

            if detected_at is not None:
                # STAGES.md: "time from a person entering to the start of the greeting: < 1
                # second". This measures everything under this project's control - detection,
                # the deliberate RECOGNITION_WAIT_S pause so a returning visitor gets their name,
                # the database lookup for the recall line - up to the moment the greeting is
                # handed to Piper. Piper's own time-to-first-audio is on top of it and is
                # reported separately by the speaker.
                elapsed_ms = (time.monotonic() - detected_at) * 1000
                verdict = "OK" if elapsed_ms <= GREETING_LATENCY_TARGET_MS else "SLOW"
                print(
                    f"[latency] detected -> greeting {elapsed_ms:.0f}ms ({verdict} vs "
                    f"{GREETING_LATENCY_TARGET_MS:.0f}ms target; includes the deliberate "
                    f"{RECOGNITION_WAIT_S:.1f}s recognition wait)"
                )

            self.greeting_sm.on_people_entered(arriving, recall=recall)
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
            # Learn from the visit BEFORE closing it out, so the extracted facts can still be
            # attached to this conversation id for auditing (robot_db's person_facts schema).
            # Inside `finally` because a visit that crashed halfway through still happened, and
            # what the visitor said before it went wrong is still worth remembering.
            self._remember_visit()
            # Record that this visit HAPPENED, which nothing did before: last_seen stayed frozen
            # at whatever it was when the person enrolled, and visit_count never moved off 1.
            # Two things quietly depended on that being wrong. purge_expired() deletes anybody
            # whose last_seen is more than RETENTION_DAYS old, so a student who came by every
            # week was still on course to be forgotten 120 days after enrolling - the exact
            # opposite of what the retention policy is for. And the "it's been a while" recall
            # line measures from last_seen, so it would have gone on aging forever.
            #
            # At the END of the visit specifically, for the reason _recall_line_for() explains:
            # the greeting reads last_seen to work out how long it has been, so it has to still
            # say the PREVIOUS visit while the greeting is being built. By here it has been read
            # and spoken, and moving it on is safe.
            self._mark_visit_recorded()
            # In `finally` so a crashed visit still closes its transcript instead of leaving a
            # row with a NULL ended_at that reads, forever, like a conversation still going on.
            self._end_conversation()
            self._woken_visit = False
            self._voice_presence_until = 0.0

    def _mark_visit_recorded(self) -> None:
        """Advances last_seen/visit_count for the person this visit belonged to.

        Only for a visitor the robot actually has a record for - somebody who declined to be
        remembered has no person_id and nothing to update, which is the same consent rule
        _remember_visit() relies on.
        """
        person_id = self._conversation_person_id
        if person_id is None:
            return
        record = self.face_store.get(person_id)
        if record is None:
            return
        # mark_seen_again() mirrors to MySQL through _mirror(), which already turns a database
        # failure into a warning rather than an exception - but this runs in a `finally` during
        # teardown, so anything unexpected here would mask whatever ended the visit.
        try:
            self.face_store.mark_seen_again(record)
        except Exception as exc:
            log.warning(f"[memory] could not record this visit against {person_id} ({exc!r}).")

    def _recall_line_for(self, person_id: str) -> str:
        """The extra sentence in a returning visitor's greeting - see stage05's visit_memory.

        Runs while somebody is walking up to the robot, so it is one indexed SELECT and pure
        string formatting: no model call, no network. Any failure returns "" - a missing recall
        is invisible to the visitor, whereas a slow or crashing one delays the greeting itself,
        which is the one thing this project measures in milliseconds (STAGES.md: under a second
        from entering to greeting).
        """
        if self.db is None:
            return ""
        try:
            facts = [row["fact"] for row in self.db.person_facts(person_id)]
            record = self.face_store.get(person_id)
            # last_seen as it stands now, which is still the PREVIOUS visit - exactly what
            # "it's been a while" should be measured against. _mark_visit_recorded() moves it
            # on at the end of the visit, deliberately after this has been read; do not move
            # that call earlier or every returning visitor is told it has been 0 days.
            last_seen = record.last_seen if record is not None else None
            return visit_memory.build_recall_line(facts, last_seen=last_seen)
        except Exception as exc:
            log.warning(f"[memory] could not build a recall line ({exc!r}) - greeting normally.")
            return ""

    def _remember_visit(self) -> None:
        """Extracts the few things worth remembering from the visit that just ended and stores
        them against the person (stage05's visit_memory + robot_db's person_facts).

        Deliberately at the END of the visit, where its cost is invisible - the visitor has
        already walked away. Doing the same work at greeting time (reading back whole
        transcripts through a model to work out what is known about somebody) is what makes
        "remembering" too slow to actually use.

        Skipped entirely for a visitor with no person_id: they either declined to be remembered
        or were never asked, and there is nowhere to attach a fact. That is the consent rule
        enforced structurally rather than by a check somebody could forget - see PRIVACY.md.
        """
        if self.db is None or self._conversation_id is None:
            return
        person_id = self._conversation_person_id
        if person_id is None:
            return

        try:
            turns = self.db.conversation_transcript(self._conversation_id)
            facts = visit_memory.extract_facts(turns)
            if facts:
                stored = self.db.add_person_facts(
                    person_id, facts, conversation_id=self._conversation_id
                )
                log.info(f"[memory] stored {stored} new fact(s) about this visitor.")
        except Exception as exc:
            # Never a reason to disturb anything: the visit is over and the transcript is
            # already saved. The robot simply knows less next time.
            log.warning(f"[memory] could not learn from this visit ({exc!r}).")

    def _detect_people(self, frame) -> list:
        """This frame's people, from whichever presence backend is running."""
        if self.face_presence is not None:
            return self.face_presence.detect_people(frame)
        return detect_people(self.yolo, frame)

    def _update_engagement(self, people, frame_width: int, now: float) -> set[int]:
        """Track ids that are near, facing the robot and have stayed (engagement.py).

        People without face landmarks (the YOLO backend gives none) cannot be judged, so the gate
        lets them through - exactly the old behaviour."""
        if not RT.ENGAGEMENT_GATE:
            return {p.track_id for p in people}
        engaged: set[int] = set()
        for p in people:
            landmarks = getattr(p, "landmarks", None)
            if not landmarks:
                engaged.add(p.track_id)
            elif self.engagement.observe(p.track_id, p.bbox, landmarks, frame_width, now):
                engaged.add(p.track_id)
        return engaged

    def _update_mouths(self, frame, people, now: float) -> list:
        """Mouth activity for this frame, and one timestamped row of the lip history the
        addressee check reads (whose lips moved, who was facing the robot)."""
        if self.mouth_detector is None:
            return []
        try:
            if self.face_presence is not None:
                # The presence tracks ARE faces, so the mesh runs on each face crop and its
                # face_id is the presence track id - no geometry join needed.
                activities = self.mouth_detector.update_faces(
                    frame, [(p.track_id, p.bbox) for p in people]
                )
            else:
                activities = self.mouth_detector.update(frame)
        except Exception as exc:
            log.warning(
                f"[mouth] detector failed ({exc!r}) - disabling active speaker detection for "
                f"the rest of this run; the robot falls back to answering whoever it "
                f"recognized first."
            )
            self.mouth_detector = None
            activities = []
        with self._state_lock:
            self._activities = activities
        samples = []
        for a in activities:
            yaw = getattr(a, "yaw", None)
            facing = True if yaw is None else abs(yaw) <= RT.ENGAGE_MAX_YAW
            samples.append(FaceSample(a.face_id, bool(a.is_speaking), facing))
        self.lip_history.record(samples, now)
        return activities

    def _recognize(self, frame, people) -> dict:
        """{track_id: FaceDetection} for this frame. With face presence, the faces it already
        found are embedded directly (describe_faces) - no second full-resolution detection."""
        if self.face_presence is not None:
            detections = self.embedder.describe_faces(
                frame, [(p.bbox, p.landmarks, p.score) for p in people]
            )
        else:
            detections = self.embedder.detect(frame)
        return match_faces_to_people(people, detections)

    def _idle_audio_loop(self) -> None:
        """Reads the microphone while no visit is running.

        Two jobs. The always-on wake word (WAKE_WORD_ENGINE=openwakeword) listens here. And the
        queue is kept empty: before this existed nothing read the microphone between visits, so
        audio_q grew by ~32KB a second for as long as the robot stood idle - over 100MB an hour,
        all of it thrown away by the next visit's stale-audio drain anyway.
        """
        detector = self.wake_detector
        while not self._shutting_down.is_set():
            if self._visit_active.is_set():
                time.sleep(0.05)
                continue
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if detector is None:
                continue
            try:
                heard = detector.process(frame)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[wake] wake word detector failed ({exc!r}) - switched off.")
                detector = None
                continue
            if heard and self._wake_requested_at is None:
                log.info(f"[wake] heard the wake word (score {detector.last_score:.2f})")
                rt_log("TURN", "woken by voice - starting a conversation")
                # Stop reading NOW: the question often follows in the same breath, and the
                # visit thread needs those frames.
                self._visit_active.set()
                self._wake_requested_at = time.monotonic()
                detector.reset()

    def run(self) -> None:
        log.info(
            "Live robot running. "
            + (
                "Stop it with Ctrl+C."
                if RT.HEADLESS
                else "Press 'q' in a robot window to quit (or Ctrl+C)."
            )
        )
        # Printed every start, because these settings decide how the robot FEELS to talk to,
        # and "it interrupted me" / "it waited too long" / "it answered my friend" are all
        # answered by these lines. Run `python stages/stage00_setup/realtime_config.py` for the
        # full list and where each value came from.
        log.warning(
            f"Conversation: "
            f"{'push-to-talk' if RT.push_to_talk() else 'continuous (hands-free)'} | "
            f"turn ends between {RT.MIN_SILENCE_DURATION}ms and {RT.MAX_SILENCE_DURATION}ms "
            f"of silence ({RT.END_OF_TURN_TIMEOUT}ms when the pitch cannot be read; "
            f"turn detector={RT.TURN_DETECTOR}) | interruption={RT.INTERRUPTION_ENABLED} | "
            f"streaming STT={RT.STREAMING_STT}, TTS={RT.STREAMING_TTS}"
        )
        log.info(
            f"Attention: presence={'faces' if self.face_presence is not None else 'YOLO'} "
            f"(up to {RT.VISION_FPS:.0f}fps) | engagement gate={RT.ENGAGEMENT_GATE} | "
            f"addressee check={RT.ADDRESSEE_CHECK} | wake word={RT.WAKE_WORD_ENGINE} | "
            f"echo cancellation={'on' if self.echo is not None else 'off'} | "
            f"fillers={RT.FILLER_MODE} | back-channels={'on' if self.backchannels else 'off'}"
        )
        pending_since: dict[int, float] = {}
        # When each track was first detected, kept separately because pending_since is popped on
        # the way to a greeting - this is the clock STAGES.md's "under a second from a person
        # entering to the start of the greeting" is measured against.
        first_seen_at: dict[int, float] = {}
        ready_to_greet: set[int] = set()
        last_recognized_at = 0.0  # monotonic clock of the last recognition pass
        last_quality_log_at = 0.0  # so the "ignoring this face" notice cannot flood the console
        visit_thread: threading.Thread | None = None
        visit_ids: set[int] = set()
        people: list = []
        seen_ids: set[int] = set()
        engaged_ids: set[int] = set()
        activities: list = []
        last_vision_at = 0.0
        # VISION_FPS caps the heavy work (detection, mouths, recognition). The camera is still
        # read every loop - frames never go stale in the driver - and the face on the screen
        # still animates at the camera's own rate.
        vision_interval = 1.0 / RT.VISION_FPS
        show_preview = RT.CAMERA_PREVIEW and not RT.HEADLESS
        any_window = show_preview or self.face_display is not None

        try:
            with self.mic_stream:
                # Measured once, with the stream already open. Only when barge-in is switched
                # on - it costs a spoken sentence at startup, which is not worth paying for a
                # threshold nothing will read.
                if BARGE_IN_ENABLED:
                    log.info(
                        f"Calibrating barge-in (stay quiet for a moment) - "
                        f"INTERRUPTION_ENABLED={RT.INTERRUPTION_ENABLED}..."
                    )
                    self._barge_in_threshold = self._calibrate_barge_in()
                    if self._barge_in_threshold is not None:
                        log.info("[barge-in] ON - you can talk over the robot and it will stop.")
                    else:
                        log.info(
                            "[barge-in] OFF - the robot will finish each answer before "
                            "listening again."
                        )
                else:
                    log.info("[barge-in] OFF (INTERRUPTION_ENABLED=false) - not calibrating.")

                # Keeps the university data current without anybody restarting the robot, and
                # catches up immediately when startup began from an expired cache.
                threading.Thread(
                    target=self._refresh_knowledge_base, name="kb-refresh", daemon=True
                ).start()
                # Reads the microphone between visits (wake word, and keeps the queue empty).
                threading.Thread(
                    target=self._idle_audio_loop, name="idle-audio", daemon=True
                ).start()

                while True:
                    ok, frame = self.cap.read()
                    if not ok:
                        log.warning("Could not read a frame - stopping.")
                        break
                    now = time.monotonic()

                    if now - last_vision_at >= vision_interval:
                        last_vision_at = now
                        people = self._detect_people(frame)
                        seen_ids = {p.track_id for p in people}
                        entered_ids, left_ids = self.presence.update(seen_ids)

                        with self._state_lock:
                            self._seen_ids = seen_ids
                            # Stamped only while somebody is actually visible, so the
                            # conversation thread can tell "nobody has been here for N seconds"
                            # from "nobody is in this single frame" - see ABSENCE_GRACE_S.
                            if seen_ids:
                                self._last_seen_at = now
                            for tid in left_ids:
                                self._resolved_names.pop(tid, None)
                                self._resolved_person_ids.pop(tid, None)
                                self._last_embedding_by_track.pop(tid, None)
                                self._identity_votes.pop(tid, None)
                                self._face_quality_by_track.pop(tid, None)
                                # Every per-track dict is cleaned up here - a box left behind
                                # is what used to put a trail of frozen boxes on screen.
                                self._face_bbox_by_track.pop(tid, None)
                                self._face_bbox_at.pop(tid, None)
                                self._newcomers_waiting.discard(tid)
                                pending_since.pop(tid, None)
                                first_seen_at.pop(tid, None)
                                ready_to_greet.discard(tid)
                        for tid in left_ids:
                            self.engagement.forget(tid)

                        for tid in entered_ids:
                            pending_since[tid] = now
                            first_seen_at[tid] = now

                        if entered_ids:
                            # Somebody just walked up: find out NOW, in the background, whether
                            # the cloud model is answering, so their first question does not
                            # have to find out the expensive way. Debounced inside probe_now().
                            self._probe_answer_backends("visitor detected")

                        engaged_ids = self._update_engagement(people, frame.shape[1], now)
                        # Mouth movement on every vision frame - the signal is how much the
                        # mouth varies over time, so it needs a steady sampling rate.
                        activities = self._update_mouths(frame, people, now)

                        if people and now - last_recognized_at >= RECOGNIZE_INTERVAL_S:
                            last_recognized_at = now
                            matches = self._recognize(frame, people)
                            for tid, det in matches.items():
                                # A face too small, too blurry or too weakly detected produces
                                # a well-formed embedding of nothing in particular - see the
                                # quality-gating note in sface_embedder.py.
                                if not det.is_usable:
                                    if now - last_quality_log_at >= QUALITY_LOG_INTERVAL_S:
                                        last_quality_log_at = now
                                        log.info(
                                            f"[face] ignoring track #{tid}: "
                                            f"{det.quality_reason()}"
                                        )
                                    continue

                                with self._state_lock:
                                    self._last_embedding_by_track[tid] = det.embedding
                                    self._face_bbox_by_track[tid] = det.bbox
                                    self._face_bbox_at[tid] = now
                                    self._face_quality_by_track[tid] = det
                                    match = self.face_store.find_match(det.embedding)
                                    # Only reinforce from a face good enough to STORE.
                                    if match and det.is_enrollable:
                                        self.face_store.reinforce(match, det.embedding)
                                    # One frame does not decide who somebody is.
                                    settled = self._vote_identity(tid, match)
                                    if settled is not None:
                                        self._resolved_names[tid] = settled.name
                                        self._resolved_person_ids[tid] = settled.person_id

                                if settled is not None and tid in pending_since:
                                    pending_since.pop(tid)
                                    ready_to_greet.add(tid)

                        for tid in [
                            t
                            for t, since in pending_since.items()
                            if now - since >= RECOGNITION_WAIT_S
                        ]:
                            pending_since.pop(tid)
                            ready_to_greet.add(tid)

                        # Drop anyone who became "ready" but walked off before a greeting.
                        ready_to_greet &= seen_ids

                    # Reap a finished visit before considering a new one.
                    if visit_thread is not None and not visit_thread.is_alive():
                        visit_thread.join()
                        visit_thread = None
                        visit_ids = set()
                        with self._state_lock:
                            self._newcomers_waiting.clear()
                            self._newcomers_acknowledged.clear()
                        self._visit_active.clear()

                    if (
                        visit_thread is None
                        and self._wake_requested_at is not None
                        and self.greeting_sm.state == RobotState.IDLE
                    ):
                        # Woken by voice: talk to whoever is there (possibly nobody the camera
                        # can see - they may have called from the side).
                        arriving_ids = sorted(seen_ids)
                        with self._state_lock:
                            arriving = [
                                Person(track_id=tid, known_name=self._resolved_names.get(tid))
                                for tid in arriving_ids
                            ]
                        for tid in arriving_ids:
                            pending_since.pop(tid, None)
                            first_seen_at.pop(tid, None)
                        ready_to_greet -= set(arriving_ids)
                        visit_ids = set(arriving_ids)
                        self._wake_requested_at = None
                        self._visit_active.set()
                        visit_thread = threading.Thread(
                            target=self._run_visit, args=(arriving, None, True), daemon=True
                        )
                        visit_thread.start()
                    elif self._wake_requested_at is not None and visit_thread is not None:
                        self._wake_requested_at = None  # already in a conversation

                    # A greeting needs someone who is ready (recognized, or given a fair chance
                    # to be) AND engaged: near, facing the robot, and staying. Somebody walking
                    # past, or standing across the lobby, is not greeted.
                    greetable = sorted(t for t in ready_to_greet if t in engaged_ids)
                    if (
                        greetable
                        and visit_thread is None
                        and self.greeting_sm.state == RobotState.IDLE
                    ):
                        with self._state_lock:
                            arriving = [
                                Person(track_id=tid, known_name=self._resolved_names.get(tid))
                                for tid in greetable
                            ]
                        ready_to_greet -= set(greetable)
                        detected_at = min(
                            (first_seen_at.get(tid, now) for tid in greetable), default=now
                        )
                        for tid in greetable:
                            first_seen_at.pop(tid, None)
                        visit_ids = set(greetable)
                        self._visit_active.set()
                        visit_thread = threading.Thread(
                            target=self._run_visit, args=(arriving, detected_at), daemon=True
                        )
                        visit_thread.start()
                    elif greetable and visit_thread is not None and RT.ACKNOWLEDGE_NEWCOMERS:
                        # Somebody new walked up while the robot is busy: queue a quick "I'll be
                        # with you in a moment" for the visit thread to say between answers.
                        fresh = set(greetable) - visit_ids - self._newcomers_acknowledged
                        if fresh:
                            with self._state_lock:
                                self._newcomers_waiting |= fresh

                    # Where the face on the screen looks: at whoever is talking, else at the
                    # nearest engaged person, else straight ahead.
                    target = None
                    speaker = (
                        self.mouth_detector.most_likely_speaker(activities)
                        if self.mouth_detector is not None and activities
                        else None
                    )
                    if speaker is not None:
                        target = speaker.bbox
                    else:
                        candidates = [p for p in people if p.track_id in engaged_ids] or people
                        if candidates:
                            target = max(candidates, key=lambda p: p.bbox[2] * p.bbox[3]).bbox
                    self._look_x = look_x_for(target, frame.shape[1]) if target else 0.0

                    if self.face_display is not None:
                        state = self.conv.state.name if self._visit_active.is_set() else "IDLE"
                        self.face_display.show(state, self.tts.level_at(), self._look_x)
                    if show_preview:
                        self._draw_overlay(frame)
                        cv2.imshow("Stage 08 - live robot (press q to quit)", frame)
                    if any_window and cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Wind the visit down HERE, still inside `with self.mic_stream`, so the
                # microphone and speaker are both still open while it does. Doing it in the
                # `finally` below meant the `with` had already closed the stream, and the visit
                # thread's next line of speech hit a PortAudio host error - see _shutting_down.
                self._wind_down(visit_thread)
        except KeyboardInterrupt:
            print("\nStopped (Ctrl+C).")
        finally:
            # Safety net for the paths that do not reach the call above (Ctrl+C, or a camera
            # failure breaking the loop early). Idempotent: joining a finished thread returns
            # immediately, and the event is already set.
            self._wind_down(visit_thread)
            self.cap.release()
            if self.face_display is not None:
                self.face_display.close()
            if any_window:
                cv2.destroyAllWindows()
            if self.mouth_detector is not None:
                self.mouth_detector.close()
            if self.db is not None:
                self.db.close()

    def _wind_down(self, visit_thread) -> None:
        """Stops an in-flight conversation cleanly, before any audio device is closed.

        Order matters, and getting it wrong is what crashed the last run:
          1. set _shutting_down, so the visit thread stops STARTING new speech or listens
             (_speak/_listen return immediately once it is set);
          2. cut off whatever is already playing, so a long answer does not hold the shutdown
             open for its full length;
          3. join, bounded - a visit stuck on something unexpected must not hang the exit.

        Bounded rather than unbounded because these are daemon threads: if one somehow outlives
        the join it dies with the interpreter. The join is what makes that almost never happen,
        which is the point - a daemon thread still running while Python finalizes is what
        produced "Fatal Python error: _enter_buffered_busy".
        """
        already = self._shutting_down.is_set()
        self._shutting_down.set()
        self.tts.stop_requested.set()
        # Stop the background LLM re-check (stage05's ChainAnswerGenerator), so a parked cloud
        # backend is not still being probed while the rest of this tears down. It is a daemon
        # thread and would not hold the process open regardless - this just means it stops when
        # asked rather than up to ten minutes later, mid-shutdown.
        closer = getattr(getattr(self, "dialogue", None), "answer_generator", None)
        if hasattr(closer, "close"):
            try:
                closer.close()
            except Exception:  # noqa: BLE001 - never let cleanup stop a shutdown
                pass
        if visit_thread is not None and visit_thread.is_alive():
            if not already:
                log.info("Waiting for the current conversation to wrap up...")
            visit_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            if visit_thread.is_alive():
                log.info("The conversation thread did not stop in time - exiting anyway.")
        # The always-running speaker stream (echo cancellation keeps one open) - after the
        # visit thread, which may still be finishing a sentence on it.
        player = getattr(self, "player", None)
        if player is not None and hasattr(player, "close"):
            try:
                player.close()
            except Exception:  # noqa: BLE001
                pass

    def _drawable_boxes(self, now: float) -> dict[int, tuple[int, int, int, int]]:
        """The face boxes it is still honest to draw, out of everything recognition has stored.

        Two filters, both of which were missing and both of which showed up as extra boxes on
        one face:

          - the track must still be present. A box belonging to a track that has gone is at
            best out of date and at worst somewhere the person no longer is.
          - the box must have been measured within FACE_BOX_MAX_AGE_S. Recognition runs a few
            times a second, so every box is slightly behind the live frame; one that has not
            been refreshed for most of a second is not slightly behind, it is a leftover, and
            drawing it puts a box over empty space beside the person it used to be on.

        Split out from _draw_overlay so this is testable without a camera - it is the part that
        was wrong, and "how many boxes are on my face" is not something a self-test can see.
        """
        with self._state_lock:
            seen = set(self._seen_ids)
            return {
                tid: bbox
                for tid, bbox in self._face_bbox_by_track.items()
                if tid in seen and now - self._face_bbox_at.get(tid, 0.0) <= FACE_BOX_MAX_AGE_S
            }

    def _draw_overlay(self, frame) -> None:
        """Draws who the robot currently thinks is talking to it, so the speaker detection is
        visible while testing instead of being an invisible internal decision. Reads a snapshot
        under the lock and does the drawing outside it - drawing is comparatively slow and must
        not hold up the interaction thread.

        Only ONE box is ever drawn per face. That sounds obvious and was not true before: boxes
        outlived the track they belonged to, they outlived the detection that measured them, and
        the active-speaker highlight drew a second rectangle on top of a face the recognizer had
        already outlined. Moving around in front of the camera produced two or three boxes on
        one face because of it. See FACE_BOX_MAX_AGE_S and _boxes_overlap below.
        """
        face_boxes = self._drawable_boxes(time.monotonic())
        with self._state_lock:
            activities = list(self._activities)
            names = dict(self._resolved_names)
            votes = {tid: list(v) for tid, v in self._identity_votes.items()}
            quality = dict(self._face_quality_by_track)

        # --- what the RECOGNIZER is doing, per tracked person ---------------------------------
        # Recognition was completely invisible on screen: a face that was
        # being ignored for being too small or too blurry looked identical to one that simply
        # did not match anybody, and a name that had been agreed by voting looked identical to a
        # single lucky frame. All three need different fixes, so all three are drawn.
        for track_id, bbox in face_boxes.items():
            x, y, w, h = bbox
            det = quality.get(track_id)
            name = names.get(track_id)
            history = votes.get(track_id, [])
            agreeing = sum(1 for v in history if v is not None)

            if det is not None and not det.is_usable:
                colour, label = (0, 165, 255), f"unusable: {det.quality_reason()}"
            elif name:
                colour = (0, 220, 0)
                label = f"{name} ({agreeing}/{len(history)} frames agree)"
            else:
                colour = (200, 200, 200)
                label = f"unknown ({agreeing}/{len(history)} votes)" if history else "checking..."

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(
                frame,
                label,
                (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )
            if det is not None and det.is_usable:
                cv2.putText(
                    frame,
                    f"{det.width}px  sharp {det.sharpness:.0f}  " f"conf {det.score:.2f}",
                    (x, y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    colour,
                    1,
                    cv2.LINE_AA,
                )

        if not activities:
            return
        speaker = (
            self.mouth_detector.most_likely_speaker(activities) if self.mouth_detector else None
        )
        if speaker is None:
            return

        # The active speaker is found by a DIFFERENT detector (mediapipe, in mouth_activity.py)
        # than the recognizer above (YuNet), so on a single person the two agree about which
        # face it is while disagreeing slightly about where its edges are. Drawing both put two
        # nearly-identical rectangles a few pixels apart on the same face - the second of the
        # "2 or 3 boxes". When the highlight lands on a face that already has a box, label that
        # box instead of drawing a competing one.
        x, y, w, h = speaker.bbox
        existing = next((b for b in face_boxes.values() if _boxes_overlap(b, speaker.bbox)), None)
        if existing is None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            x, y, w, h = existing
        cv2.putText(
            frame,
            "talking to me",
            (x, max(20, y - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def _self_test() -> None:
    """Everything in this file that can be checked without a camera, a microphone or a model.

    Run with: python main_live.py --self-test
    """
    # --- interruptions and course changes ---------------------------------------------------
    control_cases = [
        ("wait", "stop"),
        ("hold on", "stop"),
        ("stop", "stop"),
        ("just a second", "stop"),
        ("Wait! Stop.", "stop"),
        ("hold on a second", "stop"),
        ("No no, that is not what I meant", "change"),
        ("never mind", "change"),
        ("forget it", "change"),
        ("let me ask something else", "change"),
        ("no no lets change", "change"),
        ("scratch that", "change"),
        ("start over", "change"),
        # A control word on the FRONT of a real question is a question - answering it matters
        # more than acknowledging the filler, and a plain word-count rule gets this wrong.
        ("Wait, how long is the Master of Computer Science?", None),
        ("hold on, can you tell me about the engineering department", None),
        ("where is the library", None),
        ("how long is it", None),
        ("what are the entry requirements", None),
        ("", None),
        (None, None),
    ]
    for text, expected in control_cases:
        got = _detect_control_intent(text)
        assert got == expected, f"{text!r} -> {got!r}, expected {expected!r}"
    print(f"[self-test] control intents OK ({len(control_cases)} cases)")

    # --- varied phrasing --------------------------------------------------------------------
    for group in (
        _MORE_QUESTIONS,
        _GOODBYE_AFTER_HELP,
        _NOTHING_ASKED,
        _OUT_OF_QUESTIONS,
        _STOP_ACK,
        _CHANGE_ACK,
        _REPAIR_WITH_CONTEXT,
        _REPAIR_NO_CONTEXT,
    ):
        picks = [_pick(group) for _ in range(200)]
        assert all(a != b for a, b in zip(picks, picks[1:])), "a phrasing repeated back-to-back"
        assert set(picks) == set(group), "some phrasings were never used"
    print("[self-test] phrasing variety OK (no back-to-back repeats, every variant used)")

    # --- answer priming overlaps the acknowledgment ------------------------------------------
    started = threading.Event()

    def _slow():
        started.set()
        time.sleep(0.3)
        yield "first sentence. "
        yield "second bit."

    began = time.monotonic()
    primed = _prime_stream(_slow())
    assert started.wait(1.0), "priming thread never ran"
    time.sleep(0.3)  # the caller doing something else, e.g. playing a filler
    assert "".join(primed) == "first sentence. second bit."
    assert time.monotonic() - began < 0.55, "generation did not overlap the caller's own work"
    assert list(_prime_stream(iter([]))) == []
    print("[self-test] answer priming OK (generation overlaps the acknowledgment)")

    # --- yes/no and name parsing (unchanged behaviour, guarded against regressions) ----------
    assert _parse_yes_no("yes please") and _parse_yes_no("of course")
    assert not _parse_yes_no("no thanks") and not _parse_yes_no(None)
    assert _is_clear_negative("that's all") and not _is_clear_negative("yes")
    print("[self-test] consent parsing OK")

    # --- identity voting --------------------------------------------------------------------
    # One frame must not be enough to name somebody. Built on a bare instance rather than a real
    # orchestrator so this needs no camera, no models and no database.
    class _FakeRecord:
        def __init__(self, person_id, name):
            self.person_id, self.name = person_id, name

    class _FakeStore:
        def __init__(self):
            self.people = {"p1": _FakeRecord("p1", "Sara"), "p2": _FakeRecord("p2", "Ahmad")}

        def get(self, person_id):
            return self.people.get(person_id)

    voter = RobotOrchestrator.__new__(RobotOrchestrator)
    voter._identity_votes = {}
    voter.face_store = _FakeStore()
    sara, ahmad = _FakeStore().people["p1"], _FakeStore().people["p2"]

    # A single recognition is NOT enough to say a name out loud.
    assert voter._vote_identity(1, sara) is None, "one frame was enough to name somebody"
    assert voter._vote_identity(1, sara) is None
    settled = voter._vote_identity(1, sara)
    assert settled is not None and settled.name == "Sara", "three agreeing frames should settle"
    print("[self-test] identity voting OK (one frame never names anybody; three agreeing do)")

    # One bad frame among good ones must not flip the identity.
    voter._identity_votes[2] = []
    for who in (sara, sara, ahmad, sara, sara):
        settled = voter._vote_identity(2, who)
    assert (
        settled is not None and settled.name == "Sara"
    ), f"a single mismatched frame changed the identity to {settled.name if settled else None}"

    # A face nobody matches never settles on a name.
    voter._identity_votes[3] = []
    for _ in range(6):
        assert voter._vote_identity(3, None) is None
    print("[self-test] identity voting is robust to one bad frame and never invents a name")

    # --- how many boxes end up on one face ---------------------------------------------------
    # The reported symptom was two or three boxes appearing on one face while moving about.
    # Three separate causes, so three separate checks.
    face = (100, 100, 80, 80)

    # 1. The two detectors frame the same face differently - YuNet tight around the features,
    #    mediapipe wider - so "the same face" has to mean a generous overlap, not a close one.
    assert _boxes_overlap(
        face, (108, 104, 84, 88)
    ), "two views of the same face were treated as two faces - that is the duplicate box"
    assert _boxes_overlap(face, face)
    # ...while two genuinely different people must stay two boxes.
    assert not _boxes_overlap(face, (400, 100, 80, 80))
    assert not _boxes_overlap(face, (180, 100, 80, 80))  # touching edge-on is not the same face
    assert not _boxes_overlap(face, (100, 100, 0, 0))  # a degenerate box overlaps nothing

    # 2. A track that has gone away must take its box with it. This is the leak that turned
    #    moving around into a trail of frozen boxes: YOLO re-ids a moving person under a new
    #    track number, and every number they used to have kept its last box for the whole run.
    orch = RobotOrchestrator.__new__(RobotOrchestrator)
    orch._state_lock = threading.Lock()
    orch._face_bbox_by_track = {1: face, 2: (300, 100, 80, 80)}
    orch._face_bbox_at = {1: 500.0, 2: 500.0}
    orch._seen_ids = {1}
    orch._activities = []
    orch._resolved_names = {}
    orch._identity_votes = {}
    orch._face_quality_by_track = {}
    orch.mouth_detector = None

    # Track 2 has left. _draw_overlay must not draw it even before the cleanup runs, because it
    # is no longer in _seen_ids...
    drawn = orch._drawable_boxes(now=500.0)
    assert drawn == {1: face}, f"a departed track was still drawable: {drawn}"

    # ...and once the cleanup has run, it is gone from the state entirely.
    for tid in (2,):
        orch._face_bbox_by_track.pop(tid, None)
        orch._face_bbox_at.pop(tid, None)
    assert orch._face_bbox_by_track == {1: face} and orch._face_bbox_at == {1: 500.0}

    # 3. And a box whose detection has gone stale must stop being drawn even while the person is
    #    still tracked - otherwise it hangs in the air beside them as a second box.
    # Values either side of the limit rather than exactly on it - the boundary itself is a
    # float comparison and not worth pinning.
    assert orch._drawable_boxes(now=500.0 + FACE_BOX_MAX_AGE_S * 0.9) == {1: face}
    stale = orch._drawable_boxes(now=500.0 + FACE_BOX_MAX_AGE_S * 1.1)
    assert stale == {}, f"a stale box survived the age filter: {stale}"
    print("[self-test] face boxes: one per face, cleaned up on exit, dropped when stale")

    # --- an empty transcript must not end the visit -------------------------------------------
    # From a real run: the visitor spoke, the transcription came back empty, and the robot went
    # straight to "That's alright - I'm here if you think of something" without ever asking them
    # to repeat. _listen() returns "" for that and None for a timeout, and the caller treated
    # both as "they said nothing".
    def _listener(results, present=True):
        """A stand-in orchestrator that hands out `results` from _listen, one per call."""
        o = RobotOrchestrator.__new__(RobotOrchestrator)
        o._state_lock = threading.Lock()
        o.said = []
        o.listens = 0
        pending = list(results)

        def _listen(timeout_s):
            o.listens += 1
            return pending.pop(0) if pending else None

        o._listen = _listen
        o._speak = lambda text: o.said.append(text)
        o._visitor_present = lambda: present
        return o

    # A question that arrives straight away is returned as-is, with nothing said in between.
    o = _listener(["where is the library"])
    assert o._listen_with_retry(1.0) == "where is the library"
    assert o.said == [] and o.listens == 1

    # An empty transcript earns one "sorry, could you say that again" and a second chance - and
    # the question asked on that second attempt is the one that gets answered.
    o = _listener(["", "how long is the PhD"])
    assert o._listen_with_retry(1.0) == "how long is the PhD", "the retry never happened"
    assert len(o.said) == 1 and o.said[0] in _REPAIR_NO_CONTEXT
    assert o.listens == 2

    # Two empty transcripts in a row is a microphone or load problem, not something a third
    # prompt fixes: give up, having asked exactly MAX_EMPTY_LISTEN_RETRIES times.
    o = _listener(["", ""])
    assert o._listen_with_retry(1.0) is None
    assert len(o.said) == MAX_EMPTY_LISTEN_RETRIES
    assert o.listens == MAX_EMPTY_LISTEN_RETRIES + 1

    # A plain timeout is silence, not a failed transcription - nobody is talking to the robot,
    # so asking them to repeat themselves would be talking to nobody.
    o = _listener([None])
    assert o._listen_with_retry(1.0) is None
    assert o.said == [] and o.listens == 1

    # ...and the same goes for an empty transcript once the visitor has walked off: room noise
    # transcribing to nothing must not make the robot re-prompt an empty room.
    o = _listener(["", "too late"], present=False)
    assert o._listen_with_retry(1.0) is None
    assert o.said == [] and o.listens == 1
    print(
        "[self-test] an empty transcript gets a second chance; silence and an empty room " "do not"
    )

    # --- continuous vs push-to-talk conversation flow -----------------------------------------
    # The behavioural half of the real-time work: in continuous mode a follow-up question is
    # answered without the robot first asking "do you have any other questions?". That prompt is
    # what made every turn wait on the robot, and removing it from the common path is the
    # difference between a conversation and a series of voice recordings. It must still appear
    # when the visitor actually goes quiet, and push_to_talk must still behave exactly as before.
    def _conversation(results, *, push_to_talk):
        """A stand-in orchestrator that runs _handle_conversation against scripted replies."""
        o = RobotOrchestrator.__new__(RobotOrchestrator)
        o._state_lock = threading.Lock()
        o.said = []
        o.answered = []
        o._visit_history = []
        o.conv = ConversationStateMachine(debug=False)
        pending = list(results)

        def _listen(timeout_s):
            return pending.pop(0) if pending else None

        o._listen = _listen
        o._speak = lambda text, intent=None: o.said.append(text)
        o._visitor_present = lambda: True
        o._abandon_if_alone = lambda: False
        o._last_heard_reliable = lambda: True

        def _answer_one(question):
            o.answered.append(question)
            return False  # not interrupted

        o._answer_one = _answer_one
        return o

    original_mode = RT.CONVERSATION_MODE
    original_continuous = RT.CONTINUOUS_CONVERSATION
    try:
        # Continuous: two questions in a row, answered back to back, with no "anything else?"
        # between them.
        RT.CONVERSATION_MODE, RT.CONTINUOUS_CONVERSATION = "continuous", True
        o = _conversation(
            [
                "where is the engineering department",
                "how long does it take to walk there",
                None,
                None,
            ],
            push_to_talk=False,
        )
        o._handle_conversation()
        assert o.answered == [
            "where is the engineering department",
            "how long does it take to walk there",
        ], f"both questions should have been answered back to back, got {o.answered}"
        prompts = [s for s in o.said if s in _MORE_QUESTIONS]
        assert len(prompts) <= 1, f"continuous mode must not prompt between answers - said {o.said}"
        print(
            f"[self-test] continuous mode: {len(o.answered)} questions answered with "
            f"{len(prompts)} 'anything else?' prompt(s) - the follow-up needed no prompt."
        )

        # ...and the prompt still appears when they genuinely go quiet after an answer.
        o = _conversation(["where is the library", None, None, None], push_to_talk=False)
        o._handle_conversation()
        assert any(
            s in _MORE_QUESTIONS for s in o.said
        ), f"silence after an answer should still get a prompt - said {o.said}"
        print("[self-test] continuous mode still prompts when the visitor goes quiet.")

        # ...and a clear "no thanks" straight after an answer ends the visit politely, without a
        # prompt having been needed at all.
        o = _conversation(["where is the library", "no thanks"], push_to_talk=False)
        o._handle_conversation()
        assert any(
            s in _GOODBYE_AFTER_HELP for s in o.said
        ), f"a clear negative should close the visit - said {o.said}"
        assert not any(
            s in _MORE_QUESTIONS for s in o.said
        ), "the robot asked 'anything else?' after being told 'no thanks'"
        print("[self-test] continuous mode: 'no thanks' ends the visit without a prompt.")

        # push_to_talk: the original behaviour, unchanged - a prompt after every answer.
        RT.CONVERSATION_MODE, RT.CONTINUOUS_CONVERSATION = "push_to_talk", False
        o = _conversation(
            [
                "where is the engineering department",
                "how long does it take to walk there",
                None,
                None,
            ],
            push_to_talk=True,
        )
        o._handle_conversation()
        assert any(
            s in _MORE_QUESTIONS for s in o.said
        ), f"push_to_talk must prompt after each answer - said {o.said}"
        print("[self-test] push_to_talk mode still prompts after every answer, as before.")
    finally:
        RT.CONVERSATION_MODE, RT.CONTINUOUS_CONVERSATION = original_mode, original_continuous

    # --- the interrupting words are carried into the next listen -------------------------------
    # The orchestrator half of the barge-in fix (stage04 covers the capture itself). Without
    # this, _listen()'s stale-audio drain deletes the very sentence the visitor interrupted with.
    o = RobotOrchestrator.__new__(RobotOrchestrator)
    o._shutting_down = threading.Event()
    o.audio_q = queue.Queue()
    o.conv = ConversationStateMachine(debug=False)
    o._carried_audio = [b"\x00" * 960] * 12
    o._record = lambda *a, **k: None
    o.stt_engine = o.stt_listener = o.vad = None
    seen: dict = {}

    def _fake_listen(
        engine, listener, vad, audio_q, timeout_s=None, prefix_frames=None, on_turn_event=None
    ):
        seen["prefix"] = prefix_frames
        return "I meant the AI department"

    real_listen = globals()["listen_for_one_utterance"]
    globals()["listen_for_one_utterance"] = _fake_listen
    try:
        o.audio_q.put(b"\x00" * 960)  # stale echo that SHOULD be drained
        heard = o._listen(1.0)
    finally:
        globals()["listen_for_one_utterance"] = real_listen

    assert heard == "I meant the AI department"
    assert seen["prefix"] and len(seen["prefix"]) == 12, (
        "the interrupting audio must reach the listener as prefix_frames - without it the "
        "visitor has to repeat the sentence they just interrupted with"
    )
    assert o._carried_audio == [], "carried audio must be consumed exactly once"
    print("[self-test] barge-in audio survives _listen()'s stale-audio drain and is used once.")

    print(
        "\nOK: interruptions, course changes, phrasing variety, answer priming, consent "
        "parsing, face-box drawing, continuous/push-to-talk conversation flow and barge-in "
        "carry-over all behave - none of which needs hardware to verify."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)

    # --refresh forces a fresh lincoln.edu.my scrape even if the on-disk knowledge-base cache is
    # still within KB_CACHE_MAX_AGE_S - use it after the website has changed (or after editing
    # the scraper) instead of hunting down and deleting the cache file by hand.
    force_refresh = "--refresh" in sys.argv
    if force_refresh:
        print("--refresh given: ignoring the knowledge-base cache and re-scraping now.")
    # --calibrate-barge-in measures how loudly this room returns the robot's own voice to the
    # microphone and prints the threshold that would be used, without starting a visit. Run it
    # once on the real kiosk hardware to find out whether barge-in (interrupting the robot
    # mid-sentence) can work at all here - see the barge-in note in stage04's stt_listener_live.py
    # for why that is a property of the room and the microphone rather than of the code.
    if "--calibrate-barge-in" in sys.argv:
        print(
            "Calibrating barge-in. Stay QUIET - everything the mic hears now is the robot's "
            "own voice coming back."
        )
        orchestrator = RobotOrchestrator(force_refresh=False, skip_knowledge_base=True)
        threshold = orchestrator._calibrate_barge_in(manage_stream=True)
        if threshold is None:
            print("\nResult: barge-in is NOT usable on this hardware as it stands.")
        else:
            print(
                "\nResult: barge-in looks usable. INTERRUPTION_ENABLED=auto (the default) "
                "switches it on at startup; test it by talking over the robot mid-answer."
            )
        sys.exit(0)

    orchestrator = RobotOrchestrator(force_refresh=force_refresh)
    orchestrator.run()
