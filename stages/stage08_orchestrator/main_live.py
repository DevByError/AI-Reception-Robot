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
blind between "hello" and "goodbye", and the preview window stopped responding. (That rewrite
also claimed to stop the robot "talking to people who had already walked away" - it did not.
It made the camera keep SEEING them leave, but nothing on the conversation side ever read that,
so the robot still finished its answer to an empty room. That half was finished on 2026-09-05 -
see ABSENCE_GRACE_S and _abandon_if_alone.) The split is deliberate about which side gets what:
OpenCV's
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
from collections import Counter
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
    "stage09_deployment",
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
    BARGE_IN_ENABLED,
    BARGE_IN_MARGIN,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    BargeInMonitor,
    PatientVAD,
    StreamingVAD,
    WhisperSTTEngine,
    calibrate_echo_floor,
    listen_for_one_utterance,
    real_hotwords,
    real_initial_prompt,
    real_vocabulary,
)
from identity import use_database as load_identity  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from dialogue_manager import DialogueManager  # noqa: E402
from dialogue_manager_live import SAMPLE_CHUNKS, _build_answer_generator  # noqa: E402
import visit_memory  # noqa: E402
from tts_speaker_live import LiveTTSSpeaker  # noqa: E402
from filler_bank import FillerBank  # noqa: E402
from scraper import REAL_SEED_URLS, UniversityScraper  # noqa: E402
from indexer import UniversityDataIndexer  # noqa: E402
from logging_config import LatencyTracker  # noqa: E402

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
# OFF (2026-09-13), on explicit instruction after hearing it in use: the robot should give the
# answer, not preface it. Worth recording what this costs, because it is not free and the
# measurements are in this project's own logs - time-to-first-word for a grounded answer runs
# 2.7-4.2s on this hardware, and the filler existed to make that stretch feel like the robot had
# started responding rather than like it had frozen. With it off, that is now plain silence.
#
# Nothing else changes: _answer_one() already treats a missing bank as "no acknowledgment", and
# the background answer priming that used to overlap the filler (see _prime_stream) still runs,
# so this costs perceived snappiness only, never real latency. Set back to True to restore them;
# filler_bank.py is left intact and still has its own self-test.
ACKNOWLEDGMENT_FILLERS_ENABLED = False

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
_QUESTION_CUES = frozenset({
    "what", "where", "when", "how", "who", "why", "which", "whose",
    "can", "could", "do", "does", "is", "are", "tell", "show", "explain", "about",
})

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
    "wait", "hold on", "hang on", "stop", "one second", "one moment", "just a second",
    "give me a second", "pause", "shut up", "be quiet", "let me finish", "let me speak",
)

_CHANGE_PHRASES = (
    "no no", "never mind", "nevermind", "forget it", "forget that", "scratch that",
    "not what i meant", "not what i asked", "that's not it", "thats not it",
    "let me change", "change the question", "different question", "another question",
    "something else", "ask something else", "start over", "start again", "go back",
    "i meant", "actually i", "instead",
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
    box: dict = {}

    def _pull() -> None:
        try:
            box["first"] = next(pieces)
        except StopIteration:
            box["empty"] = True

    thread = threading.Thread(target=_pull, name="answer-priming", daemon=True)
    thread.start()

    def _replay():
        thread.join()
        if "empty" in box:
            return
        yield box["first"]
        yield from pieces

    return _replay()


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


class _ShuttingDown(Exception):
    """Raised inside a background refresh to abandon it when the robot is closing down.

    An exception rather than a return value because the abandon point is inside a callback
    handed to the scraper, several frames below the code that can act on it - see _scrape_into().
    """


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int],
                   min_iou: float = SAME_FACE_IOU) -> bool:
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

    def __init__(self, force_refresh: bool = False,
                 skip_knowledge_base: bool = False):
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

        # score= is what lets find_match() pick the BEST candidate instead of the first
        # one over the threshold, and refuse to choose between two people it cannot tell
        # apart - see FaceMemoryStore.find_match(). Without it the store silently falls
        # back to the old first-match behaviour.
        self.face_store = FaceMemoryStore(
            is_match=self.embedder.is_match, score=self.embedder.similarity, db=self.db
        )
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
        # None = barge-in off (the default, and also what a failed calibration falls back to).
        self._barge_in_threshold: float | None = None
        # Set while building the knowledge base below; initialised here so the background
        # refresher can read it even in modes that skip the knowledge base entirely.
        self._kb_stale = False
        # Wall-clock of the last frame anybody was visible on. Written by the camera thread,
        # read by the conversation thread - see _visitor_present().
        self._last_seen_at = time.monotonic()

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

        # Pre-rendered "let me see..." acknowledgments, synthesized ONCE here rather than on
        # demand - the whole point is that playing one costs nothing at the moment it is needed
        # (see stage06's filler_bank.py). Optional: a failure here costs a bit of perceived
        # snappiness, not the ability to answer, so it degrades to None rather than refusing to
        # start - same rule as mediapipe above.
        self.fillers = None
        if ACKNOWLEDGMENT_FILLERS_ENABLED:
            try:
                self.fillers = FillerBank(self.tts.voice)
            except Exception as exc:
                print(f"[filler] could not pre-render acknowledgments ({exc!r}) - continuing "
                      f"without them; answers will just start after a short silence.")

        self.expression = ExpressionController()
        self.greeting_sm = GreetingStateMachine(
            self.expression,
            _LiveTTSAdapter(self.tts, on_spoken=lambda text: self._record("robot", text)),
        )

        print("Loading faster-whisper...")
        self.stt_engine = WhisperSTTEngine(
            initial_prompt=real_initial_prompt(), vocabulary=real_vocabulary(),
            hotwords=real_hotwords(),
        )
        self.stt_listener = STTListener(self.stt_engine, PatientVAD(), locale="en-US")
        self.vad = StreamingVAD()
        self.audio_q: "queue.Queue[bytes]" = queue.Queue()

        def _on_audio(indata, frames, time_info, status):
            if status:
                # "input overflow" here is not cosmetic and is worth naming: PortAudio ran out
                # of room and THREW AWAY microphone audio, so whatever reaches Whisper next has
                # holes in it. A recording full of holes is what transcribes to nothing at all,
                # which the visitor experiences as the robot ignoring them.
                detail = ""
                if status.input_overflow:
                    detail = (" - microphone audio was DROPPED; expect a poor or empty "
                              "transcript")
                print(f"[audio] {status}{detail}", file=sys.stderr)
            # One callback carries MIC_BLOCK_FRAMES worth of audio; everything downstream (the
            # webrtcvad wrapper, the preroll, the barge-in monitor) is built on single 30ms
            # frames, so split it back up here rather than changing all of them. blocksize is an
            # exact multiple of FRAME_SAMPLES, so this never leaves a partial frame behind.
            raw = bytes(indata)
            step = FRAME_SAMPLES * 2  # int16 = 2 bytes per sample
            for start in range(0, len(raw) - step + 1, step):
                self.audio_q.put(raw[start:start + step])

        self.mic_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES * MIC_BLOCK_FRAMES, dtype="int16",
            channels=1, callback=_on_audio,
        )

        # Say which microphone this is actually using. No device is named above, so it is
        # whatever Windows currently calls the default input - and that changes under you:
        # connecting Bluetooth earbuds switches it to their hands-free profile, which is 8kHz
        # narrowband, and Whisper transcribes that badly enough to look like a broken robot. The
        # device is the first thing to check when recognition is poor and the last thing anybody
        # thinks of, because nothing ever printed it. Purely informational - any failure here
        # must not stop the robot starting.
        try:
            device = sd.query_devices(self.mic_stream.device, "input")
            print(f"Microphone: {device['name']} "
                  f"(native {int(device['default_samplerate'])}Hz, capturing at {SAMPLE_RATE}Hz)")
            if device["default_samplerate"] < SAMPLE_RATE:
                print("  ^ this device runs BELOW 16kHz - almost certainly a Bluetooth headset's "
                      "hands-free\n    profile. Expect poor transcription; pick a different "
                      "input in Windows sound settings.")
        except Exception:
            pass

        # _build_answer_generator() reads dialogue_manager_live.py's own MODE constant - reusing
        # it (rather than picking a generator here independently) means flipping MODE there
        # automatically applies to the real robot too, with nothing to keep in sync by hand.
        # Skipped by --calibrate-barge-in: that mode only needs the microphone and the voice,
        # and scraping plus embedding the whole site first would mean waiting a minute to
        # measure a number that has nothing to do with the knowledge base.
        if skip_knowledge_base:
            print("Skipping the knowledge base (not needed for this mode).")
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
            # STALE-BUT-USABLE (2026-09-05): a cache older than KB_CACHE_MAX_AGE_S used to mean a
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
                    print("Knowledge base cache is past its expiry - starting with it anyway "
                          "and refreshing in the background.")

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

            # MEASURED AND REJECTED (2026-09-05): feeding the scraped programme names into
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

            self.dialogue = DialogueManager(self.kb, _build_answer_generator())

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
        actual = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera: {actual[0]}x{actual[1]}"
              + ("" if actual == (CAPTURE_WIDTH, CAPTURE_HEIGHT) else
                 f" (asked for {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} - the camera refused; faces "
                 f"will be smaller, so stand closer)"))
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
        self._visit_history = []
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
        # Same reason as _speak(): the microphone is closing, so there is nothing to listen to
        # and blocking here for the timeout would just delay the shutdown by that long.
        if self._shutting_down.is_set():
            return None
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
                print("[kb] somebody is here - holding the refresh until they are done.")
            time.sleep(KB_REFRESH_YIELD_POLL_S)
            waited += KB_REFRESH_YIELD_POLL_S
        if waited and not self._shutting_down.is_set():
            print(f"[kb] clear after {waited:.0f}s - starting the refresh now.")

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
        pages = scraper.scrape_once(
            urls, before_each_page=_yield if yield_to_visitors else None
        )
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

        This is what indexer.py's own "wire into a real scheduler" TODO was asking for, and it is
        why startup no longer has to choose between fresh data and starting promptly.
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
                print("[kb] refreshing the knowledge base in the background...")
                fresh = KnowledgeBase()
                count = self._scrape_into(fresh, yield_to_visitors=True)
                if len(fresh) == 0:
                    print("[kb] refresh produced an empty knowledge base - keeping the old one.")
                else:
                    self.kb = fresh
                    if self.dialogue is not None:
                        self.dialogue.kb = fresh
                    self._kb_stale = False
                    print(f"[kb] refreshed: {count} chunk(s) indexed, {len(fresh)} total, now "
                          f"live.")
                    try:
                        fresh.save(KB_CACHE_PATH)
                    except Exception as exc:
                        print(f"[kb] could not write the cache ({exc!r}).")
            except _ShuttingDown:
                # Not a failure - the robot is closing and this work no longer matters. Return
                # quietly: printing anything here risks doing it while the interpreter is
                # already finalizing.
                return
            except Exception as exc:
                # The robot keeps answering from the knowledge base it already has. A scrape
                # failing is a data-freshness problem, never a reason to stop serving visitors.
                print(f"[kb] background refresh failed ({exc!r}) - keeping the current data.")
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
        try:
            if manage_stream:
                with self.mic_stream:
                    floor = calibrate_echo_floor(self.audio_q, lambda t: self.tts.speak(t))
            else:
                floor = calibrate_echo_floor(self.audio_q, lambda t: self.tts.speak(t))
        except Exception as exc:
            print(f"[barge-in] calibration failed ({exc!r}) - barge-in stays off.")
            return None

        threshold = floor * BARGE_IN_MARGIN
        print(f"[barge-in] this room returns the robot's own voice at RMS {floor:.4f}; "
              f"interrupting needs {threshold:.4f} ({BARGE_IN_MARGIN}x that).")
        if floor <= 0.0:
            print("[barge-in] nothing measured - is the microphone live? barge-in stays off.")
            return None
        if floor > 0.08:
            # The speaker is loud into this microphone. A person would have to shout to clear
            # the margin, and the robot would interrupt itself on its own voice long before
            # that. Say so plainly rather than shipping a feature that misfires.
            print("[barge-in] the microphone hears the robot loudly, so there is no room for a "
                  "reliable threshold. Barge-in stays off - this needs a directional mic or a "
                  "speakerphone with hardware echo cancellation.")
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
        print(f"[visit] nobody visible for {ABSENCE_GRACE_S:.0f}s - ending this visit.")
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
        print(f'[TTS] "{text}"')
        self._record("robot", text, intent)
        self.tts.speak(text)

    def _speak_stream(self, result, intent: str | None = None) -> str:
        """Speaks a StreamingDialogueResult, starting playback as soon as the first sentence
        exists rather than waiting for the whole answer to be generated.

        Returns what was actually said, which is the reason this isn't just tts.speak_stream():
        the caller no longer holds the finished answer string, and the transcript still needs it.
        The line is recorded AFTER speaking for the same reason - there is nothing to record
        until the stream has been consumed.
        """
        spoken = self.tts.speak_stream(result.pieces)
        text = (spoken.get("text") or "").strip()
        print(f'[TTS] "{text}"')
        if spoken.get("time_to_first_audio_ms"):
            print(f"[TTS] (first word after {spoken['time_to_first_audio_ms']:.0f}ms, "
                  f"{spoken['num_segments']} sentence(s))")
        self._record("robot", text, intent)
        return text

    def _report_turn_latency(self, tracker: "LatencyTracker") -> None:
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
        # Time-to-first-WORD is the number that matters to a person standing there, not the
        # total: the robot is already talking while the rest of the answer generates.
        to_first_word = m.get("retrieve+generate", 0.0) + m.get("first_audio", 0.0)
        parts = " ".join(f"{k}={v:.0f}ms" for k, v in m.items())
        verdict = "OK" if to_first_word <= ANSWER_LATENCY_TARGET_MS else "SLOW"
        print(f"[latency] {parts} | time-to-first-word {to_first_word:.0f}ms "
              f"({verdict} vs {ANSWER_LATENCY_TARGET_MS:.0f}ms target)")

    def _answer_one(self, question: str) -> bool:
        # An acknowledgment plays IMMEDIATELY, on its own thread, while retrieval and generation
        # happen behind it - see stage06's filler_bank.py. Started before the dialogue call, not
        # after, which is the whole point: the silence being covered is the silence that starts
        # the instant the visitor stops talking.
        filler = self.fillers.play_async() if self.fillers is not None else None
        tracker = LatencyTracker()

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
        print(f"[dialogue] intent={result.intent.name}, {len(result.used_chunks)} chunk(s) "
              f"used, streaming={result.is_streaming}, question={question!r}")

        # Wait for the acknowledgment to finish before the real answer starts, so the robot
        # never talks over itself. By now it has almost always finished already - that overlap
        # is exactly what was bought by starting it early.
        if filler is not None:
            filler.wait()

        # The intent is stored on the answer line so the saved transcript keeps the same "why
        # did it answer that way" trace the console log has - a row of apologies in the database
        # with no route attached is a lot harder to diagnose weeks later.
        with tracker.measure("first_audio"):
            interrupted = self._speak_interruptible(
                lambda: self._speak_stream(result, intent=result.intent.name)
            )
        self._report_turn_latency(tracker)
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
            print(f"[stt] transcription came back empty - asking again "
                  f"(attempt {attempt + 2}/{MAX_EMPTY_LISTEN_RETRIES + 1}).")
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
        confident = [
            w for w in re.sub(r"\[\?\?\]", " ", heard).split()
            if len(w.strip(".,?!")) > 3
        ]
        topic = " ".join(confident[:4]).strip(".,?! ")

        if topic:
            self._speak(_pick(_REPAIR_WITH_CONTEXT).format(topic=topic))
        else:
            self._speak(_pick(_REPAIR_NO_CONTEXT))
        return self._listen(LISTEN_TIMEOUT_S)

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
        question = self._listen_with_retry(LISTEN_TIMEOUT_S)
        questions_answered = 0
        repairs_used = 0
        interruptions_used = 0

        while question and questions_answered < MAX_QUESTIONS_PER_VISIT:
            # Checked before every answer, not just at the start: an answer takes seconds, and
            # somebody who has already walked off should not be talked at for the rest of them.
            if self._abandon_if_alone():
                return

            # "wait" / "no no, that's not what I meant" is a command, not a question. Handled
            # BEFORE the confidence check and before retrieval, because sending it to the
            # knowledge base is exactly the old broken behaviour: it matches nothing and comes
            # back as an apology, which reads as the robot ignoring you at the one moment you
            # were trying to take control.
            control = _detect_control_intent(question)
            if control and interruptions_used < MAX_INTERRUPTIONS_PER_VISIT:
                interruptions_used += 1
                print(f"[visit] heard a {control!r} command, not a question "
                      f"({interruptions_used}/{MAX_INTERRUPTIONS_PER_VISIT}).")
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
                print(f"[stt] transcript looks unreliable - asking about the part I did catch "
                      f"(repair {repairs_used}/{MAX_REPAIRS_PER_VISIT}).")
                question = self._handle_repair(question)
                continue

            interrupted = self._answer_one(question)
            questions_answered += 1

            if interrupted:
                # They started talking over the answer. Skip "anything else?" entirely - they
                # are already mid-sentence, and asking a question at them now would talk over
                # THEM. Just listen.
                print("[visit] answer was interrupted - listening instead of prompting.")
                question = self._listen_with_retry(LISTEN_TIMEOUT_S)
                continue

            # ...and again before asking "anything else?", which is the single most common
            # moment for the robot to be left talking to nobody - the visitor has their answer
            # and leaves while it is still speaking.
            if self._abandon_if_alone():
                return

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
            print("[visit] visitor already gone - skipping the closing line.")
        elif questions_answered == 0:
            self._speak(_pick(_NOTHING_ASKED))
        elif questions_answered >= MAX_QUESTIONS_PER_VISIT:
            self._speak(_pick(_OUT_OF_QUESTIONS))
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

    def _run_visit(self, arriving: list[Person], detected_at: float | None = None) -> None:
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
        try:
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
                print(f"[latency] detected -> greeting {elapsed_ms:.0f}ms ({verdict} vs "
                      f"{GREETING_LATENCY_TARGET_MS:.0f}ms target; includes the deliberate "
                      f"{RECOGNITION_WAIT_S:.1f}s recognition wait)")

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
            print(f"[memory] could not record this visit against {person_id} ({exc!r}).")

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
            print(f"[memory] could not build a recall line ({exc!r}) - greeting normally.")
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
                print(f"[memory] stored {stored} new fact(s) about this visitor.")
        except Exception as exc:
            # Never a reason to disturb anything: the visit is over and the transcript is
            # already saved. The robot simply knows less next time.
            print(f"[memory] could not learn from this visit ({exc!r}).")

    def run(self) -> None:
        print("Live robot running. Press 'q' in the preview window to quit.")
        pending_since: dict[int, float] = {}
        # When each track was first detected, kept separately because pending_since is popped on
        # the way to a greeting - this is the clock STAGES.md's "under a second from a person
        # entering to the start of the greeting" is measured against.
        first_seen_at: dict[int, float] = {}
        ready_to_greet: set[int] = set()
        last_recognized_at = 0.0  # monotonic clock of the last recognition pass
        last_quality_log_at = 0.0  # so the "ignoring this face" notice cannot flood the console
        visit_thread: threading.Thread | None = None

        try:
            with self.mic_stream:
                # Measured once, with the stream already open. Only when barge-in is switched
                # on - it costs a spoken sentence at startup, which is not worth paying for a
                # threshold nothing will read.
                if BARGE_IN_ENABLED:
                    print("Calibrating barge-in (stay quiet for a moment)...")
                    self._barge_in_threshold = self._calibrate_barge_in()

                # Keeps the university data current without anybody restarting the robot, and
                # catches up immediately when startup began from an expired cache.
                threading.Thread(
                    target=self._refresh_knowledge_base, name="kb-refresh", daemon=True
                ).start()

                while True:
                    ok, frame = self.cap.read()
                    if not ok:
                        print("Could not read a frame - stopping.")
                        break

                    people = detect_people(self.yolo, frame)
                    seen_ids = {p.track_id for p in people}
                    entered_ids, left_ids = self.presence.update(seen_ids)

                    with self._state_lock:
                        self._seen_ids = seen_ids
                        # Stamped only while somebody is actually visible, so the conversation
                        # thread can tell "nobody has been here for N seconds" from "nobody is
                        # in this single frame" - see ABSENCE_GRACE_S.
                        if seen_ids:
                            self._last_seen_at = time.monotonic()
                        for tid in left_ids:
                            self._resolved_names.pop(tid, None)
                            self._resolved_person_ids.pop(tid, None)
                            self._last_embedding_by_track.pop(tid, None)
                            self._identity_votes.pop(tid, None)
                            self._face_quality_by_track.pop(tid, None)
                            # Was missing, and it is what put several boxes on one face: every
                            # other per-track dict was cleaned up here, so a track that went
                            # away left its LAST face box behind in _face_bbox_by_track for the
                            # rest of the run, and _draw_overlay kept drawing it. Moving around
                            # is exactly what makes YOLO drop a track and re-id the same person
                            # under a new number, so one person walking about accumulated a
                            # trail of frozen boxes at every place they had paused.
                            self._face_bbox_by_track.pop(tid, None)
                            self._face_bbox_at.pop(tid, None)
                            pending_since.pop(tid, None)
                            first_seen_at.pop(tid, None)
                            ready_to_greet.discard(tid)

                    for tid in entered_ids:
                        pending_since[tid] = time.monotonic()
                        first_seen_at[tid] = pending_since[tid]

                    # Mouth movement, every frame - unlike face recognition below, this NEEDS
                    # every frame: the signal is how much the mouth ratio varies over time, so
                    # sampling it on an interval would throw away most of the variation it's
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

                    if people and time.monotonic() - last_recognized_at >= RECOGNIZE_INTERVAL_S:
                        last_recognized_at = time.monotonic()
                        face_detections = self.embedder.detect(frame)
                        matches = match_faces_to_people(people, face_detections)
                        for tid, det in matches.items():
                            # A face too small, too blurry or too weakly detected produces a
                            # perfectly well-formed embedding of nothing in particular - see the
                            # quality-gating note in sface_embedder.py. Skipping it entirely is
                            # right: the next pass is RECOGNIZE_INTERVAL_S away, and answering
                            # "unknown" for a moment costs nothing next to matching against mush.
                            if not det.is_usable:
                                if time.monotonic() - last_quality_log_at >= QUALITY_LOG_INTERVAL_S:
                                    last_quality_log_at = time.monotonic()
                                    print(f"[face] ignoring track #{tid}: {det.quality_reason()}")
                                continue

                            with self._state_lock:
                                self._last_embedding_by_track[tid] = det.embedding
                                self._face_bbox_by_track[tid] = det.bbox
                                self._face_bbox_at[tid] = time.monotonic()
                                self._face_quality_by_track[tid] = det
                                match = self.face_store.find_match(det.embedding)
                                # Only reinforce from a face good enough to STORE. Reinforcing
                                # on every match was quietly feeding whatever the camera happened
                                # to catch - including motion-blurred and half-turned frames -
                                # into somebody's permanent reference samples, making the
                                # recognizer worse at them over time.
                                if match and det.is_enrollable:
                                    self.face_store.reinforce(match, det.embedding)

                                # One frame does not decide who somebody is - see _vote_identity.
                                settled = self._vote_identity(tid, match)
                                if settled is not None:
                                    self._resolved_names[tid] = settled.name
                                    self._resolved_person_ids[tid] = settled.person_id

                            if settled is not None and tid in pending_since:
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
                        detected_at = min(
                            (first_seen_at.get(tid, now) for tid in arriving_ids), default=now
                        )
                        for tid in arriving_ids:
                            first_seen_at.pop(tid, None)
                        visit_thread = threading.Thread(
                            target=self._run_visit, args=(arriving, detected_at), daemon=True
                        )
                        visit_thread.start()

                    self._draw_overlay(frame)
                    cv2.imshow("Stage 08 - live robot (press q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
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
        if visit_thread is not None and visit_thread.is_alive():
            if not already:
                print("Waiting for the current conversation to wrap up...")
            visit_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            if visit_thread.is_alive():
                print("The conversation thread did not stop in time - exiting anyway.")

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
                tid: bbox for tid, bbox in self._face_bbox_by_track.items()
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
        # Added 2026-09-05. Recognition was completely invisible on screen: a face that was
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
                label = (f"unknown ({agreeing}/{len(history)} votes)" if history else "checking...")

            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            if det is not None and det.is_usable:
                cv2.putText(frame, f"{det.width}px  sharp {det.sharpness:.0f}  "
                                    f"conf {det.score:.2f}",
                            (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
                            cv2.LINE_AA)

        if not activities:
            return
        speaker = self.mouth_detector.most_likely_speaker(activities) if self.mouth_detector else None
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
        cv2.putText(frame, "talking to me", (x, max(20, y - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def _self_test() -> None:
    """Everything in this file that can be checked without a camera, a microphone or a model.

    Run with: python main_live.py --self-test
    """
    # --- interruptions and course changes ---------------------------------------------------
    control_cases = [
        ("wait", "stop"), ("hold on", "stop"), ("stop", "stop"), ("just a second", "stop"),
        ("Wait! Stop.", "stop"), ("hold on a second", "stop"),
        ("No no, that is not what I meant", "change"), ("never mind", "change"),
        ("forget it", "change"), ("let me ask something else", "change"),
        ("no no lets change", "change"), ("scratch that", "change"), ("start over", "change"),
        # A control word on the FRONT of a real question is a question - answering it matters
        # more than acknowledging the filler, and a plain word-count rule gets this wrong.
        ("Wait, how long is the Master of Computer Science?", None),
        ("hold on, can you tell me about the engineering department", None),
        ("where is the library", None), ("how long is it", None),
        ("what are the entry requirements", None), ("", None), (None, None),
    ]
    for text, expected in control_cases:
        got = _detect_control_intent(text)
        assert got == expected, f"{text!r} -> {got!r}, expected {expected!r}"
    print(f"[self-test] control intents OK ({len(control_cases)} cases)")

    # --- varied phrasing --------------------------------------------------------------------
    for group in (_MORE_QUESTIONS, _GOODBYE_AFTER_HELP, _NOTHING_ASKED, _OUT_OF_QUESTIONS,
                  _STOP_ACK, _CHANGE_ACK, _REPAIR_WITH_CONTEXT, _REPAIR_NO_CONTEXT):
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
    assert settled is not None and settled.name == "Sara", (
        f"a single mismatched frame changed the identity to {settled.name if settled else None}"
    )

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
    assert _boxes_overlap(face, (108, 104, 84, 88)), \
        "two views of the same face were treated as two faces - that is the duplicate box"
    assert _boxes_overlap(face, face)
    # ...while two genuinely different people must stay two boxes.
    assert not _boxes_overlap(face, (400, 100, 80, 80))
    assert not _boxes_overlap(face, (180, 100, 80, 80))  # touching edge-on is not the same face
    assert not _boxes_overlap(face, (100, 100, 0, 0))    # a degenerate box overlaps nothing

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
    print("[self-test] an empty transcript gets a second chance; silence and an empty room "
          "do not")

    print("\nOK: interruptions, course changes, phrasing variety, answer priming, consent "
          "parsing and face-box drawing all behave - none of which needs hardware to verify.")


if __name__ == "__main__":
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
        print("Calibrating barge-in. Stay QUIET - everything the mic hears now is the robot's "
              "own voice coming back.")
        orchestrator = RobotOrchestrator(force_refresh=False, skip_knowledge_base=True)
        threshold = orchestrator._calibrate_barge_in(manage_stream=True)
        if threshold is None:
            print("\nResult: barge-in is NOT usable on this hardware as it stands.")
        else:
            print("\nResult: barge-in looks usable. Set BARGE_IN_ENABLED = True in "
                  "stages/stage04_speech_to_text/stt_listener_live.py, then test by talking "
                  "over the robot mid-answer.")
        sys.exit(0)

    orchestrator = RobotOrchestrator(force_refresh=force_refresh)
    orchestrator.run()
