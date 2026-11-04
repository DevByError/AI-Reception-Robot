"""
Stage 04 (live) - the assembled single-speaker pipeline:

  microphone -> preprocessing (optional denoise/normalize) -> VAD -> ASR (Whisper, word-level
  confidence) -> domain vocabulary correction (optional, low-confidence words only) ->
  confidence gating -> final text

Reuses STTListener/VoiceActivityDetector from stt_listener.py UNCHANGED - only the pieces that
were placeholders there (MockSTTEngine, and the caller manually supplying trailing_silence_ms)
are replaced with real audio and a real per-frame VAD, same "swap only what's fake" approach as
greeting_live.py in Stage 03.

Deliberately NOT included here: speaker diarization (see diarization.py - a standalone, opt-in
module, not wired into this loop, since this pipeline is built for one person talking to the
robot at a time, matching the project's actual design - ARCHITECTURE.md/README.md treat a group
as one shared interaction, not per-speaker turns) and real fine-tuning (see README.md's
"fine-tune only when necessary" section - no training data exists yet, and Whisper's own
pretrained accuracy needs to be measured with evaluate_wer.py before fine-tuning would even be
worth considering).

English only for now (STAGES.md) - locale is threaded through unchanged so Arabic/Chinese can be
added later without touching STTListener's interface.

Install first: see README.md in this folder (includes the one-time model download note).
Run: python stt_listener_live.py
Speak into your microphone - after you stop talking for about PatientVAD's threshold_ms (see
below), it prints what it heard and starts listening for the next utterance. Press Ctrl+C to
quit.
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad

import audio_preprocessing
import vocabulary_correction
from stt_listener import STTListener, TranscriptChunk, VoiceActivityDetector

# Every tunable in this file now takes its DEFAULT from here instead of being a literal, so a
# noisy lobby or a different microphone is a .env edit rather than a code edit. The constants
# below keep their names, their homes and the comments explaining the measurement behind each
# one - only the number moved. See stage00_setup/realtime_config.py.
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from realtime_config import RT  # noqa: E402


def rt_log(tag: str, message: str) -> None:
    """One tagged line of the real-time trace ([VAD], [TURN], [STT], [INTERRUPTION]...).

    Separate from this file's existing "[stt] ..." prints, which are left alone: those describe
    what the STAGE is doing and have been read in this project's logs for a long time. These
    describe what the CONVERSATION is doing, are the trace a real-time pipeline is debugged
    from, and can be switched off as a group with REALTIME_DEBUG=false without losing the
    former.
    """
    if RT.REALTIME_DEBUG:
        print(f"[{tag}] {message}")

# The robot's own name and its creator's name are things Whisper has to hear correctly (see
# identity_vocabulary() below), and since 2026-09-04 they live in MySQL rather than in source,
# so
# they are read from the one module that owns them (Stage 05's identity.py) rather than spelled
# out again here, where a rename in the database would silently leave this list biasing Whisper
# toward the OLD name. identity.py imports nothing heavier than the standard library and only
# touches MySQL on first use, so this stays a cheap import.
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage05_nlu_dialogue"))
from identity import get_identity  # noqa: E402

SAMPLE_RATE = 16000  # webrtcvad only accepts 8000/16000/32000/48000 Hz
FRAME_MS = 30  # webrtcvad only accepts 10/20/30ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480

# Went to 3 (strictest) briefly to kill false triggers on room noise, but that backfired - it
# also flags quieter opening consonants/vowels as "not speech" mid-word, delaying when a
# recording actually starts and clipping the first word or two beyond what PREROLL_FRAMES can
# recover. Back to 1 (still filters obvious silence, keeps far more real speech) - the noise/
# hallucination problem is handled downstream instead (MIN_UTTERANCE_MS + the transcribe()
# settings below + confidence gating), not by starving the model of real audio upfront.
VAD_AGGRESSIVENESS = RT.VAD_AGGRESSIVENESS  # 0 (most permissive) - 3 (most aggressive)

# A single noisy/misclassified frame shouldn't start a recording - require a short run of
# consecutive speech-flagged frames first (same debouncing idea as Stage 01's
# ENTER_CONFIRM_FRAMES, just for audio instead of video). Kept short since MIN_UTTERANCE_MS
# below is what actually screens out pure noise now, not this.
SPEECH_START_FRAMES = RT.SPEECH_START_FRAMES  # default 4, ~120ms

# Frames of audio kept from *before* speech was confirmed, prepended once recording starts, so
# the first syllable isn't clipped while we were still debouncing SPEECH_START_FRAMES - the
# listening-side mirror of the clipped-audio fix LiveTTSSpeaker needed on the speaking side in
# Stage 03. Widened from 300ms to 600ms for extra margin.
PREROLL_FRAMES = RT.PREROLL_FRAMES  # default 20, ~600ms

MAX_UTTERANCE_S = RT.MAX_UTTERANCE_S  # default 15 - safety cap; forces a transcription
                                       # even if silence never triggers
# (stuck-open mic, constant background noise misread as speech)

# If VAD confirms "speech started" but trailing silence hits immediately after, the whole clip is
# too short to be real words - transcribing it anyway is exactly what produces Whisper's
# hallucinated filler ("Okay. Okay. Okay.") on essentially empty audio. Discard it silently
# instead of sending it to the model.
MIN_UTTERANCE_MS = RT.MIN_SPEECH_DURATION  # default 400

# A word below this probability (faster-whisper's own per-word confidence, 0-1) is treated as
# uncertain: first tried against vocabulary_correction (if a vocabulary was provided), then
# flagged with LOW_CONFIDENCE_MARKER if that doesn't resolve it either - never silently trusted.
WORD_CONFIDENCE_THRESHOLD = 0.5
LOW_CONFIDENCE_MARKER = "[??]"

# If more than this fraction of a transcript's words end up flagged, treat the WHOLE utterance
# as unreliable (not just the individual words) - see ConfidenceReport.is_reliable.
UNRELIABLE_WORD_FRACTION = 0.5
# ...or if Whisper's own segment-level "this might not even be speech" signal is this high.
UNRELIABLE_NO_SPEECH_PROB = 0.6

# Optional partial preview while a long utterance is still being recorded (streaming vs. batch
# tradeoff - STAGES.md's own note that "very small chunks: low latency + less context, large
# chunks: more context + higher latency"). Re-running Whisper on a growing buffer every couple of
# seconds is real extra CPU load for a kiosk that isn't doing live captioning, which is why this
# was off for a long time.
#
# ON by default now (STREAMING_STT=true), for a reason that is about debuggability rather than
# latency: the partial transcript is the ONLY visible evidence of what the robot is hearing while
# somebody is still talking. Without it, "why did it answer me in the middle of my sentence?" can
# only be reconstructed from silence timings, and the turn detector - which is the single most
# important thing in a conversational assistant - is a black box. Set STREAMING_STT=false to get
# the CPU back on constrained hardware; the endpointer's own behaviour does not change either way,
# because it does its own transcription at the eager threshold regardless (see _endpoint_decision
# and the _ends_mid_thought check in listen_for_one_utterance).
PARTIAL_PREVIEW_ENABLED = RT.STREAMING_STT
PARTIAL_MIN_DURATION_S = RT.PARTIAL_MIN_DURATION_S  # don't preview until the recording is
                                                    # already this long
PARTIAL_INTERVAL_S = RT.PARTIAL_INTERVAL_S  # how often to re-preview after that

# How many consecutive IDENTICAL partial transcripts mean the speaker has finished, regardless
# of what the VAD thinks - see the long note in realtime_config.py. This is the signal that
# rescues a turn on a microphone whose room noise reads as continuous speech, where neither the
# silence counter nor the pitch contour ever gets a chance to fire. 0 disables it.
TURN_STABLE_PARTIALS = RT.TURN_STABLE_PARTIALS
# ...and one more ends the turn even when the words sound unfinished. From a real run on the
# Realtek mic array: the partial stayed "[??] [??] is" for four previews in a row
# while the VAD kept reporting speech, the "ends mid-thought" guard (it ends on "is") blocked the
# rule above, and the visitor waited for the 15-second cap. At the defaults this is about five
# seconds with no new word while the microphone still "hears speech" - past that point the guard
# is protecting nobody, and a wrong guess costs one "sorry, could you say that again?".
TURN_STALL_PARTIALS = TURN_STABLE_PARTIALS + 1

# The real institution this robot is for - "Lincoln University College" (lincoln.edu.my).
# CORRECTION: earlier in this project the spelling "Lincoln University Collage" was used here,
# confirmed (twice) by the user before the real website was available to check against. Once the
# real site (lincoln.edu.my) was actually fetched, its own header/title confirm the standard
# spelling "College" is correct - Whisper had actually been transcribing it correctly the whole
# time, and the earlier "fix" (a KNOWN_TEXT_CORRECTIONS rule forcing "College" -> "Collage") was
# backwards and has been removed. Real department/school names now come from the real site
# structure (see scraper.py in stage07_data_pipeline) rather than Stage 05's earlier placeholder
# list.
INSTITUTION_PROMPT = (
    "Lincoln University College. Computer Science, Engineering, Management, Nursing, "
    "Admissions, Student Affairs, Library."
)


# --- hotword biasing ---------------------------------------------------------------------------
# faster-whisper's `hotwords` argument biases DECODING toward a list of terms, which is a
# different and better mechanism than either of the two this file already had:
#
#   initial_prompt        - one sentence of context; helps with spelling/register, but there is
#                           only room for a sentence and it competes with itself as you add terms.
#   vocabulary_correction - runs AFTER decoding, fuzzy-matching low-confidence words against a
#                           list. It can only repair a word Whisper already got CLOSE to, and by
#                           then the word has already been marked [??], which now also trips the
#                           repair prompt in Stage 08 ("sorry, I only caught...").
#
# Measured on this folder's own test_samples (2026-09-05, small.en/int8/greedy): adding hotwords
# took raw model WER on the noisy clips from 58.3% to 45.8%, and specifically turned the creator's
# name from "[??]" into "Alex". The end-to-end number after vocabulary_correction was unchanged,
# because that stage was already patching the same word - but fixing it at decode time means it
# never becomes a low-confidence word in the first place, so it doesn't count toward
# ConfidenceReport.is_reliable and doesn't provoke a needless "could you repeat that".
#
# The list is capped: faster-whisper truncates hotwords at max_length // 2 tokens (~223), and
# biasing dilutes as the list grows - fifty terms all nudged slightly is worth less than the
# twenty terms visitors actually say.
# faster-whisper truncates the hotword list at max_length // 2 tokens (~223 for this model), and
# comma-separated proper nouns run roughly four characters to the token, so ~750 characters is
# the point where terms start being silently dropped off the end. Kept under that deliberately:
# biasing also DILUTES as the list grows, so the budget is spent on the terms visitors actually
# say rather than on filling it.
HOTWORDS_MAX_CHARS = 550

# Words to keep OUT of the hotword list even when they appear in INSTITUTION_VOCABULARY. That
# list serves vocabulary_correction, which uses short function words to repair specific observed
# misrecognitions ("[??] yeah, it's [??] engineering department"). As hotwords they are pure
# waste: Whisper does not need help hearing "the", biasing toward "Where" nudges it to start
# transcripts with a question word, and each one spends budget the real proper nouns need.
_HOTWORD_STOPWORDS = frozenset({
    "where", "what", "when", "how", "who", "why", "to", "in", "on", "the", "are", "is", "so",
    "and", "of", "a", "an", "at", "for", "with", "full time",
})

def real_hotwords() -> str:
    """The always-present hotwords: the institution, its departments, and the robot's own and its
    creator's names - genuinely unusual proper nouns that Whisper has no reason to know.

    Deliberately SHORT. Extending it with the scraped programme names was tried and measured on
    2026-09-05, and made recognition slightly worse rather than better - see the note in
    main_live.py where that experiment is recorded, so it is not attempted again by accident."""
    me = get_identity()
    return merge_hotwords(
        ", ".join(INSTITUTION_VOCABULARY),
        f"{me.robot_name}, {me.creator_name}",
    )


def merge_hotwords(*parts: str, max_chars: int = HOTWORDS_MAX_CHARS) -> str:
    """Combines hotword lists into one, in order, dropping duplicates and stopwords and stopping
    at the character budget.

    Order is priority: whatever comes first survives the cap. Callers put the identity and
    institution terms ahead of the knowledge-base terms for that reason - the robot's own name
    being misheard is worse than one programme name out of two hundred being unbiased.

    Deduplication is case-insensitive and matters more than it looks: "Lincoln University
    College" is both the first hand-written term and the single most frequent phrase in the
    scraped site, so without this it would be spent twice out of a budget that only fits about
    twenty-five terms.
    """
    chosen: list[str] = []
    seen: set[str] = set()
    used = 0
    for part in parts:
        for raw in (part or "").split(","):
            term = " ".join(raw.split())
            key = term.lower()
            if not term or key in seen or key in _HOTWORD_STOPWORDS:
                continue
            if used + len(term) + 2 > max_chars:
                continue  # keep going - a later, shorter term may still fit
            chosen.append(term)
            seen.add(key)
            used += len(term) + 2
    return ", ".join(chosen)


def real_initial_prompt() -> str:
    """The initial_prompt to hand WhisperSTTEngine. A function rather than the constant it used
    to be because its last sentence names the robot and its creator, and those now come from the
    database (identity.py) - building the string at call time means an engine constructed after
    a rename is biased toward the new name, with no import-order subtlety about when the value
    was read."""
    me = get_identity()
    return (
        f"{INSTITUTION_PROMPT} My name is {me.robot_name}. "
        f"I was created by {me.creator_name}, who is from {me.creator_country}."
    )
INSTITUTION_VOCABULARY = [
    "Lincoln University College",
    "Lincoln",
    "Computer Science",
    "Engineering",
    "Management",
    "Nursing",
    "Admissions",
    "Student Affairs",
    "Library",
    "Mosque",
    # Common question/preposition words - added after a real report that these were
    # frequently coming out as LOW_CONFIDENCE_MARKER ("[??] is the Engineering Department?")
    # instead of the actual word. This is a different failure mode from the proper-noun entries
    # above, and the fix belongs here (vocabulary_correction) rather than in the initial prompt:
    # initial_prompt mainly helps Whisper's confidence on RARE tokens it wouldn't otherwise
    # recognize (see the "Lincoln" fix above) - "where"/"what"/"how" are already extremely common
    # in Whisper's training data, so biasing the prompt toward them isn't expected to move the
    # needle. What's actually happening is Whisper transcribing the word CORRECTLY but with low
    # per-word probability (short, unstressed function words at the very start of an utterance
    # legitimately score lower confidence even when right) - try_correct() only fires on
    # low-confidence words, and when the word is already correct, fuzzy-matching it against its
    # own entry here just confirms it instead of flagging it, which is exactly what removes the
    # "[??]" for cases like this.
    #
    # Risk to be aware of: MATCH_SCORE_THRESHOLD (vocabulary_correction.py) is 85, which is a
    # comfortable margin for these at their actual lengths, but very short words have less room
    # for a fuzzy match to go wrong on a genuine near-homophone (e.g. "in" vs. "on" differ by one
    # letter) - if you ever see one short word get silently swapped for a different one instead
    # of just confirmed, that's this list's short-word/short-edit-distance tradeoff showing up,
    # not a new bug.
    "Where",
    "What",
    "When",
    "How",
    "Who",
    "Why",
    "to",
    "in",
    "on",
    # Same fix, same reasoning, extended after a second real report: "the"/"are"/
    # "is"/"so" were the specific words showing up as "[??]" in that run ("[??] yeah, it's [??]
    # engineering department", "Who [??] you?"), and a corrupted query text like that is what
    # sent a garbled question into knowledge-base search at all - not a retrieval bug on its own.
    # Most of that same run's STT trouble turned out to be a separate root cause (a stale audio
    # backlog reaching listen_for_one_utterance(), see _listen()'s comment in main_live.py), but
    # these words were still worth adding on their own evidence.
    "the",
    "are",
    "is",
    "so",
]


# --- adaptive endpointing ---------------------------------------------------------------------
# A FIXED trailing-silence threshold is always wrong in one direction. PatientVAD's 900ms was
# raised from the stock 700ms because 700 cut people off mid-sentence - but 900ms is then also
# 900ms of dead air after a two-word question that obviously ended, and that pause is a large
# part of what makes a voice robot feel like a machine rather than a person.
#
# So: decide per-utterance instead of per-installation, using two signals that cost nothing.
#
#   1. PITCH CONTOUR (_pitch_trend below). A LEVEL pitch at the end of a phrase is the classic
#      continuation contour - the speaker is mid-thought and about to keep going. Falling
#      (statement) or sharply rising (question) are terminal contours. This is real linguistics,
#      it is speaker-independent, and it is ~1ms of numpy on audio already in memory.
#
#   2. THE WORDS THEMSELVES (_ends_mid_thought below). "Where is the department of..." is not a
#      finished question no matter how long the pause. Checked on the transcript we were about
#      to return anyway, so in the common case it costs NOTHING - the early transcription is the
#      one we would have done at the patient threshold regardless.
#
# The result: a clearly-finished question is answered after ~550ms instead of 900ms, and someone
# who trails off thinking gets 1500ms instead of being interrupted at 900ms. Both directions
# improve at once, which a single fixed number cannot do.
# Set False to fall straight back to PatientVAD's single fixed 900ms threshold - the behaviour
# before 2026-09-05 - if adaptive endpointing ever misbehaves on a room/microphone this was not
# tuned against. Everything below is bypassed in that case.
ADAPTIVE_ENDPOINTING_ENABLED = RT.VAD_ENABLED

ENDPOINT_EAGER_MS = RT.MIN_SILENCE_DURATION     # earliest a turn can be called finished, and
                                                # only with evidence (default 550)
ENDPOINT_PATIENT_MS = RT.MAX_SILENCE_DURATION   # hard stop - end the turn here no matter what
                                                # the signals say (default 1500)
# Used when the pitch contour cannot be measured at all (see _pitch_trend()'s "unknown", and the
# branch that consumes it in listen_for_one_utterance). Deliberately PatientVAD's old fixed
# 900ms: with no evidence either way, the right behaviour is the one from before this adaptive
# machinery existed, not the eager cut.
ENDPOINT_UNKNOWN_MS = RT.END_OF_TURN_TIMEOUT  # default 900

# How much audio before the silence to measure the pitch contour over. Long enough for a few
# voiced windows, short enough to only capture the final syllables rather than the whole phrase.
PITCH_WINDOW_MS = 400
PITCH_FRAME_MS = 40
PITCH_MIN_HZ = 70    # below a typical adult male floor
PITCH_MAX_HZ = 350   # above a typical adult female ceiling
# Semitones per second. Below this the contour counts as LEVEL (still talking); above it, the
# pitch is clearly moving and the phrase is landing. Semitones rather than Hz so one threshold
# works for a low and a high voice alike - pitch is perceived logarithmically.
PITCH_LEVEL_ST_PER_S = 1.8

# Ending on one of these means the sentence is grammatically incomplete - a pause here is
# someone thinking, not someone finished. Kept to function words that essentially cannot end an
# English sentence; content words are excluded deliberately, since "where is the library" ends
# on a noun and is complete.
_CONTINUATION_WORDS = frozenset({
    "a", "an", "the", "my", "your", "our", "their", "his", "her", "its",
    "and", "or", "but", "so", "because", "if", "when", "while", "that", "which", "who",
    "of", "for", "to", "in", "on", "at", "with", "from", "about", "into", "over", "under",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must",
    "have", "has", "had", "i", "you", "he", "she", "it", "we", "they",
    "how", "what", "where", "why", "whose", "there's", "it's",
    "some", "any", "this", "these", "those", "than", "then", "as", "like",
})


def _ends_mid_thought(text: str) -> bool:
    """Whether a transcript reads as an unfinished sentence - see _CONTINUATION_WORDS.

    Punctuation is checked first: Whisper writes a '?' or '.' when it heard a complete
    utterance, which is a stronger signal than any word list, and a trailing ',' or '...' is a
    strong signal in the other direction.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("...", ",", "-")):
        return True
    if stripped.endswith(("?", ".", "!")):
        return False

    # Strip the low-confidence marker before looking at the last word - "the [??]" should be
    # judged on "the", not on the marker.
    cleaned = stripped.replace(LOW_CONFIDENCE_MARKER, " ")
    words = re.findall(r"[a-z']+", cleaned.lower())
    if not words:
        return False
    return words[-1] in _CONTINUATION_WORDS


def _pitch_trend(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> str:
    """Classifies the pitch contour of the final moments of speech as "level", "moving", or
    "unknown" (too little voiced audio to tell).

    Autocorrelation per short frame - deliberately not a full F0 tracker. All that is needed is
    whether the pitch is holding steady or clearly going somewhere, and a real pitch tracker
    would be a dependency and a startup cost for a decision that gets re-made every 30ms.
    """
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    frame_len = int(sample_rate * PITCH_FRAME_MS / 1000)
    if samples.size < frame_len * 3:
        return "unknown"

    min_lag = max(int(sample_rate / PITCH_MAX_HZ), 2)
    max_lag = min(int(sample_rate / PITCH_MIN_HZ), frame_len - 1)
    if max_lag <= min_lag:
        return "unknown"

    times: list[float] = []
    pitches: list[float] = []
    for start in range(0, samples.size - frame_len + 1, frame_len):
        frame = samples[start:start + frame_len]
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy < 1e-6:
            continue

        corr = np.correlate(frame, frame, mode="full")[frame_len - 1:]
        window = corr[min_lag:max_lag + 1]
        if window.size == 0:
            continue
        peak = int(np.argmax(window))
        # Unvoiced audio (a fricative, or room noise) has no strong periodic peak. Requiring the
        # peak to be a real fraction of the zero-lag energy is what keeps those frames from
        # contributing a meaningless "pitch" to the slope.
        if window[peak] < 0.3 * corr[0]:
            continue
        pitches.append(sample_rate / (min_lag + peak))
        times.append((start + frame_len / 2) / sample_rate)

    if len(pitches) < 3:
        return "unknown"

    # Slope in semitones per second: 12 * log2(f) is a linear scale in semitones.
    semitones = 12.0 * np.log2(np.asarray(pitches))
    slope = float(np.polyfit(np.asarray(times), semitones, 1)[0])
    return "level" if abs(slope) < PITCH_LEVEL_ST_PER_S else "moving"


class _EndpointState:
    """Per-pause bookkeeping for the adaptive endpointer - see _endpoint_decision().

    A pause gets ONE pitch reading, not one per 30ms frame: a reading that says "keep waiting"
    means we are waiting for the patient threshold anyway, and re-measuring thirty times while
    we wait is pure CPU burn. Both fields therefore have to survive between frames, and both
    have to be cleared the moment the speaker carries on talking - the next pause is a fresh
    question and deserves its own reading.
    """

    __slots__ = ("checked", "floor_ms", "smart_p", "holding", "holding_reported")

    def __init__(self) -> None:
        self.checked = False           # the contour has had its look at this pause
        self.floor_ms: int | None = None  # fallback threshold when it could not be read
        self.smart_p: float | None = None  # smart-turn's verdict on this pause, once asked
        # Why this pause was judged "mid-sentence - keep waiting", once it has been. Reported
        # to the observer as a "holding" event (stage08 uses it for a quiet "mm-hm").
        self.holding: str | None = None
        self.holding_reported = False

    def reset(self) -> None:
        self.checked = False
        self.floor_ms = None
        self.smart_p = None
        self.holding = None
        self.holding_reported = False


# --- the end-of-turn model (added 2026-10-27, see smart_turn.py) -------------------------------
# TURN_DETECTOR=smart_turn asks a small model, once per pause, whether the visitor sounds
# FINISHED. It gets the first say, after only SMART_TURN_MIN_SILENCE_MS of quiet:
#   p >= SMART_TURN_THRESHOLD     -> finished: answer now (typically ~300ms instead of 550-900ms)
#   p <  SMART_TURN_UNFINISHED    -> clearly mid-thought: wait for the patient limit, so a
#                                    "where is the... um..." is not cut off
#   anything in between            -> no strong opinion: the pitch rules below decide, as before
# The transcript check (_ends_mid_thought) still runs after any of these, so a model mistake on
# "can you tell me where the" is still caught by the words. Without the model file (or with
# TURN_DETECTOR=pitch) the behaviour is exactly the pre-2026-10-27 one.
SMART_TURN_ENABLED = RT.TURN_DETECTOR == "smart_turn"
SMART_TURN_THRESHOLD = RT.SMART_TURN_THRESHOLD          # default 0.5
SMART_TURN_UNFINISHED = 0.2
SMART_TURN_MIN_SILENCE_MS = RT.SMART_TURN_MIN_SILENCE_MS  # default 250


def _smart_turn_model():
    if not SMART_TURN_ENABLED:
        return None
    try:
        import smart_turn
    except Exception as exc:  # noqa: BLE001
        print(f"[turn] smart-turn unavailable ({exc!r}) - using pitch endpointing.")
        return None
    return smart_turn.shared_model()


def _endpoint_decision(silence_ms: int, audio_chunks: list[bytes],
                       state: _EndpointState) -> bool:
    """Has the speaker finished their turn, given how long this pause has run and the audio
    just before it? Mutates `state` (see above).

    Three outcomes from the pitch contour, and the middle one is the whole point:

      moving   - a falling (statement) or rising (question) contour: the phrase landed. End the
                 turn now, at ENDPOINT_EAGER_MS.
      level    - a continuation contour: they are mid-thought. Wait for ENDPOINT_PATIENT_MS.
      unknown  - the contour could not be measured at all, because there was too little VOICED
                 audio in the window (see _pitch_trend). This is the ABSENCE of evidence, and it
                 used to be handled as though it were evidence the turn had ended - anything
                 that did not come back "level" ended the recording at ENDPOINT_EAGER_MS.

    That third case is common, not exotic. The window is the last 400ms before the pause, and a
    great many English phrases end on an unvoiced consonant - "computer science", "my class",
    "the campus", "the department" - so the contour is genuinely unreadable whenever somebody
    pauses for breath after one of them. The robot was cutting people off 550ms into an ordinary
    mid-sentence pause and answering half a question.

    With no evidence either way, fall back to the fixed threshold that was in use before any of
    this adaptive machinery existed (PatientVAD's 900ms): slower than an eager cut, faster than
    the hard stop, and not pretending to know something it does not.
    """
    if silence_ms >= ENDPOINT_PATIENT_MS:
        return True

    # Not once the pause has already been judged (state.checked): after "the transcript ends
    # mid-thought" the decision is to wait for the patient limit, and asking the model again on
    # every 30ms frame of that wait would re-open a question that has been answered - and burn
    # a model call plus a transcription per frame doing it.
    if (state.smart_p is None and not state.checked
            and silence_ms >= SMART_TURN_MIN_SILENCE_MS):
        model = _smart_turn_model()
        if model is not None:
            try:
                state.smart_p = model.probability(b"".join(audio_chunks))
            except Exception as exc:  # noqa: BLE001 - never lose a turn over the model
                print(f"[turn] smart-turn failed ({exc!r}) - using pitch endpointing.")
                state.smart_p = -1.0
            else:
                rt_log("TURN", f"end-of-turn model: {state.smart_p:.2f} after {silence_ms}ms "
                               f"of silence ({model.last_ms:.0f}ms to decide)")
                if state.smart_p >= SMART_TURN_THRESHOLD:
                    return True
                if state.smart_p < SMART_TURN_UNFINISHED:
                    print("[stt] (they sound mid-sentence - waiting longer)")
                    state.checked = True     # skip the pitch shortcut; patient limit applies
                    state.holding = "they sound mid-sentence"
                    return False

    if state.floor_ms is not None:
        return silence_ms >= state.floor_ms
    if silence_ms < ENDPOINT_EAGER_MS or state.checked:
        return False

    # Measure the contour over the audio immediately BEFORE the silence, not including it.
    silence_frames = max(int(silence_ms / FRAME_MS), 1)
    voiced = audio_chunks[:-silence_frames] if silence_frames < len(audio_chunks) else []
    window_frames = max(int(PITCH_WINDOW_MS / FRAME_MS), 1)
    trend = _pitch_trend(b"".join(voiced[-window_frames:]))

    state.checked = True
    if trend == "moving":
        return True
    if trend == "level":
        print("[stt] (pitch still level - they sound mid-sentence, waiting longer)")
        state.holding = "pitch still level"
        return False

    state.floor_ms = ENDPOINT_UNKNOWN_MS
    if silence_ms >= ENDPOINT_UNKNOWN_MS:
        return True
    print("[stt] (cannot read the pitch contour - using the safe "
          f"{ENDPOINT_UNKNOWN_MS}ms pause)")
    return False


def _transcribe_now(engine: "WhisperSTTEngine", listener: STTListener,
                    audio_chunks: list[bytes]) -> str | None:
    """Transcribes immediately, without consulting the VAD.

    listener.listen() (stt_listener.py, deliberately untouched) asks its own VAD whether speech
    has ended and returns None if not - correct for the fixed-threshold design it was written
    for, but the adaptive endpointer above owns that decision now and needs to transcribe at a
    moment listener.listen() would refuse to. Same two lines listener.listen() runs once it
    agrees, so the mock's protocol keeps working unchanged for everyone else.
    """
    if not audio_chunks:
        return None
    chunks = _final_transcribe(engine)(audio_chunks, listener.locale)
    final = [c for c in chunks if c.is_final]
    return final[-1].text if final else None


# --- barge-in: interrupting the robot while it is still speaking --------------------------------
# The hard problem is not detecting speech during playback, it is that WITHOUT ECHO CANCELLATION
# the microphone hears the robot's own voice through the speaker. A naive "is anyone talking"
# check fires on every sentence the robot says and it interrupts itself constantly, which is
# worse than not being interruptible at all.
#
# This is a poor-man's echo SUPPRESSOR rather than a canceller, and it works on one asymmetry:
# the robot's own voice arrives at a fairly constant level (fixed speaker, fixed distance, known
# volume), while a person talking over it arrives ON TOP of that. So the echo level is measured
# once - calibrate_echo_floor() plays a clip with nobody speaking and records what comes back -
# and anything sustained and clearly louder than that is somebody else.
#
# Whether that gap actually exists is a property of the ROOM AND THE HARDWARE, not of this code.
# A directional or close-talk microphone leaves a wide gap and this works well; an omnidirectional
# mic sitting next to a loud speaker leaves none and nothing here can rescue it - that is what
# hardware AEC (a conference speakerphone) solves. Hence BARGE_IN_ENABLED defaults to False: run
# the calibration, see the numbers for your own setup, then decide.
BARGE_IN_ENABLED = RT.barge_in_wanted()

# How far above the measured echo floor a sound has to be to count as a person rather than the
# robot's own voice. Multiplier on RMS, so 2.5x is about 8dB.
BARGE_IN_MARGIN = RT.BARGE_IN_MARGIN  # default 2.5

# Sustained, not instantaneous: a door, a cough or one bad frame must not stop the robot
# mid-sentence. About a third of a second is roughly a syllable or two of real speech.
BARGE_IN_MIN_SPEECH_MS = RT.BARGE_IN_MIN_SPEECH_MS  # default 330

# Ignored at the very start of each utterance. Playback has an attack transient, and the
# visitor's own last words can still be reverberating in the room when the robot starts its
# reply - both would otherwise read as an immediate interruption.
BARGE_IN_WARMUP_MS = RT.BARGE_IN_WARMUP_MS  # default 700


def frame_rms(frame: bytes) -> float:
    """RMS level of one PCM16 frame, 0.0-1.0."""
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


class BargeInMonitor:
    """Watches the microphone while the robot is speaking and raises a stop signal when somebody
    talks over it.

    Runs on its own thread for the duration of one spoken response, consuming the same audio
    queue the listener uses. Consuming is deliberate rather than incidental: those frames are the
    robot's own voice and would otherwise sit in the queue as backlog for the next listen - the
    exact stale-audio problem _listen() in Stage 08 already has to drain around.

    THE FRAMES THAT TRIGGERED THE INTERRUPTION ARE KEPT, in `captured_frames`, and that is not a
    detail. Everything this monitor consumes is the robot's own echo and is rightly thrown away -
    except the last few hundred milliseconds, which are the visitor's actual opening words. Those
    used to be discarded along with the echo, and then _listen() drained the queue on top, so the
    brief's own example failed in the most annoying way possible:

        AI:   "The Computer Science department is located on..."
        User: "Wait, I mean the AI department."     <- heard, used to stop playback, DELETED
        AI:   [listens from scratch, having thrown away the sentence it just reacted to]

    The visitor then has to say it again, which is precisely the wait barge-in existed to remove.
    Keeping them means the interrupting sentence is carried into the next listen as preroll and
    answered directly - see `prefix_frames` on listen_for_one_utterance(). Bounded by
    `keep_frames` so a long answer cannot accumulate the whole of its own echo in memory.
    """

    def __init__(self, audio_q, vad: "StreamingVAD", stop_event,
                 rms_threshold: float, min_speech_ms: int = BARGE_IN_MIN_SPEECH_MS,
                 warmup_ms: int = BARGE_IN_WARMUP_MS, keep_audio: bool | None = None):
        self.audio_q = audio_q
        self.vad = vad
        self.stop_event = stop_event
        self.rms_threshold = rms_threshold
        self.min_speech_ms = min_speech_ms
        self.warmup_ms = warmup_ms
        self.triggered = False
        self.triggered_at: float | None = None
        self.keep_audio = RT.BARGE_IN_KEEP_AUDIO if keep_audio is None else keep_audio
        # The interrupting speech, ready to be replayed into the next listen. Written by this
        # monitor's thread and read by the visit thread only AFTER stop() has joined it, so no
        # lock is needed - the join is the synchronisation point.
        self.captured_frames: list[bytes] = []
        # A little more than min_speech_ms, so the run of frames that proved it was speech is
        # carried over whole rather than clipped at its own trigger point. PREROLL_FRAMES again
        # on top, for the syllable or two before the VAD was convinced.
        self._keep_frames = int(self.min_speech_ms / FRAME_MS) + PREROLL_FRAMES
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        started = time.monotonic()
        speech_ms = 0
        recent: list[bytes] = []
        while self._running.is_set():
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if (time.monotonic() - started) * 1000 < self.warmup_ms:
                continue

            if self.keep_audio:
                recent.append(frame)
                if len(recent) > self._keep_frames:
                    recent.pop(0)

            # BOTH tests must pass. Loudness alone fires on a door slam; the VAD alone fires on
            # the robot's own perfectly speech-shaped voice. Together they mean "speech, and
            # louder than this room's echo of ourselves".
            if frame_rms(frame) > self.rms_threshold and self.vad.is_speech(frame):
                speech_ms += FRAME_MS
                if speech_ms >= self.min_speech_ms:
                    self.triggered = True
                    self.triggered_at = time.monotonic()
                    self.captured_frames = list(recent) if self.keep_audio else []
                    self.stop_event.set()
                    print("[barge-in] someone started talking - stopping.")
                    rt_log("INTERRUPTION", "user interrupted AI"
                           + (f" - carrying {len(self.captured_frames)} frame(s) "
                              f"({len(self.captured_frames) * FRAME_MS}ms) of what they said "
                              f"into the next listen" if self.captured_frames else ""))
                    return
            else:
                speech_ms = 0  # must be CONTINUOUS speech, not a scattering of loud frames

    def start(self) -> None:
        if self._thread is not None:
            return
        self.triggered = False
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="barge-in", daemon=True)
        self._thread.start()

    def stop(self) -> bool:
        """Stops watching. Returns whether an interruption was actually detected."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        return self.triggered

    def take_captured_audio(self) -> list[bytes]:
        """The interrupting speech, handed over exactly once.

        Cleared on the way out so a second call cannot replay the same words into a second
        listen - which would transcribe the visitor's interruption twice and answer it twice.
        Only safe to call after stop() has joined the thread.
        """
        frames, self.captured_frames = self.captured_frames, []
        return frames

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def calibrate_echo_floor(audio_q, speak, sample_text: str | None = None,
                         skip_first_s: float = 0.0) -> float:
    """Measures how loudly this room returns the robot's OWN voice to the microphone.

    Call with nobody speaking. `speak` is any callable that plays a sentence and blocks;
    everything the microphone captures while it runs is, by definition, echo. The returned value
    is the RMS to set a barge-in threshold above.

    The 95th percentile rather than the mean: what matters is the loud part of the robot's own
    speech, since that is what a threshold has to clear to avoid false triggers.

    skip_first_s: with software echo cancellation the canceller needs a second or so of the
    robot's voice to converge, and the very first startup sentence is the only time it has not
    yet; measuring that second would set the threshold for a robot that no longer exists.

    Returns -1.0 when the microphone delivered nothing at all while the robot spoke.
    """
    while True:  # start from an empty queue so old audio isn't measured as echo
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break

    speak(sample_text or "Testing, one two three, this is how loud I sound to my own microphone.")

    levels = []
    while True:
        try:
            levels.append(frame_rms(audio_q.get_nowait()))
        except queue.Empty:
            break
    skip = int(skip_first_s * 1000 / FRAME_MS)
    if skip and len(levels) > skip * 2:
        levels = levels[skip:]
    if not levels:
        return -1.0  # no audio arrived at all - distinct from an echo cancelled down to zero
    return float(np.percentile(levels, 95))


class PatientVAD(VoiceActivityDetector):
    """Same is_speech_ended() interface as VoiceActivityDetector (stt_listener.py, unchanged) -
    just a longer default trailing-silence threshold. The stock 700ms default was ending
    utterances on an ordinary mid-sentence breath/pause, cutting off real sentences early
    ("One more." instead of the rest of what was said). 900ms trades a bit of extra latency
    after you stop talking for not truncating you."""

    def is_speech_ended(
        self, silence_duration_ms: int, threshold_ms: int = 900
    ) -> bool:
        return super().is_speech_ended(silence_duration_ms, threshold_ms)


class StreamingVAD:
    """Thin wrapper around webrtcvad's per-frame speech/non-speech classifier."""

    def __init__(
        self, aggressiveness: int = VAD_AGGRESSIVENESS, sample_rate: int = SAMPLE_RATE
    ):
        self._vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate

    def is_speech(self, frame_bytes: bytes) -> bool:
        return self._vad.is_speech(frame_bytes, self.sample_rate)


def _pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    """faster-whisper expects float32 samples in [-1, 1] when given a numpy array directly,
    not raw 16-bit PCM bytes from the microphone."""
    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


@dataclass
class ConfidenceReport:
    """Per-transcription confidence summary - "don't treat every transcription as equally
    reliable" (this is what lets a caller ask "did you say X?" instead of confidently acting on
    a wrong guess)."""

    total_word_count: int
    low_confidence_word_count: (
        int  # flagged with LOW_CONFIDENCE_MARKER, unresolved even after
    )
    # trying vocabulary correction
    corrected_word_count: int  # low-confidence, but resolved via vocabulary_correction
    avg_word_probability: float | None
    no_speech_prob: float  # Whisper's own "this might not be speech at all" signal

    @property
    def is_reliable(self) -> bool:
        if self.total_word_count == 0:
            return False
        if self.no_speech_prob >= UNRELIABLE_NO_SPEECH_PROB:
            return False
        return (
            self.low_confidence_word_count / self.total_word_count
        ) <= UNRELIABLE_WORD_FRACTION


def _add_nvidia_dll_dirs() -> None:
    """Windows: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` puts the CUDA DLLs under
    site-packages/nvidia/*/bin, which is not on the DLL search path - the cause of the
    "cublas64_12.dll is not found" seen on 2026-09-05. Adding those folders fixes it."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    import site

    for base in site.getsitepackages() + [site.getusersitepackages()]:
        for bin_dir in Path(base, "nvidia").glob("*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


def _load_whisper(model_size: str, device: str = "auto"):
    """(WhisperModel, "cuda" | "cpu"). STT_DEVICE=auto tries the GPU and PROVES it works with
    one tiny transcription - CUDA problems only show up on the first transcription, not at
    load time (see the GPU note in WhisperSTTEngine) - and otherwise uses the CPU. A GPU makes
    small.en ~5-10x faster, which is most of the wait after a visitor stops talking."""
    if device in ("auto", "cuda"):
        _add_nvidia_dll_dirs()  # before faster-whisper loads CTranslate2 and its CUDA parts
    from faster_whisper import WhisperModel

    if device in ("auto", "cuda"):
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                model = WhisperModel(model_size, device="cuda", compute_type="float16")
                segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                                               language="en", beam_size=1)
                list(segments)
                print("[stt] Whisper is running on the GPU.")
                return model, "cuda"
            if device == "cuda":
                print("[stt] STT_DEVICE=cuda but no CUDA GPU is visible - using the CPU.")
        except Exception as exc:  # noqa: BLE001 - any CUDA/driver/DLL problem: use the CPU
            print(f"[stt] the GPU could not run Whisper ({str(exc)[:120]}) - using the CPU. "
                  f"For an NVIDIA GPU: pip install -r requirements-gpu.txt")
    return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu"


class WhisperSTTEngine:
    """Implements the same STTEngine protocol as MockSTTEngine (stt_listener.py) - a drop-in
    replacement, so STTListener doesn't need to change at all. Not truly incremental/streaming
    on its own (returns one final chunk once the whole utterance is in) - see PARTIAL_PREVIEW_*
    above and README's known limitations for how streaming is layered on top instead."""

    def __init__(
        self,
        model_size: str | None = None,
        initial_prompt: str | None = None,
        vocabulary: list[str] | None = None,
        denoise_enabled: bool = False,
        hotwords: str | None = None,
    ):
        """
        model_size: "base.en" (74M params) was noticeably inaccurate in real testing. "small.en"
        (244M) is the current default. It was briefly raised to "medium.en" (769M) on 2026-08-19
        chasing accuracy, then reverted the same day on explicit feedback that the robot took too
        long to answer - medium.en is genuinely more accurate, but on Windows CPU it costs
        multiple seconds per utterance, and "responds like a real human" was the higher priority.
        The accuracy lost by coming back down is partly bought back by fixes that don't cost
        latency: always-on peak normalization (see transcribe_stream), the expanded
        real_vocabulary() confirmation list, and the stale-audio-queue drain in main_live.py's
        _listen(). If you ever move to the Jetson target (which has a GPU) medium.en becomes
        cheap again and is worth revisiting - measure with evaluate_wer.py rather than guessing.
        compute_type="int8" costs a small amount of accuracy for a lot of CPU speed; try
        compute_type="float32" if you want to isolate whether quantization itself is the issue.

        ON THE GPU (checked 2026-09-05, GTX 1650 4GB): device="cuda", compute_type="float16"
        LOADS fine and then fails on the first transcription with "Library cublas64_12.dll is not
        found". CTranslate2 needs NVIDIA's CUDA 12 runtime libraries, which a CPU-only torch
        install does not bring - note that torch's own cuda availability says nothing here, since
        faster-whisper does not use torch at all. To try it:
            pip install -r requirements-gpu.txt
        (cuBLAS 12 + cuDNN 9.10.2.21 - the cuDNN version must match the cudnn64_9.dll that
        CTranslate2 4.8.4 ships, or cuDNN refuses to load its other parts.) If a GPU transcription
        ever fails mid-conversation (out of GPU memory, a driver reset), transcribe_stream moves
        Whisper to the CPU and retries, so the visitor still gets an answer.
        Two things to weigh before doing that on THIS machine: it is ~700MB of extra dependency,
        and the GPU is already hosting Ollama's chat model (~2GB resident, KEEP_ALIVE=30m - see
        ollama_client.py). small.en at float16 is ~0.5GB and would fit alongside it; medium.en at
        ~1.5GB plus Ollama plus the display is very tight on a 4GB card, and an out-of-memory
        failure mid-conversation is a worse outcome than a slower transcription.

        initial_prompt: biases Whisper's vocabulary/spelling toward expected terms (e.g. "Lincoln
        University College" - the real institution this robot is for, see INSTITUTION_PROMPT
        below) without any training/fine-tuning - helps most on rare proper nouns Whisper
        wouldn't otherwise recognize confidently (e.g. "Lincoln" going from flagged-uncertain to
        confident once this was wired in - see evaluate_wer.py's real runs).

        vocabulary: real terms for vocabulary_correction.py to fuzzy-match low-confidence words
        against (e.g. department/building/staff names) - see INSTITUTION_VOCABULARY below for what's
        actually known so far (still incomplete until Stage 07's scraper exists to source the
        rest from).

        denoise_enabled: runs audio_preprocessing.denoise() (spectral gating) before
        transcription, on top of the peak normalization below (see transcribe_stream) which now
        always runs regardless of this flag. Still off by default - A/B test it with
        evaluate_wer.py on your own recordings before trusting it, aggressive noise removal can
        hurt accuracy as easily as help it (quiet consonants look a lot like noise to a spectral
        gate). Peak normalization carries no such risk (pure gain, no information removed), which
        is why it's no longer gated behind this same flag.

        That A/B has now been run (2026-09-05, this folder's test_samples): denoising made
        accuracy WORSE and cost 2.4x the time - overall WER 16.7% -> 20.2%, and it broke a clip
        that was previously perfect (clean category 0.0% -> 5.0%), while the noisy clips it was
        supposed to help did not improve at all. The "quiet consonants look like noise" concern
        above is exactly what showed up. Leave this off unless your own recordings say otherwise.

        hotwords: comma-separated terms to bias decoding toward - see the "hotword biasing" note
        near the top of this file. Defaults to none here; callers pass real_hotwords(), and
        Stage 08 adds knowledge-base terms on top via set_hotwords().
        """
        # model_size defaults to WHISPER_MODEL (small.en); the device to STT_DEVICE (auto = the
        # GPU when faster-whisper can really run on it - see _load_whisper()).
        model_size = model_size or RT.WHISPER_MODEL
        print(
            f"Loading faster-whisper '{model_size}' (first run downloads it, cached after "
            f"that)..."
        )
        self.model_size = model_size
        self.model, self.device = _load_whisper(model_size, RT.STT_DEVICE)
        self.initial_prompt = initial_prompt
        # Terms to bias decoding toward - see the "hotword biasing" note above. Extended at
        # runtime by Stage 08 once the knowledge base is loaded (set_hotwords), because the
        # programme names visitors actually ask about only exist once the site has been scraped.
        self.hotwords = hotwords or None
        self.vocabulary = vocabulary or []
        self.denoise_enabled = denoise_enabled
        self.last_confidence: ConfidenceReport | None = (
            None  # set after every transcribe_stream
        )

    def set_hotwords(self, hotwords: str | None) -> None:
        """Replaces the decode-time bias list. Called by Stage 08 once the knowledge base is
        loaded, and again whenever it is refreshed, so a newly-scraped programme name starts
        being heard correctly without rebuilding the Whisper model (which takes seconds and
        would have to happen while somebody is standing there)."""
        self.hotwords = (hotwords or "").strip() or None

    def transcribe_stream(
        self, audio_chunks: list[bytes], locale: str
    ) -> list[TranscriptChunk]:
        audio = _pcm_bytes_to_float32(b"".join(audio_chunks))

        # Peak normalization always runs now - previously bundled behind
        # denoise_enabled, which meant a quiet/far mic never got leveled at all unless the
        # riskier spectral denoise was also switched on. Pure gain adjustment, no information
        # removed, so there's no accuracy downside to always applying it (unlike denoise() below).
        # Front-end alternatives measured 2026-09-05 against this folder's test_samples, all
        # scoring IDENTICALLY to plain peak normalization (13.1% overall, 45.8% on the noisy
        # clips): no normalization at all, RMS/loudness normalization at -17/-20/-23 dBFS, and an
        # 80Hz high-pass ahead of either. Recorded here so the next person does not re-derive
        # them: on this hardware and these recordings the front-end is not where the remaining
        # errors are. Worth re-running (evaluate_wer.py) if the kiosk ends up somewhere with real
        # low-frequency noise - an 80Hz high-pass is the first thing to try for HVAC rumble, and
        # it measured harmless here.
        audio = audio_preprocessing.normalize(audio)
        if self.denoise_enabled:
            audio = audio_preprocessing.denoise(audio, SAMPLE_RATE)
            audio = audio_preprocessing.normalize(audio)  # denoise can shift peak - re-level

        language = locale.split("-")[0]  # "en-US" -> "en"
        try:
            segments = self._decode(audio, language)
        except Exception as exc:  # noqa: BLE001 - a GPU failure must not lose the question
            if self.device != "cuda":
                raise
            print(f"[stt] Whisper failed on the GPU ({str(exc)[:120]}) - moving it to the CPU "
                  f"and trying again.")
            from faster_whisper import WhisperModel

            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self.device = "cpu"
            segments = self._decode(audio, language)
        return self._collect(segments)

    def _decode(self, audio: "np.ndarray", language: str) -> list:
        """One Whisper pass. Returns the segments as a list: faster-whisper decodes lazily while
        they are iterated, so a GPU error only shows up during that iteration."""
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=self.initial_prompt,
            word_timestamps=True,  # needed for per-word confidence, not just per-segment
            # condition_on_previous_text=True (the default) feeds each segment's own guess back
            # in as context for the next one - on a short/noisy clip that starts with a wrong
            # guess, this is exactly what causes Whisper's well-known repetition-hallucination
            # loop ("I can't. I can't. I can't."). Off = each segment decoded independently.
            condition_on_previous_text=False,
            # A second, independent silence filter (Silero VAD, bundled with faster-whisper) on
            # top of our own webrtcvad - catches near-silent stretches webrtcvad's simpler
            # per-frame classifier let through, which is what tends to trigger hallucinated
            # filler words on essentially empty audio.
            vad_filter=True,
            # Greedy decoding instead of faster-whisper's default beam_size=5 (2026-08-19, same
            # "respond like a real human" pass as the small.en revert above). Beam search decodes
            # five candidate transcripts and picks the best, so it costs roughly proportionally
            # more time for a small accuracy gain. On the short, single-sentence utterances this
            # robot actually receives ("where is the engineering department") the beam rarely
            # changes the answer - the phrasing is short and unambiguous - so this is close to
            # free latency. It matters much more on long, rambling, or noisy audio, which is not
            # this workload. Raise back to 5 (or drop the argument) if evaluate_wer.py shows a
            # real WER regression on your own recordings.
            #
            # RE-MEASURED 2026-09-05 on this folder's test_samples, and the reasoning holds up:
            # beam_size=5 produced byte-identical transcripts to beam_size=1 on all seven clips
            # (overall WER 16.7% either way) while taking 2.5s longer over the set. There is
            # nothing being bought here at this utterance length.
            beam_size=1,
            # Decode-time bias toward the institution's own proper nouns - see the "hotword
            # biasing" note near the top of this file for what this fixes and what it measured.
            hotwords=self.hotwords,
        )
        return list(segments)

    def _collect(self, segments: list) -> list[TranscriptChunk]:
        pieces: list[str] = []
        total = low_confidence = corrected = 0
        word_probs: list[float] = []
        no_speech_probs: list[float] = []

        for seg in segments:
            no_speech_probs.append(seg.no_speech_prob)
            for w in seg.words or []:
                total += 1
                word_probs.append(w.probability)
                leading_ws = w.word[: len(w.word) - len(w.word.lstrip())]

                if w.probability >= WORD_CONFIDENCE_THRESHOLD:
                    pieces.append(w.word)
                    continue

                fix = vocabulary_correction.try_correct(
                    w.word,
                    w.probability,
                    self.vocabulary,
                    WORD_CONFIDENCE_THRESHOLD,
                )
                if fix is not None:
                    pieces.append(f"{leading_ws}{fix}")
                    corrected += 1
                else:
                    pieces.append(f"{leading_ws}{LOW_CONFIDENCE_MARKER}")
                    low_confidence += 1

        text = "".join(pieces).strip()
        self.last_confidence = ConfidenceReport(
            total_word_count=total,
            low_confidence_word_count=low_confidence,
            corrected_word_count=corrected,
            avg_word_probability=(
                (sum(word_probs) / len(word_probs)) if word_probs else None
            ),
            no_speech_prob=(
                (sum(no_speech_probs) / len(no_speech_probs))
                if no_speech_probs
                else 1.0
            ),
        )
        return [TranscriptChunk(text=text, is_final=True)]


def build_stt_engine(**kwargs):
    """The recognizer STT_BACKEND asks for: Whisper (default, most accurate) or the streaming
    Zipformer (stt_sherpa.py - for small computers). Falls back to Whisper, saying why.

    With STT_CLOUD=groq the local recognizer is wrapped so that the FINISHED question goes to
    Groq's Whisper large-v3 and the local one is the backup (stt_cloud.py); partials stay local.
    """
    engine = None
    if RT.STT_BACKEND == "sherpa":
        try:
            from stt_sherpa import SherpaSTTEngine

            engine = SherpaSTTEngine(hotwords=kwargs.get("hotwords"))
            print("[stt] using the streaming Zipformer recognizer (STT_BACKEND=sherpa).")
        except Exception as exc:  # noqa: BLE001
            print(f"[stt] streaming recognizer unavailable ({exc}) - using Whisper.")
    if engine is None:
        engine = WhisperSTTEngine(**kwargs)
    return _with_cloud_final(engine)


def _with_cloud_final(engine):
    """STT_CLOUD=groq: the local engine wrapped in stt_cloud.CloudFinalEngine. Anything missing
    (no key, no requests package) leaves the local engine in charge, saying why."""
    if RT.STT_CLOUD != "groq":
        return engine
    try:
        from stt_cloud import CloudFinalEngine, CloudSTTError, GroqTranscriber
    except Exception as exc:  # noqa: BLE001
        print(f"[stt] STT_CLOUD=groq, but the cloud recognizer could not load ({exc}) - "
              f"using the local Whisper only.")
        return engine
    try:
        cloud = GroqTranscriber(
            os.environ.get("GROQ_API_KEY", "").strip(), model=RT.GROQ_STT_MODEL,
            timeout_s=RT.STT_CLOUD_TIMEOUT_S, sample_rate=SAMPLE_RATE,
        )
    except CloudSTTError as exc:
        print(f"[stt] STT_CLOUD=groq, but {exc} - using the local Whisper only. Get a free key "
              f"at https://console.groq.com/keys")
        return engine
    wrapped = CloudFinalEngine(engine, cloud, ConfidenceReport, TranscriptChunk,
                               log_transcripts=RT.LOG_TRANSCRIPTS)
    print(f"[stt] finished questions go to Groq {RT.GROQ_STT_MODEL} (waits at most "
          f"{RT.STT_CLOUD_TIMEOUT_S:.1f}s), partials and the backup stay on this computer.")
    wrapped.check_in_background()
    return wrapped


def _final_transcribe(engine):
    """The call that produces the FINISHED transcript: the cloud one when STT_CLOUD wraps the
    engine (transcribe_final), otherwise the same local call the partials use."""
    return getattr(engine, "transcribe_final", engine.transcribe_stream)


def identity_vocabulary() -> list[str]:
    """The robot's own identity (2026-08-19, see stage05's identity.py) - rare proper nouns, same
    reasoning as "Lincoln" in INSTITUTION_VOCABULARY: without a hint, Whisper has nothing in its
    training data biasing it toward an uncommon name over a more common-sounding near-homophone.

    Derived from the database-backed profile instead of being listed literally, and
    split into parts as well as the full name because a visitor says "who is Tan" at least as
    often as the full name. dict.fromkeys de-duplicates while keeping order, which matters for a
    single-word creator name where the name and its only part are the same string.
    """
    me = get_identity()
    terms = [me.robot_name, me.creator_name, *me.creator_name.split(),
             me.creator_country, me.creator_nationality]
    return [t for t in dict.fromkeys(terms) if t]


def real_vocabulary() -> list[str]:
    """Everything WhisperSTTEngine should fuzzy-match low-confidence words against: the fixed
    institution/question terms plus whoever the robot currently believes it is."""
    return INSTITUTION_VOCABULARY + identity_vocabulary()


def _print_result(
    text: str, confidence: ConfidenceReport | None, prefix: str = "you said"
) -> None:
    if not text:
        print("[stt] (heard something, but no words came out - please try again)")
        return
    if confidence is not None and not confidence.is_reliable:
        print(
            f"[stt] (low confidence: {confidence.low_confidence_word_count}/"
            f'{confidence.total_word_count} words uncertain) {prefix}: "{text}"'
        )
    else:
        print(f'[stt] {prefix}: "{text}"')


# Below this, a voice is too quiet for reliable recognition (median RMS of the voiced 30ms frames,
# dB relative to full scale). Ordinary speech at arm's length from a laptop mic is about -35 to
# -25 dBFS.
QUIET_VOICE_DBFS = -42.0
# Voice less than this far above the room: the VAD will keep "hearing speech" in the room noise.
MIN_VOICE_ABOVE_ROOM_DB = 12.0
# Above this the microphone is set too hot: speech peaks run into full scale and distort, which
# costs recognition accuracy (and makes the robot's own echo non-linear, which echo cancellation
# cannot remove). A voice at -10 dBFS was measured on 2026-10-27 with many "[??]" words.
LOUD_VOICE_DBFS = -16.0
# Share of a voice's samples at full scale above which it is reported as clipping.
CLIPPING_FRACTION = 0.001
_CLIP_LEVEL = 32000

# How many voiced frames (30ms each) of the visitor are needed before the voice gate
# (RT.VOICE_GATE_DB) trusts its idea of how loud they are.
VOICE_GATE_MIN_FRAMES = 10
VOICE_GATE_DB = RT.VOICE_GATE_DB


def _frame_dbfs(frame: bytes) -> float:
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-6))


def _clipped_samples(frame: bytes) -> int:
    return int(np.count_nonzero(np.abs(np.frombuffer(frame, dtype=np.int16).astype(np.int32))
                                >= _CLIP_LEVEL))


def _gated_by_voice_level(level_db: float, voice_db: list[float]) -> bool:
    """True when a frame the VAD calls speech is far quieter than the visitor has been - somebody
    else talking in the room, not the person in front of the robot. Compared against the loud
    end of the visitor's own voice (90th percentile), so it follows each visitor and each
    microphone level rather than a fixed number."""
    if not VOICE_GATE_DB or len(voice_db) < VOICE_GATE_MIN_FRAMES:
        return False
    return level_db < float(np.percentile(voice_db, 90)) - VOICE_GATE_DB


def _report_mic_levels(voice_db: list[float], room_db: list[float],
                       clipped: int = 0, voiced_samples: int = 0, gated: int = 0) -> None:
    """One line per question saying how loud the visitor and the room were, so a log shows at a
    glance whether bad transcripts come from the microphone level rather than the recognizer."""
    if not voice_db:
        return
    voice = float(np.median(voice_db))
    line = f"[mic] voice {voice:.0f} dBFS"
    room = float(np.median(room_db)) if room_db else None
    if room is not None:
        line += f", room {room:.0f} dBFS ({voice - room:.0f} dB apart)"
    clip_share = clipped / voiced_samples if voiced_samples else 0.0
    if clipped:
        line += f", clipping {clip_share * 100:.1f}%"
    if gated:
        line += f", ignored {gated * FRAME_MS / 1000:.1f}s of quieter background voices"
    if clip_share >= CLIPPING_FRACTION:
        line += (" - the voice is CLIPPING (too loud for the microphone): lower the microphone "
                 "level to about 50% and turn Microphone Boost off (Windows: Sound settings > "
                 "Input > device properties)")
    elif voice > LOUD_VOICE_DBFS:
        line += (" - the microphone level is very high; if words come out wrong, lower it to "
                 "about 50% and turn Microphone Boost off")
    elif voice < QUIET_VOICE_DBFS:
        line += (" - the voice is very quiet: raise the microphone level (Windows: Sound "
                 "settings > Input > Volume) or speak closer")
    elif room is not None and voice - room < MIN_VOICE_ABOVE_ROOM_DB:
        line += (" - the room is nearly as loud as the voice, so the robot keeps 'hearing "
                 "speech': move the microphone away from fans or noise")
    print(line)


def listen_for_one_utterance(
    engine: WhisperSTTEngine,
    listener: STTListener,
    vad: StreamingVAD,
    audio_q: "queue.Queue[bytes]",
    timeout_s: float | None = None,
    prefix_frames: list[bytes] | None = None,
    on_turn_event=None,
) -> str | None:
    """Blocks until one complete utterance has been heard and transcribed, then returns the
    text (same preroll/debounce/min-utterance/max-utterance real-time logic this file's own
    __main__ loop used to have inline - extracted here so Stage 08's orchestrator can listen for
    exactly one question/answer per call instead of duplicating ~80 lines of subtle real-time
    audio logic, which would risk the two copies quietly drifting apart).

    timeout_s: if set, and nobody has even started speaking within this many seconds, gives up
    and returns None instead of blocking forever - lets a caller move on if someone walked away
    without asking anything. Has no effect once speech has actually started (MAX_UTTERANCE_S
    still caps that case) - this only bounds the "waiting for someone to start talking" phase.

    prefix_frames: audio captured BEFORE this call that belongs to the utterance about to be
    heard - in practice, the words somebody interrupted the robot with (see
    BargeInMonitor.captured_frames). Supplying them starts this call already recording, with
    those frames as the beginning of the utterance, so the visitor's interrupting sentence is
    transcribed and answered instead of being thrown away and asked for again.

    on_turn_event: optional observer, called as on_turn_event(event, detail) for
    "speech_started" / "paused" / "resumed" / "finished" / "discarded" / "partial" /
    "holding" (the endpointer has decided this pause is mid-sentence and is waiting). Purely
    informational - it drives Stage 08's conversation state machine and the [VAD]/[TURN] trace,
    and nothing in this loop asks it for a decision, so a slow or broken observer can make the
    log wrong but cannot change which words are heard. Exceptions from it are swallowed for
    exactly that reason.

    Assumes the caller already has a real mic stream open and feeding audio_q (via the same
    sd.RawInputStream + callback pattern as __main__ below) - opening/closing the stream once
    per call, rather than once per session, would reintroduce the exact stream-open transient
    false-trigger bug record_sample.py's WARMUP_FRAMES was built to fix (Stage 04's README).
    """

    def _emit(event: str, detail: str = "") -> None:
        if on_turn_event is None:
            return
        try:
            on_turn_event(event, detail)
        except Exception as exc:  # noqa: BLE001 - an observer must never break the audio loop
            print(f"[stt] turn observer failed ({exc!r}) - continuing to listen.")

    preroll: list[bytes] = []
    recording = False
    speech_streak = 0
    audio_chunks: list[bytes] = []
    silence_ms = 0
    recording_started_at = 0.0
    next_partial_at: float | None = None
    wait_started_at = time.monotonic()
    endpoint = _EndpointState()  # see _endpoint_decision()
    paused = False  # whether the CURRENT pause has already been reported, so a mid-turn pause
                    # is announced once instead of once per 30ms frame
    last_partial: str | None = None  # see TURN_STABLE_PARTIALS
    stable_partials = 0
    room_db: list[float] = []   # level of the last ~3s of non-speech before the visitor spoke
    voice_db: list[float] = []  # level of each voiced frame of this utterance
    clipped = voiced_samples = gated = 0  # for the [mic] line

    if prefix_frames:
        # Start mid-utterance rather than waiting for the VAD to re-confirm speech that has
        # already been confirmed once. Re-running the SPEECH_START_FRAMES debounce on audio
        # that already cleared a stricter test (louder than the room's echo AND speech-shaped,
        # sustained for BARGE_IN_MIN_SPEECH_MS) would only risk dropping the first word again.
        recording = True
        audio_chunks = list(prefix_frames)
        recording_started_at = time.monotonic() - len(prefix_frames) * FRAME_MS / 1000
        next_partial_at = time.monotonic() + PARTIAL_INTERVAL_S
        carried_ms = len(prefix_frames) * FRAME_MS
        print(f"[stt] continuing from {carried_ms}ms of audio the visitor interrupted with.")
        rt_log("TURN", f"resuming mid-utterance with {carried_ms}ms carried over from the "
                       f"interruption")
        _emit("speech_started", f"carried over {carried_ms}ms")

    while True:
        remaining = None
        if timeout_s is not None and not recording:
            remaining = timeout_s - (time.monotonic() - wait_started_at)
            if remaining <= 0:
                return None
        try:
            frame = audio_q.get(timeout=remaining)
        except queue.Empty:
            return None

        is_speech = vad.is_speech(frame)
        level_db = _frame_dbfs(frame)
        if is_speech and recording and _gated_by_voice_level(level_db, voice_db):
            # Speech, but far quieter than this visitor: someone else in the room. Counted as
            # silence, so the visitor's turn can end when THEY stop (see RT.VOICE_GATE_DB).
            is_speech = False
            gated += 1
        if is_speech and recording:
            voice_db.append(level_db)
            clipped += _clipped_samples(frame)
            voiced_samples += len(frame) // 2
        elif not is_speech and not recording:
            room_db.append(level_db)
            if len(room_db) > 100:
                del room_db[0]

        if not recording:
            preroll.append(frame)
            if len(preroll) > PREROLL_FRAMES:
                preroll.pop(0)

            if is_speech:
                speech_streak += 1
                if speech_streak >= SPEECH_START_FRAMES:
                    recording = True
                    audio_chunks = list(preroll)
                    silence_ms = 0
                    speech_streak = 0
                    endpoint.reset()
                    recording_started_at = time.monotonic()
                    paused = False
                    next_partial_at = recording_started_at + max(
                        PARTIAL_MIN_DURATION_S, PARTIAL_INTERVAL_S
                    )
                    print("[stt] speech detected, listening...")
                    rt_log("VAD", "user started speaking")
                    _emit("speech_started")
            else:
                speech_streak = 0
            continue

        audio_chunks.append(frame)
        if is_speech:
            # They carried on talking, so whatever was decided about the pause that just ended
            # is spent - the NEXT pause is a fresh question and deserves its own pitch check.
            # Without this reset, one mid-thought pause would put the rest of the utterance on
            # the patient threshold alone: never wrong, but slower than it needs to be for
            # somebody who paused once and then finished cleanly.
            if silence_ms:
                endpoint.reset()
            if paused:
                # The single most important event in this whole file: they paused, the robot did
                # NOT answer, and they carried on. This is "where is the computer science...
                # [pause] ...department?" surviving intact.
                rt_log("TURN", f"user still speaking - the {silence_ms}ms pause was mid-thought")
                _emit("resumed", f"{silence_ms}ms pause")
                paused = False
            silence_ms = 0
        else:
            silence_ms += FRAME_MS
            if not paused and silence_ms >= FRAME_MS * 2:
                rt_log("VAD", "user paused - not treating this as the end of the turn yet")
                _emit("paused")
                paused = True
        now = time.monotonic()
        elapsed_ms = (now - recording_started_at) * 1000

        if (
            PARTIAL_PREVIEW_ENABLED
            and next_partial_at is not None
            and now >= next_partial_at
        ):
            preview = engine.transcribe_stream(audio_chunks, listener.locale)[0].text
            if preview:
                print(f'[stt] (partial) "{preview}"')
                # LOG_TRANSCRIPTS=false keeps the trace usable on an unattended kiosk without
                # writing a visitor's half-finished words into a shared log - PRIVACY.md.
                rt_log("STT", f'partial transcript: "{preview}"' if RT.LOG_TRANSCRIPTS
                       else f"partial transcript: {len(preview)} chars")
                _emit("partial", preview)
            next_partial_at = now + PARTIAL_INTERVAL_S

            # --- the words stopped changing (see TURN_STABLE_PARTIALS) -----------------------
            # Compared normalised, because Whisper re-punctuates a growing buffer freely - the
            # same sentence can come back as "What is your name" and then "What is your name?"
            # without a single new word having been spoken, and treating that as a change would
            # reset the count forever on exactly the utterance this is meant to rescue.
            normalised = re.sub(r"[^a-z0-9 ]", "", (preview or "").lower()).strip()
            if normalised and normalised == last_partial:
                stable_partials += 1
            else:
                stable_partials = 0
                last_partial = normalised

            if (
                TURN_STABLE_PARTIALS
                and stable_partials >= TURN_STABLE_PARTIALS
                and elapsed_ms >= MIN_UTTERANCE_MS
                and preview
                and (not _ends_mid_thought(preview) or stable_partials >= TURN_STALL_PARTIALS)
            ):
                # Deliberately still checks _ends_mid_thought: somebody who has trailed off on
                # "...and the" has also stopped producing new words, and cutting them off there
                # is the exact failure the patient threshold exists to prevent. Stability says
                # they stopped TALKING; the transcript is what says they stopped THINKING. Up to
                # TURN_STALL_PARTIALS, that is - see there.
                quiet_s = stable_partials * PARTIAL_INTERVAL_S
                print(f"[stt] (no new words for ~{quiet_s:.0f}s - taking that as finished)")
                if hasattr(engine, "transcribe_final"):
                    # STT_CLOUD: the preview was the local recognizer's; the finished question
                    # gets the cloud transcript like every other ending does.
                    preview = (engine.transcribe_final(audio_chunks, listener.locale)[0].text
                               or preview)
                rt_log("TURN", f"user finished speaking - transcript unchanged across "
                               f"{stable_partials + 1} partials (~{quiet_s:.0f}s of no new "
                               f"words), even though the microphone still reports speech")
                rt_log("STT", f'final transcript: "{preview}"' if RT.LOG_TRANSCRIPTS
                       else f"final transcript: {len(preview)} chars")
                _emit("finished", preview)
                _print_result(preview, engine.last_confidence)
                _report_mic_levels(voice_db, room_db, clipped, voiced_samples, gated)
                return preview

        # --- adaptive endpointing (see ENDPOINT_EAGER_MS above) ------------------------------
        # Below the eager threshold nothing is decided. Between eager and patient, the pitch
        # contour gets ONE chance to say "that phrase landed" - checked once rather than every
        # 30ms frame, because a failed check means we are waiting for the patient threshold
        # anyway and re-running Whisper 30 times while we wait would be pure CPU burn.
        if not ADAPTIVE_ENDPOINTING_ENABLED:
            speech_ended = listener.vad.is_speech_ended(silence_ms)
        else:
            speech_ended = _endpoint_decision(silence_ms, audio_chunks, endpoint)
            if endpoint.holding and not endpoint.holding_reported and not speech_ended:
                endpoint.holding_reported = True
                _emit("holding", endpoint.holding)

        if speech_ended and elapsed_ms < MIN_UTTERANCE_MS:
            # too short to be real words (a cough, a click, a stray VAD false-positive) - drop
            # it without ever calling Whisper, instead of hallucinating on it, and go back to
            # waiting (with a fresh timeout clock) rather than giving up on the whole call.
            print("[stt] (too short - ignoring, probably not real speech)")
            rt_log("VAD", f"discarded {elapsed_ms:.0f}ms of audio - below the "
                          f"{MIN_UTTERANCE_MS}ms minimum, so a cough/click/knock rather than "
                          f"words. Nothing was sent to the model.")
            _emit("discarded", f"{elapsed_ms:.0f}ms")
            recording = False
            audio_chunks = []
            voice_db = []
            clipped = voiced_samples = gated = 0
            preroll = []
            silence_ms = 0
            paused = False
            endpoint.reset()
            next_partial_at = None
            wait_started_at = time.monotonic()
            continue

        if ADAPTIVE_ENDPOINTING_ENABLED:
            text = _transcribe_now(engine, listener, audio_chunks) if speech_ended else None
        else:
            # listener.listen() itself, except that the finished words come from
            # _final_transcribe (the cloud one with STT_CLOUD) - see _transcribe_now.
            text = (_transcribe_now(engine, listener, audio_chunks)
                    if listener.vad.is_speech_ended(silence_ms) else None)

        # Second adaptive signal: the words. A grammatically unfinished transcript means the
        # pause was someone thinking, not someone finished - keep the recording running rather
        # than answering half a question. Costs nothing in the common case: this is the
        # transcription we would have done at the patient threshold anyway, just done earlier.
        if (
            ADAPTIVE_ENDPOINTING_ENABLED
            and text is not None
            and silence_ms < ENDPOINT_PATIENT_MS
            and _ends_mid_thought(text)
        ):
            print(f'[stt] (heard "{text}" - sounds unfinished, giving them a moment)')
            rt_log("TURN", "user still speaking - the transcript ends mid-thought, so this "
                           "pause is thinking, not finishing")
            _emit("resumed", "transcript ends mid-thought")
            paused = False  # so the next pause in this same turn is reported again
            # The transcript is better evidence than a contour we could not read, so drop
            # any pending "unknown" fallback and let them have the full patient threshold.
            endpoint.reset()
            endpoint.checked = True
            endpoint.holding = "transcript ends mid-thought"
            endpoint.holding_reported = True
            _emit("holding", endpoint.holding)
            continue

        if text is None and elapsed_ms >= MAX_UTTERANCE_S * 1000:
            print("[stt] max utterance length reached, transcribing what I have...")
            rt_log("TURN", f"user finished speaking - hit the {MAX_UTTERANCE_S:.0f}s cap")
            text = _final_transcribe(engine)(audio_chunks, listener.locale)[0].text

        if text is not None:
            rt_log("TURN", f"user finished speaking after {elapsed_ms:.0f}ms of audio and a "
                           f"{silence_ms}ms pause")
            rt_log("STT", f'final transcript: "{text}"' if RT.LOG_TRANSCRIPTS
                   else f"final transcript: {len(text)} chars")
            _emit("finished", text)
            _print_result(text, engine.last_confidence)
            _report_mic_levels(voice_db, room_db, clipped, voiced_samples, gated)
            return text


def _self_test_endpointing() -> None:
    """Offline checks for the two adaptive-endpointing signals, run before the live mic loop
    below so a broken heuristic shows up as a failed assertion at startup rather than as the
    robot mysteriously interrupting people. Needs no microphone and no Whisper model."""
    # --- the words ---------------------------------------------------------------------------
    finished = [
        "where is the library",
        "Where is the library?",
        "how long is the PhD programme",
        "what time does the cafeteria open?",
        "tell me about the engineering department",
    ]
    unfinished = [
        "where is the department of",
        "I want to know if",
        "tell me about the",
        "can you tell me,",
        # Whisper heard a word it could not place after "the" - we do not have a complete
        # thought, so waiting is right even though the sentence looks longer than it is.
        "I need the [??]",
    ]
    for text in finished:
        assert not _ends_mid_thought(text), f"treated as unfinished: {text!r}"
    for text in unfinished:
        assert _ends_mid_thought(text), f"treated as finished: {text!r}"

    # --- the pitch contour -------------------------------------------------------------------
    # Synthetic voiced tones: a near-flat pitch is the continuation contour (keep listening),
    # while a clear fall (statement) or rise (question) means the phrase landed.
    def _tone(start_hz: float, end_hz: float, duration_s: float = 0.4) -> bytes:
        t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
        freq = np.linspace(start_hz, end_hz, t.size)
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        wave = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
        return (wave / np.max(np.abs(wave)) * 20000).astype(np.int16).tobytes()

    assert _pitch_trend(_tone(120, 122)) == "level", "flat pitch should read as mid-sentence"
    assert _pitch_trend(_tone(180, 110)) == "moving", "falling pitch should end the turn"
    assert _pitch_trend(_tone(110, 200)) == "moving", "rising pitch should end the turn"
    # No voiced audio at all must never be mistaken for a confident decision either way.
    assert _pitch_trend(np.zeros(6400, dtype=np.int16).tobytes()) == "unknown"
    assert _pitch_trend(_tone(120, 122, 0.05)) == "unknown", "too short to judge"

    # --- the endpoint decision itself --------------------------------------------------------
    # The contour is only half the job; what the endpointer DOES with it is the half that was
    # wrong. An unreadable contour ("unknown") was being treated exactly like a terminal one, so
    # every pause after an unvoiced ending ended the turn at ENDPOINT_EAGER_MS - the robot
    # answering half a question, which is what "it isn't getting my words" looked like in use.
    def _frames(audio: bytes) -> list[bytes]:
        """Split raw PCM into FRAME_MS frames, the way the mic callback delivers it."""
        step = FRAME_SAMPLES * 2
        return [audio[i:i + step] for i in range(0, len(audio) - step + 1, step)]

    silence = _frames(np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes())

    def _decide(speech: bytes, silence_ms: int, state=None):
        """Runs the decision over one pause, feeding it the same buffer shape the loop does."""
        state = state or _EndpointState()
        chunks = _frames(speech) + silence[: max(int(silence_ms / FRAME_MS), 1)]
        return _endpoint_decision(silence_ms, chunks, state), state

    # 1. A terminal contour still ends the turn eagerly - the latency win must survive the fix.
    ended, _ = _decide(_tone(180, 110), ENDPOINT_EAGER_MS)
    assert ended, "a falling contour should end the turn at the eager threshold"

    # 2. A continuation contour still waits.
    ended, st = _decide(_tone(120, 122), ENDPOINT_EAGER_MS)
    assert not ended and st.checked and st.floor_ms is None

    # 3. THE FIX: an unreadable contour must NOT end the turn eagerly. White noise has no
    #    periodicity, the same way an unvoiced consonant ("...science", "...class") does not.
    noise = (np.random.default_rng(0).normal(0, 6000, SAMPLE_RATE // 2)
             .astype(np.int16).tobytes())
    assert _pitch_trend(noise) == "unknown", "test audio is not actually unreadable"
    ended, st = _decide(noise, ENDPOINT_EAGER_MS)
    assert not ended, (
        "an unreadable pitch contour ended the turn at the eager threshold - this is the bug "
        "that cut people off 550ms into an ordinary mid-sentence pause"
    )
    assert st.floor_ms == ENDPOINT_UNKNOWN_MS

    # ...but it must not wait forever either: the same pause ends at the 900ms fallback, well
    # before the 1500ms hard stop.
    assert _endpoint_decision(ENDPOINT_UNKNOWN_MS, [], st), "the 900ms fallback never fired"
    assert not _endpoint_decision(ENDPOINT_UNKNOWN_MS - FRAME_MS, [], st)

    # 4. The hard stop outranks everything, including a contour that says "keep waiting".
    ended, _ = _decide(_tone(120, 122), ENDPOINT_PATIENT_MS)
    assert ended, "the patient threshold must end the turn whatever the contour says"

    # 5. One reading per pause: a second call at the same pause must not re-measure.
    st = _EndpointState()
    _endpoint_decision(ENDPOINT_EAGER_MS, _frames(_tone(120, 122)) + silence[:18], st)
    assert st.checked
    assert not _endpoint_decision(ENDPOINT_EAGER_MS + FRAME_MS, [], st)

    # 6. ...and reset() genuinely clears it, so the next pause is judged fresh rather than
    #    inheriting the last one's verdict.
    st.reset()
    assert not st.checked and st.floor_ms is None

    print("[self-test] adaptive endpointing: words + pitch contour OK "
          f"(eager {ENDPOINT_EAGER_MS}ms, patient {ENDPOINT_PATIENT_MS}ms)")


def _self_test_smart_turn_policy() -> None:
    """How _endpoint_decision uses the end-of-turn model, with a stand-in model (no file
    needed): confident "finished" ends early, confident "unfinished" waits for the patient
    limit, and anything in between leaves the decision to the pitch rules."""
    global SMART_TURN_ENABLED, _smart_turn_model

    class _Fake:
        last_ms = 1.0

        def __init__(self, p):
            self.p = p

        def probability(self, _audio):
            return self.p

    def _tone(start_hz: float, end_hz: float, duration_s: float = 0.4) -> list[bytes]:
        t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
        phase = 2 * np.pi * np.cumsum(np.linspace(start_hz, end_hz, t.size)) / SAMPLE_RATE
        audio = (np.sin(phase) * 20000).astype(np.int16).tobytes()
        step = FRAME_SAMPLES * 2
        return [audio[i:i + step] for i in range(0, len(audio) - step + 1, step)]

    falling_tone = _tone(180, 110)
    quiet = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()

    def falling_then(silence_ms: int) -> list[bytes]:
        """The buffer the loop would hold: the phrase, then silence_ms of quiet."""
        return falling_tone + [quiet] * max(int(silence_ms / FRAME_MS), 1)

    saved = (SMART_TURN_ENABLED, _smart_turn_model)
    try:
        SMART_TURN_ENABLED = True
        _smart_turn_model = lambda: _Fake(0.9)  # noqa: E731
        st = _EndpointState()
        assert not _endpoint_decision(SMART_TURN_MIN_SILENCE_MS - FRAME_MS, falling_then(SMART_TURN_MIN_SILENCE_MS - FRAME_MS), st)
        assert _endpoint_decision(SMART_TURN_MIN_SILENCE_MS, falling_then(SMART_TURN_MIN_SILENCE_MS), st), \
            "a confident 'finished' should end the turn at the short silence"

        _smart_turn_model = lambda: _Fake(0.05)  # noqa: E731
        st = _EndpointState()
        assert not _endpoint_decision(SMART_TURN_MIN_SILENCE_MS, falling_then(SMART_TURN_MIN_SILENCE_MS), st)
        assert not _endpoint_decision(ENDPOINT_EAGER_MS, falling_then(ENDPOINT_EAGER_MS), st), \
            "a confident 'unfinished' must override the falling-pitch shortcut"
        assert _endpoint_decision(ENDPOINT_PATIENT_MS, falling_then(ENDPOINT_PATIENT_MS), st), "the hard stop still wins"

        _smart_turn_model = lambda: _Fake(0.35)  # noqa: E731
        st = _EndpointState()
        assert not _endpoint_decision(SMART_TURN_MIN_SILENCE_MS, falling_then(SMART_TURN_MIN_SILENCE_MS), st)
        assert _endpoint_decision(ENDPOINT_EAGER_MS, falling_then(ENDPOINT_EAGER_MS), st), \
            "an unsure model should leave the decision to the pitch rules"

        _smart_turn_model = lambda: None  # noqa: E731 - model file missing
        st = _EndpointState()
        assert not _endpoint_decision(SMART_TURN_MIN_SILENCE_MS, falling_then(SMART_TURN_MIN_SILENCE_MS), st)
        assert _endpoint_decision(ENDPOINT_EAGER_MS, falling_then(ENDPOINT_EAGER_MS), st)
    finally:
        SMART_TURN_ENABLED, _smart_turn_model = saved
    print("[self-test] end-of-turn model policy OK")


def run_self_tests() -> None:
    """Every offline self-test in this file. The pitch/turn tests describe the pitch rules on
    synthetic tones, so they run with the end-of-turn model switched off; its own policy test
    uses a stand-in model."""
    global SMART_TURN_ENABLED
    saved, SMART_TURN_ENABLED = SMART_TURN_ENABLED, False
    try:
        _self_test_endpointing()
        _self_test_hotwords()
        _self_test_barge_in_capture()
        _self_test_turn_events()
        _self_test_noisy_room()
    finally:
        SMART_TURN_ENABLED = saved
    _self_test_smart_turn_policy()


def _self_test_hotwords() -> None:
    """Offline checks for the decode-time bias list. No model, no microphone."""
    merged = merge_hotwords(
        "Lincoln University College, Where, the, Lina",
        "Lincoln University College, Bachelor of Computer Science",
    )
    terms = [t.strip() for t in merged.split(",")]
    assert terms.count("Lincoln University College") == 1, f"duplicate survived: {merged}"
    assert "Where" not in terms and "the" not in terms, f"stopword survived: {merged}"
    assert "Lina" in terms and "Bachelor of Computer Science" in terms, merged
    # Earlier parts win the budget, so identity terms are never crowded out by scraped ones.
    tight = merge_hotwords("Lina, Alex Tan", "Master of Business Administration", max_chars=20)
    assert tight.startswith("Lina"), tight

    assert merge_hotwords("") == ""
    assert len(merge_hotwords(", ".join(f"Term Number {i}" for i in range(200)))) \
        <= HOTWORDS_MAX_CHARS

    print(f"[self-test] hotword biasing OK (budget {HOTWORDS_MAX_CHARS} chars)")


def _self_test_barge_in_capture() -> None:
    """Checks that an interruption's own words survive it - no microphone, no models.

    This is the half of barge-in that is easy to get wrong and impossible to notice: stopping
    playback works either way, and the bug only shows up as a visitor having to repeat
    themselves, which reads like bad speech recognition rather than like dropped audio.
    """
    stop_event = threading.Event()
    audio_q: "queue.Queue[bytes]" = queue.Queue()

    silence = b"\x00\x00" * FRAME_SAMPLES
    # A loud 200Hz tone: periodic enough for webrtcvad to call it speech, and far above any
    # plausible echo threshold.
    t = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
    loud = (np.sin(2 * np.pi * 200 * t) * 12000).astype(np.int16).tobytes()

    monitor = BargeInMonitor(
        audio_q, StreamingVAD(), stop_event,
        rms_threshold=0.01, min_speech_ms=90, warmup_ms=0, keep_audio=True,
    )
    for _ in range(5):
        audio_q.put(silence)
    for _ in range(12):
        audio_q.put(loud)

    monitor.start()
    deadline = time.monotonic() + 5.0
    while not stop_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    interrupted = monitor.stop()

    assert interrupted, "sustained loud speech-shaped audio should have triggered barge-in"
    assert stop_event.is_set(), "barge-in must set the TTS stop event, or playback never stops"

    carried = monitor.take_captured_audio()
    assert carried, (
        "the frames that triggered the interruption must be KEPT - discarding them is what "
        "forces the visitor to repeat the sentence they just interrupted with"
    )
    assert b"".join(carried).count(loud) >= 3, (
        "the carried audio must actually contain the interrupting speech, not just trailing "
        "silence"
    )
    assert monitor.take_captured_audio() == [], (
        "captured audio must be handed over exactly once - replaying it twice would transcribe "
        "and answer the same interruption twice"
    )
    print(f"[self-test] barge-in keeps the interrupting speech: {len(carried)} frame(s), "
          f"{len(carried) * FRAME_MS}ms carried over, and only once.")

    # ...and with keeping switched off, nothing is retained but the interruption still fires.
    stop_event.clear()
    audio_q2: "queue.Queue[bytes]" = queue.Queue()
    for _ in range(12):
        audio_q2.put(loud)
    off = BargeInMonitor(audio_q2, StreamingVAD(), stop_event, rms_threshold=0.01,
                         min_speech_ms=90, warmup_ms=0, keep_audio=False)
    off.start()
    deadline = time.monotonic() + 5.0
    while not stop_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert off.stop(), "barge-in must still fire with BARGE_IN_KEEP_AUDIO off"
    assert off.take_captured_audio() == []
    print("[self-test] BARGE_IN_KEEP_AUDIO=false still interrupts, just without the carry-over.")


def _self_test_turn_events() -> None:
    """Checks the turn observer reports a mid-sentence pause as a PAUSE and not as an END.

    Drives listen_for_one_utterance with a scripted audio queue and a stub engine, so it
    exercises the real loop - the preroll, the debounce, the pause bookkeeping and the
    endpointer - without a microphone or Whisper.
    """
    silence = b"\x00\x00" * FRAME_SAMPLES
    t = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
    voiced = (np.sin(2 * np.pi * 200 * t) * 12000).astype(np.int16).tobytes()

    audio_q: "queue.Queue[bytes]" = queue.Queue()
    # "where is the computer science" ... 600ms pause ... "department" ... long pause.
    for _ in range(30):
        audio_q.put(voiced)
    for _ in range(20):      # 600ms - past the eager threshold, inside the patient one
        audio_q.put(silence)
    for _ in range(20):
        audio_q.put(voiced)
    for _ in range(60):      # 1800ms - past the hard stop, so the turn must end here
        audio_q.put(silence)

    class _StubEngine:
        """Returns an unfinished transcript first, then a complete one - which is what makes the
        linguistic half of the endpointer fire at the right moment."""

        def __init__(self):
            self.calls = 0
            self.last_confidence = None

        def transcribe_stream(self, audio_chunks, locale):
            self.calls += 1
            text = ("where is the computer science the" if self.calls <= 2
                    else "where is the computer science department?")
            return [TranscriptChunk(text=text, is_final=True)]

    engine = _StubEngine()
    events: list[tuple[str, str]] = []
    # MIN_UTTERANCE_MS is measured against the WALL CLOCK, while this test hands the loop its
    # audio all at once - so a 4-second scripted utterance is consumed in a few milliseconds and
    # would be discarded as "too short". That mismatch is real and is exactly what _listen() in
    # Stage 08 drains the stale queue to avoid; it is a property of feeding the loop faster than
    # real time, not a bug in the guard. Standing it down for this test keeps the check fast
    # (the alternative is sleeping through four real seconds at every startup) and leaves the
    # guard itself covered by the "discarded" branch, which fires on genuinely short audio.
    original_min = globals()["MIN_UTTERANCE_MS"]
    globals()["MIN_UTTERANCE_MS"] = 0
    try:
        text = listen_for_one_utterance(
            engine, STTListener(engine, PatientVAD(), locale="en-US"), StreamingVAD(),
            audio_q, timeout_s=5.0, on_turn_event=lambda e, d: events.append((e, d)),
        )
    finally:
        globals()["MIN_UTTERANCE_MS"] = original_min

    names = [e for e, _ in events]
    assert text, "the scripted utterance should have produced a transcript"
    assert names[0] == "speech_started", f"expected speech_started first, got {names}"
    assert "paused" in names, (
        "the 600ms mid-sentence gap must be reported as a PAUSE - that is the signal a naive "
        f"assistant misreads as the end of a turn. Got {names}"
    )
    assert names.index("paused") < names.index("finished"), (
        f"the pause must be reported before the turn ends, not after it. Got {names}"
    )
    assert names.count("finished") == 1, f"exactly one turn should have ended. Got {names}"
    resumed = names.count("resumed")
    assert resumed >= 1, (
        f"the speaker carried on after the pause, so at least one 'resumed' was expected - "
        f"without it the robot answered half a question. Got {names}"
    )
    print(f"[self-test] turn events OK: {' -> '.join(names)}")

    # ...and the interruption carry-over starts the loop already recording.
    audio_q2: "queue.Queue[bytes]" = queue.Queue()
    for _ in range(10):
        audio_q2.put(voiced)
    for _ in range(60):
        audio_q2.put(silence)
    engine2 = _StubEngine()
    engine2.calls = 5  # so it returns the complete transcript straight away
    events2: list[tuple[str, str]] = []
    globals()["MIN_UTTERANCE_MS"] = 0  # same wall-clock reason as above
    try:
        listen_for_one_utterance(
            engine2, STTListener(engine2, PatientVAD(), locale="en-US"), StreamingVAD(),
            audio_q2, timeout_s=5.0, prefix_frames=[voiced] * 15,
            on_turn_event=lambda e, d: events2.append((e, d)),
        )
    finally:
        globals()["MIN_UTTERANCE_MS"] = original_min
    assert events2 and events2[0][0] == "speech_started" and "carried over" in events2[0][1], (
        f"prefix_frames must begin the utterance already recording. Got {events2[:2]}"
    )
    print(f"[self-test] interruption carry-over OK: started with {events2[0][1]}.")


def _self_test_noisy_room() -> None:
    """The turn must still end when the microphone NEVER reports silence.

    Reproduces a real run (2026-09-17, Realtek mic array). The visitor asked "What is your
    name?" in about two seconds; webrtcvad then reported speech continuously for another
    nineteen, so silence_ms stayed at zero, the adaptive endpointer was never handed a pause to
    judge, and the turn only ended on the 15-second cap - by which point the visitor had waited
    twenty seconds to be told a name.

    Every frame here is voiced, exactly like that room. The only thing that changes is the
    transcript, which stops growing - and that has to be enough.
    """
    t = np.arange(FRAME_SAMPLES) / SAMPLE_RATE
    voiced = (np.sin(2 * np.pi * 200 * t) * 12000).astype(np.int16).tobytes()

    class _DeafVAD:
        """A VAD that never reports silence - the failure exactly as it happened. Modelled
        directly rather than by feeding it noisy audio, because what matters is the CONSEQUENCE
        (silence_ms can never rise above zero) and that must be reproduced every run, not
        whenever a synthetic signal happens to fool webrtcvad."""

        def is_speech(self, frame_bytes: bytes) -> bool:
            return True

    audio_q: "queue.Queue[bytes]" = queue.Queue()
    for _ in range(700):  # 21 seconds, not one frame of silence in it
        audio_q.put(voiced)

    class _StubEngine:
        """Grows the transcript for the first couple of calls, then repeats itself - a speaker
        who stopped talking while the room carried on making noise."""

        def __init__(self):
            self.calls = 0
            self.last_confidence = None

        def transcribe_stream(self, audio_chunks, locale):
            self.calls += 1
            text = "What is" if self.calls == 1 else "What is your name?"
            return [TranscriptChunk(text=text, is_final=True)]

    engine = _StubEngine()
    events: list[tuple[str, str]] = []
    original_min = globals()["MIN_UTTERANCE_MS"]
    original_partial = globals()["PARTIAL_PREVIEW_ENABLED"]
    original_gap = globals()["PARTIAL_INTERVAL_S"]
    original_first = globals()["PARTIAL_MIN_DURATION_S"]
    globals()["MIN_UTTERANCE_MS"] = 0  # wall-clock guard; see _self_test_turn_events
    globals()["PARTIAL_PREVIEW_ENABLED"] = True
    # Partials are scheduled on the WALL CLOCK, and this test hands the loop 21 seconds of audio
    # in a few milliseconds - so at the shipped 1.5s spacing not one partial would ever come due
    # and the signal under test could never fire. Zeroing the spacing makes every frame a
    # partial, which is the same sequence of transcripts the real robot sees, just without
    # sleeping through it. The real spacing is what turns "3 identical partials" into "~3
    # seconds of no new words"; that arithmetic is not what is being tested here.
    globals()["PARTIAL_INTERVAL_S"] = 0.0
    globals()["PARTIAL_MIN_DURATION_S"] = 0.0
    try:
        text = listen_for_one_utterance(
            engine, STTListener(engine, PatientVAD(), locale="en-US"), _DeafVAD(),
            audio_q, timeout_s=5.0, on_turn_event=lambda e, d: events.append((e, d)),
        )
    finally:
        globals()["MIN_UTTERANCE_MS"] = original_min
        globals()["PARTIAL_PREVIEW_ENABLED"] = original_partial
        globals()["PARTIAL_INTERVAL_S"] = original_gap
        globals()["PARTIAL_MIN_DURATION_S"] = original_first

    assert text == "What is your name?", f"expected the settled transcript, got {text!r}"
    assert not audio_q.empty(), (
        "the turn consumed all 21 seconds of audio - it ended on the length cap again, which "
        "is the exact bug this test exists for"
    )
    consumed_s = (700 - audio_q.qsize()) * FRAME_MS / 1000
    assert consumed_s < MAX_UTTERANCE_S, (
        f"the turn ran {consumed_s:.1f}s, past the {MAX_UTTERANCE_S}s cap - the stability "
        f"signal did not fire"
    )
    assert [e for e, _ in events].count("finished") == 1
    print(f"[self-test] a mic that never reports silence: turn ended after {consumed_s:.1f}s of "
          f"audio on transcript stability alone (the cap is {MAX_UTTERANCE_S:.0f}s).")

    # ...and an UNFINISHED sentence gets one partial more than a finished one, then it ends too.
    # Until 2026-10-27 it was never ended this way, so a mic that never reports silence held a
    # visitor whose words stopped on "is" for the full 15-second cap (a real run). Counted in
    # transcriptions, because PARTIAL_INTERVAL_S is zeroed here: one more partial is ~1.5s more of
    # patience on the real robot.
    audio_q2: "queue.Queue[bytes]" = queue.Queue()
    for _ in range(700):
        audio_q2.put(voiced)

    class _TrailedOffEngine:
        last_confidence = None

        def __init__(self):
            self.calls = 0

        def transcribe_stream(self, audio_chunks, locale):
            self.calls += 1
            return [TranscriptChunk(text="I wanted to ask about the", is_final=True)]

    engine2 = _TrailedOffEngine()
    globals()["MIN_UTTERANCE_MS"] = 0
    globals()["PARTIAL_PREVIEW_ENABLED"] = True
    globals()["PARTIAL_INTERVAL_S"] = 0.0
    globals()["PARTIAL_MIN_DURATION_S"] = 0.0
    try:
        text2 = listen_for_one_utterance(
            engine2, STTListener(engine2, PatientVAD(), locale="en-US"), _DeafVAD(),
            audio_q2, timeout_s=5.0,
        )
    finally:
        globals()["MIN_UTTERANCE_MS"] = original_min
        globals()["PARTIAL_PREVIEW_ENABLED"] = original_partial
        globals()["PARTIAL_INTERVAL_S"] = original_gap
        globals()["PARTIAL_MIN_DURATION_S"] = original_first
    assert text2 == "I wanted to ask about the", f"got {text2!r}"
    if TURN_STABLE_PARTIALS:
        assert engine2.calls == TURN_STALL_PARTIALS + 1, (
            f"an unfinished transcript ended after {engine2.calls} partials; expected "
            f"{TURN_STALL_PARTIALS + 1} - one more than a finished sentence gets, no fewer"
        )
    print(f"[self-test] ...and a transcript ending mid-thought gets one more partial of patience "
          f"({engine2.calls} partials), then ends instead of waiting for the "
          f"{MAX_UTTERANCE_S:.0f}s cap.")

if __name__ == "__main__":
    run_self_tests()

    engine = build_stt_engine(
        initial_prompt=real_initial_prompt(), vocabulary=real_vocabulary(),
        hotwords=real_hotwords(),
    )
    listener = STTListener(engine, PatientVAD(), locale="en-US")
    vad = StreamingVAD()

    audio_q: queue.Queue[bytes] = queue.Queue()

    def _on_audio(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    print("Live listening running. Speak, then pause - press Ctrl+C to quit.")

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        callback=_on_audio,
    )

    try:
        with stream:
            while True:
                listen_for_one_utterance(engine, listener, vad, audio_q)
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
