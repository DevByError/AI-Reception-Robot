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
VAD_AGGRESSIVENESS = (
    1  # 0 (most permissive) - 3 (most aggressive about rejecting non-speech)
)

# A single noisy/misclassified frame shouldn't start a recording - require a short run of
# consecutive speech-flagged frames first (same debouncing idea as Stage 01's
# ENTER_CONFIRM_FRAMES, just for audio instead of video). Kept short since MIN_UTTERANCE_MS
# below is what actually screens out pure noise now, not this.
SPEECH_START_FRAMES = 4  # ~120ms

# Frames of audio kept from *before* speech was confirmed, prepended once recording starts, so
# the first syllable isn't clipped while we were still debouncing SPEECH_START_FRAMES - the
# listening-side mirror of the clipped-audio fix LiveTTSSpeaker needed on the speaking side in
# Stage 03. Widened from 300ms to 600ms for extra margin.
PREROLL_FRAMES = 20  # ~600ms

MAX_UTTERANCE_S = (
    15  # safety cap - forces a transcription even if silence never triggers
)
# (stuck-open mic, constant background noise misread as speech)

# If VAD confirms "speech started" but trailing silence hits immediately after, the whole clip is
# too short to be real words - transcribing it anyway is exactly what produces Whisper's
# hallucinated filler ("Okay. Okay. Okay.") on essentially empty audio. Discard it silently
# instead of sending it to the model.
MIN_UTTERANCE_MS = 400

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
# chunks: more context + higher latency"). Off by default: re-running Whisper on a growing buffer
# every couple of seconds is real extra CPU load for a kiosk that isn't doing live captioning -
# flip PARTIAL_PREVIEW_ENABLED on if you want to see/tune the latency-vs-accuracy tradeoff
# yourself before deciding whether it's worth it for Stage 08's UX.
PARTIAL_PREVIEW_ENABLED = False
PARTIAL_MIN_DURATION_S = (
    2.0  # don't bother previewing until the recording's already this long
)
PARTIAL_INTERVAL_S = 2.0  # how often to re-preview after that

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
    # Common question/preposition words - added after a real report (2026-08-12) that these were
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
    # Same fix, same reasoning, extended after a second real report (2026-08-19): "the"/"are"/
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
ADAPTIVE_ENDPOINTING_ENABLED = True

ENDPOINT_EAGER_MS = 550     # earliest a turn can be called finished, and only with evidence
ENDPOINT_PATIENT_MS = 1500  # hard stop - end the turn here no matter what the signals say
# Used when the pitch contour cannot be measured at all (see _pitch_trend()'s "unknown", and the
# branch that consumes it in listen_for_one_utterance). Deliberately PatientVAD's old fixed
# 900ms: with no evidence either way, the right behaviour is the one from before this adaptive
# machinery existed, not the eager cut.
ENDPOINT_UNKNOWN_MS = 900

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

    __slots__ = ("checked", "floor_ms")

    def __init__(self) -> None:
        self.checked = False           # the contour has had its look at this pause
        self.floor_ms: int | None = None  # fallback threshold when it could not be read

    def reset(self) -> None:
        self.checked = False
        self.floor_ms = None


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
    chunks = engine.transcribe_stream(audio_chunks, listener.locale)
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
BARGE_IN_ENABLED = False

# How far above the measured echo floor a sound has to be to count as a person rather than the
# robot's own voice. Multiplier on RMS, so 2.5x is about 8dB.
BARGE_IN_MARGIN = 2.5

# Sustained, not instantaneous: a door, a cough or one bad frame must not stop the robot
# mid-sentence. About a third of a second is roughly a syllable or two of real speech.
BARGE_IN_MIN_SPEECH_MS = 330

# Ignored at the very start of each utterance. Playback has an attack transient, and the
# visitor's own last words can still be reverberating in the room when the robot starts its
# reply - both would otherwise read as an immediate interruption.
BARGE_IN_WARMUP_MS = 700


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
    """

    def __init__(self, audio_q, vad: "StreamingVAD", stop_event,
                 rms_threshold: float, min_speech_ms: int = BARGE_IN_MIN_SPEECH_MS,
                 warmup_ms: int = BARGE_IN_WARMUP_MS):
        self.audio_q = audio_q
        self.vad = vad
        self.stop_event = stop_event
        self.rms_threshold = rms_threshold
        self.min_speech_ms = min_speech_ms
        self.warmup_ms = warmup_ms
        self.triggered = False
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        started = time.monotonic()
        speech_ms = 0
        while self._running.is_set():
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if (time.monotonic() - started) * 1000 < self.warmup_ms:
                continue

            # BOTH tests must pass. Loudness alone fires on a door slam; the VAD alone fires on
            # the robot's own perfectly speech-shaped voice. Together they mean "speech, and
            # louder than this room's echo of ourselves".
            if frame_rms(frame) > self.rms_threshold and self.vad.is_speech(frame):
                speech_ms += FRAME_MS
                if speech_ms >= self.min_speech_ms:
                    self.triggered = True
                    self.stop_event.set()
                    print("[barge-in] someone started talking - stopping.")
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def calibrate_echo_floor(audio_q, speak, sample_text: str | None = None) -> float:
    """Measures how loudly this room returns the robot's OWN voice to the microphone.

    Call with nobody speaking. `speak` is any callable that plays a sentence and blocks;
    everything the microphone captures while it runs is, by definition, echo. The returned value
    is the RMS to set a barge-in threshold above.

    The 95th percentile rather than the mean: what matters is the loud part of the robot's own
    speech, since that is what a threshold has to clear to avoid false triggers.
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
    if not levels:
        return 0.0
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


class WhisperSTTEngine:
    """Implements the same STTEngine protocol as MockSTTEngine (stt_listener.py) - a drop-in
    replacement, so STTListener doesn't need to change at all. Not truly incremental/streaming
    on its own (returns one final chunk once the whole utterance is in) - see PARTIAL_PREVIEW_*
    above and README's known limitations for how streaming is layered on top instead."""

    def __init__(
        self,
        model_size: str = "small.en",
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
            pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
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
        from faster_whisper import WhisperModel

        print(
            f"Loading faster-whisper '{model_size}' (first run downloads it, cached after "
            f"that)..."
        )
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
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

        # Peak normalization always runs now (2026-08-19) - previously bundled behind
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


def identity_vocabulary() -> list[str]:
    """The robot's own identity (2026-08-19, see stage05's identity.py) - rare proper nouns, same
    reasoning as "Lincoln" in INSTITUTION_VOCABULARY: without a hint, Whisper has nothing in its
    training data biasing it toward an uncommon name over a more common-sounding near-homophone.

    Derived from the database-backed profile instead of being listed literally (2026-09-04), and
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


def listen_for_one_utterance(
    engine: WhisperSTTEngine,
    listener: STTListener,
    vad: StreamingVAD,
    audio_q: "queue.Queue[bytes]",
    timeout_s: float | None = None,
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

    Assumes the caller already has a real mic stream open and feeding audio_q (via the same
    sd.RawInputStream + callback pattern as __main__ below) - opening/closing the stream once
    per call, rather than once per session, would reintroduce the exact stream-open transient
    false-trigger bug record_sample.py's WARMUP_FRAMES was built to fix (Stage 04's README).
    """
    preroll: list[bytes] = []
    recording = False
    speech_streak = 0
    audio_chunks: list[bytes] = []
    silence_ms = 0
    recording_started_at = 0.0
    next_partial_at: float | None = None
    wait_started_at = time.monotonic()
    endpoint = _EndpointState()  # see _endpoint_decision()

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
                    next_partial_at = recording_started_at + max(
                        PARTIAL_MIN_DURATION_S, PARTIAL_INTERVAL_S
                    )
                    print("[stt] speech detected, listening...")
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
            silence_ms = 0
        else:
            silence_ms += FRAME_MS
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
            next_partial_at = now + PARTIAL_INTERVAL_S

        # --- adaptive endpointing (see ENDPOINT_EAGER_MS above) ------------------------------
        # Below the eager threshold nothing is decided. Between eager and patient, the pitch
        # contour gets ONE chance to say "that phrase landed" - checked once rather than every
        # 30ms frame, because a failed check means we are waiting for the patient threshold
        # anyway and re-running Whisper 30 times while we wait would be pure CPU burn.
        if not ADAPTIVE_ENDPOINTING_ENABLED:
            speech_ended = listener.vad.is_speech_ended(silence_ms)
        else:
            speech_ended = _endpoint_decision(silence_ms, audio_chunks, endpoint)

        if speech_ended and elapsed_ms < MIN_UTTERANCE_MS:
            # too short to be real words (a cough, a click, a stray VAD false-positive) - drop
            # it without ever calling Whisper, instead of hallucinating on it, and go back to
            # waiting (with a fresh timeout clock) rather than giving up on the whole call.
            print("[stt] (too short - ignoring, probably not real speech)")
            recording = False
            audio_chunks = []
            preroll = []
            silence_ms = 0
            endpoint.reset()
            next_partial_at = None
            wait_started_at = time.monotonic()
            continue

        if ADAPTIVE_ENDPOINTING_ENABLED:
            text = _transcribe_now(engine, listener, audio_chunks) if speech_ended else None
        else:
            text = listener.listen(audio_chunks, trailing_silence_ms=silence_ms)

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
            # The transcript is better evidence than a contour we could not read, so drop
            # any pending "unknown" fallback and let them have the full patient threshold.
            endpoint.reset()
            endpoint.checked = True
            continue

        if text is None and elapsed_ms >= MAX_UTTERANCE_S * 1000:
            print("[stt] max utterance length reached, transcribing what I have...")
            text = engine.transcribe_stream(audio_chunks, listener.locale)[0].text

        if text is not None:
            _print_result(text, engine.last_confidence)
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


if __name__ == "__main__":
    _self_test_endpointing()
    _self_test_hotwords()

    engine = WhisperSTTEngine(
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
