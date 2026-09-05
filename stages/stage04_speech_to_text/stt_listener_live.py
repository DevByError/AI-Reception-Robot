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
import sys
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
        """
        from faster_whisper import WhisperModel

        print(
            f"Loading faster-whisper '{model_size}' (first run downloads it, cached after "
            f"that)..."
        )
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.initial_prompt = initial_prompt
        self.vocabulary = vocabulary or []
        self.denoise_enabled = denoise_enabled
        self.last_confidence: ConfidenceReport | None = (
            None  # set after every transcribe_stream
        )

    def transcribe_stream(
        self, audio_chunks: list[bytes], locale: str
    ) -> list[TranscriptChunk]:
        audio = _pcm_bytes_to_float32(b"".join(audio_chunks))

        # Peak normalization always runs now (2026-08-19) - previously bundled behind
        # denoise_enabled, which meant a quiet/far mic never got leveled at all unless the
        # riskier spectral denoise was also switched on. Pure gain adjustment, no information
        # removed, so there's no accuracy downside to always applying it (unlike denoise() below).
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
            beam_size=1,
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
                    recording_started_at = time.monotonic()
                    next_partial_at = recording_started_at + max(
                        PARTIAL_MIN_DURATION_S, PARTIAL_INTERVAL_S
                    )
                    print("[stt] speech detected, listening...")
            else:
                speech_streak = 0
            continue

        audio_chunks.append(frame)
        silence_ms = 0 if is_speech else silence_ms + FRAME_MS
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

        speech_ended = listener.vad.is_speech_ended(silence_ms)
        if speech_ended and elapsed_ms < MIN_UTTERANCE_MS:
            # too short to be real words (a cough, a click, a stray VAD false-positive) - drop
            # it without ever calling Whisper, instead of hallucinating on it, and go back to
            # waiting (with a fresh timeout clock) rather than giving up on the whole call.
            print("[stt] (too short - ignoring, probably not real speech)")
            recording = False
            audio_chunks = []
            preroll = []
            silence_ms = 0
            next_partial_at = None
            wait_started_at = time.monotonic()
            continue

        text = listener.listen(audio_chunks, trailing_silence_ms=silence_ms)
        if text is None and elapsed_ms >= MAX_UTTERANCE_S * 1000:
            print("[stt] max utterance length reached, transcribing what I have...")
            text = engine.transcribe_stream(audio_chunks, listener.locale)[0].text

        if text is not None:
            _print_result(text, engine.last_confidence)
            return text


if __name__ == "__main__":
    engine = WhisperSTTEngine(
        initial_prompt=real_initial_prompt(), vocabulary=real_vocabulary()
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
