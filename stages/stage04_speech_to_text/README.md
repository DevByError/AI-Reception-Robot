# Stage 04 — Listening (Speech-to-Text)

## Files here

- `stt_listener.py` — the mock-based version, keep this working as a regression test of the
  VAD-triggers-transcription logic. No hardware needed.
- `stt_listener_live.py` — the real, assembled single-speaker pipeline (see diagram below). It
  reuses `STTListener`/`VoiceActivityDetector` from `stt_listener.py` completely unchanged —
  only `MockSTTEngine` is swapped for a real `WhisperSTTEngine`, and real audio replaces the
  manually-supplied `trailing_silence_ms`.
- `audio_preprocessing.py` — optional noise suppression + normalization (opt-in).
- `vocabulary_correction.py` — optional domain vocabulary correction for low-confidence words
  only (opt-in, empty vocabulary by default).
- `diarization.py` — speaker diarization ("who said what"). Standalone, **not** wired into the
  default pipeline (see "What's in vs. out" below).
- `evaluate_wer.py` + `test_samples/manifest.json` — the actual Stage 04 test from STAGES.md,
  turned into a repeatable script with a per-category WER breakdown.
- `record_sample.py` — records real test samples through your mic and fills in
  `test_samples/manifest.json` automatically (correct WAV format, auto-stop-on-silence, no
  hand-edited JSON). Run this before `evaluate_wer.py`. Recordings stay local: `test_samples/` is
  gitignored because it holds real voices.

## Pipeline

```
microphone
   │
   v
preprocessing (audio_preprocessing.py - opt-in: denoise + normalize)
   │
   v
VAD (webrtcvad, streaming) ── decides when an utterance starts/ends
   │
   v
ASR (faster-whisper, word-level confidence)
   │
   v
domain vocabulary correction (vocabulary_correction.py - opt-in, low-confidence words only)
   │
   v
confidence gating (ConfidenceReport.is_reliable)
   │
   v
final text
```

Punctuation/capitalization isn't a separate stage here — Whisper produces it natively as part of
transcription, unlike older ASR engines that need a dedicated punctuation model bolted on
afterward.

## What's wired in by default vs. opt-in

| Stage | Status | Why |
|---|---|---|
| Preprocessing (denoise/normalize) | **Off by default** | Aggressive noise removal can destroy speech information as easily as it removes noise — flip `denoise_enabled=True` on `WhisperSTTEngine` and A/B test it with `evaluate_wer.py` on your own recordings before trusting it, don't assume it helps. |
| VAD | **On** | Required to know when someone's done talking. |
| Confidence scoring | **On** | Free (faster-whisper already computes it) — no reason not to use it. |
| Vocabulary correction + initial_prompt | **On, with real terms** | `stt_listener_live.py`'s `real_vocabulary()`/`real_initial_prompt()` (university name + Stage 05's department names, plus the robot's and its creator's names read from MySQL via `identity.py`) are passed into `WhisperSTTEngine()` — still incomplete until Stage 07's scraper exists to source staff names etc. automatically, but no longer a no-op. |
| Speaker diarization | **Not wired in at all** | This robot's design (ARCHITECTURE.md/README.md) treats a group as one shared interaction with one greeting, not per-speaker turns — there's no current use for "who said what." `diarization.py` exists standalone for if/when Stage 08 makes multi-speaker turn-taking a deliberate product decision. |
| Real-time partial transcription | **Off by default** (`PARTIAL_PREVIEW_ENABLED`) | Re-running Whisper on a growing buffer every couple of seconds is real CPU load a kiosk doing turn-based dialogue doesn't need — flip it on if you want to see the latency/accuracy tradeoff yourself. |
| Fine-tuning | **Not implemented** | See "On accuracy generally" below — the wrong tool until pretrained + tuning + vocabulary correction have all been tried and measured. |

## Install (one time, venv already active)

`sounddevice`/`numpy` are from Stage 00, `torch` (reused by `diarization.py`) is from Stage 01.
This stage adds:

```powershell
pip install -r stages\stage04_speech_to_text\requirements.txt
```

## Run the live demo

```powershell
python stages\stage04_speech_to_text\stt_listener_live.py
```

First run downloads the `small.en` Whisper model (~500MB, needs internet once, cached after
that — `base.en` was tried first and was noticeably inaccurate in real testing. `medium.en` was
briefly the default on 2026-08-19 chasing accuracy, then reverted the same day: it's genuinely
more accurate but costs multiple seconds per utterance on Windows CPU, and answering quickly
turned out to matter more. Decoding is also greedy (`beam_size=1`) for the same reason — see
`WhisperSTTEngine`'s docstring for both tradeoffs). Then just talk: it prints `speech detected, listening...` once you start talking, and after you pause
for about 900ms (`PatientVAD`'s trailing-silence threshold) it prints what it heard. A
low-confidence transcript is flagged rather than trusted silently — either individual words show
as `[??]`, or if too much of the utterance is uncertain, the whole line is prefixed with a
confidence warning. Press Ctrl+C to quit — audio only, no preview window.

## Why it doesn't clip the first (or last) word

The mic is captured continuously in 30ms frames even while nothing is being said yet, and the
last `PREROLL_FRAMES` (~600ms) of that "silence" buffer get prepended once speech is confirmed
(`SPEECH_START_FRAMES` consecutive speech-flagged frames) — otherwise the first syllable, spoken
during the brief debounce window itself, would be lost. This is the listening-side mirror of the
clipped-audio fix `LiveTTSSpeaker` needed on the speaking side in Stage 03. On the other end,
`PatientVAD` waits a bit longer (900ms vs. the mock's 700ms default) before deciding you're done
talking, so an ordinary mid-sentence breath doesn't truncate a real sentence early.

Getting this right took two passes: `VAD_AGGRESSIVENESS` was briefly set to 3 (strictest) to stop
false triggers on room noise, but that also flagged quieter opening syllables as "not speech,"
delaying recording start and clipping real words beyond what the pre-roll buffer could recover.
It's back to 1 (permissive) — the noise/hallucination problem is handled downstream instead (see
below), not by starving the model of real audio upfront.

## Avoiding hallucinated garbage output

Whisper has a well-known failure mode on short/near-silent clips: it hallucinates repeated
filler like `"I can't. I can't. I can't."` instead of admitting there's nothing there. Guards
against it: `MIN_UTTERANCE_MS` (400ms) silently discards a clip that turned out too short instead
of ever sending it to Whisper; `condition_on_previous_text=False` + `vad_filter=True` on the
`transcribe()` call stop it from spiraling into a repetition loop on whatever's left; and
confidence gating (below) catches whatever slips through by flagging it instead of trusting it.

## Confidence scoring

faster-whisper exposes per-word confidence (`word.probability`, with `word_timestamps=True`) and
a per-segment "this might not even be speech" signal (`no_speech_prob`) for free — no extra
model, no extra dependency. `WhisperSTTEngine` uses both: any word below
`WORD_CONFIDENCE_THRESHOLD` (0.5) is first tried against `vocabulary_correction` (see below), and
if that doesn't resolve it, gets flagged inline with `[??]` rather than silently trusted — e.g.
`"The patient has [??] pain for three weeks."` `ConfidenceReport.is_reliable` additionally flags
the *whole* utterance as unreliable if too many words ended up uncertain, or if Whisper's own
no-speech signal is high, so the caller can show/ask for confirmation instead of confidently
acting on a bad guess. This is also exactly the mechanism Stage 09's planned fallback phrase
("Sorry, could you repeat that?") should hook into later.

## Domain vocabulary correction

`vocabulary_correction.py` fuzzy-matches only low-confidence words against a supplied vocabulary
list and swaps in a correction only if it's a close match — confident words are never touched,
since a wrong "correction" there would make things worse. No longer a no-op: `stt_listener_live.py`
now defines `real_vocabulary()`/`real_initial_prompt()` — real terms (the university's actual
name, "Lincoln University College", the department names from Stage 05's `SAMPLE_CHUNKS`, and the
robot's/creator's names, which come from the `robot_identity` table so a rename biases Whisper
toward the new name instead of the old one) — and
passes both into `WhisperSTTEngine()` in its own `__main__`, and `evaluate_wer.py` uses the same
config so measured numbers reflect what's actually running. Still incomplete (no staff names, for
example) until Stage 07's scraper exists to source the rest automatically.

`initial_prompt` mainly helps with rare proper nouns Whisper wouldn't otherwise transcribe
confidently — "Lincoln" going from flagged-uncertain to confident once it was wired in, measured
directly via `evaluate_wer.py`. (An earlier version of this section covered a "Collage" vs.
"College" spelling correction based on a name confirmation that turned out to be wrong once the
real university website was actually checked — Whisper had been transcribing it correctly the
whole time. Worth remembering: a human confirming something twice still isn't as reliable as
checking the real source.)

## Speaker diarization

`diarization.py` groups a recording into per-speaker segments using resemblyzer (speaker
embeddings) + clustering — no Hugging Face account or gated model needed, unlike the more
accurate `pyannote.audio`. It's a standalone utility, not called from `stt_listener_live.py`'s
default loop (see "What's in vs. out" above for why). Try it directly against a two-person WAV —
see the module's own `__main__` output for a snippet.

## Accents

Not a separate engineering stage — it comes from model choice, not custom code. Whisper's
`small.en`/`medium.en` models are already trained on a large, accent-diverse set of real-world
English audio; the lever here was already pulled (upgrading from `base.en`). If accented speech
is still a weak spot once `evaluate_wer.py` has real accent-tagged samples to measure against,
the next lever is `medium.en`, not a bespoke accent-detection pipeline.

## Evaluating accuracy properly

`evaluate_wer.py` is the actual test STAGES.md defines for this stage, made repeatable: record a
handful of samples (clean audio, noisy audio, different accents, quick 1-2 word answers), write
the expected text into `test_samples/manifest.json` tagged by category, and run it. It reports
WER per category as well as overall — an overall average hides exactly the cases that matter
most (STAGES.md's own point: 5% overall can still mean 20%+ on noisy or accented speech). This is
also what makes every other knob in this stage (denoise on/off, model size, VAD thresholds)
something you can actually measure instead of eyeballing single print statements.

```powershell
python stages\stage04_speech_to_text\record_sample.py
python stages\stage04_speech_to_text\evaluate_wer.py
```

`record_sample.py` does the recording and manifest bookkeeping for you (correct WAV format,
auto-stops when you pause, same VAD as `stt_listener_live.py`) — no real ASR evaluation exists
until `manifest.json` has real samples in it, so this is the step to do first if the numbers so
far have only been "sounds a lot better" rather than an actual measured WER.

## On accuracy generally: fine-tuning is not the first lever

If accuracy is still insufficient after the above, the right order of operations is:

```
pretrained ASR → evaluate (evaluate_wer.py) → preprocessing (denoise on/off) →
vocabulary/context (initial_prompt, vocabulary_correction) → evaluate again
```

Only if it's *still* insufficient after that does fine-tuning become worth considering — and
even then: collect real domain-specific audio, get human transcripts, clean/normalize the
dataset, fine-tune, and evaluate on **unseen** speakers (never on the recordings used to train,
or the number is meaningless). This isn't implemented here — there's no training data or GPU
training infrastructure yet, and Whisper was already trained on 680k+ hours of real-world speech,
which a small homemade dataset won't out-train for *general* accuracy anyway. Fine-tuning's real
value is narrow: teaching the model specific proper nouns/jargon it keeps getting wrong (a
professor's name, a building name) — which `vocabulary_correction.py`'s `initial_prompt`/fuzzy-
match approach already covers without any training, once Stage 07 provides real terms to feed
it.

## Known limitations (by design, for now)

- Not truly streaming/incremental in normal operation — `WhisperSTTEngine.transcribe_stream()`
  waits for the whole utterance (until trailing silence) and transcribes it in one shot.
  `PARTIAL_PREVIEW_ENABLED` layers a periodic re-transcription on top for a preview during long
  utterances, but it's off by default (see "What's in vs. out" above).
- English only (`small.en`) — matches the project's English-first scope
  (STAGES.md/ARCHITECTURE.md §6). Swapping in a multilingual model later is a `WhisperSTTEngine`
  change, not an `STTListener` interface change.
- `MAX_UTTERANCE_S` (15s) hard-cuts a stuck-open recording (e.g. constant background noise
  misread as speech) — tune against real testing.
- Not yet linked to Stage 03's `LISTENING` state — this stage is tested standalone with an
  always-on mic loop. Stage 08 (Orchestrator) is what will start/stop listening in sync with the
  greeting state machine instead of listening continuously.
- `diarization.py`'s speaker labels (`speaker_0`, `speaker_1`, ...) are per-recording cluster
  IDs, not linked to Stage 02's actual face-recognized identities — that linkage, if ever wanted,
  is a Stage 08 integration question.
