# Stage 06 — Speaking (Text-to-Speech)

## Files here

- `tts_speaker.py` — the mock-based version, keep this working as a regression test of the
  sentence-splitting/measurement logic. No hardware needed.
- `tts_speaker_live.py` — the real version: `LiveTTSSpeaker`, real streaming synthesis and
  playback via [Piper](https://github.com/OHF-Voice/piper1-gpl) (local neural TTS) +
  `sounddevice`.
- `evaluate_clarity.py` — objective clarity test: synthesizes known sentences, transcribes them
  back with real Whisper STT, scores the round-trip Word Error Rate.

## Why `LiveTTSSpeaker` isn't a drop-in `TTSEngine`

Every other stage's live file reuses its mock's core class completely unchanged, swapping only
the fake engine for a real one. This stage is the exception, deliberately: `TTSEngine.
synthesize_stream()` (in `tts_speaker.py`) returns a fully materialized list of `AudioChunk`s
before `TTSSpeaker.speak()` plays anything — fine for a mock where everything is instant, but
structurally incompatible with *real* streaming, since "return the whole list" means the last
chunk has to exist before the first one can be played. So this stage gets its own
`LiveTTSSpeaker` with real incremental playback instead of forcing genuine streaming through a
batch-shaped interface. `tts_speaker.py` itself is untouched and still passes its own self-test —
same kind of deliberate interface evolution as Stage 05's new `GENERAL_CHAT` intent, not a
regression.

## Install (one time, venv already active)

```powershell
pip install -r stages\stage06_text_to_speech\requirements.txt
```

## Download a voice (one time)

```powershell
cd stages\stage06_text_to_speech
python -m piper.download_voices en_US-amy-medium
```

This downloads two files into the current folder: `en_US-amy-medium.onnx` (the model) and
`en_US-amy-medium.onnx.json` (its config) — both required, both gitignored (same pattern as
Stage 02's face-recognition models). `en_US-amy-medium` (female, US English) is the current
default — swapped 2026-08-19 from the earlier `en_US-lessac-medium` (male), same "medium"
quality/speed tier so this alone doesn't change CPU latency; see "Trying a different voice" below
to compare others — [listen to samples first](https://rhasspy.github.io/piper-samples) before
downloading a bigger one.

## Run the live demo

```powershell
python stages\stage06_text_to_speech\tts_speaker_live.py
```

Speaks a sample three-sentence answer out loud through your speakers and reports the real
Time-To-First-Audio — STAGES.md's own success criterion (< 1-2 seconds). Playback starts as soon
as the first sentence's audio is ready, not after the whole answer is synthesized.

## Measuring clarity, not just latency

```powershell
python stages\stage06_text_to_speech\evaluate_clarity.py
```

Synthesizes a handful of sentences (including one with floor/room numbers, since Stage 05's real
answers are full of those) and transcribes each one back with real Whisper STT, reporting the
Word Error Rate between what was supposed to be said and what an independent model actually
heard. This is the same "measure, don't eyeball it" discipline as Stage 04's `evaluate_wer.py` —
a low round-trip WER is a real, objective signal that the voice is intelligible, not just a
subjective "sounds fine to me." If the numbers sentence scores worse than the others, that's
specifically a digit-pronunciation problem; if everything scores poorly, it's the voice model or
pipeline itself.

## Trying a different voice

Piper voices trade off quality against synthesis speed — [listen to samples](
https://rhasspy.github.io/piper-samples) before choosing. To switch:

```powershell
python -m piper.download_voices en_US-libritts_r-high   # example: higher quality, slower
```

...then change `VOICE_NAME` at the top of `tts_speaker_live.py`. Higher-quality voices take
longer to synthesize per sentence, which directly trades against the Time-To-First-Audio target
— re-run the live demo and `evaluate_clarity.py` after switching to see both sides of that
tradeoff for real, rather than assuming "higher quality" is strictly better for this use case.

## Known limitations (by design, for now)

- Sentences are synthesized and played strictly one after another — the next sentence only
  starts synthesizing once the current one has finished playing, not pipelined in the
  background. Fine for keeping Time-To-First-Audio low (that only depends on the *first*
  sentence), but there can be a brief gap between sentences on longer answers. A background-
  thread pipeline (synthesize sentence N+1 while sentence N plays) would close that gap — not
  implemented yet, worth revisiting if multi-sentence answers sound choppy in practice.
- CPU-only here (`use_cuda=False` default) — Piper supports GPU acceleration
  (`use_cuda=True`, needs the `onnxruntime-gpu` package) for the eventual Jetson deployment,
  same "CPU now, GPU later" pattern as Stage 01's torch setup.
- `_sanitize_for_speech()` is a belt-and-suspenders defense against stray markdown from an LLM
  answer (Stage 05's own prompts already forbid it) — not a full text-normalization pipeline.
  Numbers, dates, and abbreviations are handled by Piper's embedded espeak-ng phonemizer, not by
  any custom logic here; `evaluate_clarity.py` is exactly how to check whether that's good enough
  for this robot's actual content.
- Not yet linked to Stage 05 (dialogue) — tested here with hardcoded sample text, not real
  generated answers. Stage 08 (Orchestrator) wires the full pipeline together, and should also
  finally replace Stage 03's temporary `pyttsx3`-based `LiveTTSSpeaker` stand-in with this real
  one.
- **Licensing**: Piper (OHF-Voice/piper1-gpl) is GPL-3.0-or-later — stronger copyleft than
  Ultralytics' AGPL-3.0 (Stage 01's own licensing note). Fine while prototyping, but needs legal
  review before shipping this closed-source to the university. Voice models have their own
  separate licenses too (check each voice's `MODEL_CARD` at
  [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)) — most are
  permissive, but verify before shipping.
