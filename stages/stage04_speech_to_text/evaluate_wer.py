"""
Stage 04 test harness - this is the actual test STAGES.md defines for this stage ("run a sample
audio file -> compare the resulting text against the expected text (acceptable WER)"), turned
into a repeatable script instead of a one-off manual check, with a per-category breakdown -
a single average WER hides exactly the cases you most need to see (noisy clips, accents,
short/quick utterances), so this reports each category separately as well as the overall number.

Setup: this needs real recordings of a real voice on real hardware to mean anything - there's no
way to fake this input. Easiest path: run

  python record_sample.py

which records through your mic (auto-stops when you pause, same VAD as stt_listener_live.py),
saves a compliant 16kHz mono 16-bit WAV, and writes the manifest.json entry for you - see that
file's own docstring for suggested phrases across the clean/noise/accent/short categories. Ideally
include other speakers/accents too - accent robustness can't be evaluated from one speaker alone.

If you'd rather do it by hand instead, manifest.json just needs entries like:

{
  "clean_01.wav":  {"expected": "where is the computer science department", "category": "clean"},
  "noise_01.wav":  {"expected": "how do I get to the library",             "category": "noise"},
  "accent_01.wav": {"expected": "what time does the cafeteria open",       "category": "accent"},
  "quick_01.wav":  {"expected": "restrooms",                               "category": "short"}
}

and each WAV needs to be 16kHz mono 16-bit - record_sample.py handles that formatting
automatically; doing it manually means matching it yourself (e.g. via Audacity's export settings
or `sd.rec()` + `wave` directly).

Run: python evaluate_wer.py

Note: `expected` text is compared case/punctuation-insensitively (see the `normalize` transform
in main()) - jiwer does no normalization by default, and without it, "computer" vs "Computer" or
a missing comma would count as a full word error despite being correct. Number FORMAT still
matters though (Whisper writing "4th" as "fourth" isn't auto-normalized) - write `expected` the
way the words are actually meant to sound to keep comparisons fair.
"""

from __future__ import annotations

import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from stt_listener_live import (
    WhisperSTTEngine, real_hotwords, real_initial_prompt, real_vocabulary,
)

SAMPLES_DIR = Path(__file__).resolve().parent / "test_samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"


def _load_wav_as_pcm16_bytes(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"{path.name}: expected 16kHz mono 16-bit WAV, got "
                f"{wf.getframerate()}Hz / {wf.getnchannels()}ch / {wf.getsampwidth() * 8}bit - "
                f"re-export it in that format first."
            )
        return wf.readframes(wf.getnframes())


def main() -> None:
    try:
        import jiwer
    except ImportError:
        print("jiwer isn't installed - run: pip install -r requirements.txt")
        return

    # jiwer's wer() does NO normalization by default - "Hi there!" vs "Hi there," or
    # "Computer" vs "computer" would score as full word errors even though the actual words
    # match. Case and punctuation don't affect intelligibility/correctness here, so strip them
    # before comparing - otherwise this measures transcription formatting, not accuracy. (Digit
    # vs. word-form numbers, e.g. Whisper writing "4th" as "fourth", aren't covered by this - see
    # this file's docstring: write `expected` the way it's actually meant to sound.)
    normalize = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH} - see this file's docstring to set one up.")
        return

    manifest = json.loads(MANIFEST_PATH.read_text())
    if not manifest:
        print("Manifest is empty - add some test samples first (see this file's docstring).")
        return

    # Same real config as stt_listener_live.py's __main__ - measuring against the bare defaults
    # would score a setup nobody actually runs; this is what tells you whether
    # the initial prompt/vocabulary are actually helping (re-run with WhisperSTTEngine()
    # bare, no args, to A/B against the unbiased baseline if you want to confirm that directly).
    engine = WhisperSTTEngine(
        initial_prompt=real_initial_prompt(), vocabulary=real_vocabulary(),
        hotwords=real_hotwords(),
    )

    rows: list[tuple[str, str, str, str, float]] = []
    by_category: dict[str, list[float]] = defaultdict(list)

    for filename, meta in manifest.items():
        wav_path = SAMPLES_DIR / filename
        if not wav_path.exists():
            print(f"[skip] {filename} listed in manifest but file not found")
            continue

        raw_pcm = _load_wav_as_pcm16_bytes(wav_path)
        result = engine.transcribe_stream([raw_pcm], locale="en-US")
        hypothesis = result[0].text
        expected = meta["expected"]
        category = meta.get("category", "uncategorized")

        sample_wer = (
            jiwer.wer(expected, hypothesis, reference_transform=normalize,
                      hypothesis_transform=normalize)
            if expected.strip() else float("nan")
        )
        rows.append((filename, category, expected, hypothesis, sample_wer))
        by_category[category].append(sample_wer)

    if not rows:
        print("No valid samples found - nothing to evaluate.")
        return

    print(f"\n{'file':<20} {'category':<12} {'WER':>6}  expected / got")
    print("-" * 90)
    for filename, category, expected, hypothesis, sample_wer in rows:
        print(f"{filename:<20} {category:<12} {sample_wer:>5.1%}  "
              f"{expected!r} / {hypothesis!r}")

    print(f"\n{'category':<12} {'avg WER':>8}  {'n':>4}")
    print("-" * 30)
    for category, values in sorted(by_category.items()):
        avg = sum(values) / len(values)
        print(f"{category:<12} {avg:>7.1%}  {len(values):>4}")

    overall = [w for values in by_category.values() for w in values]
    print("-" * 30)
    print(f"{'overall':<12} {sum(overall) / len(overall):>7.1%}  {len(overall):>4}")
    print(
        "\nA single overall number hides where it's actually failing - watch the per-category "
        "breakdown, not just the bottom line (see this file's docstring)."
    )


if __name__ == "__main__":
    main()
