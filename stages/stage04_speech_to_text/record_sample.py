"""
Stage 04 test-sample recorder - fills in test_samples/manifest.json for evaluate_wer.py, which
has been sitting empty (see this project's own "measure, don't eyeball it" philosophy - that only
works once there's something real to measure against). Records through your real microphone using
the same auto-stop-on-silence VAD logic as stt_listener_live.py's own main loop (PatientVAD /
StreamingVAD, PREROLL_FRAMES, SPEECH_START_FRAMES, MIN_UTTERANCE_MS) - so samples aren't clipped,
padded with dead air, or accidentally captured from a stray noise - then saves a compliant 16kHz
mono 16-bit WAV and appends the entry to manifest.json automatically. No hand-written "python -c"
one-liners, no manually edited JSON.

Run interactively (keeps prompting until you type "done" or Ctrl+C):
  python record_sample.py

Or one-shot, non-interactively:
  python record_sample.py "where is the computer science department" --category clean

Categories (match evaluate_wer.py's own per-category breakdown):
  clean   - quiet room, normal speaking voice - the baseline.
  noise   - the SAME kind of phrases, but with real background noise (TV, fan, people talking) -
            this is what actually tells you whether the denoise step is pulling its weight.
  accent  - a different speaker/accent, if you have access to one - accent robustness can't be
            measured from a single speaker talking to themselves.
  short   - one or two words, quick utterances (tests MIN_UTTERANCE_MS / short-clip handling).

A representative first batch, using real content from this project (Stage 05's building
directory) rather than generic filler - feel free to say your own instead:
  clean/noise:  "where is the computer science department"
                "where is the engineering department"
                "where is the management department"
                "where is the nursing department"
                "where is the library"
                "when does freshman orientation start"
  short:        "restrooms"   "library"   "admissions"

Files save as test_samples/<category>_<NN>.wav, auto-numbered per category so re-running this
later adds more samples instead of overwriting earlier ones.

Install first: same requirements.txt as the rest of this stage (sounddevice + webrtcvad-wheels
are both already in it).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from stt_listener_live import (
    FRAME_MS,
    FRAME_SAMPLES,
    MAX_UTTERANCE_S,
    MIN_UTTERANCE_MS,
    PREROLL_FRAMES,
    SAMPLE_RATE,
    SPEECH_START_FRAMES,
    PatientVAD,
    StreamingVAD,
)

SAMPLES_DIR = Path(__file__).resolve().parent / "test_samples"
MANIFEST_PATH = SAMPLES_DIR / "manifest.json"

CATEGORIES = ("clean", "noise", "accent", "short")

# How long to wait for you to start speaking before giving up on one attempt - not a project
# constant, just a sanity cap so a misfiring mic doesn't hang the script forever.
WAIT_FOR_SPEECH_S = 10

# Opening a fresh audio stream can produce a brief click/pop/DC-offset transient in the very
# first frames - webrtcvad at aggressiveness=1 (lenient by design, see stt_listener_live.py's own
# comment on this) can easily misread that as "speech" and trigger recording before you've said
# anything. record_one() opens a brand new stream for every single sample, so without a warm-up
# this fired reliably on every attempt (first real bug report: it saved before any speaking
# happened, twice, on two separate runs). Same category of fix as Stage 03's TTS "primer"
# utterance, just on the input side instead of output - discard a short settle period before
# evaluating any frame for speech.
WARMUP_FRAMES = 15  # ~450ms

# A false trigger (the bug above, or a single loud click/knock) can satisfy SPEECH_START_FRAMES
# without ever being followed by real sustained speech. Require at least this many total
# speech-classified frames across the whole recording - not just the initial debounce - before
# trusting it enough to save. Catches false triggers that WARMUP_FRAMES alone might miss.
MIN_SPEECH_FRAMES = 8  # ~240ms of frames actually flagged as speech, cumulative

# Peak sample amplitude (int16, 0-32767) below which the mic is almost certainly too quiet for
# VAD to reliably tell speech from silence at all - flagged in the level meter / failure messages
# below rather than guessed at blind. -3dBFS full scale is 32767; this is roughly -26dBFS.
QUIET_PEAK_THRESHOLD = 1500


def _next_filename(category: str) -> str:
    SAMPLES_DIR.mkdir(exist_ok=True)
    n = 1
    while (SAMPLES_DIR / f"{category}_{n:02d}.wav").exists():
        n += 1
    return f"{category}_{n:02d}.wav"


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _peak(frame_bytes: bytes) -> int:
    """Peak absolute sample amplitude in a frame (0-32767) - a cheap, real number to look at
    instead of guessing whether the mic is picking anything up, same "measure, don't eyeball it"
    reasoning as evaluate_wer.py itself."""
    samples = np.frombuffer(frame_bytes, dtype=np.int16)
    return int(np.abs(samples).max()) if len(samples) else 0


def _level_bar(peak: int, width: int = 20) -> str:
    filled = min(width, int(peak / 32767 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_input_device() -> None:
    try:
        info = sd.query_devices(kind="input")
        print(
            f"Input device: {info['name']!r} (default) - if this isn't your mic, run "
            f"stage00_setup/list_audio_devices.py to find the right one and set "
            f"sd.default.device before recording."
        )
    except Exception as exc:  # pragma: no cover - purely informational
        print(f"(couldn't query the default input device: {exc})")


def record_one() -> bytes | None:
    """Records one utterance with the same VAD auto-stop logic as stt_listener_live.py's main
    loop - waits for real speech to start (debounced via SPEECH_START_FRAMES), prepends
    PREROLL_FRAMES so the first syllable isn't clipped, and stops after the same trailing-silence
    threshold PatientVAD uses elsewhere in this stage. Discards a short WARMUP_FRAMES settle
    period right after opening the stream (see that constant's comment) and double-checks
    MIN_SPEECH_FRAMES before accepting the result, so a stream-open transient can't masquerade as
    a real recording. Returns raw PCM16 bytes, or None if nothing usable was captured (silence-
    only, too short, or a false trigger)."""
    vad = StreamingVAD()
    patient = PatientVAD()
    audio_q: queue.Queue[bytes] = queue.Queue()

    def _on_audio(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    preroll: list[bytes] = []
    recording = False
    speech_streak = 0
    speech_frame_count = 0
    audio_chunks: list[bytes] = []
    silence_ms = 0
    recording_started_at = 0.0
    waiting_peak_max = 0
    recording_peak_max = 0
    frame_i = 0

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        callback=_on_audio,
    )
    with stream:
        # Drain and discard the warm-up period before evaluating anything for speech - see
        # WARMUP_FRAMES' comment above for why this exists.
        for _ in range(WARMUP_FRAMES):
            audio_q.get()

        print("  (speak now - recording starts automatically, stops after you pause)")
        wait_started_at = time.monotonic()

        while True:
            frame = audio_q.get()
            is_speech = vad.is_speech(frame)
            peak = _peak(frame)

            if not recording:
                waiting_peak_max = max(waiting_peak_max, peak)
                # Live level meter, updated in place (\r) every few frames - so you can watch it
                # move while you talk instead of guessing whether the mic is even picking you up.
                frame_i += 1
                if frame_i % 3 == 0:
                    sys.stdout.write(f"\r  listening... level {_level_bar(peak)} {peak:>5}  ")
                    sys.stdout.flush()

                preroll.append(frame)
                if len(preroll) > PREROLL_FRAMES:
                    preroll.pop(0)

                if is_speech:
                    speech_streak += 1
                    if speech_streak >= SPEECH_START_FRAMES:
                        recording = True
                        audio_chunks = list(preroll)
                        speech_frame_count = speech_streak
                        recording_peak_max = waiting_peak_max
                        silence_ms = 0
                        speech_streak = 0
                        recording_started_at = time.monotonic()
                        sys.stdout.write("\n")
                        print("  [recording...]")
                        continue
                else:
                    speech_streak = 0

                if time.monotonic() - wait_started_at > WAIT_FOR_SPEECH_S:
                    sys.stdout.write("\n")
                    print(f"  (no speech detected in {WAIT_FOR_SPEECH_S}s - skipping)")
                    _report_quiet_if_needed(waiting_peak_max)
                    return None
                continue

            audio_chunks.append(frame)
            recording_peak_max = max(recording_peak_max, peak)
            if is_speech:
                speech_frame_count += 1
            silence_ms = 0 if is_speech else silence_ms + FRAME_MS
            elapsed_ms = (time.monotonic() - recording_started_at) * 1000

            if patient.is_speech_ended(silence_ms) or elapsed_ms >= MAX_UTTERANCE_S * 1000:
                duration_s = len(audio_chunks) * FRAME_MS / 1000
                if elapsed_ms < MIN_UTTERANCE_MS:
                    print(f"  (too short, {duration_s:.1f}s - try again, speak a bit longer)")
                    _report_quiet_if_needed(recording_peak_max)
                    return None
                if speech_frame_count < MIN_SPEECH_FRAMES:
                    # Recording ran long enough, but hardly any of it was actually classified as
                    # speech - almost certainly a false trigger (mic click/pop), not real words.
                    print(
                        f"  (probably a false trigger, not real speech, {duration_s:.1f}s "
                        f"captured but barely any of it was speech - try again)"
                    )
                    _report_quiet_if_needed(recording_peak_max)
                    return None
                print(f"  [done, {duration_s:.1f}s, peak level {recording_peak_max}]")
                return b"".join(audio_chunks)


def _report_quiet_if_needed(peak_max: int) -> None:
    if peak_max < QUIET_PEAK_THRESHOLD:
        print(
            f"  peak level observed: {peak_max}/32767 - that's quite quiet. If you were "
            f"speaking normally, check the input device (see stage00_setup/list_audio_devices.py) "
            f"and your mic's input volume/gain in Windows sound settings."
        )


def save_wav(raw_pcm: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_pcm)


def add_sample(expected: str, category: str) -> bool:
    if category not in CATEGORIES:
        print(f"Unknown category {category!r} - expected one of {CATEGORIES}")
        return False
    raw = record_one()
    if raw is None:
        return False
    filename = _next_filename(category)
    save_wav(raw, SAMPLES_DIR / filename)
    manifest = _load_manifest()
    manifest[filename] = {"expected": expected, "category": category}
    _save_manifest(manifest)
    print(f"  saved {filename} -> manifest.json ({len(manifest)} sample(s) total)\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("text", nargs="?", help="expected text (one-shot mode)")
    parser.add_argument("--category", choices=CATEGORIES, help="sample category (one-shot mode)")
    args = parser.parse_args()

    _print_input_device()

    if args.text:
        if not args.category:
            print("One-shot mode needs --category too, e.g. --category clean")
            return
        add_sample(args.text, args.category)
        return

    print(f"Interactive mode - samples save to {SAMPLES_DIR}\\")
    print("Type the expected text exactly as you're about to say it, then Enter.")
    print("Type 'done' (or Ctrl+C) when you have enough samples.\n")
    try:
        while True:
            expected = input("Expected text (or 'done'): ").strip()
            if not expected or expected.lower() == "done":
                break
            category = input(f"Category {CATEGORIES} [clean]: ").strip().lower() or "clean"
            add_sample(expected, category)
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")

    manifest = _load_manifest()
    print(f"\n{len(manifest)} sample(s) now in manifest.json. Run: python evaluate_wer.py")


if __name__ == "__main__":
    main()
