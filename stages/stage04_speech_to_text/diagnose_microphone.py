"""
Stage 04 - is the microphone actually working?

"The robot isn't getting my speech" has a handful of causes that are indistinguishable from the
outside, and the running robot cannot tell you which: it says "sorry, I didn't catch that" for
all of them. This walks the audio path in order and reports where it breaks.

  1. Which device is Windows handing us?   - not always the one you think; connecting Bluetooth
                                             earbuds silently switches it to an 8kHz profile
  2. Is there any SIGNAL on it?            - a muted microphone reads as a perfectly valid
                                             stream of near-zero samples, with no error anywhere
  3. Is the level sane?                    - too quiet and Whisper has nothing to work with,
                                             too hot and it clips, which transcribes just as badly
  4. Does the VAD find speech in it?       - what decides when a recording starts and stops
  5. Does Whisper produce words?           - the only step that is actually transcription

Step 2 is the one worth knowing about. A muted device is the single most common cause and the
hardest to spot, because nothing fails: PortAudio opens the stream, delivers frames on schedule,
and every sample is silence. That was the real cause of a run where every question came back
"heard something, but no words came out" - see the verdict text below for how to fix it.

Run:  python diagnose_microphone.py
      python diagnose_microphone.py --no-whisper    (skip the model, just check the audio)
      python diagnose_microphone.py --seconds 6
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
import webrtcvad
from stt_listener_live import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, VAD_AGGRESSIVENESS

# Peak amplitude (of int16's 32768) below which a device is treated as producing nothing at all.
# A muted microphone still delivers frames; they just sit at the dither floor, a handful of
# counts at most. Real room tone in a quiet room is comfortably above this.
SILENT_PEAK = 50
# Above this, the input is saturating and words are being squared off at the top.
CLIPPING_PEAK = 32700
# Rough healthy window for the loudest moment of ordinary speech.
QUIET_PEAK = 800
HOT_PEAK = 26000


def _record(seconds: float, device=None) -> bytes:
    """Captures raw 16-bit mono PCM at SAMPLE_RATE, the same way the robot does."""
    q: queue.Queue[bytes] = queue.Queue()

    def _cb(indata, frames, time_info, status):
        if status:
            print(f"  [audio] {status}", file=sys.stderr)
        q.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES * 3,
        dtype="int16",
        channels=1,
        device=device,
        callback=_cb,
    )
    with stream:
        time.sleep(seconds)
    out = b""
    while not q.empty():
        out += q.get()
    return out


def _frames(raw: bytes) -> list[bytes]:
    step = FRAME_SAMPLES * 2  # int16
    return [raw[i : i + step] for i in range(0, len(raw) - step + 1, step)]


def _stats(raw: bytes) -> tuple[float, float]:
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if a.size == 0:
        return 0.0, 0.0
    return float(np.sqrt((a**2).mean())), float(np.abs(a).max())


def list_devices() -> None:
    print("=" * 78)
    print("1. Input devices")
    print("=" * 78)
    apis = {i: a["name"] for i, a in enumerate(sd.query_hostapis())}
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        mark = "  <== DEFAULT (what the robot will use)" if i == default_in else ""
        # The name can contain characters the Windows console cannot encode; it is not worth
        # crashing a diagnostic over, so replace anything unprintable.
        name = d["name"].encode("ascii", "replace").decode("ascii").replace("\n", " ")
        print(
            f"  {i:3d} [{apis[d['hostapi']]:<20}] {int(d['default_samplerate']):>6}Hz  "
            f"{name[:44]}{mark}"
        )
        if i == default_in and d["default_samplerate"] < SAMPLE_RATE:
            print(
                f"        ^ BELOW {SAMPLE_RATE}Hz - almost certainly a Bluetooth headset's "
                f"hands-free profile.\n          Whisper transcribes that badly. Pick another "
                f"input in Windows sound settings."
            )


def check_signal(seconds: float) -> tuple[bytes, float, float]:
    print()
    print("=" * 78)
    print(
        f"2-3. Signal and level  ({seconds:.0f}s - say something, normally, from where you "
        f"stand)"
    )
    print("=" * 78)
    raw = _record(seconds)
    rms, peak = _stats(raw)

    # A crude level meter per half second, so a dead channel is obvious at a glance and so is
    # "it only picked up the second half of what I said".
    step = int(SAMPLE_RATE * 0.5) * 2
    for i in range(0, max(len(raw) - step + 1, 0), step):
        _, chunk_peak = _stats(raw[i : i + step])
        bars = int(min(chunk_peak / 32768 * 50, 50))
        print(f"  {i / 2 / SAMPLE_RATE:4.1f}s |{'#' * bars:<50}| {chunk_peak:6.0f}")

    print(f"\n  overall rms={rms:.1f}  peak={peak:.0f}  (int16 full scale is 32768)")
    return raw, rms, peak


def check_vad(raw: bytes) -> int:
    print()
    print("=" * 78)
    print("4. Voice activity detection")
    print("=" * 78)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frames = _frames(raw)
    flags = [vad.is_speech(f, SAMPLE_RATE) for f in frames]
    speech = sum(flags)
    print(
        f"  {speech}/{len(frames)} frames of {FRAME_MS}ms flagged as speech "
        f"(aggressiveness {VAD_AGGRESSIVENESS})"
    )
    if frames:
        per_frame = np.array([_stats(f)[0] for f in frames])
        print(
            f"  per-frame rms: p10={np.percentile(per_frame, 10):.0f} "
            f"median={np.median(per_frame):.0f} p90={np.percentile(per_frame, 90):.0f}"
        )
        if speech == len(frames):
            print(
                "  ^ EVERY frame reads as speech. Either the room is genuinely noisy or the\n"
                "    input level is too high - the robot will start recording immediately and\n"
                "    struggle to tell when you have finished talking."
            )
        elif speech == 0:
            print("  ^ NO frame reads as speech. Nothing will ever start a recording.")
    return speech


def check_whisper(raw: bytes) -> str | None:
    print()
    print("=" * 78)
    print("5. Transcription")
    print("=" * 78)
    from stt_listener_live import (
        WhisperSTTEngine,
        real_hotwords,
        real_initial_prompt,
        real_vocabulary,
    )

    engine = WhisperSTTEngine(
        initial_prompt=real_initial_prompt(),
        vocabulary=real_vocabulary(),
        hotwords=real_hotwords(),
    )
    chunks = engine.transcribe_stream([raw], "en-US")
    text = (chunks[0].text if chunks else "") or ""
    report = engine.last_confidence
    print(f'  transcript: "{text}"')
    if report is not None:
        print(
            f"  confidence: {report.low_confidence_word_count}/{report.total_word_count} "
            f"words uncertain, reliable={report.is_reliable}"
        )
    return text


def verdict(rms: float, peak: float, speech_frames: int, text: str | None) -> None:
    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)

    if peak < SILENT_PEAK:
        print(
            "  THE MICROPHONE IS DELIVERING SILENCE. Nothing downstream can work, and nothing\n"
            "  reports an error - the stream opens fine and every sample is zero.\n"
        )
        print(
            "  Almost always one of these, in this order of likelihood:\n"
            "    1. The microphone is MUTED in Windows. Right-click the speaker icon ->\n"
            "       Sound settings -> Input -> your microphone -> make sure it is not muted\n"
            "       and the volume is around 50-75%.\n"
            "    2. A hardware mic-mute key is on. Many laptops have one on F4/F8, sometimes\n"
            "       with its own LED.\n"
            "    3. Microphone access is blocked: Settings -> Privacy & security ->\n"
            "       Microphone -> 'Let desktop apps access your microphone'.\n"
            "    4. Another application has the device open exclusively."
        )
        return

    if peak >= CLIPPING_PEAK:
        print(
            f"  THE INPUT IS CLIPPING (peak {peak:.0f}, full scale 32768). Loud parts of your\n"
            f"  speech are being squared off, which transcribes about as badly as being too\n"
            f"  quiet. Turn the input volume DOWN in Windows sound settings - try 50%."
        )
    elif peak < QUIET_PEAK:
        print(
            f"  The signal is very quiet (peak {peak:.0f}). Whisper has little to work with.\n"
            f"  Raise the input volume in Windows sound settings, or move closer."
        )
    elif peak > HOT_PEAK:
        print(
            f"  The signal is hot (peak {peak:.0f}) and close to clipping. Consider turning\n"
            f"  the input volume down a little."
        )
    else:
        print(f"  Level looks healthy (rms {rms:.0f}, peak {peak:.0f}).")

    if speech_frames == 0:
        print("  But the VAD found no speech, so a recording would never even start.")
        return

    if text is None:
        print(
            "  Transcription was skipped (--no-whisper), so steps 1-4 are all that was " "checked."
        )
    elif not text.strip():
        print(
            "  ...but Whisper still produced NOTHING from it. If the level above looks fine\n"
            "  and you definitely spoke, the likely cause is dropped audio: watch for\n"
            "  '[audio] input overflow' in the robot's own log, which means the machine was\n"
            "  too busy to keep up with the microphone and threw frames away."
        )
    else:
        print(f'  ...and Whisper heard: "{text.strip()}"')
        print("  The audio path is working end to end.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="how long to record (default 5)")
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="skip loading the model; only check the audio path",
    )
    args = parser.parse_args()

    list_devices()
    raw, rms, peak = check_signal(args.seconds)
    if not raw:
        print("\nThe stream returned no audio at all - the device could not be read.")
        return
    speech = check_vad(raw)

    text = None
    # No point spending 20 seconds loading Whisper to transcribe silence.
    if not args.no_whisper and peak >= SILENT_PEAK:
        text = check_whisper(raw)

    verdict(rms, peak, speech, text)


if __name__ == "__main__":
    sys.exit(main())
