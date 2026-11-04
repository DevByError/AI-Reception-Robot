"""
Stage 00 - Basic hardware check (camera, microphone, speaker).

Goal: one script to run as soon as the real hardware is connected, printing pass/fail for
each device instead of discovering the problem after all the stages on top are built.

Run: python test_hardware.py
"""

import sys


def check_camera(index: int = 0) -> bool:
    """Tries to open the camera and read one frame."""
    try:
        import cv2
    except ImportError:
        print("FAIL Camera: opencv-python not installed (pip install opencv-python)")
        return False

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"FAIL Camera: could not open camera index={index}")
        return False

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("FAIL Camera: opened but could not read a frame")
        return False

    print(f"PASS Camera: OK - frame shape {frame.shape}")
    return True


# Input-level thresholds, in dBFS (0 = the loudest a sample can be).
#
# WHY THIS IS CHECKED AT ALL: a badly-levelled microphone degrades everything downstream
# invisibly. Whisper gets quieter, less distinct audio, more words fall under the confidence
# threshold, more of them come out as "[??]", and the robot starts asking people to repeat
# themselves - which reads as "the robot can't understand me" rather than "the gain is too low".
# It is also the single easiest thing to fix once you know, and impossible to guess at otherwise.
#
# The previous version of this check recorded a second of audio and confirmed the array was not
# empty. Its docstring claimed it confirmed "not just zeros", but it never looked at a single
# sample - a muted microphone passed it.
MIC_SILENT_DBFS = -60.0  # below this there is no signal at all: muted, or the wrong device
MIC_QUIET_DBFS = -40.0  # audible but weak - Whisper will struggle
MIC_CLIP_FRACTION = 0.001  # more than 0.1% of samples at full scale = distortion
# Gap between a visitor's voice and the room underneath it. Below roughly this, the words and
# the background are competing, and that is a microphone/placement problem rather than anything
# tuning can rescue.
MIC_MIN_SNR_DB = 12.0


def check_microphone(seconds: float = 3.0, interactive: bool = True) -> bool:
    """Records a few seconds and reports the actual INPUT LEVEL, not just that bytes arrived.

    Speak normally, from where a visitor would stand, while this runs - the numbers only mean
    something if there is real speech in them. Recorded in silence it still gives you the room's
    noise floor, which is worth knowing on its own.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("FAIL Mic: sounddevice/numpy not installed")
        return False

    if interactive:
        print(f"     Mic: say something normally for the next {seconds:.0f} seconds...")

    try:
        recording = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
    except Exception as e:  # noqa: BLE001 - diagnostic script, print any hardware error
        print(f"FAIL Mic: recording failed - {e}")
        return False

    if recording is None or len(recording) == 0:
        print("FAIL Mic: no data returned")
        return False

    audio = np.asarray(recording, dtype=np.float32).ravel()
    clipped = float(np.mean(np.abs(audio) >= 0.99))

    def _dbfs(value: float) -> float:
        return 20.0 * np.log10(value) if value > 1e-9 else -120.0

    # Split into short frames and read the loud end and the quiet end separately, rather than
    # averaging the whole recording into one number. A single average cannot tell "a clear voice
    # in a quiet room" from "no voice at all in a noisy room" - both land in the middle - and
    # those two need opposite fixes. The 90th percentile is the speech, the 10th is what the room
    # is doing underneath it, and the gap between them is what actually decides whether Whisper
    # can hear a visitor over the lobby.
    frame = 480  # 30ms at 16kHz, same frame size the real VAD uses
    usable = audio[: len(audio) - len(audio) % frame]
    frames = usable.reshape(-1, frame) if usable.size else audio.reshape(1, -1)
    frame_rms = (
        np.sqrt(np.mean(frames**2, axis=1)) if frames.size else np.array([0.0], dtype=np.float32)
    )

    signal_db = _dbfs(float(np.percentile(frame_rms, 90)))
    noise_db = _dbfs(float(np.percentile(frame_rms, 10)))
    peak_db = _dbfs(float(np.max(np.abs(audio))))
    snr_db = signal_db - noise_db

    rms_db = signal_db  # the level checks below are about the SPEECH, not the room
    stats = (
        f"speech {signal_db:.0f} dBFS, background {noise_db:.0f} dBFS, "
        f"signal-to-noise {snr_db:.0f} dB, peak {peak_db:.0f} dBFS"
    )

    try:
        device_name = sd.query_devices(kind="input")["name"]
    except Exception:  # noqa: BLE001
        device_name = "default input"

    if rms_db < MIC_SILENT_DBFS:
        print(f"FAIL Mic: essentially silent ({stats}) on '{device_name}'.")
        print(
            "     Nothing is reaching the microphone. Check it isn't muted, and that this is "
            "the right input device - run list_audio_devices.py to see the alternatives."
        )
        return False

    if clipped > MIC_CLIP_FRACTION:
        print(
            f"WARN Mic: clipping - {clipped * 100:.1f}% of samples are at full scale "
            f"({stats}) on '{device_name}'."
        )
        print(
            "     Distorted audio transcribes badly. Lower the input gain, or move the "
            "microphone further from the speaker."
        )
        return True

    if rms_db < MIC_QUIET_DBFS:
        print(f"WARN Mic: quiet ({stats}) on '{device_name}'.")
        print(
            "     This still works, but Whisper does noticeably better with a stronger "
            "signal. Raise the input gain in the OS sound settings, or move the microphone "
            "closer to where people actually stand."
        )
        return True

    if snr_db < MIC_MIN_SNR_DB:
        print(
            f"WARN Mic: little separation between voice and background ({stats}) on "
            f"'{device_name}'."
        )
        print(
            "     Either nothing was said while this ran - in which case just re-run it and "
            "speak - or the room is genuinely loud relative to a visitor's voice. The second "
            "case is the one that hurts: it is what a directional or close-talk microphone "
            "fixes, and what no amount of software tuning does."
        )
        return True

    print(f"PASS Mic: good level ({stats}) on '{device_name}'.")
    return True


def check_speaker() -> bool:
    """Plays a short test tone (requires human confirmation by listening)."""
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("FAIL Speaker: sounddevice/numpy not installed")
        return False

    try:
        fs = 44100
        duration_s = 1.5
        t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        sd.play(tone, fs)
        sd.wait()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL Speaker: playback failed - {e}")
        return False

    print("PASS Speaker: played a 440Hz tone for 1.5s - confirm you actually heard it")
    return True


if __name__ == "__main__":
    print("== Stage 00: hardware check ==")
    results = {
        "camera": check_camera(),
        "microphone": check_microphone(),
        "speaker": check_speaker(),
    }
    print("\n== Summary ==")
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'} {name}")

    if not all(results.values()):
        sys.exit(1)
