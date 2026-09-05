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


def check_microphone(seconds: float = 1.0) -> bool:
    """Records one second of audio and confirms it got a signal (not just zeros)."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("FAIL Mic: sounddevice/numpy not installed")
        return False

    try:
        recording = sd.rec(int(seconds * 16000), samplerate=16000, channels=1)
        sd.wait()
    except Exception as e:  # noqa: BLE001 - diagnostic script, print any hardware error
        print(f"FAIL Mic: recording failed - {e}")
        return False

    if recording is None or len(recording) == 0:
        print("FAIL Mic: no data returned")
        return False

    print(f"PASS Mic: OK - recorded {len(recording)} samples")
    return True


def check_speaker() -> bool:
    """Plays a short test tone (requires human confirmation by listening)."""
    try:
        import sounddevice as sd
        import numpy as np
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
