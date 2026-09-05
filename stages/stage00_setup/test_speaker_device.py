"""
Stage 00 - diagnostic helper: play a louder, longer test tone on a specific output device index,
so you can confirm exactly which device actually makes sound. Use after list_audio_devices.py.

Run: python test_speaker_device.py           (uses the default output device)
Run: python test_speaker_device.py 5         (forces output device index 5)
"""

import sys

import numpy as np
import sounddevice as sd

device_index = int(sys.argv[1]) if len(sys.argv) > 1 else None

fs = 44100
duration_s = 2.0
t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # louder (0.5) and longer (2s) than the Stage 00 check

if device_index is not None:
    print(f"Playing 2s test tone on device index {device_index} "
          f"({sd.query_devices(device_index)['name']})...")
else:
    default_out = sd.default.device[1]
    print(f"Playing 2s test tone on the DEFAULT output device "
          f"({sd.query_devices(default_out)['name']})...")

sd.play(tone, fs, device=device_index)
sd.wait()
print("Done. Did you hear a 2-second tone?")
