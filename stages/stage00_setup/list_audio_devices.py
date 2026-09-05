"""
Stage 00 - diagnostic helper: list all audio devices sounddevice can see, and mark which ones
are the current defaults. Use this if test_hardware.py says the speaker/mic test "passed" but
you didn't actually hear/verify anything - the wrong device may be selected as default.

Run: python list_audio_devices.py
"""

import sounddevice as sd

print("== Audio devices ==\n")
devices = sd.query_devices()
default_in, default_out = sd.default.device

for i, dev in enumerate(devices):
    marker = []
    if i == default_in:
        marker.append("DEFAULT INPUT")
    if i == default_out:
        marker.append("DEFAULT OUTPUT")
    tag = f"  <-- {', '.join(marker)}" if marker else ""
    print(f"[{i}] {dev['name']}  "
          f"(in={dev['max_input_channels']}, out={dev['max_output_channels']}){tag}")

print(f"\nDefault input index: {default_in}")
print(f"Default output index: {default_out}")
print("\nIf the default output above is not your laptop speakers (e.g. it's an HDMI/digital "
      "output or a disconnected device), note its correct index from the list and pass it to "
      "test_speaker_device.py")
