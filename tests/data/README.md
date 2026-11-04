# Test data

`speech/` holds five short synthetic speech clips (16 kHz mono WAV) generated with the Kokoro
voice. The echo-cancellation tests in `tests/test_audio_io.py` use them as both the robot's voice
and the visitor's voice, so no recordings of real people are needed.

`face.jpg` is not in the repository, because a photo of a real person should not be published
with the code. Add a photo of one face looking at the camera (at least 200x200 pixels) to run:

- `tests/test_perception.py::test_yunet_presence_finds_a_real_face`
- `tests/test_end_to_end.py` (the simulated camera shows this face to the robot)

Without it those tests are skipped.
