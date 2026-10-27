# Test data

`face.jpg` - a photo of one face, looking at the camera, at least 200x200 pixels. A selfie
works. It is not in the repository (a photo of a real person should not be published with the
code), so add your own before running:

- `tests/test_perception.py::test_yunet_presence_finds_a_real_face`
- `tests/test_end_to_end.py` (the simulated camera shows this face to the robot)

Without it those tests are skipped; everything else runs.
