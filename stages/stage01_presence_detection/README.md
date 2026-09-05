# Stage 01 — Presence Detection

Two files here:

- `presence_detector.py` — the mock-based version (from earlier), keep this working as a
  regression test of the entered/left/present state logic. No hardware needed.
- `presence_detector_live.py` — the real version: your webcam + YOLOv8n (ultralytics),
  detecting actual people in real time, with a preview window (dev tool only — the real kiosk
  has no screen).

## Install (one time, venv already active)

One command — torch (CPU build) and ultralytics are both pinned in `requirements.txt`, with
torch pulled from the PyTorch CPU wheel index:

```powershell
pip install -r stages\stage01_presence_detection\requirements.txt
```

## Run the live detector

```powershell
python stages\stage01_presence_detection\presence_detector_live.py
```

First run downloads the YOLOv8n weights automatically (~6MB, cached after that). A window opens
showing your webcam feed with green boxes around anyone detected as "near" and orange for "far"
(distance is a rough placeholder based on box size — see `NEAR_THRESHOLD_AREA`, not calibrated
yet), labeled with a stable per-person track ID. The terminal prints
`[event] person(s) entered - ids=[...]` / `person(s) left - ids=[...]` as people step in and out
of frame — using real tracking (ByteTrack via `model.track()`), so each person keeps the same ID
across frames instead of getting a new one every frame. Press `q` in the window to quit.

## Tracking + debouncing

Detection alone isn't enough to know "is this the same person still standing there" — a single
frame can miss someone from a pose change or motion blur. Two layers handle this:

1. **`model.track()`** (ByteTrack) keeps a stable ID per person across frames, tolerating brief
   gaps in detection.
2. **`PerTrackPresenceTracker`** debounces on top of that: an "entered" event only fires after a
   track has been seen for `ENTER_CONFIRM_FRAMES` consecutive frames, and "left" only after it's
   been missing for `LEAVE_CONFIRM_FRAMES` frames — so the real robot's greeting logic doesn't
   fire repeatedly off of one shaky frame.

This is what Stage 02 (face memory) will build on — a track ID stable enough to say "link this
face to the same person across the whole interaction."

## Licensing note

`ultralytics` is AGPL-3.0. Fine while prototyping. Revisit before shipping this as closed-source
to the university — you'd need either an Ultralytics commercial license or to open-source the
parts of this project that depend on it.
