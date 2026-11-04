"""
Stage 01 (live) - real person detection + tracking using a webcam + YOLOv8n (ultralytics).

This replaces the mock in presence_detector.py with a real model, so you can see it detect
you (and others) live through the laptop camera. Bounding boxes are drawn in a preview window
purely for development/debugging - the real kiosk has no screen for this, this window is a
dev tool only.

Uses model.track() (ByteTrack under the hood) instead of plain detection, so each person gets
a STABLE id across frames - required before Stage 02 (face memory) can reliably say "this is
the same person I saw 3 seconds ago" instead of treating every frame as a brand-new person.

Install first: see README.md in this folder.
Run: python presence_detector_live.py
Press 'q' in the preview window to quit.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
from presence_detector import DetectedPerson

if TYPE_CHECKING:  # imported lazily below: ultralytics pulls in PyTorch (~2GB), which the
    from ultralytics import YOLO  # default face-based presence (face_presence.py) never needs

PERSON_CLASS_ID = 0  # COCO class 0 = "person"
CONFIDENCE_THRESHOLD = 0.5
NEAR_THRESHOLD_AREA = 15000  # same placeholder value as presence_detector.py

# Debouncing per tracked person: even with a real tracker, a brief occlusion or a bad frame can
# drop a track for a frame or two. Require a few consecutive frames of agreement before firing
# a confirmed entered/left event - avoids the real robot re-greeting the same person repeatedly.
ENTER_CONFIRM_FRAMES = 3  # ~0.1s at 30fps - confirm "entered" fast, greeting should feel snappy
LEAVE_CONFIRM_FRAMES = 20  # ~0.7s at 30fps - confirm "left" slower, avoid false exits


class PerTrackPresenceTracker:
    """
    Turns the tracker's raw per-frame set of track IDs into confirmed entered/left events,
    per individual person - important for groups, where one person can leave while another
    stays.
    """

    def __init__(
        self, enter_frames: int = ENTER_CONFIRM_FRAMES, leave_frames: int = LEAVE_CONFIRM_FRAMES
    ):
        self.enter_frames = enter_frames
        self.leave_frames = leave_frames
        self._present_streak: dict[int, int] = {}
        self._absent_streak: dict[int, int] = {}
        self.confirmed_ids: set[int] = set()

    def update(self, seen_track_ids: set[int]) -> tuple[list[int], list[int]]:
        """Feed this frame's raw track IDs. Returns (entered_ids, left_ids) confirmed this
        frame (usually empty lists - most frames confirm nothing new)."""
        entered: list[int] = []
        left: list[int] = []

        for tid in seen_track_ids:
            self._present_streak[tid] = self._present_streak.get(tid, 0) + 1
            self._absent_streak[tid] = 0
            if tid not in self.confirmed_ids and self._present_streak[tid] >= self.enter_frames:
                self.confirmed_ids.add(tid)
                entered.append(tid)

        missing_ids = set(self._present_streak) - seen_track_ids
        for tid in missing_ids:
            self._present_streak[tid] = 0
            self._absent_streak[tid] = self._absent_streak.get(tid, 0) + 1
            if tid in self.confirmed_ids and self._absent_streak[tid] >= self.leave_frames:
                self.confirmed_ids.discard(tid)
                left.append(tid)
                del self._present_streak[tid]
                del self._absent_streak[tid]

        return entered, left


INFERENCE_SIZE = 320  # downscale for inference - on CPU, a smaller frame means faster inference,
# which means less motion between processed frames, which is what
# ByteTrack actually needs to keep the same ID across frames. Boxes are
# returned scaled back to the original frame size automatically.


def detect_people(model: YOLO, frame) -> list[DetectedPerson]:
    """Runs detection + tracking on one frame. Returns only people with a confirmed track id
    (the tracker needs a frame or two before it assigns one - that's normal)."""
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        imgsz=INFERENCE_SIZE,
    )[0]
    people: list[DetectedPerson] = []

    if results.boxes is None or results.boxes.id is None:
        return people  # tracker hasn't confirmed any ids yet this frame

    for box, raw_id in zip(results.boxes, results.boxes.id):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id != PERSON_CLASS_ID or conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area = w * h
        distance = "near" if area >= NEAR_THRESHOLD_AREA else "far"

        people.append(
            DetectedPerson(track_id=int(raw_id), bbox=(x1, y1, w, h), distance_estimate=distance)
        )

    return people


def draw_boxes(frame, people: list[DetectedPerson]):
    for p in people:
        x, y, w, h = p.bbox
        color = (0, 200, 0) if p.distance_estimate == "near" else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"person #{p.track_id} ({p.distance_estimate})",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


if __name__ == "__main__":
    from ultralytics import YOLO

    print("Loading YOLOv8n weights (first run downloads them automatically, ~6MB)...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index=0")

    print("Live tracking running. Press 'q' in the preview window to quit.")
    tracker_state = PerTrackPresenceTracker()

    frame_count = 0
    fps_window_start = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame - stopping.")
                break

            people = detect_people(model, frame)
            seen_ids = {p.track_id for p in people}

            entered_ids, left_ids = tracker_state.update(seen_ids)
            if entered_ids:
                print(f"[event] person(s) entered - ids={entered_ids}")
            if left_ids:
                print(f"[event] person(s) left - ids={left_ids}")

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.monotonic() - fps_window_start
                fps = 30 / elapsed if elapsed > 0 else 0
                print(f"[perf] ~{fps:.1f} FPS over the last 30 frames")
                fps_window_start = time.monotonic()

            frame = draw_boxes(frame, people)
            cv2.imshow("Stage 01 - live person tracking (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
