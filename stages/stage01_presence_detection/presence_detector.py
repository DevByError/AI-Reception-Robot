"""
Stage 01 - Presence Detection

Goal: detect when a person/group enters the camera's field of view, and estimate proximity.
This file provides an interface + a placeholder implementation, so you can test the "person
entered / left" logic without a real camera, and later plug in a real model (YOLOv8n/MediaPipe)
without changing any code in the stages that come after this one.

Success criterion (from STAGES.md): detection in < 500ms of a person appearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class DetectedPerson:
    track_id: int  # temporary identity for the person while they're in frame (used to
    # re-link with face recognition later)
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixels
    distance_estimate: str  # "far" | "near" | "very_near" - a rough estimate from bbox size


class FrameSource(Protocol):
    """Any frame source (real camera, test video, or mock) must implement this."""

    def read_frame(self):
        """Returns one frame (numpy array) or None if there is no frame."""
        ...


class MockFrameSource:
    """A fake source for testing: returns a pre-scripted sequence of events instead of real
    images."""

    def __init__(self, scripted_events: list[list[DetectedPerson]]):
        self._events = scripted_events
        self._i = 0

    def read_frame(self):
        if self._i >= len(self._events):
            return None
        frame = self._events[self._i]
        self._i += 1
        return frame  # here the "frame" is directly a list of detected people, for testing


class PresenceDetector:
    """Mock presence detector used by the self-tests. The camera-backed versions are
    presence_detector_live.py (YOLOv8n) and face_presence.py (YuNet)."""

    NEAR_THRESHOLD_AREA = 15000  # placeholder: bbox area in pixels considered "close enough to
    # interact"

    def __init__(self, frame_source: FrameSource):
        self.frame_source = frame_source
        self._currently_present: dict[int, DetectedPerson] = {}

    def detect_in_frame(self, frame) -> list[DetectedPerson]:
        """
        Real implementation: runs a detection model on the frame and returns a list of
        DetectedPerson. Mock: the frame is already a ready list of DetectedPerson (see
        MockFrameSource).
        """
        if frame is None:
            return []
        return frame

    def step(self) -> dict:
        """
        Called on each iteration of the main loop. Returns a dict with:
          - entered: new people who appeared this iteration (used to trigger the Stage 03
            greeting)
          - left: people who were present and disappeared
          - present: everyone currently present
        """
        frame = self.frame_source.read_frame()
        detected = self.detect_in_frame(frame)
        detected_ids = {p.track_id: p for p in detected}

        entered = [p for tid, p in detected_ids.items() if tid not in self._currently_present]
        left_ids = [tid for tid in self._currently_present if tid not in detected_ids]
        left = [self._currently_present[tid] for tid in left_ids]

        self._currently_present = detected_ids
        return {"entered": entered, "left": left, "present": list(detected_ids.values())}


if __name__ == "__main__":
    # Self-test without hardware: simulate a person entering the frame, then a group entering,
    # then everyone leaving.
    scripted = [
        [],  # no one
        [DetectedPerson(track_id=1, bbox=(100, 100, 120, 200), distance_estimate="near")],
        [
            DetectedPerson(track_id=1, bbox=(100, 100, 120, 200), distance_estimate="near"),
            DetectedPerson(track_id=2, bbox=(300, 100, 110, 190), distance_estimate="near"),
        ],
        [],  # everyone left
    ]

    detector = PresenceDetector(MockFrameSource(scripted))
    for i in range(len(scripted)):
        result = detector.step()
        print(
            f"[frame {i}] entered={[p.track_id for p in result['entered']]} "
            f"left={[p.track_id for p in result['left']]} "
            f"present={[p.track_id for p in result['present']]}"
        )

    print(
        "\nOK: expected sequence was person 1 entering, then person 2 entering (group), "
        "then 1 and 2 both leaving together."
    )
