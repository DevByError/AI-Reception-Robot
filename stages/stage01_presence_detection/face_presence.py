"""
Stage 01 (light) - presence detection from FACES instead of whole bodies.

Why: the YOLOv8 person tracker (presence_detector_live.py) was the single most expensive thing
the robot did - ~170ms per frame on a laptop CPU, which held the whole camera loop to ~5fps -
and it needed PyTorch (~2GB) and ultralytics (AGPL-3.0) for it. A reception robot only ever
needs to talk to people whose FACE it can see: somebody walking past with their back to it is
exactly who it should ignore. YuNet, which stage 02 already uses, finds faces in ~10-20ms on a
640px frame and gives eye/nose landmarks for free, which is what the engagement gate
(stage08's engagement.py) uses to tell "facing the robot" from "walking past".

Detection runs on a downscaled copy of the frame (DETECT_WIDTH). That also acts as a distance
filter: a face too small to find at 640px is too far away to talk to.

Same output shape as presence_detector_live.detect_people() - a list of DetectedPerson with a
stable track_id - so the orchestrator and PerTrackPresenceTracker do not care which one runs.
The person's "bbox" here is the face box, and every DetectedPerson also carries the YuNet
landmarks (see FacePerson).

Run: python face_presence.py    (live webcam demo, 'q' to quit)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from presence_detector import DetectedPerson

DETECTION_MODEL = (
    Path(__file__).resolve().parents[1]
    / "stage02_face_recognition"
    / "face_detection_yunet_2023mar.onnx"
)

DETECT_WIDTH = 640  # detection runs at this width; boxes are scaled back up
SCORE_THRESHOLD = 0.6  # a little looser than recognition's 0.7 - presence, not identity
NEAR_FACE_FRACTION = 0.05  # face width / frame width at or above which a face counts as "near"

# Tracker: how far (as a fraction of the larger box side) a face may move between two processed
# frames and still be the same person, and how many processed frames a track survives unseen.
MATCH_DISTANCE = 1.0
MAX_MISSED = 6


@dataclass
class FacePerson(DetectedPerson):
    """A DetectedPerson found by its face. Adds what the engagement gate needs."""

    landmarks: tuple[tuple[int, int], ...] = ()
    score: float = 1.0
    frame_width: int = 0


@dataclass
class _Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    missed: int = 0
    hits: int = 0
    last: FacePerson | None = field(default=None, repr=False)


def _center(b):
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


class FaceTracker:
    """Gives each face a stable id across frames by nearest-centre matching.

    Deliberately simple: a kiosk sees a handful of faces that move slowly, and a greedy nearest
    match is enough. A track survives MAX_MISSED processed frames without a detection, so a
    blink of occlusion or a head turn does not produce a "new person" and a second greeting.
    """

    def __init__(self, match_distance: float = MATCH_DISTANCE, max_missed: int = MAX_MISSED):
        self.match_distance = match_distance
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        """Returns one track id per input box (same order)."""
        assigned: list[int | None] = [None] * len(boxes)
        free = list(self._tracks)
        pairs = []
        for i, box in enumerate(boxes):
            cx, cy = _center(box)
            for t in free:
                tx, ty = _center(t.bbox)
                scale = max(box[2], box[3], t.bbox[2], t.bbox[3], 1)
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / scale
                if d <= self.match_distance:
                    pairs.append((d, i, t))
        pairs.sort(key=lambda p: p[0])
        used_tracks: set[int] = set()
        for _d, i, t in pairs:
            if assigned[i] is not None or t.track_id in used_tracks:
                continue
            assigned[i] = t.track_id
            used_tracks.add(t.track_id)
            t.bbox, t.missed = boxes[i], 0
            t.hits += 1
        for i, box in enumerate(boxes):
            if assigned[i] is None:
                t = _Track(self._next_id, box, hits=1)
                self._next_id += 1
                self._tracks.append(t)
                assigned[i] = t.track_id
                used_tracks.add(t.track_id)
        for t in self._tracks:
            if t.track_id not in used_tracks:
                t.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return [a for a in assigned if a is not None]


class FacePresenceDetector:
    """YuNet on a downscaled frame + FaceTracker. One detect_people(frame) call per frame."""

    def __init__(self, detect_width: int = DETECT_WIDTH, score_threshold: float = SCORE_THRESHOLD):
        if not DETECTION_MODEL.exists():
            raise FileNotFoundError(
                f"{DETECTION_MODEL} is missing - run "
                f"python stages/stage02_face_recognition/download_models.py"
            )
        self.detect_width = detect_width
        self._detector = cv2.FaceDetectorYN.create(
            str(DETECTION_MODEL), "", (detect_width, detect_width), score_threshold, 0.3, 5000
        )
        self._size = (0, 0)
        self.tracker = FaceTracker()

    def detect_faces(
        self, frame
    ) -> list[tuple[tuple[int, int, int, int], tuple[tuple[int, int], ...], float]]:
        """(bbox, landmarks, score) per face, in FULL-frame pixel coordinates."""
        h, w = frame.shape[:2]
        scale = min(1.0, self.detect_width / float(w))
        small = (
            frame
            if scale == 1.0
            else cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        )
        sh, sw = small.shape[:2]
        if (sw, sh) != self._size:
            self._detector.setInputSize((sw, sh))
            self._size = (sw, sh)
        _, faces = self._detector.detect(small)
        out = []
        if faces is None:
            return out
        inv = 1.0 / scale
        for row in faces:
            x, y, fw, fh = (row[:4] * inv).astype(int)
            pts = row[4:14].reshape(5, 2) * inv
            landmarks = tuple((int(px), int(py)) for px, py in pts)
            out.append(((int(x), int(y), int(fw), int(fh)), landmarks, float(row[14])))
        return out

    def detect_people(self, frame) -> list[FacePerson]:
        faces = self.detect_faces(frame)
        ids = self.tracker.update([f[0] for f in faces])
        width = frame.shape[1]
        people = []
        for (bbox, landmarks, score), tid in zip(faces, ids):
            near = bbox[2] >= NEAR_FACE_FRACTION * width
            people.append(
                FacePerson(
                    track_id=tid,
                    bbox=bbox,
                    distance_estimate="near" if near else "far",
                    landmarks=landmarks,
                    score=score,
                    frame_width=width,
                )
            )
        return people


def _demo() -> None:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "stage08_orchestrator"))
    from engagement import face_yaw

    detector = FacePresenceDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index=0")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for p in detector.detect_people(frame):
                x, y, w, h = p.bbox
                yaw = face_yaw(p.landmarks)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                label = (
                    f"#{p.track_id} {p.distance_estimate} yaw={yaw:+.2f}"
                    if yaw is not None
                    else f"#{p.track_id} {p.distance_estimate}"
                )
                cv2.putText(
                    frame, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
                )
            cv2.imshow("face presence (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
