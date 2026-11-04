"""
Stage 02 add-on - active speaker detection from mouth movement.

The problem this solves, in the robot's real setting: two people stand in front of the reception
robot and one of them asks a question. The microphone is mono and has no idea which of them
spoke. Before this module, the orchestrator resolved that by guessing (whoever the face-matching
happened to return first), which means the robot could answer person A while looking at and
remembering person B - and, worse, could attach the wrong face to a name during the consent flow.

The visual signal that actually disambiguates it is lip motion: the person whose mouth is moving
while audio is arriving is the one talking. This module measures that per visible face.

KEY DESIGN POINT - movement, not openness. The naive version of this checks "is the mouth open?"
That's wrong in both directions: a person can sit with their mouth open and not be speaking, and
a person mid-consonant is briefly closed while very much speaking. What actually separates
speech from a resting face is that the mouth-open ratio KEEPS CHANGING. So the score below is the
rolling standard deviation of the mouth-open ratio over the last ~half second, not its value.

Scale invariance: the raw vertical lip gap is in pixels, so someone standing closer to the camera
would score higher purely for being closer. Dividing by the mouth's own width (which scales the
same way with distance) cancels that out, so the ratio means the same thing at any distance.

Deliberately NOT audio-visual correlation: a stronger version of this would correlate lip motion
against the audio envelope over time, so that someone silently mouthing words or chewing couldn't
win. That needs synchronized audio/video timestamps, which the current single-threaded capture
doesn't provide honestly. This module's contract is narrower and achievable: "of the faces
visible right now, whose mouth is moving most." The orchestrator only asks it that question
during a window when it already knows someone IS speaking (the VAD said so), which covers most
of the gap in practice.

LICENSING: mediapipe is Apache-2.0 - compatible with this project's existing Apache-2.0
components (YuNet/SFace), and notably less restrictive than Stage 01's ultralytics (AGPL-3.0) or
Stage 06's Piper (GPL-3.0). See README.md's licensing note.

Install: pip install -r requirements.txt (in this folder)
Run (real webcam demo, shows a live speaking/quiet readout per face):
    python mouth_activity.py
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# MediaPipe's face mesh gives 468 3D facial landmarks. These four are all this module needs -
# indices confirmed against MediaPipe's own published landmark map (see README.md for the
# reference link):
#   13  - centre of the UPPER lip
#   14  - centre of the LOWER lip      -> vertical gap between these two = how open the mouth is
#   61  - LEFT mouth corner
#   291 - RIGHT mouth corner           -> distance between these = mouth width, used to normalize
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
# For the head-turn estimate: the nose tip and the inner/outer corners of each eye (their
# midpoints approximate the eye centres YuNet reports, so both yaw numbers share one scale).
NOSE_TIP = 1
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

# How many recent frames of mouth-open ratio to keep per face when computing the movement score.
# At a typical ~20-25 fps webcam this is roughly half a second - long enough to span several
# syllables (so normal speech reliably shows variation), short enough that the robot notices
# someone starting or stopping talking without a visible lag.
DEFAULT_HISTORY_FRAMES = 12

# Rolling standard deviation of the mouth-open ratio above which a face is called "speaking".
#
# THIS IS A STARTING POINT, NOT A MEASURED CONSTANT - same warning as SIMILARITY_THRESHOLD in
# stage05's knowledge_base.py, and it deserves the same treatment. It was chosen to sit well
# above the jitter a still face produces from landmark noise alone, and below what ordinary
# speech produces, but the right value depends on your camera's frame rate, resolution and
# lighting. Run this file directly (see __main__) to print live scores for your own face, and
# adjust: read your score while sitting quietly, then while talking normally, and put this
# threshold in the gap between them. A too-low value makes a quiet face register as speaking
# (the robot answers the wrong person); a too-high value makes a real speaker look quiet (the
# robot falls back to its old "closest face" guess, which is no worse than before this existed).
SPEAKING_ACTIVITY_THRESHOLD = 0.018

# Frame-to-frame identity: a face in this frame is considered the same face as one from the
# previous frame if their centres are within this fraction of the frame width. Mouth-movement
# history is meaningless without stable identity across frames - if faces got renumbered every
# frame, every history would be one sample long and every score would be zero. Generous enough
# to survive normal head movement between frames, tight enough that two people standing apart
# don't get confused for each other.
MATCH_DISTANCE_FRACTION = 0.15


@dataclass
class MouthActivity:
    """One visible face's current mouth state."""

    face_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h - derived from the landmark mesh extent
    mouth_open_ratio: float          # current vertical lip gap / mouth width
    activity_score: float            # rolling std-dev of the above - the real "is talking" signal
    is_speaking: bool
    samples: int                     # how many frames of history back this score (see below)
    # Head turn, same scale as stage08's engagement.face_yaw(): 0 = looking at the camera,
    # ~+/-0.5 at 45 degrees. None if the landmarks could not give one. Added 2026-10-27 so the
    # addressee check can ask "were they FACING the robot while their lips moved?".
    yaw: float | None = None

    @property
    def is_confident(self) -> bool:
        """A score computed from only two or three frames isn't yet meaningful - it takes a few
        frames of history before the standard deviation means anything. Callers should prefer a
        confident speaking face over an unconfident one rather than acting on a score that's
        really just startup noise."""
        return self.samples >= 5


@dataclass
class _TrackedFace:
    face_id: int
    center: tuple[float, float]
    history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_FRAMES))


class MouthActivityDetector:
    """Tracks every visible face's mouth movement across frames.

    Usage is one update(frame) call per camera frame; it returns the current state of every face
    it can see. Call most_likely_speaker() to get the single best "this is who's talking to you"
    answer, or None when nothing looks like speech.
    """

    def __init__(
        self,
        max_faces: int = 4,
        history_frames: int = DEFAULT_HISTORY_FRAMES,
        speaking_threshold: float = SPEAKING_ACTIVITY_THRESHOLD,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for active speaker detection - run: pip install -r "
                "stages/stage02_face_recognition/requirements.txt"
            ) from exc

        # Legacy mp.solutions API rather than the newer Tasks API (FaceLandmarker). It's still
        # shipped and is markedly simpler for this use - no separate .task model file to download
        # and version-manage, which matters for a project that already asks the user to fetch
        # YOLO/YuNet/SFace/Piper models by hand. requirements.txt pins a mediapipe version known
        # to include it; if a future upgrade drops mp.solutions entirely, this class is the only
        # place that needs rewriting against FaceLandmarker.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  # video mode - tracks between frames, much faster per frame
            max_num_faces=max_faces,
            refine_landmarks=True,    # the attention mesh - materially better lip precision,
                                       # which is the one thing this module actually measures
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.history_frames = history_frames
        self.speaking_threshold = speaking_threshold
        self._tracked: list[_TrackedFace] = []
        self._next_id = 0
        # Crop mode - see update_faces(). A separate single-face mesh in
        # static mode, because each call sees a different crop.
        self._crop_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._crop_history: dict[int, deque] = {}

    # Crop mode: how much context to keep around a detected face box, and the smallest crop
    # side handed to the mesh (it is upscaled to this if smaller).
    CROP_EXPAND = 1.8
    CROP_MIN_SIDE = 192

    def update_faces(self, frame: np.ndarray,
                     faces: list[tuple[int, tuple[int, int, int, int]]]) -> list[MouthActivity]:
        """Mouth activity for faces that ANOTHER detector already found.

        faces: (track_id, (x, y, w, h)) pairs, e.g. from stage01's face_presence.py. Each face is
        cropped (with margin) and the mesh runs on the crop.

        Why: the whole-frame mesh (update()) uses MediaPipe's short-range detector, which works
        on a downscaled frame and stops finding faces once they are smaller than ~1/5 of the
        frame width. At 1280x720 that is anybody more than about a metre away - measured here, a
        160px face in a 1280px frame was not found at all, while the same face cropped was found
        down to 60px, in ~8ms per face. It also makes face_id the PRESENCE track id, so "who is
        talking" no longer has to be joined back to a person by box geometry.
        """
        h, w = frame.shape[:2]
        activities: list[MouthActivity] = []
        seen: set[int] = set()
        for track_id, (x, y, fw, fh) in faces:
            side = max(fw, fh) * self.CROP_EXPAND
            cx, cy = x + fw / 2.0, y + fh / 2.0
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            if min(ch, cw) < self.CROP_MIN_SIDE:
                factor = self.CROP_MIN_SIDE / float(min(ch, cw))
                crop = cv2_resize(crop, (int(cw * factor), int(ch * factor)))
            result = self._crop_mesh.process(np.ascontiguousarray(crop[:, :, ::-1]))
            if not result.multi_face_landmarks:
                continue
            points = result.multi_face_landmarks[0].landmark
            ratio = self._mouth_open_ratio(points)
            history = self._crop_history.setdefault(
                track_id, deque(maxlen=self.history_frames))
            history.append(ratio)
            seen.add(track_id)
            score = statistics.pstdev(history) if len(history) >= 2 else 0.0
            activities.append(MouthActivity(
                face_id=track_id, bbox=(x, y, fw, fh), mouth_open_ratio=ratio,
                activity_score=score, is_speaking=score >= self.speaking_threshold,
                samples=len(history), yaw=self._yaw(points),
            ))
        for gone in set(self._crop_history) - {t for t, _ in faces}:
            del self._crop_history[gone]
        return activities

    def update(self, frame: np.ndarray) -> list[MouthActivity]:
        """Process one BGR camera frame (OpenCV's native format) and return the current mouth
        state of every visible face."""
        h, w = frame.shape[:2]
        # MediaPipe expects RGB; OpenCV gives BGR. Getting this backwards doesn't error - it just
        # silently degrades detection, which would look like "the speaker detection is bad" rather
        # than an obvious bug, so it's worth being explicit about.
        rgb = frame[:, :, ::-1]
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._tracked = []
            return []

        activities: list[MouthActivity] = []
        still_tracked: list[_TrackedFace] = []

        for landmarks in results.multi_face_landmarks:
            points = landmarks.landmark
            ratio = self._mouth_open_ratio(points)
            bbox, center = self._bbox_and_center(points, w, h)

            tracked = self._match_or_create(center, w)
            tracked.center = center
            tracked.history.append(ratio)
            still_tracked.append(tracked)

            # Standard deviation needs at least two samples; one frame in, there's no movement to
            # measure yet, so report a flat zero rather than raising.
            score = statistics.pstdev(tracked.history) if len(tracked.history) >= 2 else 0.0
            activities.append(MouthActivity(
                face_id=tracked.face_id,
                bbox=bbox,
                mouth_open_ratio=ratio,
                activity_score=score,
                is_speaking=score >= self.speaking_threshold,
                samples=len(tracked.history),
                yaw=self._yaw(points),
            ))

        # Drop faces that left the frame, so their stale mouth history can't be inherited by a
        # different person who later happens to stand in a similar spot.
        self._tracked = still_tracked
        return activities

    def most_likely_speaker(self, activities: list[MouthActivity]) -> MouthActivity | None:
        """The single face most likely to be the one talking, or None if nothing looks like
        speech. Prefers faces with enough history to be trustworthy (see MouthActivity.
        is_confident) before falling back to the highest raw score."""
        speaking = [a for a in activities if a.is_speaking]
        if not speaking:
            return None
        confident = [a for a in speaking if a.is_confident]
        return max(confident or speaking, key=lambda a: a.activity_score)

    def _mouth_open_ratio(self, points) -> float:
        """Vertical lip gap divided by mouth width - see the module docstring on why the division
        matters. Landmark coordinates are already normalized to 0-1 by MediaPipe, so this is
        resolution-independent as well as distance-independent."""
        upper = points[UPPER_LIP_CENTER]
        lower = points[LOWER_LIP_CENTER]
        left = points[LEFT_MOUTH_CORNER]
        right = points[RIGHT_MOUTH_CORNER]

        vertical = abs(upper.y - lower.y)
        horizontal = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
        if horizontal < 1e-6:  # face turned fully side-on - no meaningful width to divide by
            return 0.0
        return vertical / horizontal

    @staticmethod
    def _yaw(points) -> float | None:
        try:
            rx = (points[RIGHT_EYE_CORNERS[0]].x + points[RIGHT_EYE_CORNERS[1]].x) / 2
            ry = (points[RIGHT_EYE_CORNERS[0]].y + points[RIGHT_EYE_CORNERS[1]].y) / 2
            lx = (points[LEFT_EYE_CORNERS[0]].x + points[LEFT_EYE_CORNERS[1]].x) / 2
            ly = (points[LEFT_EYE_CORNERS[0]].y + points[LEFT_EYE_CORNERS[1]].y) / 2
            eye = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
            if eye < 1e-4:
                return None
            return (points[NOSE_TIP].x - (rx + lx) / 2) / eye
        except (IndexError, AttributeError):
            return None

    def _bbox_and_center(self, points, frame_w: int, frame_h: int):
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        x0, x1 = min(xs) * frame_w, max(xs) * frame_w
        y0, y1 = min(ys) * frame_h, max(ys) * frame_h
        bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        return bbox, center

    def _match_or_create(self, center: tuple[float, float], frame_w: int) -> _TrackedFace:
        max_distance = frame_w * MATCH_DISTANCE_FRACTION
        best = None
        best_distance = max_distance
        for candidate in self._tracked:
            distance = (
                (candidate.center[0] - center[0]) ** 2
                + (candidate.center[1] - center[1]) ** 2
            ) ** 0.5
            if distance < best_distance:
                best = candidate
                best_distance = distance

        if best is not None:
            return best

        new = _TrackedFace(
            face_id=self._next_id,
            center=center,
            history=deque(maxlen=self.history_frames),
        )
        self._next_id += 1
        self._tracked.append(new)
        return new

    def close(self) -> None:
        self._face_mesh.close()
        self._crop_mesh.close()


def cv2_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def match_activity_to_bbox(
    activities: list[MouthActivity], bbox: tuple[int, int, int, int]
) -> MouthActivity | None:
    """Links a mouth-activity face to a face box found by a DIFFERENT detector (YuNet, via
    stage02's SFaceEmbedder) by centre proximity.

    Needed because the two detectors are separate: SFace produces the recognition embedding that
    identifies WHO someone is, while this module produces the movement score that says WHO IS
    TALKING. They have to be joined on geometry, since neither knows about the other's numbering.
    Returns None if nothing is close enough, so the caller can fall back rather than act on a
    bad pairing."""
    if not activities:
        return None
    x, y, w, h = bbox
    target = (x + w / 2, y + h / 2)
    tolerance = max(w, h)  # within roughly one face-width counts as the same face

    best = None
    best_distance = float("inf")
    for activity in activities:
        ax, ay, aw, ah = activity.bbox
        center = (ax + aw / 2, ay + ah / 2)
        distance = ((center[0] - target[0]) ** 2 + (center[1] - target[1]) ** 2) ** 0.5
        if distance < best_distance:
            best = activity
            best_distance = distance

    return best if best_distance <= tolerance else None


if __name__ == "__main__":
    # Live calibration aid, not a pass/fail test - SPEAKING_ACTIVITY_THRESHOLD can only be set
    # honestly by watching real numbers from your own camera (see that constant's comment).
    import cv2

    print(__doc__)
    print("Calibrating. Sit still for a few seconds and note the score, then talk normally and "
          "note it again. Set SPEAKING_ACTIVITY_THRESHOLD between the two.")
    print(f"(current threshold: {SPEAKING_ACTIVITY_THRESHOLD})\n")
    print("Press 'q' in the preview window to quit.\n")

    detector = MouthActivityDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index=0")

    quiet_scores: list[float] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            activities = detector.update(frame)
            speaker = detector.most_likely_speaker(activities)

            for activity in activities:
                x, y, w, h = activity.bbox
                is_speaker = speaker is not None and speaker.face_id == activity.face_id
                color = (0, 255, 0) if is_speaker else (160, 160, 160)
                label = (
                    f"#{activity.face_id} {'TALKING' if is_speaker else 'quiet'} "
                    f"score={activity.activity_score:.4f} open={activity.mouth_open_ratio:.3f}"
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if not activity.is_speaking:
                    quiet_scores.append(activity.activity_score)

            cv2.imshow("mouth activity (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    if quiet_scores:
        # A rough sanity readout: if the resting-face noise floor is already near or above the
        # configured threshold, that threshold cannot possibly separate quiet from talking on
        # this camera, and every still face will read as speaking.
        peak_quiet = max(quiet_scores)
        print(f"\nHighest score seen while NOT classified as speaking: {peak_quiet:.4f}")
        if peak_quiet >= SPEAKING_ACTIVITY_THRESHOLD * 0.8:
            print(f"WARNING: that's close to the threshold ({SPEAKING_ACTIVITY_THRESHOLD}) - "
                  f"raise SPEAKING_ACTIVITY_THRESHOLD, or a still face will read as talking.")
        else:
            print(f"OK: comfortably below the threshold ({SPEAKING_ACTIVITY_THRESHOLD}) - a "
                  f"still face should not be mistaken for a speaking one on this camera.")
