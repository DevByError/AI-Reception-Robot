"""
Stage 02 - real face detection + embedding using OpenCV's YuNet (detection) + SFace
(recognition), both Apache-2.0, loaded as ONNX through cv2. See requirements.txt and
download_models.py in this folder for setup.

YuNet finds face bounding boxes + 5 landmarks in a frame. SFace turns an aligned face crop
into a 128-d embedding, compared via cosine similarity using cv2's own built-in match().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent
DETECTION_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# SFace's own recommended cosine similarity threshold for "same person" (from opencv_zoo's
# official demo.py) - higher score = more similar. Tune experimentally against real users
# later (STAGES.md success metric: >95% recognition accuracy on a real test sample).
COSINE_MATCH_THRESHOLD = 0.363
DETECTION_SCORE_THRESHOLD = 0.7  # YuNet's own confidence threshold for "this is a face"

# --- face quality gating ----------------------------------------------------------
# Not every detected face is worth recognizing FROM. A face 30 pixels wide at the far end of a
# lobby, or motion-blurred as somebody walks past, still produces a perfectly well-formed 128-d
# embedding - it is just an embedding of mush, and nothing downstream can tell the difference.
#
# That does two kinds of damage, and the second is permanent:
#   1. Matching against a bad embedding gives a near-random score, which is how somebody gets
#      greeted by the wrong name.
#   2. ENROLLING or reinforcing with one writes that mush into the database as a reference sample
#      for a real person, so every future comparison against them is degraded. A bad recognition
#      is one bad moment; a bad stored sample is a permanently worse recognizer.
#
# So quality is measured once, at detection, and callers decide what they are willing to use it
# for - see FaceDetection.is_usable / is_enrollable below.
MIN_FACE_WIDTH_PX = 60  # smaller than this and SFace is working from too few real pixels
MIN_BLUR_VARIANCE = 40.0  # variance of the Laplacian - the standard cheap sharpness measure
# Enrolling is held to a higher standard than matching: a stored sample is compared against for
# months, so it is worth waiting a few frames for a good one.
MIN_ENROLL_FACE_WIDTH_PX = 90
MIN_ENROLL_BLUR_VARIANCE = 80.0
MIN_ENROLL_DETECTION_SCORE = 0.90


def blur_variance(gray_crop: np.ndarray) -> float:
    """Sharpness of a crop, as the variance of its Laplacian. Low = blurry.

    Cheap (one convolution on a small crop) and the standard measure for exactly this job. It is
    scale-dependent, which is why it is only ever compared against a fixed threshold on a face
    crop rather than between faces of different sizes.
    """
    if gray_crop is None or gray_crop.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray  # 128-d SFace feature vector
    # YuNet's own confidence that this is a face at all. Was being discarded before - it is
    # already computed, and it is one of the three cheap signals that say whether this embedding
    # can be trusted.
    score: float = 1.0
    # Sharpness of the face crop - see blur_variance().
    sharpness: float = 1e9
    # The 5 landmarks YuNet returns (right eye, left eye, nose, right mouth corner, left mouth
    # corner) as (x, y) pairs. Kept because they are free and they are what a frontality check
    # would need later; mouth_activity.py uses mediapipe for a denser version of the same idea.
    landmarks: tuple[tuple[int, int], ...] = ()

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def is_usable(self) -> bool:
        """Good enough to compare against people the robot already knows."""
        return (
            self.width >= MIN_FACE_WIDTH_PX
            and self.sharpness >= MIN_BLUR_VARIANCE
            and self.score >= DETECTION_SCORE_THRESHOLD
        )

    @property
    def is_enrollable(self) -> bool:
        """Good enough to STORE as a reference sample for somebody - a higher bar, because this
        one is permanent (see the quality-gating note above)."""
        return (
            self.width >= MIN_ENROLL_FACE_WIDTH_PX
            and self.sharpness >= MIN_ENROLL_BLUR_VARIANCE
            and self.score >= MIN_ENROLL_DETECTION_SCORE
        )

    def quality_reason(self) -> str:
        """Why this face was rejected, for the log. Empty when it is fine."""
        problems = []
        if self.width < MIN_FACE_WIDTH_PX:
            problems.append(f"too small ({self.width}px wide, need {MIN_FACE_WIDTH_PX})")
        if self.sharpness < MIN_BLUR_VARIANCE:
            problems.append(
                f"too blurry (sharpness {self.sharpness:.0f}, " f"need {MIN_BLUR_VARIANCE:.0f})"
            )
        if self.score < DETECTION_SCORE_THRESHOLD:
            problems.append(f"low detector confidence ({self.score:.2f})")
        return ", ".join(problems)


REFINE_CROP = 1.6  # crop side for describe_faces(), as a multiple of the face box's longer side


class SFaceEmbedder:
    def __init__(self, frame_size: tuple[int, int] = (640, 480)):
        if not DETECTION_MODEL.exists() or not RECOGNITION_MODEL.exists():
            raise FileNotFoundError(
                "Model files missing - run 'python download_models.py' in "
                "stages/stage02_face_recognition first."
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTION_MODEL),
            "",
            frame_size,
            score_threshold=DETECTION_SCORE_THRESHOLD,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNITION_MODEL), "")
        self._last_size = frame_size

    def detect(self, frame) -> list[FaceDetection]:
        h, w = frame.shape[:2]
        if (w, h) != self._last_size:
            self.detector.setInputSize((w, h))
            self._last_size = (w, h)

        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return self._describe_rows(frame, faces)

    def describe_faces(self, frame, faces) -> list[FaceDetection]:
        """Embeddings for faces ANOTHER detector already found - (bbox, landmarks, score)
        tuples in full-frame pixels, e.g. from stage01's face_presence.py.

        Skips this class's own YuNet pass, which at 1280x720 measured ~146ms - the single most
        expensive step left in the camera loop once YOLO was gone, and pure duplication when
        presence detection has just found the same faces (with the same landmarks, which are
        all SFace's alignment needs).

        Landmarks found on a downscaled frame are a pixel or two off at full size, and SFace's
        alignment is sensitive to that (measured: cosine 0.87-0.94 against the full-resolution
        result for the same image). So each face is re-detected in a small full-resolution crop
        around it first - a few milliseconds - which gives the same landmarks a full-frame pass
        would. If that finds nothing, the given landmarks are used as they are.

        The crop size matters more than expected: YuNet's landmarks on a small face vary with
        how much context surrounds it. Measured on 25 placements of one face (sizes 200-460px,
        five positions), similarity to an enrolled sample: full-frame detection mean 0.856,
        worst 0.397; this path with a 1.6x crop mean 0.882, worst 0.780 (2.0x: 0.830 / 0.419).
        The match threshold is 0.363.
        """
        rows = []
        fh_, fw_ = frame.shape[:2]
        for bbox, landmarks, score in faces:
            x, y, w, h = bbox
            side = int(max(w, h) * REFINE_CROP)
            cx, cy = x + w // 2, y + h // 2
            x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
            x1, y1 = min(fw_, x0 + side), min(fh_, y0 + side)
            refined = None
            crop = frame[y0:y1, x0:x1]
            if crop.size:
                ch, cw = crop.shape[:2]
                if (cw, ch) != self._last_size:
                    self.detector.setInputSize((cw, ch))
                    self._last_size = (cw, ch)
                _, found = self.detector.detect(np.ascontiguousarray(crop))
                if found is not None and len(found):
                    best = max(found, key=lambda r: float(r[14]))
                    # Row layout [x, y, w, h, 5 x (lx, ly), score]: shift the box corner and
                    # the landmarks into frame coordinates, leave w and h alone.
                    refined = best.copy()
                    refined[0] += x0
                    refined[1] += y0
                    refined[4:14:2] += x0
                    refined[5:14:2] += y0
            if refined is not None:
                rows.append(refined)
            elif landmarks and len(landmarks) >= 5:
                flat = [float(v) for point in landmarks[:5] for v in point]
                rows.append(np.array([x, y, w, h, *flat, score], dtype=np.float32))
        return self._describe_rows(frame, rows)

    def _describe_rows(self, frame, faces) -> list[FaceDetection]:
        results: list[FaceDetection] = []
        for face_row in faces:
            x, y, fw, fh = face_row[:4].astype(int)
            aligned = self.recognizer.alignCrop(frame, face_row)
            embedding = self.recognizer.feature(aligned)

            # Sharpness is measured on the ALIGNED CROP rather than the raw frame region: that
            # is the actual image SFace saw, already rotated and scaled to a fixed size, so the
            # number means the same thing for a face near the camera and one across the room.
            # Measuring the raw box instead would mostly measure how big the face was.
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned

            # YuNet's row layout: [x, y, w, h, 5x(lx, ly), score] - 15 values. The landmarks and
            # the score were both being thrown away.
            landmarks = (
                tuple((int(face_row[4 + i * 2]), int(face_row[5 + i * 2])) for i in range(5))
                if len(face_row) >= 14
                else ()
            )
            score = float(face_row[14]) if len(face_row) >= 15 else 1.0

            results.append(
                FaceDetection(
                    bbox=(int(x), int(y), int(fw), int(fh)),
                    embedding=embedding,
                    score=score,
                    sharpness=blur_variance(gray),
                    landmarks=landmarks,
                )
            )

        return results

    def similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Raw cosine similarity, higher = more alike.

        Exposed alongside is_match() because a boolean is not enough to pick between several
        candidates: with more than one person enrolled, "does this clear the threshold" cannot
        distinguish a borderline match from a certain one, and answering that question is the
        difference between greeting the right student and the wrong one. See
        FaceMemoryStore.find_match().
        """
        return float(
            self.recognizer.match(embedding_a, embedding_b, cv2.FaceRecognizerSF_FR_COSINE)
        )

    def is_match(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> bool:
        """Matches FaceMemoryStore's is_match(embedding, rec.embedding) signature - pass this
        directly as FaceMemoryStore(is_match=embedder.is_match)."""
        return self.similarity(embedding_a, embedding_b) >= COSINE_MATCH_THRESHOLD


if __name__ == "__main__":
    # --- quality gating (no camera, no models needed) -----------------------------------------
    # Sharpness first: the measure has to actually separate a sharp image from a blurred one, or
    # the whole gate is decoration.
    rng = np.random.default_rng(0)
    sharp = (rng.integers(0, 2, (112, 112)) * 255).astype(np.uint8)  # hard black/white edges
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)
    sharp_v, blurred_v = blur_variance(sharp), blur_variance(blurred)
    print(f"blur_variance: sharp={sharp_v:.0f}, blurred={blurred_v:.0f}")
    assert sharp_v > blurred_v * 3, "the sharpness measure cannot tell blur from detail"
    assert blur_variance(np.zeros((0, 0), dtype=np.uint8)) == 0.0

    def _face(width=120, sharpness=500.0, score=0.99):
        return FaceDetection(
            bbox=(0, 0, width, width),
            embedding=np.zeros(128, dtype=np.float32),
            score=score,
            sharpness=sharpness,
        )

    good = _face()
    assert good.is_usable and good.is_enrollable and good.quality_reason() == ""

    # Each rejection reason has to actually reject, and to say WHY - a face silently ignored is
    # indistinguishable from a recognizer that simply does not work.
    tiny = _face(width=40)
    assert not tiny.is_usable and "too small" in tiny.quality_reason()

    fuzzy = _face(sharpness=10.0)
    assert not fuzzy.is_usable and "too blurry" in fuzzy.quality_reason()

    unsure = _face(score=0.5)
    assert not unsure.is_usable and "low detector confidence" in unsure.quality_reason()

    # Enrolling is deliberately stricter than matching: a stored sample is compared against for
    # months, so a face that is fine to RECOGNIZE from can still be too poor to LEARN from.
    borderline = _face(width=70, sharpness=50.0, score=0.95)
    assert borderline.is_usable, "a decent face was rejected for matching"
    assert not borderline.is_enrollable, (
        "a borderline face was accepted as a permanent reference sample - that damage does not "
        "wash out, it makes every future comparison against that person worse"
    )
    print(
        "OK: quality gating separates usable faces from unusable ones, holds enrollment to a "
        "higher bar, and reports why."
    )

    print("\nLoading YuNet + SFace models...")
    try:
        embedder = SFaceEmbedder()
    except FileNotFoundError as exc:
        print(f"SKIPPED: {exc}")
        raise SystemExit(1) from None
    print("OK: models loaded successfully. Run face_memory_live.py for the full webcam demo.")
