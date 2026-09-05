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


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray            # 128-d SFace feature vector


class SFaceEmbedder:
    def __init__(self, frame_size: tuple[int, int] = (640, 480)):
        if not DETECTION_MODEL.exists() or not RECOGNITION_MODEL.exists():
            raise FileNotFoundError(
                "Model files missing - run 'python download_models.py' in "
                "stages/stage02_face_recognition first."
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTION_MODEL), "", frame_size, score_threshold=DETECTION_SCORE_THRESHOLD,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNITION_MODEL), "")
        self._last_size = frame_size

    def detect(self, frame) -> list[FaceDetection]:
        h, w = frame.shape[:2]
        if (w, h) != self._last_size:
            self.detector.setInputSize((w, h))
            self._last_size = (w, h)

        _, faces = self.detector.detect(frame)
        results: list[FaceDetection] = []
        if faces is None:
            return results

        for face_row in faces:
            x, y, fw, fh = face_row[:4].astype(int)
            aligned = self.recognizer.alignCrop(frame, face_row)
            embedding = self.recognizer.feature(aligned)
            results.append(FaceDetection(bbox=(int(x), int(y), int(fw), int(fh)),
                                          embedding=embedding))

        return results

    def is_match(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> bool:
        """Matches FaceMemoryStore's is_match(embedding, rec.embedding) signature - pass this
        directly as FaceMemoryStore(is_match=embedder.is_match)."""
        score = self.recognizer.match(embedding_a, embedding_b, cv2.FaceRecognizerSF_FR_COSINE)
        return score >= COSINE_MATCH_THRESHOLD


if __name__ == "__main__":
    print("Loading YuNet + SFace models...")
    embedder = SFaceEmbedder()
    print("OK: models loaded successfully. Run face_memory_live.py for the full webcam demo.")
