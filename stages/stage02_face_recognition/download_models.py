"""
Stage 02 - one-time download of the YuNet (face detection) + SFace (face recognition) ONNX
models from opencv/opencv_zoo (Apache-2.0). These are Git-LFS files on GitHub, so we use the
media.githubusercontent.com endpoint (not raw.githubusercontent.com, which only serves a small
LFS pointer stub, not the actual binary).

Run: python download_models.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent

# (filename, download URL, expected sha256) - hashes confirmed directly against the LFS pointer
# metadata for these exact files in opencv/opencv_zoo, so a mismatch means a corrupted/partial
# download, not an intentionally different file.
MODELS = [
    (
        "face_detection_yunet_2023mar.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    (
        "face_recognition_sface_2021dec.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(filename: str, url: str, expected_sha256: str) -> None:
    dest = MODELS_DIR / filename
    if dest.exists() and sha256_of(dest) == expected_sha256:
        print(f"OK  {filename} already present and verified.")
        return

    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, dest)

    actual = sha256_of(dest)
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {filename}: expected {expected_sha256}, got {actual}. "
            f"Download was corrupted or incomplete - deleted, re-run this script."
        )
    print(f"OK  {filename} downloaded and verified ({dest.stat().st_size:,} bytes).")


if __name__ == "__main__":
    for filename, url, expected_sha256 in MODELS:
        download(filename, url, expected_sha256)
    print("\nAll models ready.")
