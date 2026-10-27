"""
Stage 00 - one command to fetch every model the 2026-10-27 upgrade added, into models/.

    python stages/stage00_setup/download_models.py            # everything (~370MB download)
    python stages/stage00_setup/download_models.py --list     # what is there / missing
    python stages/stage00_setup/download_models.py --only kokoro,zipformer

What it fetches (all free licences, all run on a CPU):

  kokoro      Kokoro-82M voice, fp16 ONNX + the voice pack           Apache-2.0   ~200MB
  zipformer   sherpa-onnx streaming English recognizer (int8)        Apache-2.0    ~70MB*
  smartturn   Pipecat smart-turn v3.2 end-of-turn model              BSD-2-Clause   ~9MB
  minilm      all-MiniLM-L6-v2 sentence embeddings as ONNX           Apache-2.0    ~90MB
  wakeword    openWakeWord feature models + the stock "hey jarvis"   Apache-2.0     ~4MB

  * the archive is ~300MB; only the int8 files the robot uses are kept.

Every file is checked against the SHA-256 of the exact file the tests were run with, so a
corrupted or changed download is caught here and not as a strange error at startup.

The face models (YuNet/SFace) and the Piper voice already ship inside stages/. Whisper
downloads itself the first time it runs (it needs huggingface.co to be reachable once).

Standard library only, like the rest of stage00.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from model_paths import MODELS_DIR  # noqa: E402

KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
ZIPFORMER = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
             "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2")
MINILM = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
PIPECAT_WHEEL = ("https://files.pythonhosted.org/packages/38/31/"
                 "29c72ca8d3a16b59fd12580fcb99a7cef641d9ee07fe7b0321c1b65ca3cf/"
                 "pipecat_ai-1.14.0-py3-none-any.whl")

ZF = "sherpa-onnx-streaming-zipformer-en-2023-06-26"

# group -> list of (file under models/, source, member inside the archive or None, sha256)
MODELS: dict[str, list[tuple[str, str, str | None, str]]] = {
    "kokoro": [
        ("kokoro/kokoro-v1.0.fp16.onnx", KOKORO + "kokoro-v1.0.fp16.onnx", None,
         "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204"),
        ("kokoro/voices-v1.0.bin", KOKORO + "voices-v1.0.bin", None,
         "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
    ],
    "zipformer": [
        (f"{ZF}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx", ZIPFORMER,
         f"{ZF}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
         "5022b2eca5b19d1bc104fcf33e26bc32604b7df553cd2e1f62e31dc7b05e9c87"),
        (f"{ZF}/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx", ZIPFORMER,
         f"{ZF}/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
         "780c63ee94c7cfa314211172e5d09b406c0da2beab5c40ea2f54cc95670b76a5"),
        (f"{ZF}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx", ZIPFORMER,
         f"{ZF}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
         "abd5e30f3f16fc510605c6029dba33f10e4386bd75c5bdc30cf94076864db10d"),
        (f"{ZF}/tokens.txt", ZIPFORMER, f"{ZF}/tokens.txt",
         "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb"),
        (f"{ZF}/bpe.model", ZIPFORMER, f"{ZF}/bpe.model",
         "c53433de083c4a6ad12d034550ef22de68cec62c4f58932a7b6b8b2f1e743fa5"),
    ],
    "smartturn": [
        ("smart-turn/smart-turn-v3.2-cpu.onnx", PIPECAT_WHEEL,
         "pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx",
         "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"),
    ],
    "minilm": [
        (f"all-MiniLM-L6-v2-onnx/{name}", MINILM, f"onnx/{name}", sha)
        for name, sha in (
            ("model.onnx", "4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6"),
            ("tokenizer.json", "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
            ("vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
            ("config.json", "b567c7d5a55b636c95186aaf993f9a8920842b7e05a9e703e68b23cab2c3a670"),
            ("tokenizer_config.json",
             "7702051bbc4953b94d47fa1d61b42ed4cbb3c71b501a8dd7183a823f8bea1f20"),
            ("special_tokens_map.json",
             "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"),
        )
    ],
    "wakeword": [
        ("openwakeword/melspectrogram.onnx", OWW + "melspectrogram.onnx", None,
         "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"),
        ("openwakeword/embedding_model.onnx", OWW + "embedding_model.onnx", None,
         "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"),
        ("openwakeword/hey_jarvis_v0.1.onnx", OWW + "hey_jarvis_v0.1.onnx", None,
         "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"),
    ],
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    """Downloads into memory with a progress line (the largest file is ~200MB)."""
    name = url.rsplit("/", 1)[-1]
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            buf.write(block)
            done += len(block)
            if total:
                print(f"\r  {name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        return buf.getvalue()


def _extract(blob: bytes, url: str, member: str) -> bytes:
    if url.endswith(".whl") or url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            return zf.read(member)
    mode = "r:bz2" if url.endswith(".bz2") else "r:gz"
    with tarfile.open(fileobj=io.BytesIO(blob), mode=mode) as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{member} is not in {url}")
        return handle.read()


def status(dest: Path, groups: list[str]) -> dict[str, list[tuple[str, bool]]]:
    out = {}
    for group in groups:
        out[group] = [(rel, (dest / rel).exists()) for rel, _u, _m, _s in MODELS[group]]
    return out


def download(groups: list[str], dest: Path = MODELS_DIR, verify_existing: bool = True) -> bool:
    ok = True
    archives: dict[str, bytes] = {}
    for group in groups:
        print(f"[{group}]")
        for rel, url, member, sha in MODELS[group]:
            target = dest / rel
            if target.exists() and (not verify_existing or sha256_of(target) == sha):
                print(f"  OK       {rel}")
                continue
            try:
                if member is None:
                    data = _fetch(url)
                else:
                    if url not in archives:
                        archives[url] = _fetch(url)
                    data = _extract(archives[url], url, member)
            except Exception as exc:  # noqa: BLE001 - report every failure, keep going
                print(f"  FAILED   {rel}: {exc}")
                ok = False
                continue
            actual = hashlib.sha256(data).hexdigest()
            if actual != sha:
                print(f"  FAILED   {rel}: checksum {actual[:12]}... does not match - the "
                      f"download was corrupted or the file changed upstream. Not saved.")
                ok = False
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                tmp.write(data)
            shutil.move(tmp.name, target)
            print(f"  saved    {rel} ({len(data) / 1e6:.1f} MB)")
        archives.clear()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help=f"comma-separated groups: {','.join(MODELS)}")
    parser.add_argument("--list", action="store_true", help="show what is present and stop")
    parser.add_argument("--dest", type=Path, default=MODELS_DIR, help="models folder")
    args = parser.parse_args()
    groups = list(MODELS) if not args.only else [g.strip() for g in args.only.split(",")]
    unknown = [g for g in groups if g not in MODELS]
    if unknown:
        parser.error(f"unknown group(s) {unknown}; choose from {list(MODELS)}")
    if args.list:
        for group, files in status(args.dest, groups).items():
            for rel, present in files:
                print(f"{'OK     ' if present else 'missing'}  [{group}] {rel}")
        return 0
    print(f"Downloading into {args.dest}")
    ok = download(groups, args.dest)
    print("\nAll models are in place." if ok else
          "\nSome downloads failed (see above) - run this again; finished files are kept.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
