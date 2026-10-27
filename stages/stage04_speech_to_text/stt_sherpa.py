"""
Stage 04 - a streaming speech recognizer for very small computers. Added 2026-10-27.

STT_BACKEND=sherpa swaps Whisper for sherpa-onnx's streaming Zipformer (Apache-2.0, ~70MB
int8). The difference is WHEN the work happens: Whisper transcribes the whole utterance after
the visitor stops (0.6-1.6s on a laptop CPU for a short question), while the Zipformer decodes
as the audio arrives, so the text is ready the moment the turn ends and only the last fraction
of a second is left to decode. It also runs comfortably on a Raspberry Pi 5.

The cost is accuracy, and it is not small. Measured on this project's test_samples (see
docs/RESULTS.md): word error rate 41% for the Zipformer vs 18% for Whisper small.en, almost all
of it on the noisy clips. Use it only when Whisper is too slow on the hardware you have.

Same protocol as WhisperSTTEngine (transcribe_stream / last_confidence / set_hotwords), so
listen_for_one_utterance() and the orchestrator do not change. listen_for_one_utterance()
passes the SAME growing list on every partial and on the final call, so this engine feeds only
the frames it has not seen yet - the final call costs a few milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from model_paths import ZIPFORMER_DIR  # noqa: E402

from stt_listener import TranscriptChunk  # noqa: E402

SAMPLE_RATE = 16000

# How strongly the hotwords (the robot's name, "Lincoln", the departments) are preferred.
# Measured on Kokoro voices saying "Nova, what is your name?": greedy decoding heard "Over what
# is your name" every time; modified beam search with these hotwords at 1.5 still did; at 3.0 it
# heard "Nova" for all three voices. Higher than ~3 starts inserting the word where it was not
# said.
HOTWORDS_SCORE = 3.0

# The model's input needs this much audio AFTER the last word before it decodes that word (its
# right context). Measured: with 300ms of real pause, "...engineering offer" came back as
# "...engineering of" and "where is the library" as "where is the lib"; 0.3s of padding fixed
# every case. So during a pause, silence is added to reach the end of the words.
TAIL_PADDING_S = 0.4
# The last 240ms counts as a pause when it is this much quieter than the loudest part of the
# utterance so far (relative, so a noisy lobby still has pauses), or below the absolute floor.
QUIET_FRACTION = 0.2
QUIET_RMS = 0.004


def _ensure_bpe_vocab(model_dir: Path) -> Path | None:
    """Hotword biasing needs the BPE vocabulary as text (bpe.vocab); the model ships only
    bpe.model. Written once from it with sentencepiece (optional - no sentencepiece, no
    hotwords)."""
    vocab = model_dir / "bpe.vocab"
    if vocab.exists():
        return vocab
    model = model_dir / "bpe.model"
    if not model.exists():
        return None
    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(str(model))
        lines = [f"{sp.id_to_piece(i)}\t{sp.get_score(i)}" for i in range(sp.get_piece_size())]
        vocab.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return vocab
    except Exception:  # noqa: BLE001 - optional feature
        return None


def _first(pattern: str, folder: Path) -> str:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} in {folder} - run "
                                f"python stages/stage00_setup/download_models.py")
    return str(matches[0])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


class SherpaSTTEngine:
    def __init__(self, model_dir: Path = ZIPFORMER_DIR, threads: int = 2,
                 hotwords: str | None = None, **_ignored):
        import sherpa_onnx

        model_dir = Path(model_dir)
        self._sherpa = sherpa_onnx
        self._files = dict(
            tokens=_first("tokens.txt", model_dir),
            encoder=_first("encoder*.int8.onnx", model_dir),
            decoder=_first("decoder*.onnx", model_dir),
            joiner=_first("joiner*.int8.onnx", model_dir),
        )
        self._bpe = _ensure_bpe_vocab(model_dir)
        self._threads = threads
        self.hotwords: str | None = None
        self._recognizer = None
        self.set_hotwords(hotwords)
        self.last_confidence = None      # the Zipformer gives no per-word confidence
        self.device = "cpu"
        self._stream = None
        self._first_chunk: bytes | None = None
        self._fed = 0
        self._peak = 0.0

    def set_hotwords(self, hotwords: str | None) -> None:
        """Hotword biasing needs modified beam search and the model's BPE vocabulary; without
        those, decoding stays greedy (fastest) and the hotwords are simply not used."""
        self.hotwords = (hotwords or "").strip() or None
        kwargs = dict(self._files, num_threads=self._threads, sample_rate=SAMPLE_RATE,
                      feature_dim=80, decoding_method="greedy_search")
        if self.hotwords and self._bpe is not None:
            words = "\n".join(w.strip().upper() for w in self.hotwords.split(",") if w.strip())
            hotwords_file = Path(self._files["tokens"]).with_name("_hotwords.txt")
            try:
                hotwords_file.write_text(words, encoding="utf-8")
                kwargs.update(decoding_method="modified_beam_search", max_active_paths=4,
                              hotwords_file=str(hotwords_file), hotwords_score=HOTWORDS_SCORE,
                              modeling_unit="bpe", bpe_vocab=str(self._bpe))
            except OSError:
                pass
        self._recognizer = self._sherpa.OnlineRecognizer.from_transducer(**kwargs)
        self._stream = None

    def transcribe_stream(self, audio_chunks: list[bytes], locale: str) -> list[TranscriptChunk]:
        if not audio_chunks:
            return [TranscriptChunk(text="", is_final=True)]
        same_utterance = (self._stream is not None and audio_chunks[0] is self._first_chunk
                          and len(audio_chunks) >= self._fed)
        if not same_utterance:
            self._stream = self._recognizer.create_stream()
            self._first_chunk = audio_chunks[0]
            self._fed = 0
            self._peak = 0.0
        fresh = audio_chunks[self._fed:]
        self._fed = len(audio_chunks)
        if fresh:
            for chunk in fresh:   # the loudest 30ms so far = this utterance's speech level
                self._peak = max(self._peak, _rms(np.frombuffer(chunk, np.int16) / 32768.0))
            samples = np.frombuffer(b"".join(fresh), dtype=np.int16).astype(np.float32) / 32768.0
            self._stream.accept_waveform(SAMPLE_RATE, samples)
            # In a pause, pad to the model's right context so the LAST word is decoded now,
            # not after the next word arrives (TAIL_PADDING_S). Only in a pause: silence added
            # to silence changes nothing if they carry on, silence inserted mid-word would.
            tail = np.frombuffer(b"".join(audio_chunks[-8:]), dtype=np.int16) / 32768.0
            if tail.size and _rms(tail) < max(QUIET_RMS, QUIET_FRACTION * self._peak):
                self._stream.accept_waveform(
                    SAMPLE_RATE, np.zeros(int(TAIL_PADDING_S * SAMPLE_RATE), np.float32))
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream)
        text = text if isinstance(text, str) else getattr(text, "text", "")
        text = text.strip().lower()
        return [TranscriptChunk(text=text[:1].upper() + text[1:], is_final=True)]


if __name__ == "__main__":
    import soundfile as sf

    engine = SherpaSTTEngine()
    for path in sys.argv[1:]:
        data, rate = sf.read(path, dtype="int16")
        data = data[:, 0] if data.ndim > 1 else data
        frames = [data[i:i + 480].tobytes() for i in range(0, data.size, 480)]
        frames += [bytes(960)] * 10
        print(engine.transcribe_stream(frames, "en-US")[0].text, "|", path)
