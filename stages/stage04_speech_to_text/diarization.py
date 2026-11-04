"""
Stage 04 add-on - speaker diarization ("who said what" when more than one person is talking in
the same recording).

NOT wired into stt_listener_live.py's default loop. That pipeline is built for one person
talking to the robot at a time - matching this project's actual design (ARCHITECTURE.md/
README.md treat a group approaching the robot as one shared interaction with one group greeting,
not per-speaker turns). Whether the robot should ever track individual speakers within a group
conversation is a real product decision for Stage 08 (Orchestrator) to make deliberately, not
something to force into the base listening pipeline here. This module exists so that decision
has something real to build on if/when it's made.

Uses resemblyzer (a small, freely-downloadable speaker-embedding model, no account or license
acceptance needed) + agglomerative clustering, instead of the more accurate but heavier
pyannote.audio (which needs a Hugging Face account, accepting gated model terms, and an
HF_TOKEN) - consistent with this project's running preference for no-account/no-compiler-
toolchain dependencies (see stage02's opencv choice over dlib, stage04's webrtcvad-wheels choice
over webrtcvad). Good enough to tell speakers apart for a "who asked this" hint, not
medical-transcription-grade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# resemblyzer's own recommended defaults for reliable per-window speaker embeddings: how long
# each analysis window is, and how much consecutive windows overlap.
WINDOW_S = 1.6
OVERLAP = 0.5


@dataclass
class SpeakerSegment:
    speaker: str  # "speaker_0", "speaker_1", ... - a cluster label, not a real identity
    # (that linkage, if ever wanted, is a Stage 02 face-memory question,
    # not this module's job)
    start_s: float
    end_s: float


def diarize(
    audio: np.ndarray, sample_rate: int, num_speakers: int | None = None
) -> list[SpeakerSegment]:
    """Splits `audio` into overlapping windows, embeds each with resemblyzer, clusters the
    embeddings, and collapses consecutive same-cluster windows into contiguous segments.

    num_speakers: pass the known count if you have it (e.g. Stage 01's live headcount for this
    interaction) - clustering is far more reliable given the real number of speakers than
    guessing it from the audio alone. Left as None, this guesses a small cap instead.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering

    wav = preprocess_wav(audio, source_sr=sample_rate)
    encoder = VoiceEncoder()
    _, embeds, slices = encoder.embed_utterance(
        wav,
        return_partials=True,
        rate=1 / (WINDOW_S * (1 - OVERLAP)),
        min_coverage=0.5,
    )

    if len(embeds) < 2:
        # too short to say anything meaningful about multiple speakers - call it one
        return [SpeakerSegment(speaker="speaker_0", start_s=0.0, end_s=len(audio) / sample_rate)]

    n_clusters = num_speakers or min(len(embeds), 4)  # small default cap without a headcount
    # hint, rather than one cluster per window
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeds)

    segments: list[SpeakerSegment] = []
    for label, sl in zip(labels, slices):
        start_s, end_s = sl.start / sample_rate, sl.stop / sample_rate
        speaker = f"speaker_{label}"
        if segments and segments[-1].speaker == speaker:
            segments[-1] = SpeakerSegment(speaker, segments[-1].start_s, end_s)
        else:
            segments.append(SpeakerSegment(speaker, start_s, end_s))
    return segments


if __name__ == "__main__":
    print(
        "This module has no synthetic self-test - clustering meaningfully needs real distinct "
        "voices, which can't be faked with a sine wave the way stage04's other self-tests do.\n"
        "\n"
        "To try it for real: record a short WAV of two people talking (16kHz mono), then:\n"
        "\n"
        "  import wave, numpy as np\n"
        "  from diarization import diarize\n"
        "\n"
        "  with wave.open('two_people.wav', 'rb') as wf:\n"
        "      audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)\n"
        "      audio = audio.astype(np.float32) / 32768.0\n"
        "  for seg in diarize(audio, sample_rate=16000, num_speakers=2):\n"
        "      print(f'{seg.speaker}: {seg.start_s:.1f}s - {seg.end_s:.1f}s')"
    )
