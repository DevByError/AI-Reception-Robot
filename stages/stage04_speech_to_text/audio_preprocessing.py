"""
Stage 04 add-on - optional noise suppression + normalization, applied to a full recorded
utterance right before transcription (not per-frame in real time - spectral gating needs more
signal than one 30ms frame to build a noise profile from).

Important, and easy to get backwards: aggressive noise removal can destroy speech information
just as easily as it removes noise - quiet consonants and breathy sounds look a lot like noise
to a spectral gate. That's why this is toggleable (see DENOISE_ENABLED in
stt_listener_live.py's WhisperSTTEngine) rather than always-on - it should be A/B tested against
evaluate_wer.py's WER numbers on your own real recordings, not assumed to help.
"""

from __future__ import annotations

import numpy as np

# Below this many samples there isn't enough signal for noisereduce to estimate a reliable noise
# profile from - just skip it rather than risk mangling an already-short clip.
MIN_SAMPLES_FOR_DENOISE_S = 0.25


def denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Stationary spectral-gating noise reduction (noisereduce) - estimates a noise profile
    automatically from the signal itself, then attenuates that profile across the whole clip.
    Falls back to returning the audio unchanged if noisereduce isn't installed, or if anything
    about the reduction itself fails (never let a preprocessing step crash the pipeline)."""
    if len(audio) < sample_rate * MIN_SAMPLES_FOR_DENOISE_S:
        return audio
    try:
        import noisereduce as nr
    except ImportError:
        return audio
    try:
        return nr.reduce_noise(y=audio, sr=sample_rate, stationary=True)
    except Exception:
        return audio


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalizes so a quiet speaker/mic isn't systematically under-amplified going into
    VAD/ASR. This doesn't remove noise - it just brings the whole clip (signal and noise both)
    up to a consistent level, which is a different, complementary fix from denoise() above."""
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        return audio
    return audio * (target_peak / peak)


if __name__ == "__main__":
    # Quick self-check with a synthetic signal, since there's no real audio in this file's own
    # test - confirms the functions run and don't blow up shape/dtype, not that they sound good
    # (that needs real ears - use evaluate_wer.py with DENOISE_ENABLED flipped for the real test).
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # quiet fake "speech"
    noise = (0.02 * np.random.randn(sr)).astype(np.float32)         # fake background hiss
    noisy = tone + noise

    cleaned = denoise(noisy, sr)
    normalized = normalize(cleaned)

    assert cleaned.shape == noisy.shape
    assert normalized.shape == noisy.shape
    assert np.max(np.abs(normalized)) <= 0.9 + 1e-6
    print("OK: denoise()/normalize() run without error and preserve shape - real quality "
          "comparison needs evaluate_wer.py against your own recordings.")
