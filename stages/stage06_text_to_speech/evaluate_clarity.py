"""
Stage 06 clarity evaluation - STAGES.md only specifies a latency test for this stage
(Time-To-First-Audio), but "clear" needs its own objective measure, same philosophy as Stage 04's
evaluate_wer.py: don't eyeball it, measure it. This closes the loop with infrastructure already
built: synthesize known text with Piper, transcribe the resulting audio with a real Whisper
model, and compute the Word Error Rate between the original text and what Whisper heard. A low
round-trip WER means the synthesized speech is intelligible enough for a real speech-recognition
model to transcribe correctly - a reasonable objective proxy for "a human would understand this
clearly" too.

TEST_SENTENCES deliberately includes floor numbers - digit pronunciation is exactly the kind of
thing that's easy to get wrong, and directly relevant to this robot's real answers (Stage 05's
grounded answers are full of floor numbers - no room numbers, per the current building
directory).

Install first: needs this stage's requirements.txt AND stage04's (faster-whisper, jiwer) in the
same venv - already true if you've set up stage04.
Run: python evaluate_clarity.py
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

from tts_speaker_live import LiveTTSSpeaker, VOICE_NAME, _sanitize_for_speech, _split_sentences

# Numbers written as words ("fourth", not "4th") - Piper/espeak-ng correctly SAYS "4th" as
# "fourth", but comparing spoken "fourth" against a written reference of "4th" would count as a
# word error despite the pronunciation being completely correct. Writing the reference the way
# it's actually meant to be spoken keeps the WER comparison fair (see the `normalize` transform
# in main() below for the other half of this fix - case/punctuation).
TEST_SENTENCES = [
    "Hi there! How can I help you today?",
    "The Computer Science department is on the fourth floor.",
    "Take the elevator to the fourth floor, then it's the second door on your right.",
    "I'm sorry, I don't have that information. You can check the university's official website.",
]


def _synthesize_to_wav(speaker: LiveTTSSpeaker, text: str, out_path: Path) -> bool:
    """Synthesizes without playing it out loud - writes a real WAV file instead of streaming to
    the speaker, since this is a scoring pass, not a listening demo. Returns False if the text
    produced no audio at all (shouldn't happen for real sentences, but guards against it)."""
    sentences = _split_sentences(_sanitize_for_speech(text))
    wav_file: wave.Wave_write | None = None
    try:
        for sentence in sentences:
            for chunk in speaker.voice.synthesize(sentence):
                if wav_file is None:
                    wav_file = wave.open(str(out_path), "wb")
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setframerate(chunk.sample_rate)
                wav_file.writeframes(chunk.audio_int16_bytes)
    finally:
        if wav_file is not None:
            wav_file.close()
    return wav_file is not None


def main() -> None:
    try:
        import jiwer
        from faster_whisper import WhisperModel
    except ImportError:
        print("Missing faster-whisper/jiwer - these are stage04's dependencies, needed here "
              "too to actually score clarity. Run: pip install -r "
              "..\\stage04_speech_to_text\\requirements.txt")
        return

    # jiwer's wer() does NO normalization by default - "Hi there!" vs "Hi there," or
    # "Computer" vs "computer" score as full word errors even though the actual words match.
    # Case and punctuation don't affect intelligibility, so strip them before comparing -
    # otherwise this eval measures transcription formatting, not clarity.
    normalize = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    print(f"Loading Piper voice '{VOICE_NAME}' + Whisper STT (pulls in both stages' models)...")
    speaker = LiveTTSSpeaker()
    stt_model = WhisperModel("small.en", device="cpu", compute_type="int8")

    print(f"\n{'sentence':<70} {'WER':>6}")
    print("-" * 90)

    all_wer: list[float] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, text in enumerate(TEST_SENTENCES):
            wav_path = Path(tmp_dir) / f"sample_{i}.wav"
            if not _synthesize_to_wav(speaker, text, wav_path):
                print(f"[skip] {text!r} produced no audio")
                continue

            segments, _info = stt_model.transcribe(
                str(wav_path), language="en", condition_on_previous_text=False,
            )
            hypothesis = " ".join(seg.text.strip() for seg in segments).strip()

            sample_wer = (
                jiwer.wer(text, hypothesis, reference_transform=normalize,
                          hypothesis_transform=normalize)
                if hypothesis else 1.0
            )
            all_wer.append(sample_wer)
            print(f"{text[:68]:<70} {sample_wer:>5.1%}")
            print(f"  heard: {hypothesis!r}\n")

    if not all_wer:
        print("No samples scored - nothing to report.")
        return

    print("-" * 90)
    print(f"{'average round-trip WER':<70} {sum(all_wer) / len(all_wer):>5.1%}")
    print(
        "\nA low WER here means Piper's output is intelligible enough for a real STT model to "
        "transcribe correctly - the same 'measure, don't eyeball it' philosophy as Stage 04's "
        "evaluate_wer.py. High WER specifically on the numbers sentence points at digit "
        "pronunciation as the thing to fix; high WER across everything points at voice quality "
        "or the recording pipeline itself."
    )


if __name__ == "__main__":
    main()
