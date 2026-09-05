"""
Stage 04 add-on - domain vocabulary correction.

Whisper is a general-purpose model - it has no way to know "Cudahy" or "Damen" are real campus
building names, or how a specific professor's surname is spelled, unless told. This module
doesn't fix that by training anything: it fuzzy-matches only the words Whisper itself was
already UNSURE about against a supplied vocabulary list, and swaps in the closest match if (and
only if) it's a strong enough match. Words Whisper was already confident about are never
touched - a "correction" there would make things worse, not better.

The vocabulary list is empty by default - there's no real one yet (Stage 07's knowledge base
pipeline doesn't exist yet to source real department/staff/building names from). Wire a real list
in later, e.g.:

    WhisperSTTEngine(vocabulary=knowledge_base.all_known_terms())
"""

from __future__ import annotations

# rapidfuzz score is 0-100 - only accept a fuzzy correction this close to the vocabulary entry;
# otherwise leave the low-confidence word as-is (still gets flagged by the caller) rather than
# confidently swapping in a wrong "correction".
MATCH_SCORE_THRESHOLD = 85


def try_correct(word_text: str, probability: float, vocabulary: list[str],
                 confidence_threshold: float) -> str | None:
    """Returns a corrected word string if `word_text` was low-confidence AND a close-enough
    vocabulary match exists, or None otherwise (either it was already confident, there's no
    vocabulary to check against, or nothing matched closely enough) - callers should keep
    treating a None result as unresolved/flaggable, not as "checked and fine"."""
    if probability >= confidence_threshold or not vocabulary:
        return None

    try:
        from rapidfuzz import fuzz, process, utils
    except ImportError:
        return None

    # processor=utils.default_process lowercases (and strips punctuation) before scoring, so a
    # pure case difference never costs match score. Without this, WRatio's score is proportional
    # to string length, so a case mismatch alone was already silently hurting the odds for any
    # vocabulary entry - but it's fatal for short words specifically: "engineering" vs.
    # "Engineering" still scores 90.9 (11 characters, one differs - barely dents the ratio) and
    # clears MATCH_SCORE_THRESHOLD anyway, but "how" vs. "How" scores only 66.7 (3 characters,
    # the SAME one-character difference costs a third of the string) and fails it outright. This
    # surfaced from a real report (2026-08-12 - "where"/"what"/"how"/etc. kept coming out as
    # "[??]") once short common words were added to stt_listener_live's vocabulary, but it was already a latent
    # risk for any borderline-length proper noun before that - this fixes it for all callers.
    match = process.extractOne(
        word_text.strip(), vocabulary, scorer=fuzz.WRatio, processor=utils.default_process
    )
    if match and match[1] >= MATCH_SCORE_THRESHOLD:
        return match[0]
    return None


if __name__ == "__main__":
    vocabulary = ["Cudahy", "Damen", "Loyola", "Piper Hall"]

    # A low-confidence near-miss should get corrected...
    corrected = try_correct("cudahey", probability=0.3, vocabulary=vocabulary,
                             confidence_threshold=0.5)
    print(f"Low-confidence near-miss 'cudahey' -> {corrected!r}")
    assert corrected == "Cudahy"

    # ...but a word Whisper was already confident about should never be touched, even if it
    # happens to be close to a vocabulary entry.
    unchanged = try_correct("cudahey", probability=0.9, vocabulary=vocabulary,
                             confidence_threshold=0.5)
    print(f"High-confidence 'cudahey' (left alone) -> {unchanged!r}")
    assert unchanged is None

    # A low-confidence word with no real vocabulary match shouldn't be forced into one.
    no_match = try_correct("banana", probability=0.2, vocabulary=vocabulary,
                            confidence_threshold=0.5)
    print(f"Low-confidence 'banana' (no good match) -> {no_match!r}")
    assert no_match is None

    print("\nOK: only low-confidence words get corrected, and only when there's a strong "
          "enough vocabulary match.")
