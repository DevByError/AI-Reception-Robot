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

# A SECOND gate, on plain whole-string similarity, applied to whatever WRatio picked.
#
# WRatio is a composite: part of what it scores is partial-ratio, which asks "does the short
# string appear inside the long one" and deliberately does NOT penalise the length difference.
# That is the right behaviour for matching a phrase against a longer phrase, and completely wrong
# here, where a single low-confidence WORD is being replaced. Measured against this project's own
# vocabulary:
#
#     heard "a"     -> "Lina"         WRatio 90.0   <- accepted, and badly wrong
#     heard "Alex"  -> "Alex Tan"  WRatio 90.0   <- accepted, replaced a name with a longer one
#
# Both cleared the 85 threshold purely on containment. The damage is worse than leaving the word
# flagged: a "[??]" tells the rest of the pipeline the word is doubtful (and now prompts the robot
# to ask about it - see _handle_repair in Stage 08), whereas a confident wrong word goes straight
# into knowledge-base retrieval as if the visitor had said it.
#
# fuzz.ratio scores the whole strings against each other, so length mismatch costs. On the same
# vocabulary it separates the two cases cleanly - the real corrections this module exists for
# score 86-100, the containment accidents score 40-53:
#
#     "Enginering" -> "Engineering"   ratio  95.2   (real typo fix, kept)
#     "engineering"-> "Engineering"   ratio 100.0   (case only, kept)
#     "Librari"    -> "Library"       ratio  85.7   (real typo fix, kept)
#     "Alex"       -> "Alex Tan"   ratio  53.3   (rejected)
#     "a"          -> "Lina"          ratio  40.0   (rejected)
MIN_WHOLE_WORD_RATIO = 70


def try_correct(
    word_text: str, probability: float, vocabulary: list[str], confidence_threshold: float
) -> str | None:
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
    if not match or match[1] < MATCH_SCORE_THRESHOLD:
        return None

    # Second gate - see MIN_WHOLE_WORD_RATIO. WRatio chose the candidate; plain ratio decides
    # whether it is actually the same WORD rather than merely contained in it.
    whole = fuzz.ratio(utils.default_process(word_text.strip()), utils.default_process(match[0]))
    if whole < MIN_WHOLE_WORD_RATIO:
        return None
    return match[0]


if __name__ == "__main__":
    vocabulary = ["Cudahy", "Damen", "Loyola", "Piper Hall"]

    # A low-confidence near-miss should get corrected...
    corrected = try_correct(
        "cudahey", probability=0.3, vocabulary=vocabulary, confidence_threshold=0.5
    )
    print(f"Low-confidence near-miss 'cudahey' -> {corrected!r}")
    assert corrected == "Cudahy"

    # ...but a word Whisper was already confident about should never be touched, even if it
    # happens to be close to a vocabulary entry.
    unchanged = try_correct(
        "cudahey", probability=0.9, vocabulary=vocabulary, confidence_threshold=0.5
    )
    print(f"High-confidence 'cudahey' (left alone) -> {unchanged!r}")
    assert unchanged is None

    # A low-confidence word with no real vocabulary match shouldn't be forced into one.
    no_match = try_correct(
        "banana", probability=0.2, vocabulary=vocabulary, confidence_threshold=0.5
    )
    print(f"Low-confidence 'banana' (no good match) -> {no_match!r}")
    assert no_match is None

    # --- containment accidents ---------------------------------------------------
    # A short low-confidence word must never be "corrected" into a longer one that merely
    # CONTAINS it. Both of these scored 90 on WRatio - above MATCH_SCORE_THRESHOLD - purely
    # because of the partial-ratio component, and were being accepted before MIN_WHOLE_WORD_RATIO
    # was added. See that constant's comment for the measurements.
    real_vocab = ["Lincoln University College", "Alex Tan", "Lina", "Engineering", "Library"]

    nickname = try_correct("Alex", probability=0.3, vocabulary=real_vocab, confidence_threshold=0.5)
    print(f"Short name inside a longer one 'Alex' -> {nickname!r}")
    assert nickname is None, (
        f"'Alex' was rewritten as {nickname!r} - a confident wrong word goes straight into "
        f"knowledge-base retrieval, which is worse than leaving it flagged as uncertain"
    )

    stray = try_correct("a", probability=0.2, vocabulary=real_vocab, confidence_threshold=0.5)
    print(f"Stray single letter 'a' -> {stray!r}")
    assert stray is None, f"a stray letter became {stray!r}"

    # ...while the corrections this module actually exists for still happen.
    assert try_correct("Enginering", 0.3, real_vocab, 0.5) == "Engineering"
    assert try_correct("Librari", 0.3, real_vocab, 0.5) == "Library"
    # Case-only differences on short words must still pass - this is the 2026-08-12 fix that
    # processor=default_process was added for, and the new gate must not undo it.
    assert try_correct("engineering", 0.3, real_vocab, 0.5) == "Engineering"
    assert try_correct("how", 0.3, ["How", "Where", "What"], 0.5) == "How"
    print("Real typo/case fixes still applied: 'Enginering', 'Librari', 'engineering', 'how'")

    print(
        "\nOK: only low-confidence words get corrected, only when there's a strong enough "
        "vocabulary match, and never into a longer word that merely contains what was heard."
    )
