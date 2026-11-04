"""
Stage 05 retrieval evaluation - same "measure, don't eyeball it" discipline as Stage 04's
evaluate_wer.py and Stage 06's evaluate_clarity.py, applied to knowledge_base.py's
SIMILARITY_THRESHOLDS.

Why this needs its own harness rather than just trusting a picked number: search() has two ways
to be wrong that pull in opposite directions -
  - too LOW a threshold -> irrelevant chunks get returned as "relevant" (e.g. a cafeteria
    question matching a department chunk) -> handle_question() wrongly grounds an answer instead
    of correctly falling through to the university-related classifier / apology.
  - too HIGH a threshold -> genuinely relevant chunks get rejected as "no match" -> a real
    question about, say, the nursing department gets treated as out-of-scope.
There's no way to know which side of that a given number lands on without testing it against
real query/chunk pairs - so this script does exactly that, using the real building directory
(SAMPLE_CHUNKS from dialogue_manager_live.py) and a mix of paraphrased positive queries (should
match) and genuinely unrelated negative queries (should not match).

Also covers a second, different failure mode: even when the directory chunk
IS above the relevance threshold, it can still lose the #1 ranking spot to an unrelated but
keyword-dense chunk from Stage 07's real scraped corpus - a threshold can't fix that, only
KnowledgeChunk.boost can (see DIRECTORY_BOOST in dialogue_manager_live.py). ADVERSARIAL_CHUNKS
below are stand-ins for the real scraped chunks that actually won in live testing, added to the
same knowledge base as SAMPLE_CHUNKS so this script exercises that ranking contest directly
instead of only ever testing the directory in isolation (which is exactly why this class of bug
shipped unnoticed the first time).

Measures whichever embedding backend is active (LUC_EMBEDDING_BACKEND), and its result is only
valid for that model - thresholds do not transfer between models, which is why
SIMILARITY_THRESHOLDS in knowledge_base.py is keyed by model name. Run it once per backend:

    python evaluate_retrieval.py                                    # sentence-transformers
    $env:LUC_EMBEDDING_BACKEND="ollama"; python evaluate_retrieval.py   # Ollama embeddings

Then put the suggested value into SIMILARITY_THRESHOLDS under that model's name - and if the
adversarial check reports any losses, raise DIRECTORY_BOOST in dialogue_manager_live.py and
re-run.
"""

from __future__ import annotations

import logging
import sys

from dialogue_manager_live import SAMPLE_CHUNKS
from knowledge_base import KnowledgeBase, KnowledgeChunk

# Stand-ins for real stage07-scraped lincoln.edu.my content, added 2026-08-19 after a real
# regression this harness completely missed: it only ever tested SAMPLE_CHUNKS in isolation, so
# it never reproduced the actual production KB (directory + 14,000+ real scraped chunks) where
# "Where is the Engineering Department?" started matching a real page-title chunk ("Civil
# Engineering Department" - no location info) ahead of the correct directory sentence, and
# "where is the mosque for women" separately matched an unrelated real course-title chunk
# ("Islam: Women Rights, Culture and Civilization Perspectives") in the same live run. These are
# deliberately title/heading-shaped text with no boost (source_url tagged "adversarial-test:" -
# not a real page) - close enough in wording to plausibly out-score a plain directory sentence on
# raw cosine similarity, same as the real ones did. If DIRECTORY_BOOST (dialogue_manager_live.py)
# is ever changed, re-run this file and confirm the directory chunk still wins each case below
# despite these sitting in the same knowledge base.
ADVERSARIAL_CHUNKS = [
    KnowledgeChunk(
        chunk_id="adversarial-civil-engineering",
        text="Civil Engineering Department",
        source_url="adversarial-test:civil-engineering-title",
        last_updated=0.0,
    ),
    KnowledgeChunk(
        chunk_id="adversarial-cs-faculty",
        text="Faculty of Computer Science and Multimedia",
        source_url="adversarial-test:cs-faculty-title",
        last_updated=0.0,
    ),
    KnowledgeChunk(
        chunk_id="adversarial-nursing-programmes",
        text="School of Nursing Programmes",
        source_url="adversarial-test:nursing-programmes-title",
        last_updated=0.0,
    ),
    KnowledgeChunk(
        chunk_id="adversarial-management-programme",
        text="Management Studies Postgraduate Programme",
        source_url="adversarial-test:management-programme-title",
        last_updated=0.0,
    ),
    # The exact real chunk that won over mosque-women in a live run (2026-08-17 session) - a
    # course title sharing the word "women" but otherwise unrelated to building locations at all.
    KnowledgeChunk(
        chunk_id="adversarial-islam-women-course",
        text="Islam: Women Rights, Culture and Civilization Perspectives",
        source_url="adversarial-test:islam-women-course-title",
        last_updated=0.0,
    ),
]

# (query, expected_chunk_id) - None means "should NOT match anything" (genuinely unrelated to the
# knowledge base, the same kind of question that should fall through to GENERAL_CHAT/UNKNOWN).
# Several of these are deliberately paraphrased rather than copy-pasted from the chunk text
# itself (e.g. "where can men pray" vs. the chunk's actual wording "the men's mosque is on the
# first floor") - a real embedding model should still get these right; a word-overlap approach
# never could, since there's barely any shared vocabulary.
TEST_CASES: list[tuple[str, str | None]] = [
    ("where is the computer science department", "dept-cs"),
    ("what floor is the CS department on", "dept-cs"),
    ("where is the engineering department", "dept-engineering"),
    ("where is the management department", "dept-management"),
    ("where is the nursing department", "dept-nursing"),
    ("where is the library", "library"),
    ("where is the admissions office", "dept-admissions"),
    ("where is the student affairs department", "dept-student-affairs"),
    ("where is the mosque for women", "mosque-women"),
    ("where can men pray", "mosque-men"),
    ("when does freshman orientation start", "event-orientation"),
    ("what's the weather like today", None),
    ("what's on the cafeteria menu today", None),
    ("tell me a joke", None),
    ("who won the football game last night", None),
    ("what time is it", None),
]

# Candidate thresholds to sweep - fine-grained enough to find a real gap between "relevant" and
# "irrelevant" score clusters, if one exists.
#
# Range widened from 0.20-0.60 to 0.20-1.00 on 2026-09-04, when the Ollama embedding backend was
# added. Different models put their scores in completely different places: all-MiniLM-L6-v2 rates
# an unrelated query around 0.2-0.33 and a relevant one around 0.35-0.65, while nomic-embed-text
# rates that SAME unrelated query around 0.40-0.64 - higher than MiniLM rates a genuinely
# relevant one - and a relevant query around 0.70-0.89. Both separate relevant from irrelevant
# perfectly well; the gap is just somewhere else. With the old 0.60 ceiling this script reported
# "0.60, 4/5 false positives" for nomic and made the model look bad, when the right threshold was
# simply above the range being searched.
CANDIDATE_THRESHOLDS = [round(0.20 + 0.025 * i, 3) for i in range(33)]  # 0.20 .. 1.00


def main() -> None:
    kb = KnowledgeBase()
    # Whatever LUC_EMBEDDING_BACKEND selected. Printed because the measured number this script
    # produces is only valid for THIS model - pasting a nomic-embed-text result into MiniLM's
    # row of SIMILARITY_THRESHOLDS would be worse than not measuring at all.
    current_threshold = kb.similarity_threshold
    print(f"Embedding model: {kb.embedding_model}\n")
    for chunk in SAMPLE_CHUNKS:
        kb.upsert(chunk)
    for chunk in ADVERSARIAL_CHUNKS:
        kb.upsert(chunk)

    # For every test case, get the FULL unfiltered ranking (score_all_detailed, not search) so
    # every candidate threshold can be evaluated against the same underlying similarity numbers,
    # instead of re-running the model once per threshold.
    #
    # score_all_detailed gives (raw, ranked, chunk) and this script must use each for the same
    # job search() uses it for, or it measures a threshold the real code never applies: RAW is
    # what gets compared against the threshold, RANKED (raw + boost) is what decides order. Using
    # the boosted score for both - which is what this script did before 2026-09-04 - meant every
    # measured threshold silently had DIRECTORY_BOOST baked into it, and was then applied to
    # Stage 07's unboosted scraped chunks.
    all_scores: list[tuple[str, str | None, list[tuple[float, float, KnowledgeChunk]]]] = [
        (query, expected, kb.score_all_detailed(query)) for query, expected in TEST_CASES
    ]

    print(f"{'query':<45} {'expected':<22} {'top match':<22} {'raw':>7} {'ranked':>7}")
    print("-" * 100)
    adversarial_wins = []
    for query, expected, scores in all_scores:
        top_id = scores[0][2].chunk_id if scores else "(none)"
        top_raw = scores[0][0] if scores else 0.0
        top_ranked = scores[0][1] if scores else 0.0
        print(
            f"{query[:43]:<45} {str(expected):<22} {top_id:<22} {top_raw:>7.3f} "
            f"{top_ranked:>7.3f}"
        )
        if expected is not None and top_id != expected and top_id.startswith("adversarial-"):
            adversarial_wins.append((query, expected, top_id))

    # This is the exact regression class that slipped through before ADVERSARIAL_CHUNKS existed
    # (see its own comment above) - a fake-scraped-content chunk outranking the correct directory
    # answer. Called out on its own, separately from the general threshold sweep below, since
    # it's a ranking (boost) problem, not a threshold problem - a higher/lower threshold
    # can't fix a wrong chunk winning, only whether a match happens at all.
    print()
    if adversarial_wins:
        print(
            "WARNING: an adversarial (fake-scraped-content) chunk outranked the real directory "
            "answer for these queries - DIRECTORY_BOOST (dialogue_manager_live.py) needs to go "
            "up:"
        )
        for query, expected, top_id in adversarial_wins:
            print(f"  {query!r}: expected {expected!r}, got {top_id!r}")
    else:
        print(
            "OK: the directory answer beat every adversarial chunk for every query that "
            "should match it - DIRECTORY_BOOST is currently sufficient against this test set."
        )

    print(
        f"\nSweeping candidate thresholds (current threshold for "
        f"{kb.embedding_model!r} = {current_threshold})...\n"
    )
    print(
        f"{'threshold':>9}  {'correct positives':>18}  "
        f"{'false positives (negatives wrongly matched)':>36}"
    )
    print("-" * 70)

    best_threshold = None
    best_score = -1
    for threshold in CANDIDATE_THRESHOLDS:
        correct_positive = 0
        total_positive = 0
        false_positive = 0
        total_negative = 0
        for _query, expected, scores in all_scores:
            # Exactly what search() does: threshold the RAW similarity, keep the boosted order.
            matched = [c for raw, _, c in scores if raw >= threshold]
            if expected is not None:
                total_positive += 1
                if matched and matched[0].chunk_id == expected:
                    correct_positive += 1
            else:
                total_negative += 1
                if matched:
                    false_positive += 1

        # Simple objective: maximize correct positives, then minimize false positives - a wrong
        # "no match" (false negative) just means an extra fallback question; a false positive
        # means grounding an answer in the wrong fact, which is the worse failure mode given this
        # project's anti-hallucination requirement, so it's weighted more heavily below.
        combined = correct_positive - (false_positive * 2)
        marker = ""
        if combined > best_score:
            best_score = combined
            best_threshold = threshold
            marker = "  <- best so far"
        print(
            f"{threshold:>9.3f}  {correct_positive:>6}/{total_positive:<10}  "
            f"{false_positive:>6}/{total_negative:<10}{marker}"
        )

    print(
        f"\nSuggested threshold for {kb.embedding_model!r}: {best_threshold} "
        f"(currently in use: {current_threshold})"
    )
    if best_threshold != current_threshold:
        print(
            "These differ - put the measured value into SIMILARITY_THRESHOLDS in "
            "knowledge_base.py, keyed by this model:\n"
            f'    "{kb.embedding_model}": {best_threshold},\n'
            "rather than trusting the placeholder value that shipped with the code."
        )
    else:
        print("Matches the current value - no change needed based on this test set.")

    print(
        "\nNote: this test set is deliberately small and hand-written - if real usage surfaces "
        "queries this doesn't cover (or a threshold that works here misbehaves in practice), add "
        "more cases to TEST_CASES above and re-run. Same spirit as Stage 04's evaluate_wer.py "
        "manifest - the test set should grow with real usage, not stay fixed forever."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
