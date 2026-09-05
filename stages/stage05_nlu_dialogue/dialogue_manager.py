"""
Stage 05 (part 2) - Understanding & Dialogue (NLU + Dialogue Manager)

Goal: classify the intent of a question (general question about the university / general
conversation about anything else / outside the knowledge base) and generate an answer - grounded
ONLY in retrieved information whenever it's actually about the university (no hallucinating a
university-specific fact), but free to talk about anything else when it's not (per the project's
requirement: the robot should be able to hold a normal conversation, not just answer university
questions).

Intents supported:
  - "general_info": a question the knowledge base actually found something for -> answered,
    strictly grounded in the retrieved chunks. This covers "where is X" questions too - the
    knowledge base's building-directory chunks (see dialogue_manager_live.py's SAMPLE_CHUNKS)
    already state the floor for each department/service, so a floor-only answer comes straight
    out of the normal grounded path. There's deliberately no separate wayfinding/directions stage
    generating step-by-step routes - the project decided a floor number is enough, and a
    dedicated stage for that wasn't worth the added surface area.
  - "general_chat": the knowledge base found nothing, AND it doesn't even look like a
    university-specific question -> answered openly, no grounding constraint (small talk,
    general knowledge, anything)
  - "unknown": the knowledge base found nothing, AND it looks like it SHOULD have been a
    university question -> the generator still gets to try answering (from its own knowledge,
    or a live search - see GeminiAnswerGenerator in dialogue_manager_live.py), with instructions
    to flag it as unverified/not from the university's own data and point to the official
    website for confirmation, rather than refusing outright. The fixed FALLBACK_ANSWER apology
    is now only used as a last resort - when the generator comes back with nothing at all (a
    blank answer) - never a silent/empty response, but no longer a blanket refusal either. This
    was changed 2026-08-17 after real testing showed the previous "always apologize if it sounds
    university-related" rule was too restrictive - a real LLM with search access can often
    genuinely help even with questions the scraped knowledge base doesn't cover.

TODO (real implementation later):
  - classify_intent() only distinguishes "did the knowledge base find something" from "does this
    look university-related" - a real NLU model/entity extractor for more nuanced intents is
    future work, same as before. This file's actual real upgrade is answer generation (see
    AnswerGenerator below) and the general_chat/unknown split, which is the part that needed a
    real language model to do at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from identity import detect_identity_answer
from knowledge_base import KnowledgeBase, KnowledgeChunk


FALLBACK_ANSWER = (
    "I'm sorry, I don't have that information. "
    "You can check the university's official website, "
    "or ask a staff member at the front office for help."
)

# Phrases GROUNDED_SYSTEM_PROMPT's own instruction ("say plainly that you don't have that
# information") tends to produce when a grounded call succeeds but genuinely can't answer from
# the retrieved chunks. Heuristic, same spirit as main_live.py's NEGATIVE_PHRASES/
# AFFIRMATIVE_PHRASES - a phrase list to catch real observed wording, not a strict parser; add to
# this if a real "I don't know" response slips through unrecognized.
#
# Added 2026-08-19: without this, a grounded answer that honestly said "I don't have that
# information" was treated as a final, successful answer - correct in that it didn't hallucinate,
# but it stopped short of trying a live search, which is what an explicit request asked for ("such
# answers should come from the API if we don't have it - read from the website"). See
# handle_question() below for where this is used.
_GROUNDED_NO_INFO_PHRASES = (
    "don't have that information", "do not have that information",
    "don't have information", "do not have information",
    "doesn't contain", "does not contain",
    "doesn't mention", "does not mention",
    "doesn't say", "does not say",
    "don't have access to", "do not have access to",
    "i don't know", "i do not know",
    "not mentioned in", "not provided in",
)


def _looks_like_no_info(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _GROUNDED_NO_INFO_PHRASES)


class Intent(Enum):
    GENERAL_INFO = auto()
    GENERAL_CHAT = auto()
    UNKNOWN = auto()
    IDENTITY = auto()  # "what's your name" / "who made you" - see identity.py


@dataclass
class DialogueResult:
    intent: Intent
    answer: str
    used_chunks: list[KnowledgeChunk]


class AnswerGenerator(Protocol):
    """Two different jobs, deliberately kept separate rather than one free-form "answer this"
    call, because they have opposite rules: one must never go beyond the given text, the other
    is specifically allowed to."""

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        """Called when the knowledge base found relevant chunks. Must answer ONLY from the given
        chunks and say so plainly if they don't actually answer the question - never invent a
        university-specific fact that isn't in the retrieved text.

        May raise on failure (a network/API problem) rather than catch-and-return a fallback
        itself - see handle_question() below for why that responsibility moved to the caller
        (2026-08-19): a single generator-level fallback ("just read the top chunk verbatim") was
        too blunt, since a raw scraped chunk isn't reliably a complete sentence the way the
        curated directory chunks are."""
        ...

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        """Called when the knowledge base found NOTHING for the question, OR (2026-08-19) as a
        second-tier fallback when it found something but generate_grounded_answer() couldn't
        actually answer from it (failed outright, or honestly said it didn't have the info) -
        see handle_question() below. Returns (is_university_related, answer):
          - is_university_related=True: this looks like it SHOULD have been answerable from the
            knowledge base but wasn't. The generator should still try to help (own knowledge, or
            a live search) but flag it as unverified and point to the official website rather
            than stating an unconfirmed university-specific fact as certain - see
            GeminiAnswerGenerator's GENERAL_CLASSIFY_PROMPT in dialogue_manager_live.py for the
            actual instructions given to the model. `answer` can be blank if the generator
            genuinely has nothing - the caller then uses the fixed FALLBACK_ANSWER apology as a
            last resort, never a silent response.
          - is_university_related=False: genuine general conversation/knowledge, unrelated to
            this specific university - `answer` is the real, open response to use.
        """
        ...


class MockAnswerGenerator:
    """Placeholder for this file's own self-test only - no real language understanding, just
    enough to prove the routing logic in handle_question() works. The real implementation is
    GroqAnswerGenerator in dialogue_manager_live.py."""

    # A tiny hint list standing in for what a real classifier would actually reason about -
    # nowhere near real NLU, just enough to demonstrate the UNKNOWN vs. GENERAL_CHAT split below.
    UNIVERSITY_HINT_WORDS = {
        "department", "professor", "class", "course", "campus", "building",
        "office", "faculty", "tuition", "registrar", "dorm", "cafeteria", "menu",
    }

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        return chunks[0].text

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        q = question.lower()
        if any(word in q for word in self.UNIVERSITY_HINT_WORDS):
            return True, ""  # looks university-related but we have no data - caller apologizes
        return False, f"(mock general chat reply to: {question!r})"


class DialogueManager:
    def __init__(self, kb: KnowledgeBase, answer_generator: AnswerGenerator):
        self.kb = kb
        self.answer_generator = answer_generator

    def handle_question(self, question: str) -> DialogueResult:
        # Checked FIRST, before the knowledge base or any LLM call - these are fixed facts about
        # the robot itself (name, creator), not university content, and the explicit requirement
        # was that they answer correctly with no network at all. detect_identity_answer() is a
        # plain offline string lookup (see identity.py) - it can't be slow, can't fail, and can't
        # be affected by the knowledge base or which AnswerGenerator backend is active. Checking
        # here rather than after an empty kb.search() also means it can't accidentally be shadowed
        # by a coincidental knowledge-base match.
        identity_answer = detect_identity_answer(question)
        if identity_answer is not None:
            return DialogueResult(intent=Intent.IDENTITY, answer=identity_answer, used_chunks=[])

        chunks = self.kb.search(question)
        if chunks:
            # Fast path (added 2026-08-19): the best-matching chunk is already a finished,
            # spoken-ready sentence, so speak it as-is instead of paying a real 1-3 second LLM
            # round-trip to rephrase a sentence into itself. See KnowledgeChunk.is_direct_answer
            # in knowledge_base.py for exactly which chunks qualify and why scraped web content
            # deliberately does not. This is a pure latency optimization with no behavior change
            # in what gets said - the LLM was already instructed (GROUNDED_SYSTEM_PROMPT) to
            # answer using only this text, and for a one-sentence directory fact there's nothing
            # left for it to add. It also removes the LLM as a failure point for the robot's most
            # common question type: on 2026-08-19 a Gemini 503 on exactly this path is what made
            # the robot read a raw page title aloud as an answer.
            if chunks[0].is_direct_answer:
                return DialogueResult(
                    intent=Intent.GENERAL_INFO, answer=chunks[0].text, used_chunks=chunks
                )

            # Three-tier fallback for a non-direct-answer chunk (2026-08-19, replacing a single
            # generator-owned fallback). Real bug this fixes: "where is Lincoln University
            # College located" hit a Gemini 503 on the call below, and the old behavior was to
            # silently speak chunks[0].text verbatim - which for a raw scraped chunk can be a
            # page-title fragment ("Lincoln University College"), not an actual answer. Per
            # explicit request, an answer that isn't available locally should come from a live
            # API search (Gemini's Google Search grounding) instead of a guess at reading a
            # fragment aloud:
            #   1. try to answer strictly from the retrieved chunks (the normal, best case)
            #   2. if that fails outright, OR succeeds but honestly admits it doesn't have the
            #      info (_looks_like_no_info) - try a live search instead, same call used for
            #      out-of-KB questions (classify_and_answer_general already does its own
            #      search-then-plain-call retry internally, see dialogue_manager_live.py)
            #   3. only if BOTH of those come back empty - speak the matched chunk verbatim, so
            #      the robot still says something real and on-topic rather than going silent.
            # UniversityOnlyAnswerGenerator (no API key, MODE="university_only") is unaffected:
            # its generate_grounded_answer() never raises and never produces "no info" wording
            # (it just returns the chunk text), and its classify_and_answer_general() always
            # returns blank - so it falls straight through to tier 3, identical to its old
            # single-tier behavior.
            answer = None
            try:
                answer = self.answer_generator.generate_grounded_answer(question, chunks)
            except Exception as exc:
                print(f"[dialogue] grounded-answer generation failed ({exc!r}).")

            if answer and not _looks_like_no_info(answer):
                return DialogueResult(intent=Intent.GENERAL_INFO, answer=answer, used_chunks=chunks)

            print("[dialogue] falling through to a live search instead of the matched chunk "
                  f"verbatim (question={question!r}).")
            try:
                _, general_answer = self.answer_generator.classify_and_answer_general(question)
            except Exception as exc:
                print(f"[dialogue] live-search fallback also failed ({exc!r}).")
                general_answer = ""

            if general_answer:
                return DialogueResult(
                    intent=Intent.GENERAL_INFO, answer=general_answer, used_chunks=chunks
                )

            return DialogueResult(
                intent=Intent.GENERAL_INFO, answer=chunks[0].text, used_chunks=chunks
            )

        is_university_related, general_answer = (
            self.answer_generator.classify_and_answer_general(question)
        )
        intent = Intent.UNKNOWN if is_university_related else Intent.GENERAL_CHAT
        if general_answer:
            # Changed 2026-08-17: previously is_university_related=True always forced the fixed
            # FALLBACK_ANSWER here, even if the generator had a real answer to offer - real
            # testing showed that too restrictive (a question naming the university by name,
            # e.g. asking its ranking, got refused outright even though the LLM could plausibly
            # help). Now the generator's own answer is used whenever it has one - the prompt
            # instructs it to flag unverified Lincoln-specific facts and point to the official
            # website rather than state them as certain, so the "never confidently invent a
            # university fact" guarantee lives in the prompt/generator now, not a hard router
            # rule. The grounded path above (generate_grounded_answer, when the KB DID find
            # something) is unaffected and still 100% restricted to the retrieved text.
            return DialogueResult(intent=intent, answer=general_answer, used_chunks=[])

        # The generator came back with nothing at all - either it genuinely declined (a real
        # LLM recognizing it isn't confident enough to answer, e.g. seen for real 2026-08-17 on
        # "what is the rank of Lincoln University College"), or this backend doesn't attempt
        # general answers in the first place (UniversityOnlyAnswerGenerator). Never leave that
        # as a silent/empty response - use the fixed apology if it looked Lincoln-specific
        # (matches the old behavior for this one case), or a short honest admission otherwise.
        fallback = FALLBACK_ANSWER if is_university_related else "Sorry, I'm not sure about that one."
        return DialogueResult(intent=intent, answer=fallback, used_chunks=[])


if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.upsert(KnowledgeChunk(
        chunk_id="dept-cs",
        text="The Computer Science department is on the 4th floor.",
        source_url="https://university.example.edu/departments/cs",
        last_updated=0.0,
    ))

    dm = DialogueManager(kb, MockAnswerGenerator())

    r1 = dm.handle_question("Where is the computer science department?")
    print(f"'Where is X' question -> intent={r1.intent.name}, answer={r1.answer!r}")
    assert r1.intent == Intent.GENERAL_INFO  # answered straight from the KB (floor-only) - no
                                              # separate wayfinding/directions stage
    assert "4th floor" in r1.answer

    r2 = dm.handle_question("What time does the cafeteria open?")
    print(f"University question, no KB data -> intent={r2.intent.name}, answer={r2.answer!r}")
    assert r2.intent == Intent.UNKNOWN
    assert r2.answer == FALLBACK_ANSWER

    r3 = dm.handle_question("What's your favorite color?")
    print(f"Unrelated general question -> intent={r3.intent.name}, answer={r3.answer!r}")
    assert r3.intent == Intent.GENERAL_CHAT
    assert r3.answer != FALLBACK_ANSWER

    r4 = dm.handle_question("Tell me about the computer science department.")
    print(f"University question, KB has data -> intent={r4.intent.name}, answer={r4.answer!r}")
    assert r4.intent == Intent.GENERAL_INFO
    assert r4.used_chunks

    # --- is_direct_answer fast path (2026-08-19) -------------------------------------------
    # A generator that LOUDLY fails if it's called at all, so "the fast path skipped the LLM" is
    # actually proven rather than assumed - a plain mock returning chunks[0].text would produce
    # an identical answer whether or not the optimization worked, and would silently keep passing
    # if someone later removed the fast path.
    class ExplodingAnswerGenerator:
        def generate_grounded_answer(self, question, chunks):
            raise AssertionError(
                "generate_grounded_answer() was called for a chunk marked is_direct_answer=True "
                "- the fast path in handle_question() is not working, so every directory answer "
                "is paying a real LLM round-trip it doesn't need."
            )

        def classify_and_answer_general(self, question):
            raise AssertionError("classify_and_answer_general() should not be reached here")

    fast_kb = KnowledgeBase()
    fast_kb.upsert(KnowledgeChunk(
        chunk_id="dept-engineering",
        text="The Engineering department is on the third floor.",
        source_url="facilities-directory:engineering",
        last_updated=0.0,
        boost=0.15,
        is_direct_answer=True,
    ))
    # A scraped-style chunk: real page text, NOT a spoken-ready sentence, no flags set. Present
    # so the fast path has something it must NOT mistakenly fire on, and so this test reproduces
    # the real production mix (curated directory + scraped web content) rather than a clean
    # single-chunk KB - the exact blind spot that let the 2026-08-19 ranking regression ship.
    fast_kb.upsert(KnowledgeChunk(
        chunk_id="scraped-civil-eng",
        text="Civil Engineering Department",
        source_url="https://www.lincoln.edu.my/example-scraped-page/",
        last_updated=0.0,
    ))

    r5 = DialogueManager(fast_kb, ExplodingAnswerGenerator()).handle_question(
        "Where is the engineering department?"
    )
    print(f"Direct-answer chunk -> intent={r5.intent.name}, answer={r5.answer!r}")
    assert r5.intent == Intent.GENERAL_INFO
    assert r5.answer == "The Engineering department is on the third floor.", (
        f"expected the directory sentence verbatim, got {r5.answer!r} - if this is the scraped "
        f"'Civil Engineering Department' chunk, the boost-based ranking regressed too"
    )

    # --- identity routing (2026-08-19) -------------------------------------------------------
    # Proves handle_question() actually wires to detect_identity_answer() and short-circuits
    # BEFORE the knowledge base or the LLM - identity.py's own __main__ already exercises the
    # detection logic thoroughly, this is only about the wiring. Uses the same
    # ExplodingAnswerGenerator as above: if identity routing broke and fell through to the normal
    # path, this would fail loudly instead of silently returning a wrong-but-plausible answer.
    # get_identity() rather than the module constants this used to import: the values live in
    # MySQL now (see identity.py), so the assertions below have to ask for whatever the robot
    # actually loaded - hard-coding "Lina" here would turn a legitimate rename into a test
    # failure.
    from identity import get_identity

    me = get_identity()

    r6 = DialogueManager(fast_kb, ExplodingAnswerGenerator()).handle_question(
        "do you have a name?"
    )
    print(f"Identity question -> intent={r6.intent.name}, answer={r6.answer!r}")
    assert r6.intent == Intent.IDENTITY
    assert me.robot_name in r6.answer

    # Also confirm it wins over a knowledge-base chunk that happens to exist for the same query -
    # identity must never be shadowed by a coincidental KB match, since it's checked first.
    identity_kb = KnowledgeBase()
    identity_kb.upsert(KnowledgeChunk(
        chunk_id="decoy",
        text="This chunk should never be reached for an identity question.",
        source_url="test:decoy",
        last_updated=0.0,
        boost=999.0,  # would win any ranking contest, to make the test meaningful
        is_direct_answer=True,
    ))
    r7 = DialogueManager(identity_kb, ExplodingAnswerGenerator()).handle_question(
        "who created you?"
    )
    assert r7.intent == Intent.IDENTITY
    assert me.creator_name in r7.answer
    assert "should never be reached" not in r7.answer

    # --- three-tier grounded-answer fallback (2026-08-19) -----------------------------------
    # Real bug this reproduces: "where is Lincoln University College located" hit a Gemini 503
    # on generate_grounded_answer() and the OLD behavior silently spoke the raw matched chunk
    # ("Lincoln University College" - a title fragment, not an answer). Each generator below
    # controls exactly one tier's outcome, so a passing test here proves handle_question() itself
    # is escalating correctly - not just that some answer came back.
    fallback_kb = KnowledgeBase()
    fallback_kb.upsert(KnowledgeChunk(
        chunk_id="scraped-fragment",
        text="Lincoln University College",  # a real example of an incomplete raw chunk
        source_url="https://www.lincoln.edu.my/example/",
        last_updated=0.0,
    ))

    class RaisesThenSearchGenerator:
        """Tier 1 fails outright (simulates the real 503) - tier 2 (live search) should be used
        instead of tier 3 (the raw chunk)."""

        def generate_grounded_answer(self, question, chunks):
            raise RuntimeError("simulated 503 UNAVAILABLE")

        def classify_and_answer_general(self, question):
            return True, "The main campus is at Wisma Lincoln, Jalan Lima."

    r8 = DialogueManager(fallback_kb, RaisesThenSearchGenerator()).handle_question(
        "where is Lincoln University College located"
    )
    print(f"Tier-1 exception -> intent={r8.intent.name}, answer={r8.answer!r}")
    assert r8.intent == Intent.GENERAL_INFO
    assert r8.answer == "The main campus is at Wisma Lincoln, Jalan Lima.", (
        f"expected the tier-2 live-search answer, got {r8.answer!r} - if this is the raw chunk "
        f"('Lincoln University College'), the fallback regressed to the old single-tier behavior"
    )

    class SucceedsButNoInfoGenerator:
        """Tier 1 succeeds (no exception) but honestly admits it doesn't know - this should
        ALSO escalate to tier 2, not be accepted as a final answer."""

        def generate_grounded_answer(self, question, chunks):
            return "I don't have that information in what I was given."

        def classify_and_answer_general(self, question):
            return True, "The main campus is at Wisma Lincoln, Jalan Lima."

    r9 = DialogueManager(fallback_kb, SucceedsButNoInfoGenerator()).handle_question(
        "where is Lincoln University College located"
    )
    print(f"Tier-1 'no info' answer -> intent={r9.intent.name}, answer={r9.answer!r}")
    assert r9.answer == "The main campus is at Wisma Lincoln, Jalan Lima."

    class EverythingFailsGenerator:
        """Both tiers fail - must fall back to the raw chunk (tier 3) rather than crash or
        return nothing. Worse than ideal, but the "never go silent" guarantee still holds."""

        def generate_grounded_answer(self, question, chunks):
            raise RuntimeError("simulated failure")

        def classify_and_answer_general(self, question):
            raise RuntimeError("simulated failure")

    r10 = DialogueManager(fallback_kb, EverythingFailsGenerator()).handle_question(
        "where is Lincoln University College located"
    )
    print(f"Both tiers failed -> intent={r10.intent.name}, answer={r10.answer!r}")
    assert r10.answer == "Lincoln University College"  # tier 3: the raw chunk, never a crash

    print("\nOK: grounded-info/general-chat/unknown all route correctly, only genuinely "
          "university-shaped unanswered questions get the fixed apology, a directory chunk is "
          "spoken verbatim without ever calling the language model, identity questions about the "
          "robot itself are answered offline before the knowledge base is even searched, and a "
          "non-direct-answer chunk that can't be grounded (failure or honest 'I don't know') "
          "falls through to a live search before ever falling back to a raw fragment.")
