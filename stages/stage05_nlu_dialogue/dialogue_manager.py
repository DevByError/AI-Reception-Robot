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
from typing import Iterator, Protocol

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


@dataclass
class StreamingDialogueResult:
    """What handle_question_stream() returns: the routing decision up front, and the answer text
    as an iterator that is still being produced.

    `intent` and `used_chunks` are known BEFORE a single word is spoken - which is what lets the
    caller log the route and start playback at the same moment. `pieces` yields the answer in
    fragments; `is_streaming` says whether those fragments are arriving live from a model or are
    just one already-finished string handed back in the same shape (the identity answer, the
    direct-answer fast path, general conversation), so a caller can log the difference without
    having to care about it otherwise.
    """
    intent: Intent
    used_chunks: list[KnowledgeChunk]
    pieces: "Iterator[str]"
    is_streaming: bool


# How much of a streamed answer to hold back before speaking any of it - see
# DialogueManager._stream_grounded(). The first sentence has to be complete and inspected before
# it can be trusted, because that is where an honest "I don't have that information" appears, and
# that answer must trigger the live-search fallback instead of being spoken (the exact bug
# _GROUNDED_NO_INFO_PHRASES was added for on 2026-08-19). Everything after the first sentence
# streams straight through with no buffering at all.
_FIRST_SENTENCE_MAX_CHARS = 400  # a model that never emits a terminator can't stall the answer


# --- follow-up questions within one visit -------------------------------------------------------
# Until 2026-09-05 every question was answered in complete isolation: handle_question() took a
# string and nothing else. That is fine for the first question of a visit and wrong for every one
# after it, because the second thing a person says is almost never self-contained:
#
#     "Tell me about the Master of Computer Science."   -> answered
#     "How long is it?"                                 -> "it" means nothing; retrieval returns
#                                                          whatever "how long is it" happens to
#                                                          match, and the answer is unrelated
#
# Two separate things need the history, and they need it differently:
#
#   RETRIEVAL needs a self-contained query, because the knowledge base matches on meaning and
#   "how long is it" carries none. _contextualize() below rewrites a follow-up into something
#   searchable by prepending what was last asked about.
#
#   THE MODEL needs the actual dialogue, so it can resolve the pronoun in its own answer and not
#   repeat itself. That goes through as conversation turns (see _history_block).
#
# Deliberately NOT a coreference model: this is a receptionist answering three or four questions,
# and prepending the previous question is both cheap and enough. The failure mode is a slightly
# over-broad search, which the similarity threshold already handles.

# Openers that mean "this refers to what we were just talking about". Checked as prefixes on the
# cleaned question, so "and how long is it" and "what about the fees" both match.
_FOLLOW_UP_PREFIXES = (
    "and ", "what about", "how about", "and what", "and how", "ok what", "okay what",
    "so what", "so how", "then what", "then how", "also",
)

# Bare pronouns/determiners standing in for the previous subject. Only meaningful when the
# question is SHORT - "is it accredited" is a follow-up, "is it true that Malaysia has..." is not.
_FOLLOW_UP_PRONOUNS = frozenset({"it", "that", "this", "they", "them", "those", "these", "one"})
_FOLLOW_UP_MAX_WORDS = 8


def looks_like_follow_up(question: str) -> bool:
    """Whether this question only makes sense in the context of the previous one."""
    cleaned = " ".join(question.lower().replace("?", " ").replace(",", " ").split())
    if not cleaned:
        return False
    if cleaned.startswith(_FOLLOW_UP_PREFIXES):
        return True
    words = cleaned.split()
    if len(words) <= _FOLLOW_UP_MAX_WORDS and _FOLLOW_UP_PRONOUNS.intersection(words):
        return True
    return False


def _contextualize(question: str, history: list[tuple[str, str]] | None) -> str:
    """The string to SEARCH the knowledge base with - the question itself, unless it is a
    follow-up, in which case the last thing the visitor asked is prepended so the query carries
    the subject the pronoun is standing in for.

    Only the previous VISITOR turn is used, never the robot's answer: the answer is usually far
    longer than the question and would dominate the embedding, turning a specific follow-up into
    a search for the general area of the previous answer.
    """
    if not history or not looks_like_follow_up(question):
        return question
    for role, text in reversed(history):
        if role == "visitor" and text.strip():
            return f"{text.strip()} {question.strip()}"
    return question


def _history_block(history: list[tuple[str, str]] | None, max_turns: int = 6) -> str:
    """The recent dialogue, rendered for the model. Empty string when there is nothing yet."""
    if not history:
        return ""
    lines = [
        f"{'Student' if role == 'visitor' else 'You'}: {text.strip()}"
        for role, text in history[-max_turns:]
        if text and text.strip()
    ]
    if not lines:
        return ""
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"


def _question_with_context(question: str, history: list[tuple[str, str]] | None) -> str:
    """The question as handed to the ANSWER GENERATOR, with the recent dialogue in front of it.

    Folded into the question string rather than added as a new AnswerGenerator protocol
    parameter, deliberately. There are seven generator implementations plus the test doubles in
    each file's self-test, and two of them (Claude, Groq) cannot be exercised here for lack of an
    API key - changing a shared signature across all of them to carry a string that every one of
    them would immediately concatenate into its prompt anyway is churn and risk for no gain. The
    generators keep taking "the question to answer"; it just carries its own context now.
    """
    block = _history_block(history)
    if not block:
        return question
    return f"{block}The student has now asked: {question}"


def _first_sentence_ready(buffer: str) -> bool:
    """Whether `buffer` already contains a complete first sentence worth inspecting."""
    stripped = buffer.strip()
    if len(stripped) >= _FIRST_SENTENCE_MAX_CHARS:
        return True
    # Require a little text before treating a terminator as a sentence end, so an answer opening
    # with "Sure." isn't inspected (and judged) on one word.
    return len(stripped) >= 20 and stripped[-1] in ".!?"


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


_GENERAL_HEADER_MAX_CHARS = 160  # a model that ignores the two-line format can't stall the answer


def strip_general_header(pieces: Iterator[str], info: dict) -> Iterator[str]:
    """Turns a streamed GENERAL_CLASSIFY_PROMPT reply into speakable answer text.

    The reply is "UNIVERSITY_RELATED: yes|no" then "ANSWER: ...". The label line is a handful of
    tokens, so it is held back, parsed into info["university_related"], and everything after
    "ANSWER:" streams straight through. A model that ignores the format is spoken as-is once
    _GENERAL_HEADER_MAX_CHARS have arrived without an ANSWER: marker. info["answered"] says
    whether any answer text came out at all.
    """
    import re

    info.setdefault("university_related", True)  # same fail-safe as _parse_university_related
    info["answered"] = False
    buffer = ""
    header_done = False
    for piece in pieces:
        if header_done:
            if piece:
                info["answered"] = True
                yield piece
            continue
        buffer += piece
        related = re.search(r"UNIVERSITY_RELATED:\s*(yes|no)", buffer, re.IGNORECASE)
        if related:
            info["university_related"] = related.group(1).lower() == "yes"
        marker = re.search(r"ANSWER:\s*", buffer, re.IGNORECASE)
        if marker:
            header_done = True
            rest = buffer[marker.end():]
            if rest.strip():
                info["answered"] = True
                yield rest
        elif len(buffer) >= _GENERAL_HEADER_MAX_CHARS:
            header_done = True
            rest = re.sub(r"^\s*UNIVERSITY_RELATED:[^\n]*\n?", "", buffer, flags=re.IGNORECASE)
            if rest.strip():
                info["answered"] = True
                yield rest
    if not header_done:
        rest = re.sub(r"^\s*UNIVERSITY_RELATED:[^\n]*\n?", "", buffer, flags=re.IGNORECASE)
        rest = re.sub(r"^\s*ANSWER:\s*", "", rest, flags=re.IGNORECASE)
        if rest.strip():
            info["answered"] = True
            yield rest


class DialogueManager:
    def __init__(self, kb: KnowledgeBase, answer_generator: AnswerGenerator,
                 stream_general: bool = True):
        self.kb = kb
        self.answer_generator = answer_generator
        # Stream general (no knowledge-base match) answers too - added 2026-10-27. Before,
        # those always waited for the model's complete reply, because of the two-line format;
        # strip_general_header() now parses that format on the fly.
        self.stream_general = stream_general
        # (contextualized query, chunks) computed early from a partial transcript - see
        # prefetch(). A plain attribute swap, so no lock is needed.
        self._prefetched: tuple[str, list[KnowledgeChunk]] | None = None

    @staticmethod
    def _query_key(text: str) -> str:
        import re

        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    def prefetch(self, question: str, history: list[tuple[str, str]] | None = None) -> None:
        """Runs the knowledge-base search for a PARTIAL transcript while the visitor is still
        talking, so it is already done when they stop. Used only when the final question turns
        out to be the same words (the common case: the last partial and the final transcript
        usually match). Never raises - this is an optimisation, not a step anything waits on."""
        try:
            query = _contextualize(question, history)
            self._prefetched = (self._query_key(query), self.kb.search(query))
        except Exception as exc:  # noqa: BLE001
            print(f"[dialogue] prefetch skipped ({exc!r}).")

    def _search(self, question: str, history: list[tuple[str, str]] | None) -> list[KnowledgeChunk]:
        query = _contextualize(question, history)
        cached = self._prefetched
        if cached is not None and cached[0] == self._query_key(query):
            return cached[1]
        return self.kb.search(query)

    def _answer_after_failed_grounded(
        self, question: str, chunks: list[KnowledgeChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        """Tiers 2 and 3 of the fallback chain described in handle_question(): the grounded call
        either failed or honestly admitted it had nothing, so try a live search, and fall back to
        the matched chunk verbatim only if that comes back empty too.

        Extracted (2026-09-05) so the streaming path can reuse the identical policy - the whole
        point of that chain is that it is ONE decision about what the robot says when grounding
        fails, and a second copy for the streaming caller would be a second place to keep in
        sync."""
        print("[dialogue] falling through to a live search instead of the matched chunk "
              f"verbatim (question={question!r}).")
        try:
            _, general_answer = self.answer_generator.classify_and_answer_general(
                _question_with_context(question, history)
            )
        except Exception as exc:
            print(f"[dialogue] live-search fallback also failed ({exc!r}).")
            general_answer = ""

        return general_answer or chunks[0].text

    def handle_question_stream(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> StreamingDialogueResult:
        """Same routing as handle_question(), but the grounded-LLM answer comes back as a live
        stream so Stage 06 can start speaking sentence one while sentence two is still being
        generated.

        WHY (2026-09-05): handle_question() cannot return until the model has produced the whole
        answer, so Piper - which has streamed sentence-by-sentence since Stage 06 was written -
        sat idle for the full generation (~1-2s) before the first word reached the speaker. That
        silence is the single biggest thing that made the robot feel like a machine rather than a
        person. Streaming moves time-to-first-word to roughly one sentence of generation.

        Only the grounded-LLM path actually streams. Everything else returns its finished string
        in the same shape, because there is nothing to gain:
          - IDENTITY is an offline string lookup, already instant.
          - The is_direct_answer fast path never calls an LLM at all.
          - The general/unknown path has to parse a two-line format out of the reply before it
            knows what the answer even is (see AnswerGenerator.classify_and_answer_general).
        Callers therefore treat all four identically and just speak `pieces`.
        """
        identity_answer = detect_identity_answer(question)
        if identity_answer is not None:
            return StreamingDialogueResult(
                intent=Intent.IDENTITY, used_chunks=[],
                pieces=iter([identity_answer]), is_streaming=False,
            )

        chunks = self._search(question, history)
        self._prefetched = None
        streamer = getattr(self.answer_generator, "stream_grounded_answer", None)
        # A generator may HAVE the method and still not be able to stream - ChainAnswerGenerator
        # defines it unconditionally but can only deliver if one of its backends does. Taking the
        # streaming route on a generator that can't would skip tier 1 of the fallback chain (the
        # ordinary grounded call) and go straight to a live search, which is a quiet downgrade in
        # answer quality. Generators that don't publish the property are taken at their word.
        if streamer is not None and not getattr(self.answer_generator, "supports_streaming", True):
            streamer = None

        if chunks and not chunks[0].is_direct_answer and streamer is not None:
            return StreamingDialogueResult(
                intent=Intent.GENERAL_INFO, used_chunks=chunks,
                pieces=self._stream_grounded(question, chunks, streamer, history),
                is_streaming=True,
            )

        general_streamer = getattr(self.answer_generator, "stream_general_answer", None)
        if (not chunks and self.stream_general and general_streamer is not None
                and getattr(self.answer_generator, "supports_general_streaming", True)):
            result = StreamingDialogueResult(
                intent=Intent.GENERAL_CHAT, used_chunks=[], pieces=iter(()), is_streaming=True,
            )
            result.pieces = self._stream_general(question, history, general_streamer, result)
            return result

        # No streaming route applies - run the ordinary logic and hand the finished answer back
        # in the streaming shape. Deliberately re-running the whole of handle_question() rather
        # than reimplementing the remaining branches here: one router, one set of rules.
        result = self.handle_question(question, history, _chunks=chunks)
        return StreamingDialogueResult(
            intent=result.intent, used_chunks=result.used_chunks,
            pieces=iter([result.answer]), is_streaming=False,
        )

    def _stream_general(self, question: str, history, streamer, result):
        """Streaming counterpart of handle_question()'s general branch. result.intent starts as
        GENERAL_CHAT and is corrected to UNKNOWN once the label line says the question was about
        the university - before anything is spoken, since the label comes first."""
        info: dict = {}
        try:
            raw = streamer(_question_with_context(question, history))
            for piece in strip_general_header(raw, info):
                result.intent = Intent.UNKNOWN if info.get("university_related") else Intent.GENERAL_CHAT
                yield piece
        except Exception as exc:
            print(f"[dialogue] streamed general answer failed ({exc!r}).")
        if not info.get("answered"):
            related = info.get("university_related", True)
            result.intent = Intent.UNKNOWN if related else Intent.GENERAL_CHAT
            yield FALLBACK_ANSWER if related else "Sorry, I'm not sure about that one."

    def _stream_grounded(self, question: str, chunks: list[KnowledgeChunk], streamer,
                         history: list[tuple[str, str]] | None = None):
        """The streaming half of handle_question()'s three-tier fallback chain.

        The safety property that must not be lost: a grounded answer which honestly says "I
        don't have that information" has to trigger a live search instead of being spoken (see
        _GROUNDED_NO_INFO_PHRASES). That is a judgement about the answer's CONTENT, so some of
        the answer has to exist before it can be made - which is why the first sentence is
        buffered and inspected before any of it is spoken, and only then does the rest stream
        through untouched. Buffering one sentence costs a fraction of what waiting for the whole
        answer did, and keeps the guarantee intact.

        Once the first sentence is out we are committed: it is already audible, so a later
        failure ends the answer where it stands rather than switching to a different one
        mid-breath.
        """
        # One iterator, held across both phases - the first sentence is consumed from it and the
        # remainder continues from exactly where that left off. (Calling the streamer a second
        # time would start a whole new generation and speak the opening twice.)
        stream = streamer(_question_with_context(question, history), chunks)

        buffer = ""
        failed = False
        try:
            for piece in stream:
                buffer += piece
                if _first_sentence_ready(buffer):
                    break
        except Exception as exc:
            print(f"[dialogue] streamed grounded answer failed ({exc!r}).")
            failed = True

        if failed or not buffer.strip() or _looks_like_no_info(buffer):
            # Nothing has been spoken yet, so falling back is still free - identical policy to
            # the blocking path, via the same helper.
            yield self._answer_after_failed_grounded(question, chunks, history)
            return

        yield buffer
        try:
            yield from stream
        except Exception as exc:
            print(f"[dialogue] streamed answer ended early ({exc!r}) - speaking what arrived.")

    def handle_question(
        self, question: str, history: list[tuple[str, str]] | None = None,
        _chunks: list[KnowledgeChunk] | None = None,
    ) -> DialogueResult:
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

        # Follow-ups are searched with the previous question prepended - see _contextualize().
        chunks = _chunks if _chunks is not None else self._search(question, history)
        self._prefetched = None
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
                answer = self.answer_generator.generate_grounded_answer(
                    _question_with_context(question, history), chunks
                )
            except Exception as exc:
                print(f"[dialogue] grounded-answer generation failed ({exc!r}).")

            if answer and not _looks_like_no_info(answer):
                return DialogueResult(intent=Intent.GENERAL_INFO, answer=answer, used_chunks=chunks)

            return DialogueResult(
                intent=Intent.GENERAL_INFO,
                answer=self._answer_after_failed_grounded(question, chunks, history),
                used_chunks=chunks,
            )

        is_university_related, general_answer = (
            self.answer_generator.classify_and_answer_general(
                _question_with_context(question, history)
            )
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

    # --- streaming route (2026-09-05) -----------------------------------------------------
    # handle_question_stream() must make exactly the same decisions as handle_question(), just
    # delivering the grounded answer incrementally. Three things are checked here, all of them
    # ways the streaming path could quietly be WORSE than the blocking one it replaces:
    #   1. it actually streams (more than one piece out of the generator, first piece early)
    #   2. a streamed "I don't have that information" still falls through to the live search,
    #      instead of being spoken - the safety property _GROUNDED_NO_INFO_PHRASES exists for
    #   3. a backend with no streaming support still answers, via the blocking path
    stream_kb = KnowledgeBase()
    stream_kb.upsert(KnowledgeChunk(
        chunk_id="scraped-msc-cs",
        text=("The Master of Computer Science is a two year programme with intakes in January "
              "and September."),
        source_url="https://www.lincoln.edu.my/example-scraped-page/",
        last_updated=0.0,
    ))

    class StreamingGenerator:
        """Yields an answer word by word, the way a real LLM produces one."""

        def __init__(self, answer: str):
            self.answer = answer
            self.stream_calls = 0

        def generate_grounded_answer(self, question, chunks):
            raise AssertionError(
                "the blocking grounded call was used even though this backend can stream"
            )

        def stream_grounded_answer(self, question, chunks):
            self.stream_calls += 1
            for word in self.answer.split(" "):
                yield word + " "

        def classify_and_answer_general(self, question):
            return True, "I checked the website and it's a two year programme."

    gen = StreamingGenerator("It's a two year programme. Intakes are in January and September.")
    r11 = DialogueManager(stream_kb, gen).handle_question_stream(
        "how long is the master of computer science"
    )
    assert r11.is_streaming, "a streamable backend should have taken the streaming route"
    assert r11.intent == Intent.GENERAL_INFO
    pieces = list(r11.pieces)
    answer_11 = "".join(pieces).strip()
    print(f"Streamed grounded answer -> {len(pieces)} piece(s), answer={answer_11!r}")
    assert "two year programme" in answer_11
    assert "September" in answer_11, "the rest of the stream after the first sentence was lost"
    assert len(pieces) > 1, (
        "the whole answer arrived as one piece - the caller would wait for full generation "
        "before speaking, which is exactly what streaming is here to avoid"
    )
    assert gen.stream_calls == 1, (
        f"the backend was asked to generate {gen.stream_calls} times - the stream is being "
        f"restarted, which would speak the opening of the answer twice"
    )

    class NoInfoStreamingGenerator(StreamingGenerator):
        """Streams an honest "I don't have that" - must NOT be spoken; it has to fall through to
        the live search, same as the blocking path does."""

        def classify_and_answer_general(self, question):
            return True, "I looked it up: intakes are in January and September."

    r12 = DialogueManager(
        stream_kb, NoInfoStreamingGenerator("I don't have that information in front of me.")
    ).handle_question_stream("how long is the master of computer science")
    answer_12 = "".join(r12.pieces).strip()
    print(f"Streamed 'no info' -> answer={answer_12!r}")
    assert "I looked it up" in answer_12, (
        "a streamed 'I don't have that information' was spoken as the final answer instead of "
        "triggering the live-search fallback"
    )
    assert "don't have that information" not in answer_12

    class NonStreamingGenerator:
        """No stream_grounded_answer() at all - the common case for older/simpler backends."""

        def generate_grounded_answer(self, question, chunks):
            return "It's a two year programme."

        def classify_and_answer_general(self, question):
            raise AssertionError("should have been answered from the chunks")

    r13 = DialogueManager(stream_kb, NonStreamingGenerator()).handle_question_stream(
        "how long is the master of computer science"
    )
    assert not r13.is_streaming
    assert "".join(r13.pieces).strip() == "It's a two year programme."

    # Identity still short-circuits before the knowledge base on the streaming path too.
    r14 = DialogueManager(stream_kb, StreamingGenerator("x")).handle_question_stream(
        "what is your name"
    )
    assert r14.intent == Intent.IDENTITY and not r14.is_streaming

    print("\nOK: the streaming route makes the same decisions as the blocking one - it really "
          "streams (so speech can start before generation finishes), it never restarts the "
          "generation mid-answer, a streamed 'I don't have that information' still falls "
          "through to a live search instead of being spoken, a backend that can't stream still "
          "answers, and identity questions are still answered offline first.")

    # A generator that advertises the method but cannot deliver (the ChainAnswerGenerator shape)
    # must fall back to the BLOCKING grounded call, not skip it - see handle_question_stream().
    class UnstreamableChain(StreamingGenerator):
        supports_streaming = False

        def generate_grounded_answer(self, question, chunks):
            return "It's a two year programme."

        def stream_grounded_answer(self, question, chunks):
            raise AssertionError(
                "the streaming route was taken on a generator that reported it cannot stream - "
                "tier 1 of the fallback chain (the ordinary grounded call) would be skipped"
            )

    r15 = DialogueManager(stream_kb, UnstreamableChain("unused")).handle_question_stream(
        "how long is the master of computer science"
    )
    assert not r15.is_streaming
    assert "".join(r15.pieces).strip() == "It's a two year programme."
    print("A chain that can't actually stream still uses the ordinary grounded call - OK")

    # --- follow-up questions within one visit (2026-09-05) --------------------------------
    # The second thing a person says is rarely self-contained. Before this, "how long is it?"
    # after a question about a programme searched the knowledge base for the literal words "how
    # long is it", matched something unrelated or nothing, and answered accordingly.
    assert looks_like_follow_up("how long is it?")
    assert looks_like_follow_up("What about the fees")
    assert looks_like_follow_up("and the entry requirements")
    assert looks_like_follow_up("is it accredited")
    # ...but a self-contained question is NOT a follow-up, even when it contains "it".
    assert not looks_like_follow_up("where is the library")
    assert not looks_like_follow_up("how long is the master of computer science")
    assert not looks_like_follow_up(
        "is it true that Lincoln University College has campuses in several countries"
    ), "a long question that merely contains 'it' was mistaken for a follow-up"
    assert not looks_like_follow_up("")

    hist = [("visitor", "Tell me about the Master of Computer Science"),
            ("robot", "It's a two year programme with three intakes.")]
    # Retrieval gets the subject back...
    searched = _contextualize("how long is it?", hist)
    assert "Master of Computer Science" in searched and "how long is it" in searched, searched
    # ...but the ROBOT's answer is never folded into the search query - it is far longer than the
    # question and would drown the follow-up's own meaning in the embedding.
    assert "two year programme" not in searched, searched
    # A self-contained question is searched exactly as asked.
    assert _contextualize("where is the library", hist) == "where is the library"
    assert _contextualize("how long is it?", None) == "how long is it?"

    # The model gets the actual dialogue, so it can resolve the pronoun in its own words.
    contextual = _question_with_context("how long is it?", hist)
    assert "Student: Tell me about the Master of Computer Science" in contextual
    assert "You: It's a two year programme" in contextual
    assert contextual.rstrip().endswith("how long is it?"), contextual
    assert _question_with_context("where is the library", None) == "where is the library"

    # End to end: the follow-up must retrieve the programme chunk, not the library one.
    followup_kb = KnowledgeBase()
    followup_kb.upsert(KnowledgeChunk(
        chunk_id="mcs",
        text=("The Master of Computer Science is a two year programme with intakes in January, "
              "May and September."),
        source_url="https://www.lincoln.edu.my/mcs/", last_updated=0.0))
    followup_kb.upsert(KnowledgeChunk(
        chunk_id="library", text="The Library is on the second floor.",
        source_url="facilities-directory:library", last_updated=0.0,
        boost=0.15, is_direct_answer=True))

    class EchoGenerator:
        def generate_grounded_answer(self, question, chunks):
            return chunks[0].text

        def classify_and_answer_general(self, question):
            return True, ""

    dm_follow = DialogueManager(followup_kb, EchoGenerator())
    blind = dm_follow.handle_question("how long is it?")
    with_context = dm_follow.handle_question("how long is it?", history=hist)
    print(f"Follow-up without history -> {blind.answer!r}")
    print(f"Follow-up with history    -> {with_context.answer!r}")
    assert "two year programme" in with_context.answer, (
        "a follow-up still could not find what it was referring to"
    )

    print("\nOK: a follow-up question is searched with the subject it refers to, the robot's own "
          "answer never pollutes that search, self-contained questions are left alone, and the "
          "model receives the dialogue so it can resolve the pronoun itself.")
