"""
Stage 05 (live) - a real answer-generation backend wired into DialogueManager, tested with a
console REPL. Stage 04 (STT) isn't wired into Stage 05 yet - that full loop is Stage 08's job -
so this stage's own test is understanding + answering as text, not speaking (Stage 06 isn't
built yet either).

AnswerGenerator implementations, all implementing the same protocol from dialogue_manager.py so
DialogueManager never cares which one is running:
  - GeminiAnswerGenerator (Google, gemini-3.7-flash) / GroqAnswerGenerator - cloud LLMs. Gemini
    is the strongest option here because it can search the live web (see its
    classify_and_answer_general), which matters for a smaller institution the model's training
    data barely covers.
  - OllamaAnswerGenerator - a real LLM running on THIS machine. No API key, no quota, no
    network. Weaker than Gemini and with no live search, but it cannot 503, cannot run out of
    free tier, and works with the network unplugged.
  - ChainAnswerGenerator - tries the above in order, using the first that actually answers.
  - UniversityOnlyAnswerGenerator - no LLM at all: speak the retrieved knowledge-base text
    verbatim. Zero hallucination risk, no general conversation.
  - ClaudeAnswerGenerator (Anthropic) - optional cloud backend, used only when "claude" is
    listed in ANSWER_BACKENDS and ANTHROPIC_API_KEY is set.

WHY THE CHAIN EXISTS: this project has twice been bitten by depending on a single
cloud backend - a Gemini 503 mid-answer and a 429 on the very first question of a
session. Both times the robot fell all the way through to reading a raw scraped
page-title fragment aloud as if it were an answer. MODE = "auto" now assembles whatever backends
are actually usable at startup, cloud first and local second, so a cloud outage costs one retry
instead of one wrong answer. Nothing about DialogueManager or the knowledge base changes; the
chain is just another AnswerGenerator.

Retrieval itself has a matching local option: knowledge_base.py can take its embeddings from
Ollama too (LUC_EMBEDDING_BACKEND=ollama). That one is an accuracy choice rather than an
availability one - the default sentence-transformers model already runs offline in-process.

Reuses DialogueManager from dialogue_manager.py UNCHANGED (and everything it relies on internally
- FALLBACK_ANSWER, the Intent routing logic) - only MockAnswerGenerator is swapped for a real
generator below, same "swap only what's fake" pattern as every other _live file in this project.

Preloaded with the real building directory (SAMPLE_CHUNKS below) purely so the grounded-answer
path has something real to retrieve - Stage 07's actual scraper doesn't exist yet. Try:
  - "where is the computer science department"     -> GENERAL_INFO, floor-only answer straight
                                                        from the KB (no separate wayfinding stage
                                                        - see dialogue_manager.py's own note)
  - "tell me about student orientation"             -> GENERAL_INFO, grounded in SAMPLE_CHUNKS
  - "what's the weather like" / "tell me a joke"     -> GENERAL_CHAT, open answer
  - "what's on the cafeteria menu today"             -> UNKNOWN, fixed apology (sounds like a
                                                        university question, but there's no data)
  - "who made you" / "what's your name"              -> IDENTITY, answered offline from
                                                        identity.py - no KB, no LLM, works even
                                                        with no API key or network at all

Install first: see README.md. Nothing here is mandatory - with no API key and no Ollama it falls
back to university-only answers and says so. For the full behaviour, either set GEMINI_API_KEY or
install Ollama (or both, which is the intended setup).
Run: python dialogue_manager_live.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading  # ChainAnswerGenerator's background backend re-check
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Loaded before GEMINI_API_KEY/GROQ_API_KEY are read in _build_answer_generator(), so an API key
# can live in the project-root .env alongside every other setting instead of having to be a real
# environment variable - see stage00_setup/project_env.py.
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from project_env import load_env_file  # noqa: E402
from realtime_config import RT  # noqa: E402

load_env_file()

from dialogue_manager import DialogueManager  # noqa: E402
from identity import get_identity, identity_source  # noqa: E402
from knowledge_base import KnowledgeBase, KnowledgeChunk  # noqa: E402

CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # fast/cheap tier - matches this stage's low-latency
# voice-interaction goal; swap to "claude-sonnet-5"
# if answer quality needs to be better and the extra
# latency/cost is acceptable
# From GROQ_MODEL in .env (default llama-3.3-70b-versatile). llama-3.1-8b-instant is faster
# still and fine for short receptionist answers if you want to squeeze the last 100ms.
GROQ_MODEL = RT.GROQ_MODEL
GEMINI_MODEL = (
    "gemini-3.7-flash"  # confirmed current/GA model name via Google's own docs
    # (ai.google.dev) as of 2026-08 - fast tier, matches this
    # stage's low-latency voice-interaction goal same as
    # CLAUDE_MODEL's choice above
)

# Gemini's output cap has to be MUCH larger than the answer you actually want, because
# gemini-3.7-flash is a thinking model and its internal reasoning tokens are billed against
# max_output_tokens before a single word of the answer is emitted.
#
# Measured here on 2026-09-05, asking "how long is the master of computer science" against one
# retrieved chunk:
#     cap of 300  ->  thoughts=288, answer=8 tokens, finish_reason=MAX_TOKENS
#                     spoken result: "It's a two-year programme"   <- cut off mid-sentence
#     no cap      ->  thoughts=164, answer=23 tokens, finish_reason=STOP
#                     spoken result: "It's a two-year programme. Would you like to know about
#                     the upcoming intake dates or entry requirements?"
#
# So the old 300 was silently truncating essentially EVERY grounded Gemini answer mid-sentence,
# on the non-streaming path as much as the streaming one. Setting thinking_config with
# thinking_budget=0 does not help - this model ignores it and thinks anyway (measured: 289
# thought tokens with the budget set to zero).
#
# 1200 leaves room for a few hundred tokens of thinking plus a short spoken answer, while still
# bounding a runaway. Brevity is NOT this number's job and never was: a token cap produces a
# sentence that stops mid-word, which is worse than a long answer. Length is controlled by
# SPOKEN_STYLE_RULES below ("about two sentences"), which the measurement above shows working.
GEMINI_MAX_OUTPUT_TOKENS = 1200

# Deliberately two separate prompts/calls, not one free-form "answer this" call - they have
# opposite rules (never go beyond the given text vs. free to talk about anything), matching
# AnswerGenerator's two methods in dialogue_manager.py.

# Shared by BOTH prompts below rather than written twice, so the robot's speaking voice can't
# drift apart between the grounded path and the general-conversation path - a visitor asking two
# questions in one visit routinely crosses between them, and hearing the register change
# mid-conversation is exactly the kind of seam that reads as "there are two different systems in
# here" rather than as one person talking.
#
# WHY THESE RULES: the previous prompts already said "no markdown, spoken
# sentences", which stopped asterisks being read aloud but did nothing about REGISTER. What came
# out was correct, complete, and unmistakably machine-generated: full paragraphs delivered in one
# breath, the question restated before answering it, "based on the information provided", and the
# same opening every single time. Each rule below targets one of those observed tells. None of
# them change what the robot is allowed to say - only how it sounds saying it.
SPOKEN_STYLE_RULES = (
    "You are speaking OUT LOUD to someone standing in front of you. There is no screen and no "
    "text - they only hear you. Sound like a real receptionist having a conversation, not like "
    "a document being read.\n"
    "  - Plain spoken sentences only: no markdown, no bullet points, no headings, no asterisks, "
    "no numbered lists. If you need to list things, say them in a sentence.\n"
    "  - Keep it to about two sentences. Give the one thing they actually asked for, then stop. "
    "If there is more worth knowing, offer it in a short question instead of saying it all - "
    "for example 'Want the full entry requirements?'\n"
    "  - Use contractions and everyday words: 'it's on the third floor', not 'it is located on "
    "the third floor'.\n"
    "  - Never restate their question back to them before answering. Start with the answer.\n"
    "  - Never mention where your information came from. Do not say 'based on the information "
    "provided', 'according to the text', 'the knowledge base', 'the context' or anything like "
    "them - a receptionist doesn't narrate their own memory. Just answer.\n"
    "  - Never refer to yourself as an AI, a model, or a language model.\n"
    "  - Vary how you open. Do not begin every answer the same way.\n"
    "  - Numbers, dates and times get spoken the way a person says them: 'the third floor', "
    "'about two years', 'nine in the morning'."
)

GROUNDED_SYSTEM_PROMPT = (
    "You are a friendly university front-desk robot receptionist.\n\n"
    f"{SPOKEN_STYLE_RULES}\n\n"
    "Answer the student's question using ONLY the information given below. If that information "
    "doesn't actually answer the question, say plainly that you don't have that information - "
    "never invent or guess a university-specific fact that isn't in the text you were given."
)

# Changed 2026-08-17 (real testing): this used to instruct the model to leave ANSWER blank
# whenever UNIVERSITY_RELATED is yes, so a question naming the university by name (e.g. asking
# its ranking) always got refused outright, even though a real LLM (especially one with live
# search, see GeminiAnswerGenerator below) could often actually help. Now it's told to attempt
# an answer either way, with an honesty/caveat rule specifically for the Lincoln-specific case -
# the "never confidently invent a university fact" guarantee now lives here (in the prompt) and
# in dialogue_manager.py's fallback-only-when-blank handling, rather than a router rule that
# refused to even try. The strictly-grounded path (GROUNDED_SYSTEM_PROMPT above, used when the
# knowledge base DID find something) is untouched - still 100% restricted to retrieved text.
GENERAL_CLASSIFY_PROMPT = (
    "You are a friendly university front-desk robot receptionist.\n\n"
    f"{SPOKEN_STYLE_RULES}\n\n"
    "A student just asked something, and the university's own knowledge base had no matching "
    "information for it. First decide: does this question sound like it's specifically ABOUT "
    "this university (a department, staff member, building, policy, event, class, tuition, "
    "ranking, reputation, accreditation, number of branches/campuses, or any other fact "
    "specific to this institution, etc.) - the kind of thing that SHOULD have been in the "
    "knowledge base but wasn't? Or is it general conversation or a general-knowledge question, "
    "unrelated to this specific university?\n\n"
    "Either way, try to actually answer using what you know (and search for, if you have that "
    "ability) rather than refusing outright. Two rules for HOW you answer:\n"
    "  - If the question is specifically about Lincoln University College and you're not fully "
    "confident in a specific fact or number (e.g. an exact ranking or statistic), say so "
    "plainly instead of stating it as certain, and suggest checking the official website "
    "(lincoln.edu.my) or asking a staff member for the most accurate, up-to-date answer - never "
    "state an unverified Lincoln-specific fact as if it's certain.\n"
    "  - If you genuinely don't know or can't find anything useful at all, say so honestly "
    "instead of making something up - leave ANSWER blank only in that case.\n\n"
    "Respond in exactly this two-line format:\n"
    "UNIVERSITY_RELATED: yes or no\n"
    "ANSWER: your natural spoken answer, following the two rules above"
)


def _parse_university_related(raw: str) -> bool:
    """Shared by both backends below - parses the GENERAL_CLASSIFY_PROMPT's required
    "UNIVERSITY_RELATED: yes/no" line."""
    for line in raw.splitlines():
        if line.strip().upper().startswith("UNIVERSITY_RELATED:"):
            return "yes" in line.lower()
    return True  # fail-safe: if the model didn't follow the format, default to the apology path
    # rather than risk an ungrounded guess at a university-specific fact


def _parse_answer(raw: str) -> str:
    """Shared by both backends below - parses the GENERAL_CLASSIFY_PROMPT's required
    "ANSWER: ..." line."""
    for line in raw.splitlines():
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return raw  # format wasn't followed - fall back to the whole response rather than silently
    # returning nothing


# --- streaming grounded answers ---------------------------------------------------------------
# Only the GROUNDED path streams, deliberately. classify_and_answer_general() has to parse a
# two-line "UNIVERSITY_RELATED: / ANSWER:" format out of the model's reply (see _parse_answer
# above), which means it cannot act on the reply until the reply is finished - streaming it would
# buy nothing. The grounded path has no such structure: every token is answer text, so Stage 06
# can start speaking sentence one while sentence two is still being written.
#
# Backends that don't implement stream_grounded_answer() keep working unchanged - the caller
# (dialogue_manager.py's handle_question_stream) checks for the method and falls back to the
# ordinary blocking call, which is what UniversityOnlyAnswerGenerator and MockAnswerGenerator do.


def _grounded_prompt(question: str, chunks: list[KnowledgeChunk]) -> str:
    """The one place the grounded user-message is built, shared by every backend's streaming and
    non-streaming call alike - four copies of this f-string had already drifted into four
    slightly different spellings of the same thing."""
    context = "\n\n".join(f"- {c.text} (source: {c.source_url})" for c in chunks)
    return f"Information:\n{context}\n\nQuestion: {question}"


class ClaudeAnswerGenerator:
    """Implements the same AnswerGenerator protocol as MockAnswerGenerator
    (dialogue_manager.py) - a drop-in replacement, so DialogueManager itself never changes.
    Uses Anthropic's Claude API - new accounts get a small one-time free credit (no card
    needed); see README.md."""

    def __init__(self, model: str = CLAUDE_MODEL, api_key: str | None = None):
        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No Anthropic API key found - set the ANTHROPIC_API_KEY environment variable "
                "(see README.md for how to get one)."
            )
        self.client = Anthropic(api_key=key)
        self.model = model

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        context = "\n\n".join(f"- {c.text} (source: {c.source_url})" for c in chunks)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=GROUNDED_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Information:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,  # low - this is meant to be a precise, grounded answer, not creative
        )
        return message.content[0].text.strip()

    def stream_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]):
        """Same call as generate_grounded_answer(), yielding text as it arrives - see the
        "streaming grounded answers" note above."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=300,
            system=GROUNDED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _grounded_prompt(question, chunks)}],
            temperature=0.2,
        ) as stream:
            yield from stream.text_stream

    def stream_general_answer(self, question: str):
        with self.client.messages.stream(
            model=self.model,
            max_tokens=300,
            system=GENERAL_CLASSIFY_PROMPT,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        ) as stream:
            yield from stream.text_stream

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=GENERAL_CLASSIFY_PROMPT,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,  # higher - general chat can be more natural/varied
        )
        raw = message.content[0].text.strip()
        return _parse_university_related(raw), _parse_answer(raw)


def _brief(exc: Exception) -> str:
    """An API error in one short line - the status and nothing else.

    A raw google-genai error repr is a full JSON body with help links in it, and three of those
    per question turns a console log into something nobody reads. The full text is still printed
    where it is genuinely diagnostic (the final give-up path); this is for the routine
    "that backend is down, moving on" lines.
    """
    text = str(exc)
    for code, label in (
        ("429", "quota exhausted"),
        ("503", "model overloaded"),
        ("500", "server error"),
        ("401", "bad API key"),
        ("403", "access denied"),
    ):
        if code in text[:200]:
            return f"{code} {label}"
    return type(exc).__name__


def _is_backend_unavailable(exc: Exception) -> bool:
    """Whether this failure means the whole backend is out, rather than one feature of it.

    Kept as a string check on purpose. The alternative is importing google.genai.errors and
    matching exception classes, which would tie this decision to one SDK - and the same three
    status codes mean the same three things for Groq and for Claude, both of which are still
    live code paths in this file. A status code in the message is the one thing they all share.
    """
    text = str(exc)[:200]
    return any(code in text for code in ("429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))


class GeminiAnswerGenerator:
    """Implements the same AnswerGenerator protocol as MockAnswerGenerator
    (dialogue_manager.py) - a drop-in replacement, so DialogueManager itself never changes.
    Uses Google's Gemini API (google-genai SDK).
    """

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found - set the GEMINI_API_KEY environment variable (see "
                "README.md for how to get one from Google AI Studio)."
            )

        # --- WHY THIS CLIENT HAS AN EXPLICIT TIMEOUT (measured 2026-09-17) -------------------
        # The google-genai SDK retries 429 and 503 INTERNALLY, with exponential backoff, before
        # it raises anything at all. That is sensible for a batch job and catastrophic for a
        # robot with somebody standing in front of it: measured here, ONE plain call against a
        # quota-exhausted key took 70.6 SECONDS to fail. The fallback chain below is built on
        # the assumption that a dead backend fails quickly and the next one takes over - and
        # that assumption was simply false, so the chain never got its turn. A visitor asking
        # "how are you today?" waited 249 seconds for an answer that Ollama, already installed
        # and warm, could produce in 0.4.
        #
        # So: a hard wall-clock ceiling per call. Past it the call raises, the chain moves on to
        # Ollama, and the visitor gets an answer. The number is a latency budget, not a guess
        # about how long Gemini "should" take - STAGES.md's own target for the whole turn is
        # 2-3 seconds, so anything past a few seconds has already lost the race to the local
        # model whatever it eventually returns.
        #
        # THE 10-SECOND FLOOR IS THE API'S, NOT A PREFERENCE. Gemini rejects a shorter deadline
        # outright:
        #     400 INVALID_ARGUMENT - Manually set deadline 8s is too short.
        #                            Minimum allowed deadline is 10s.
        # and since that is a 400 rather than a timeout, it fails EVERY call, including the ones
        # that would have succeeded. Found here by setting 8s and watching Gemini stop answering
        # entirely while the log filled with 400s - a "fix" that made the backend permanently
        # unavailable instead of merely slow. Anything below the floor is raised to it and said
        # out loud, rather than being passed through to fail.
        #
        # 10s is therefore the worst case for ONE call, not for every question: the circuit
        # breaker in ChainAnswerGenerator parks a backend after two consecutive failures, so a
        # sustained outage costs that 10s roughly once every BREAKER_COOLDOWN_S and every other
        # question goes straight to the local model.
        #
        # Tune with GEMINI_TIMEOUT_S (seconds). Raise it if you are on a slow link and see
        # healthy calls being cut off; the [gemini] log line names the timeout when that happens.
        GEMINI_MIN_TIMEOUT_S = 10.0
        requested = float(
            os.environ.get("GEMINI_TIMEOUT_S")
            or os.environ.get("LUC_GEMINI_TIMEOUT_S")
            or GEMINI_MIN_TIMEOUT_S
        )
        timeout_s = max(requested, GEMINI_MIN_TIMEOUT_S)
        if requested < GEMINI_MIN_TIMEOUT_S:
            log.warning(
                f"[gemini] GEMINI_TIMEOUT_S={requested:g}s is below the API's own "
                f"{GEMINI_MIN_TIMEOUT_S:g}s minimum, which it rejects with a 400 on every "
                f"call - using {timeout_s:g}s instead."
            )
        self.timeout_s = timeout_s
        try:
            from google.genai import types as _types

            # http_options.timeout is in MILLISECONDS.
            self.client = genai.Client(
                api_key=key,
                http_options=_types.HttpOptions(timeout=int(timeout_s * 1000)),
            )
        except Exception as exc:  # noqa: BLE001 - an SDK too old to accept http_options
            log.warning(
                f"[gemini] could not set a {timeout_s:.0f}s call timeout ({exc!r}) - falling "
                f"back to the SDK default, which can hang for over a minute on a 429/503. "
                f"Upgrade with: pip install -U google-genai"
            )
            self.client = genai.Client(api_key=key)
        self.model = model

    def health_check(self) -> None:
        """A minimal call whose only job is to find out whether this key can talk to Gemini at
        all right now. Raises the API's own error if not; returns None if it can.

        Called from ChainAnswerGenerator's background prober (never from a visitor's question),
        which is the whole point: when the quota is exhausted, SOMETHING has to keep asking
        whether it has come back, and that something must not be a person standing in front of
        the robot waiting for an answer.

        Deliberately as small a request as the API allows. The response TEXT is not read and
        does not matter - a truncated reply still proves the key, the quota and the model are
        all working, which is the only question being asked. What matters is whether this
        raises.
        """
        from google.genai import types

        self.client.models.generate_content(
            model=self.model,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=16, temperature=0.0),
        )

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        # Deliberately NOT catching exceptions here (changed 2026-08-19 - this call used to
        # swallow a failure and quietly return chunks[0].text itself). That single-tier fallback
        # is what produced a real bad answer: a 503 on this call meant the raw chunk
        # ("Lincoln University College" - a page-title fragment, not a sentence) was spoken as
        # if it were the answer to "where is Lincoln University College located". A raw scraped
        # fragment is not reliably a complete answer the way the curated facilities-directory
        # chunks are (see KnowledgeChunk.is_direct_answer) - it needed a smarter fallback than
        # "read whatever text happened to rank first", which is a decision AnswerGenerator itself
        # shouldn't own. dialogue_manager.py's handle_question() now owns the full fallback
        # chain (this call -> a live search -> the raw chunk only as a genuine last resort) - see
        # its own comment for that policy. Letting the exception propagate here is what makes
        # that possible.
        from google.genai import types

        context = "\n\n".join(f"- {c.text} (source: {c.source_url})" for c in chunks)
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"Information:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=GROUNDED_SYSTEM_PROMPT,
                temperature=0.2,  # low - a precise, grounded answer, not creative
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
        return response.text.strip()

    def stream_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]):
        """Same call as generate_grounded_answer(), yielding text as it arrives - see the
        "streaming grounded answers" note above. Exceptions propagate here for exactly the same
        reason they do there: the fallback chain belongs to the caller, not to one backend."""
        from google.genai import types

        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=_grounded_prompt(question, chunks),
            config=types.GenerateContentConfig(
                system_instruction=GROUNDED_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
        for chunk in stream:
            piece = getattr(chunk, "text", None)
            if piece:
                yield piece

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        from google.genai import types

        # Google Search grounding lets Gemini look something up instead of answering only from
        # its training data, which matters most for a smaller/less-documented institution like Lincoln University
        # College where the model's own training knowledge may be thin or stale. Deliberately
        # only enabled on this call, not generate_grounded_answer() above - the grounded path
        # must stay 100% restricted to the retrieved knowledge-base text, never go searching for
        # something else instead. GroqAnswerGenerator/ClaudeAnswerGenerator below don't have an
        # equivalent built-in search tool, so they rely on GENERAL_CLASSIFY_PROMPT's honesty
        # rule alone (no live verification) if you switch back to either of those.
        #
        # Search grounding is separately metered from plain generation (Google's own docs: a
        # monthly free-prompt quota, then billed per grounded request) - a real run
        # hit a 429 RESOURCE_EXHAUSTED on the very first question, well under any
        # published free quota, most likely because Google now requires billing to be set up on
        # an account before some features activate at all, even at $0 (a 2026 AI Studio change).
        # Rather than let that crash the whole interaction, try search first but fall back to a
        # plain (non-search) call if it fails for any reason - the robot still attempts to help
        # via the model's own knowledge instead of going silent because one quota-gated feature
        # wasn't available.
        base_config = {
            "system_instruction": GENERAL_CLASSIFY_PROMPT,
            "temperature": 0.7,  # higher - general chat can be more natural/varied
            "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
        }
        try:
            if RT.GEMINI_SEARCH:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                config = types.GenerateContentConfig(**base_config, tools=[search_tool])
            else:
                # GEMINI_SEARCH=false (the default since 2026-10-27): a plain call. Search adds
                # seconds and is separately metered; with Groq answering first, Gemini is the
                # second opinion rather than the main voice.
                config = types.GenerateContentConfig(**base_config)
            response = self.client.models.generate_content(
                model=self.model,
                contents=question,
                config=config,
            )
        except Exception as exc:
            # The retry is only worth making when dropping SEARCH could plausibly be the
            # difference. Search grounding is separately metered, so "the search quota is used
            # up" is a real and common failure that a plain call does fix - that is why this
            # retry exists and it stays.
            #
            # But a plain 429 on the ACCOUNT, or a 503 on the model itself, has nothing to do
            # with search: the second call fails exactly like the first, and the only thing it
            # reliably adds is another full timeout of silence with somebody standing there.
            # Measured 2026-09-17, before the client timeout above existed: two calls, 70s
            # each, on a question the already-installed local model answered in 0.4s.
            #
            # So: skip the retry when the error says the whole backend is unavailable, and go
            # straight to the next generator in the chain (Ollama), which is the entire reason
            # the chain exists.
            if _is_backend_unavailable(exc):
                log.warning(
                    f"[gemini] backend unavailable ({_brief(exc)}) - not retrying without "
                    f"search (it would fail the same way); handing over to the next backend."
                )
                return True, ""
            log.warning(
                f"[gemini] search-grounded call failed ({exc!r}) - retrying without search..."
            )
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=question,
                    config=types.GenerateContentConfig(**base_config),
                )
            except Exception as exc2:
                log.warning(
                    f"[gemini] retry without search also failed ({exc2!r}) - giving up on "
                    f"this question."
                )
                # Same fail-safe direction as _parse_university_related() below: when truly
                # unable to tell, assume it might be university-related rather than risk an
                # unverified answer - the caller (dialogue_manager.py) uses FALLBACK_ANSWER for
                # a blank answer here, never a crash or silence.
                return True, ""

        raw = response.text.strip()
        return _parse_university_related(raw), _parse_answer(raw)

    def stream_general_answer(self, question: str):
        """Streamed general answer. With GEMINI_SEARCH on, search grounding is not streamed
        (the SDK returns it whole), so the finished reply is handed back as one piece - the
        caller treats both shapes the same."""
        if RT.GEMINI_SEARCH:
            related, answer = self.classify_and_answer_general(question)
            if answer:
                yield f"UNIVERSITY_RELATED: {'yes' if related else 'no'}\nANSWER: {answer}"
            return
        from google.genai import types

        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=GENERAL_CLASSIFY_PROMPT,
                temperature=0.7,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
        for chunk in stream:
            piece = getattr(chunk, "text", None)
            if piece:
                yield piece


class GroqAnswerGenerator:
    """Implements the same AnswerGenerator protocol as MockAnswerGenerator
    (dialogue_manager.py) - a drop-in replacement, so DialogueManager itself never changes.
    """

    def __init__(self, model: str = GROQ_MODEL, api_key: str | None = None):
        from groq import Groq

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "No Groq API key found - set the GROQ_API_KEY environment variable (see "
                "README.md for how to get a free key)."
            )
        # A short timeout and at most one retry, for the same reason Gemini has one (see its
        # comment): the SDK's own retry-with-backoff is right for a batch job and wrong for a
        # visitor standing in front of the robot. A failed call should hand over to the next
        # backend in under ten seconds, not after a minute of retries.
        try:
            self.client = Groq(api_key=key, timeout=8.0, max_retries=1)
        except TypeError:  # a very old SDK without those arguments
            self.client = Groq(api_key=key)
        self.model = model

    def health_check(self) -> None:
        """Smallest possible call - lets ChainAnswerGenerator park and re-check Groq the same
        way it does Gemini. Raises on failure, returns None when Groq is answering."""
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

    def stream_general_answer(self, question: str):
        """GENERAL_CLASSIFY_PROMPT's two-line reply, streamed. The caller strips the
        UNIVERSITY_RELATED line as soon as it arrives and speaks the answer as it generates."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERAL_CLASSIFY_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        context = "\n\n".join(f"- {c.text} (source: {c.source_url})" for c in chunks)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Information:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,  # low - this is meant to be a precise, grounded answer, not creative
        )
        return completion.choices[0].message.content.strip()

    def stream_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]):
        """Same call as generate_grounded_answer(), yielding text as it arrives - see the
        "streaming grounded answers" note above."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": _grounded_prompt(question, chunks)},
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERAL_CLASSIFY_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,  # higher - general chat can be more natural/varied
        )
        raw = completion.choices[0].message.content.strip()
        return _parse_university_related(raw), _parse_answer(raw)


# Ranking nudge for the curated facilities directory below, added after a real regression
# where a scraped page-title chunk outranked the correct directory sentence for a
# plain "Where is the Engineering Department?" question - see KnowledgeChunk.boost's docstring
# in knowledge_base.py. 0.15 is a starting point, not a measured constant: the per-model entries
# in knowledge_base.py's SIMILARITY_THRESHOLDS were tuned with evaluate_retrieval.py against real
# query/chunk pairs; this wasn't, yet, because that harness only tested SAMPLE_CHUNKS in isolation and
# never actually reproduced the real production mix (directory + 14,000+ scraped chunks) that
# the regression came from - evaluate_retrieval.py now includes an adversarial case for exactly
# this scenario (see its ADVERSARIAL_CHUNKS), so re-run it after changing this value rather than
# eyeballing a new one.
#
# Since 2026-09-04 this affects ORDER ONLY, never whether a chunk counts as relevant at all -
# KnowledgeBase.search() thresholds the raw similarity and uses the boosted score purely to
# rank. Before that, a boosted chunk needed less genuine similarity than an unboosted one to be
# retrieved, which contradicted this constant's own "ranking nudge, not a substitute for
# relevance" description and quietly folded 0.15 into every threshold ever measured. So raising
# this can change which chunk wins, but it can no longer drag an irrelevant chunk over the bar.
DIRECTORY_BOOST = 0.15

SAMPLE_CHUNKS = [
    # Real building directory (floor-only, no room numbers - deliberately no step-by-step
    # directions either, just which floor). Ground floor's department and floor 1's department
    # are placeholders for now (subject to change); everything else is the real layout.
    #
    # source_url note: this is physical facility/floor-placement data (which floor something is
    # on) - it isn't published anywhere on lincoln.edu.my, so there's no real webpage to point
    # it at. Earlier this used a fake "https://university.example.edu/..." URL, which looked
    # like a real citation but wasn't - if it ever leaked into a prompt or log it would read as
    # a real source when it's actually just placeholder junk. Using an explicit
    # "facilities-directory:" tag instead - not a fetchable URL, just an honest label meaning
    # "manually maintained building directory, not scraped from the web." Once Stage 07's real
    # scraper (stages/stage07_data_pipeline/) is run and its output reviewed, department entries
    # here could additionally link out to that department's real school page on lincoln.edu.my
    # (e.g. Computer Science -> https://www.lincoln.edu.my/school-of-ai-computing-multimedia/)
    # as a "learn more" reference - but that's a separate, unverified follow-up, not done here.
    #
    # boost=DIRECTORY_BOOST on every entry below (2026-08-19, see KnowledgeChunk's own docstring
    # in knowledge_base.py for the full "why"): once Stage 07's real scraper filled the KB with
    # 14,000+ real page chunks, a wayfinding question like "Where is the Engineering Department?"
    # started matching a real scraped page-title chunk ("Civil Engineering Department" - a
    # heading with no floor/location info in it at all) ahead of THIS directory's actual answer,
    # purely because dense keyword-matching real content can out-score a short, deliberately
    # plain directory sentence on raw cosine similarity. These are the one part of the knowledge
    # base that's manually verified ground truth for "which floor is X on" - they should reliably
    # win that specific kind of query over anything merely keyword-adjacent from the scraped
    # corpus.
    #
    # is_direct_answer=True on every entry too (same date, see that field's docstring in
    # knowledge_base.py): each `text` below is deliberately written as a complete, natural spoken
    # sentence, so when one of these is the top match the robot speaks it immediately instead of
    # spending 1-3 seconds asking a language model to rewrite a finished sentence into itself.
    # KEEP THAT TRUE when editing these - if you ever make an entry a sentence fragment, a bare
    # label, or a list, clear is_direct_answer on it, or the robot will read that fragment aloud
    # verbatim as if it were an answer.
    KnowledgeChunk(
        chunk_id="dept-admissions",
        text="The Admissions Office is on the ground floor.",
        source_url="facilities-directory:admissions",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="library",
        text="The Library is on the first floor.",
        source_url="facilities-directory:library",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="dept-student-affairs",
        text="The Student Affairs department is on the first floor.",
        source_url="facilities-directory:student-affairs",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="mosque-men",
        text="The men's mosque is on the first floor.",
        source_url="facilities-directory:mosque-men",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="dept-management",
        text="The Management department is on the second floor.",
        source_url="facilities-directory:management",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="mosque-women",
        text="The women's mosque is on the second floor.",
        source_url="facilities-directory:mosque-women",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="dept-engineering",
        text="The Engineering department is on the third floor.",
        source_url="facilities-directory:engineering",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="dept-cs",
        text="The Computer Science department is on the fourth floor.",
        source_url="facilities-directory:computer-science",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="dept-nursing",
        text="The Nursing department is on the fifth floor.",
        source_url="facilities-directory:nursing",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
    KnowledgeChunk(
        chunk_id="event-orientation",
        text="Freshman orientation starts next Monday at 9am in the main auditorium.",
        source_url="facilities-directory:orientation-event",
        last_updated=0.0,
        boost=DIRECTORY_BOOST,
        is_direct_answer=True,
    ),
]


class OllamaAnswerGenerator:
    """A real LLM running on this machine (see ollama_client.py) - no API key, no quota, no
    network. Implements the same AnswerGenerator protocol as every other generator here.

    This is the answer to "what happens when the API isn't working", which for this project is
    not hypothetical: Gemini has produced a 503 mid-answer and a 429 on the first question of a
    session, and each time the robot degraded to reading a raw scraped page-title fragment
    aloud. A local model is worse than Gemini at general knowledge and has no live web search,
    but it is always there, and a slightly weaker sentence beats a nonsense one.

    Deliberately does NOT catch its own exceptions, for the same reason GeminiAnswerGenerator
    doesn't (see its comment): the fallback policy belongs to the caller - ChainAnswerGenerator
    below, and dialogue_manager.py's three-tier chain - not to an individual backend.
    """

    def __init__(self, model: str | None = None):
        import ollama_client

        self._client = ollama_client
        self.model = model or ollama_client.CHAT_MODEL
        if not ollama_client.is_available():
            raise RuntimeError(
                f"No Ollama server answering at {ollama_client.OLLAMA_HOST} - see README.md "
                f"('Local LLM via Ollama') for the one-time install."
            )
        if not ollama_client.has_model(self.model):
            raise RuntimeError(
                f"Ollama is running but {self.model!r} is not pulled. Run: "
                f"ollama pull {self.model}"
            )

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        context = "\n\n".join(f"- {c.text} (source: {c.source_url})" for c in chunks)
        return self._client.chat(
            f"Information:\n{context}\n\nQuestion: {question}",
            system=GROUNDED_SYSTEM_PROMPT,
            temperature=0.2,  # low - a precise, grounded answer, not creative
            model=self.model,
        )

    def stream_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]):
        """Same call as generate_grounded_answer(), yielding text as it arrives - see the
        "streaming grounded answers" note above. This matters MORE for the local model than for
        the cloud one: a 3B model on this machine is the slowest backend in the chain, so it is
        where hiding generation time behind the first spoken sentence buys the most."""
        yield from self._client.chat_stream(
            _grounded_prompt(question, chunks),
            system=GROUNDED_SYSTEM_PROMPT,
            temperature=0.2,
            model=self.model,
        )

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        # No search tool here, unlike GeminiAnswerGenerator - a local model can only answer from
        # what it learned during training. GENERAL_CLASSIFY_PROMPT's honesty rule ("never state
        # an unverified Lincoln-specific fact as if it's certain, point at lincoln.edu.my
        # instead") is doing all the work in that case, which is exactly the situation it was
        # written for. A small model follows that rule less reliably than a large one, which is
        # why Gemini stays first in the chain when its key is available.
        raw = self._client.chat(
            question, system=GENERAL_CLASSIFY_PROMPT, temperature=0.7, model=self.model
        )
        return _parse_university_related(raw), _parse_answer(raw)

    def stream_general_answer(self, question: str):
        yield from self._client.chat_stream(
            question, system=GENERAL_CLASSIFY_PROMPT, temperature=0.7, model=self.model
        )


class ChainAnswerGenerator:
    """Tries several real generators in order and uses the first one that actually produces an
    answer. This is what makes "the API stopped working" a non-event instead of a bad answer.

    Order matters and is set by the builder below: a cloud model with live search first (best
    answers when it works), the local model second (always there). Each failure is printed with
    the backend's name - a chain that silently downgrades would hide exactly the API problem the
    operator needs to know about, and would make a run on the weaker model indistinguishable
    from a run on the stronger one when reading the log afterwards.

    Exhausting the chain re-raises, rather than inventing an answer: dialogue_manager.py's
    handle_question() already owns what happens next (a live-search attempt, then the matched
    chunk verbatim, then the fixed apology), and duplicating that policy here would mean two
    places to keep in sync.
    """

    def __init__(self, generators: list[tuple[str, object]]):
        if not generators:
            raise ValueError("ChainAnswerGenerator needs at least one generator")
        self.generators = generators
        # --- the circuit breaker (added 2026-09-17, from a measured 249s answer) -------------
        # A chain that re-tries a dead backend on every question pays that backend's failure
        # cost on every question. When a Gemini key is out of quota it stays out of quota - the
        # quota resets on Google's clock, not between two sentences - so the first visitor
        # pays the timeout, and so does the second, and the third, for the whole day.
        #
        # So after BREAKER_THRESHOLD consecutive failures a backend is PARKED - taken out of the
        # chain entirely - and the robot runs on the next one down (Ollama) with no waiting at
        # all. Nothing about a visitor's question ever touches a parked backend again.
        #
        # HOW IT COMES BACK, and why this is a background thread rather than a timer checked on
        # the next question. The obvious design is "park it for N seconds, then let the next
        # question try it again", and that was the first version of this. It is wrong for the
        # reason the user put plainly: the retry still lands on somebody's question. Whoever
        # happens to ask the first question after the cooldown expires pays the full 10-second
        # timeout, and if the quota is still exhausted they pay it again every cooldown, all
        # day. The robot is fast for most people and mysteriously slow for a few.
        #
        # So the re-check runs on its own thread instead, every RECHECK_INTERVAL_S, using a
        # deliberately tiny health_check() call. A visitor's question NEVER probes a parked
        # backend. When the probe finally succeeds - the quota reset, the outage ended, billing
        # was enabled - the backend is un-parked and the very next question gets the better
        # model, with nobody having waited for the discovery.
        #
        # Backends with no health_check() (Ollama) are never parked in the first place: the
        # local model is the floor of the chain, and parking the floor would leave nothing.
        #
        # Deliberately per-INSTANCE rather than global: the self-tests build throwaway chains
        # from deliberately-broken stubs, and a global breaker would let one test's failures
        # change what the next test sees.
        self._lock = threading.Lock()
        self._consecutive_failures: dict[str, int] = {}
        self._parked: set[str] = set()
        # Backends with a health check in flight. Treated as unavailable while probing - see
        # probe_now(): "we are still finding out" must never become "let the visitor wait".
        self._probing: set[str] = set()
        self._last_probe_at = -1e9  # so the first probe_now() is never debounced away
        self._prober: threading.Thread | None = None
        self._stop_probing = threading.Event()

    BREAKER_THRESHOLD = 2  # consecutive failures before a backend is parked

    # How often a parked backend is quietly re-checked in the background. Ten minutes by
    # default: an exhausted Gemini quota resets on Google's clock (hourly or daily depending on
    # the limit hit), so probing much more often is just noise, and probing much less often
    # means running on the weaker model for longer than necessary after it recovers.
    RECHECK_INTERVAL_S = float(
        os.environ.get("LLM_RECHECK_INTERVAL_S")
        or os.environ.get("LUC_LLM_RECHECK_INTERVAL_S")
        or 600.0
    )

    def _recheck_every(self) -> str:
        """The re-check interval as a phrase, in whichever unit reads naturally. A short test
        interval printing as "every 0 minute(s)" is the kind of small wrongness that makes a
        reader distrust the rest of the log."""
        secs = self.RECHECK_INTERVAL_S
        if secs < 60:
            return f"{secs:.0f} second(s)"
        return f"{secs / 60:.0f} minute(s)"

    # How long probe_now() stays quiet after a probe, so a group of people arriving together -
    # or one person stepping in and out of frame - fires one check rather than a burst of them.
    PROBE_DEBOUNCE_S = 30.0

    def probe_now(self, reason: str = "") -> None:
        """Check every health-checkable backend RIGHT NOW, in the background. Returns
        immediately; nothing waits on the result.

        WHY THIS IS CALLED WHEN THE ROBOT SEES SOMEBODY
        -----------------------------------------------
        The expensive question is "is Gemini answering?", and the only two places to ask it are
        both bad: on a visitor's question (they wait up to the full timeout for the answer) or
        on a timer (the first visitor after each interval waits instead). Both put the cost on
        a person standing in front of the robot.

        But there is a third moment, and it is free. Between the camera noticing somebody and
        that person finishing their first question, several seconds pass with nothing else
        happening: they walk up, the robot greets them, face recognition votes, they start
        talking. Asking during THAT window costs nobody anything, and by the time a question
        actually arrives the answer is already known.

        So the orchestrator calls this the moment a visitor is detected. If Gemini is healthy it
        is used for their whole visit. If it is not, it is parked before their first question is
        even finished, and the entire visit runs on Ollama at full speed - the visitor never
        experiences the timeout at all, because it happened while they were still walking over.

        A backend being probed counts as unavailable until the probe comes back (see
        _available). That is deliberate: mid-probe we do not KNOW it works, and the entire point
        here is to never gamble a visitor's wait on it. The cost is that a question arriving
        during a probe on a HEALTHY system is answered by Ollama instead of Gemini - slightly
        weaker, never slower, and only for that one question.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_probe_at < self.PROBE_DEBOUNCE_S:
                return
            candidates = [
                (n, g)
                for n, g in self.generators
                if hasattr(g, "health_check") and n not in self._probing
            ]
            if not candidates:
                return
            self._last_probe_at = now
            for name, _ in candidates:
                self._probing.add(name)

        threading.Thread(
            target=self._probe_once,
            args=(candidates, reason),
            name="llm-probe",
            daemon=True,
        ).start()

    def _probe_once(self, candidates: list[tuple[str, object]], reason: str) -> None:
        """One round of health checks, on a throwaway thread. Never raises."""
        for name, generator in candidates:
            try:
                generator.health_check()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._probing.discard(name)
                    newly_parked = name not in self._parked
                    self._parked.add(name)
                if newly_parked:
                    # One probe failure is enough to park, unlike BREAKER_THRESHOLD on the
                    # answer path. A probe exists only to answer this question and it answered
                    # it; requiring a second opinion would just mean a visitor's question
                    # supplying it, which is the whole thing being avoided.
                    where = f" ({reason})" if reason else ""
                    why = " ".join(str(exc).split())
                    # never print an API key, should an error message ever echo one back
                    why = re.sub(r"(AIza|gsk_)[0-9A-Za-z_\-]{8,}", r"\1...", why)[:160]
                    why = why or type(exc).__name__
                    log.info(f"[chain] {name} is not responding{where}: {why}")
                    log.info(
                        f"[chain] keeping {name} out of the chain so this visit runs on the "
                        f"next backend with no waiting. Re-checking every "
                        f"{self._recheck_every()}."
                    )
                self._start_prober()
                continue

            with self._lock:
                self._probing.discard(name)
                was_parked = name in self._parked
                self._parked.discard(name)
                self._consecutive_failures.pop(name, None)
            if was_parked:
                log.info(
                    f"[chain] {name} is available again - putting it back at the front of "
                    f"the chain."
                )

    def _available(self) -> list[tuple[str, object]]:
        """The generators worth trying right now, in order.

        Excludes both parked backends and ones with a probe in flight - see probe_now() for why
        "we are still finding out" has to count as unavailable rather than as available.

        Never returns empty: if every backend is parked, the parking is ignored and the full
        chain is tried anyway. "All the backends are parked" must degrade to "try them all and
        let the caller's fallback handle it", never to "there is nothing to try" - the latter
        would turn a recoverable outage into a hard failure for the visitor.
        """
        with self._lock:
            unavailable = self._parked | self._probing
        live = [(n, g) for n, g in self.generators if n not in unavailable]
        return live or self.generators

    def _note_failure(self, name: str, exc: Exception) -> None:
        with self._lock:
            count = self._consecutive_failures.get(name, 0) + 1
            self._consecutive_failures[name] = count
            if count < self.BREAKER_THRESHOLD or name in self._parked:
                return
            # Only park something that can tell us when it is well again. Parking a backend
            # with no health_check() would take it out with nothing to bring it back, which is
            # a permanent downgrade dressed up as a temporary one.
            generator = dict(self.generators).get(name)
            if not hasattr(generator, "health_check"):
                return
            self._parked.add(name)

        log.warning(
            f"[chain] {name} has failed {count} times in a row ({_brief(exc)}) - taking it "
            f"out of the chain so no question waits on it again. Re-checking quietly in the "
            f"background every {self._recheck_every()}; it comes back by itself the moment "
            f"it works."
        )
        self._start_prober()

    def _note_success(self, name: str) -> None:
        """One success clears the record completely.

        Not a decrement: a backend that answers is working, and holding a grudge over failures
        from before an outage ended would park it again on the next single blip.
        """
        with self._lock:
            had_failures = self._consecutive_failures.pop(name, None)
            was_parked = name in self._parked
            self._parked.discard(name)
        if had_failures or was_parked:
            log.info(f"[chain] {name} is answering again.")

    def _start_prober(self) -> None:
        """Starts the background re-check thread, at most once."""
        with self._lock:
            if self._prober is not None and self._prober.is_alive():
                return
            self._stop_probing.clear()
            self._prober = threading.Thread(
                target=self._probe_loop, name="llm-recheck", daemon=True
            )
            thread = self._prober
        thread.start()

    def _probe_loop(self) -> None:
        """Re-checks parked backends on a timer until they all come back.

        Daemon thread, and it exits on its own once nothing is parked - so a chain that
        recovers stops costing anything at all rather than pinging a healthy API forever.
        """
        while not self._stop_probing.is_set():
            # wait() rather than sleep(), so close() takes effect immediately at shutdown
            # instead of after up to ten minutes.
            if self._stop_probing.wait(self.RECHECK_INTERVAL_S):
                return

            with self._lock:
                parked = list(self._parked)
            if not parked:
                return

            for name in parked:
                if self._stop_probing.is_set():
                    return
                generator = dict(self.generators).get(name)
                check = getattr(generator, "health_check", None)
                if check is None:
                    continue
                try:
                    check()
                except Exception as exc:  # noqa: BLE001 - still down; try again next round
                    log.warning(
                        f"[chain] {name} is still unavailable ({_brief(exc)}) - staying "
                        f"out of the chain, next check in {self._recheck_every()}."
                    )
                    continue
                with self._lock:
                    self._parked.discard(name)
                    self._consecutive_failures.pop(name, None)
                log.info(
                    f"[chain] {name} is available again - putting it back at the front of "
                    f"the chain. The next question will use it."
                )

    def close(self) -> None:
        """Stops the background re-check. Safe to call more than once, and safe to never call -
        the thread is a daemon and will not hold the process open either way."""
        self._stop_probing.set()

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.generators]

    @property
    def supports_streaming(self) -> bool:
        """Whether ANY backend in this chain can actually stream.

        A chain always HAS a stream_grounded_answer method, so a caller checking for the method
        alone would take the streaming route even for a chain of backends where none of them can
        stream - and then, when that route failed on the first token, skip straight past tier 1
        of the fallback chain (the ordinary grounded call) to tier 2 (a live search). That would
        be a silent downgrade in answer quality caused purely by how the capability was
        detected. dialogue_manager.py checks this property before committing.
        """
        return any(hasattr(g, "stream_grounded_answer") for _, g in self.generators)

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        errors = []
        for name, generator in self._available():
            try:
                answer = generator.generate_grounded_answer(question, chunks)
            except Exception as exc:
                log.warning(
                    f"[chain] {name} failed on the grounded answer ({exc!r}) - trying the "
                    f"next backend."
                )
                self._note_failure(name, exc)
                errors.append(f"{name}: {exc!r}")
                continue
            if answer and answer.strip():
                self._note_success(name)
                if errors:
                    log.warning(f"[chain] answered by {name} after {len(errors)} failure(s).")
                return answer
            log.info(f"[chain] {name} returned an empty grounded answer - trying the next backend.")
            errors.append(f"{name}: empty answer")
        raise RuntimeError(f"every answer backend failed ({'; '.join(errors)})")

    def stream_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]):
        """Streaming counterpart to generate_grounded_answer(), with the same "first backend
        that actually works" behaviour - but the failover window closes as soon as the first
        piece of text is out.

        That is the important difference from the non-streaming version and the reason this
        isn't a plain loop: once a fragment has been yielded, Stage 06 may already be SPEAKING
        it, so silently restarting on a different backend would make the robot say the first
        half of one answer followed by the whole of a different one. So each backend is given
        its chance to produce its FIRST fragment inside the try - a backend that is down fails
        there, before anything is audible, and the next one is tried cleanly. After that first
        fragment we are committed: a mid-stream failure ends the answer where it stands (the
        caller still has the partial text and speaks what it got) rather than starting over.

        Backends without a stream_grounded_answer() are skipped rather than treated as failures
        - see the module note above; the caller falls back to the blocking path if the whole
        chain has nobody who can stream.
        """
        errors = []
        for name, generator in self._available():
            streamer = getattr(generator, "stream_grounded_answer", None)
            if streamer is None:
                continue

            stream = streamer(question, chunks)
            try:
                first = next(stream)
            except StopIteration:
                log.info(
                    f"[chain] {name} streamed an empty grounded answer - trying the next "
                    f"backend."
                )
                errors.append(f"{name}: empty stream")
                continue
            except Exception as exc:
                log.warning(
                    f"[chain] {name} failed to start streaming ({exc!r}) - trying the next "
                    f"backend."
                )
                self._note_failure(name, exc)
                errors.append(f"{name}: {exc!r}")
                continue

            self._note_success(name)
            if errors:
                log.warning(f"[chain] streaming from {name} after {len(errors)} failure(s).")
            yield first
            try:
                yield from stream
            except Exception as exc:
                # Already committed - part of this answer is out of the speaker. Stop here and
                # let the caller speak what it has instead of contradicting itself.
                log.warning(
                    f"[chain] {name} failed mid-stream ({exc!r}) - ending this answer with "
                    f"what was already produced."
                )
            return

        detail = "; ".join(errors) if errors else "no backend supports streaming"
        raise RuntimeError(f"no streaming answer backend succeeded ({detail})")

    @property
    def supports_general_streaming(self) -> bool:
        return any(hasattr(g, "stream_general_answer") for _, g in self.generators)

    def stream_general_answer(self, question: str):
        """Same failover rule as stream_grounded_answer(): a backend gets its chance to produce
        a FIRST piece, and only after that is the answer committed to it."""
        errors = []
        for name, generator in self._available():
            streamer = getattr(generator, "stream_general_answer", None)
            if streamer is None:
                continue
            stream = streamer(question)
            try:
                first = next(stream)
            except StopIteration:
                errors.append(f"{name}: empty stream")
                continue
            except Exception as exc:
                log.warning(
                    f"[chain] {name} failed to start the general answer ({_brief(exc)}) - "
                    f"trying the next backend."
                )
                self._note_failure(name, exc)
                errors.append(f"{name}: {exc!r}")
                continue
            self._note_success(name)
            yield first
            try:
                yield from stream
            except Exception as exc:
                log.warning(
                    f"[chain] {name} failed mid-answer ({exc!r}) - ending with what was said."
                )
            return
        detail = "; ".join(errors) if errors else "no backend supports streaming"
        raise RuntimeError(f"no streaming general backend succeeded ({detail})")

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        for name, generator in self._available():
            try:
                is_university_related, answer = generator.classify_and_answer_general(question)
            except Exception as exc:
                log.warning(
                    f"[chain] {name} failed on the general answer ({exc!r}) - trying the "
                    f"next backend."
                )
                continue
            if answer and answer.strip():
                return is_university_related, answer
            log.info(f"[chain] {name} had no general answer - trying the next backend.")
        # Every backend declined or failed. True (= "assume it might be university-related") is
        # the fail-safe direction the rest of this file already uses: the caller turns a blank
        # answer into the fixed apology rather than an unverified guess.
        return True, ""


class UniversityOnlyAnswerGenerator:
    """Real answer generator (not a test mock - this is the current phase's actual behavior),
    needing no LLM API at all: current project decision is to get the robot speaking about the
    university first, using only real retrieved knowledge-base text, before turning on general
    conversation once a working LLM API account is sorted out (see MODE below).

    Implements the same AnswerGenerator protocol as every other generator in this file - a
    drop-in swap for ClaudeAnswerGenerator/GroqAnswerGenerator later. DialogueManager itself
    doesn't change either way."""

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        # No LLM to synthesize/rephrase multiple chunks - just speak the single best-matching
        # chunk verbatim. Zero hallucination risk since there's no generation step at all, but
        # it'll sound exactly like however the knowledge-base text was written - Stage 07 will
        # want to write/clean scraped chunks with that in mind (full spoken sentences, not raw
        # HTML fragments).
        return chunks[0].text

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        # General chat is deliberately off in this phase - always route to the fixed apology
        # instead of attempting an open answer with no LLM to actually generate one.
        return True, ""


# "auto" (default since 2026-09-04): build a fallback chain out of whatever is actually usable
#   on this machine right now - a cloud API if its key is set, a local Ollama model if a server
#   is answering - and drop to UniversityOnlyAnswerGenerator if neither is. This is the mode the
#   robot should normally run in: it uses the best backend available and survives any one of them
#   being down, instead of the previous all-or-nothing "an API key exists / it doesn't".
# "full": require a cloud API key, no local fallback. Kept for testing the cloud path in
#   isolation - if the key is missing or the API is down, that failure is visible rather than
#   quietly covered for.
# "ollama_only": force the local model, never call out to an API. The way to check what the
#   robot will actually do with the network unplugged, without having to unplug the network.
# "university_only": no LLM at all - real KB retrieval, zero hallucination risk, general chat
#   disabled (routes to the fixed apology instead).
MODE = "auto"  # "auto" | "full" | "ollama_only" | "university_only"


_CLOUD_BACKENDS = {
    # name in ANSWER_BACKENDS -> (display name, env var holding the key, constructor)
    "groq": ("Groq", "GROQ_API_KEY", lambda: GroqAnswerGenerator()),
    "gemini": ("Gemini", "GEMINI_API_KEY", lambda: GeminiAnswerGenerator()),
    "claude": ("Claude", "ANTHROPIC_API_KEY", lambda: ClaudeAnswerGenerator()),
}


def _build_cloud_generators() -> list[tuple[str, object]]:
    """Every cloud backend that has a key, in ANSWER_BACKENDS order.

    Before, only ONE cloud backend was ever used - Gemini if its key existed, otherwise Groq - so
    a robot with both keys still waited on Gemini's thinking time on every question, and lost
    Groq as a fallback. Now every backend with a key joins the chain, fastest first.
    """
    built = []
    for key in RT.answer_backend_order():
        if key not in _CLOUD_BACKENDS:
            continue
        display, env_var, make = _CLOUD_BACKENDS[key]
        if not os.environ.get(env_var):
            continue
        try:
            built.append((display, make()))
        except Exception as exc:  # e.g. the SDK for that backend is not installed
            log.warning(
                f"[answers] {display} has a key but could not start ({exc!r}) - skipping it."
            )
    return built


def _build_cloud_generator():
    """Kept for anything that still calls it: the FIRST cloud backend, or None."""
    built = _build_cloud_generators()
    return built[0] if built else None


def _build_local_generator(warm: bool = True):
    """The local Ollama backend, or None if it isn't usable. Never raises - an absent local model
    is a normal, expected state (nobody has installed Ollama yet), not an error worth stopping
    for."""
    try:
        generator = OllamaAnswerGenerator()
    except Exception as exc:
        log.warning(f"[ollama] local model unavailable: {exc}")
        return None

    if warm:
        # Pay the model load now, at startup, instead of on the first visitor's first question -
        # see ollama_client.warm_up() for why that difference is worth a few seconds here.
        try:
            import ollama_client

            elapsed = ollama_client.warm_up(generator.model)
            log.info(f"[ollama] {generator.model} warmed up in {elapsed:.1f}s.")
        except Exception as exc:
            # It answered is_available() a moment ago, so this is worth reporting but not worth
            # dropping the backend over - the first real question will simply pay the load.
            log.warning(f"[ollama] warm-up failed ({exc!r}) - keeping the backend anyway.")
    return "Ollama", generator


def _build_answer_generator():
    if MODE == "university_only":
        log.info(
            "Running in university-only mode (no LLM at all) - answers are quoted directly "
            "from the knowledge base, and anything else gets the fixed apology instead of open "
            'conversation. Set MODE = "auto" in this file to use a cloud API and/or a local '
            "Ollama model when they're available."
        )
        return UniversityOnlyAnswerGenerator()

    if MODE == "ollama_only":
        local = _build_local_generator()
        if local is None:
            raise RuntimeError(
                'MODE = "ollama_only" but the local model is not usable (see the message above). '
                'Install/pull it, or switch MODE back to "auto".'
            )
        log.info(f"Using the local model only: {local[1].model}.")
        return local[1]

    if MODE == "full":
        cloud = _build_cloud_generator()
        if cloud is None:
            raise RuntimeError(
                'MODE = "full" but no API key found - set GEMINI_API_KEY or GROQ_API_KEY (see '
                'README.md), or use MODE = "auto" to fall back to a local Ollama model.'
            )
        log.info(f'Using {cloud[0]} only (MODE = "full" - no local fallback).')
        return cloud[1]

    # --- MODE == "auto" ---------------------------------------------------------------------
    if MODE != "auto":
        log.info(f'Unknown MODE={MODE!r} - treating it as "auto".')

    order = RT.answer_backend_order()
    backends = list(_build_cloud_generators())
    if "ollama" in order:
        local = _build_local_generator()
        if local is not None:
            # Ollama goes where ANSWER_BACKENDS puts it (normally last).
            position = sum(
                1
                for k in order[: order.index("ollama")]
                if k != "ollama" and any(n.lower() == k for n, _ in backends)
            )
            backends.insert(position, local)

    if not backends:
        log.warning(
            "No LLM backend is available - no GEMINI_API_KEY/GROQ_API_KEY is set, and no local "
            "Ollama model is usable. Falling back to university-only answers: real knowledge-base "
            "retrieval still works and is still accurate, but general conversation is off and "
            "questions the knowledge base can't answer get the fixed apology. Install Ollama "
            "(README.md) to get a free local model with no API key."
        )
        return UniversityOnlyAnswerGenerator()

    if len(backends) == 1:
        name, generator = backends[0]
        note = (
            "no local Ollama fallback - if this API goes down mid-conversation the robot "
            "drops to knowledge-base answers only"
            if name != "Ollama"
            else "no cloud API key set, so this is also the fallback"
        )
        log.info(f"Answer backend: {name} ({note}).")
        return generator

    chain = ChainAnswerGenerator(backends)
    log.warning(
        f"Answer backends, in order: {' -> '.join(chain.names)}. Each question tries them in "
        f"turn, so an API failure costs a retry rather than a bad answer."
    )
    return chain


def _run_chain_self_test() -> int:
    """`python dialogue_manager_live.py --self-test`

    Proves the failover actually happens rather than assuming it. The interesting case can't be
    tested by just running the robot and hoping Gemini breaks, so the first backend here is a
    stand-in that fails exactly the way Gemini has failed for real (a 503, and a 429), with the
    genuine local Ollama model behind it. Needs Ollama running; skips loudly if it isn't, rather
    than passing vacuously.
    """
    print("Chain failover self-test\n" + "-" * 60)

    local = _build_local_generator(warm=True)
    if local is None:
        print(
            "\nSKIPPED: this test needs a working local Ollama backend (see the message "
            "above, and README.md's 'Local LLM via Ollama')."
        )
        return 1

    class _BrokenCloudGenerator:
        """Fails the way the real one has failed in this project, on both methods."""

        def __init__(self, error: str):
            self.error = error
            self.grounded_calls = 0
            self.general_calls = 0

        def generate_grounded_answer(self, question, chunks):
            self.grounded_calls += 1
            raise RuntimeError(self.error)

        def classify_and_answer_general(self, question):
            self.general_calls += 1
            raise RuntimeError(self.error)

    class _SilentCloudGenerator:
        """Doesn't raise - just returns nothing. A backend that answers with an empty string is
        a different failure from one that errors, and a chain that only handled exceptions would
        pass this straight through as "the answer"."""

        def generate_grounded_answer(self, question, chunks):
            return "   "

        def classify_and_answer_general(self, question):
            return True, ""

    chunk = KnowledgeChunk(
        chunk_id="test-chunk",
        text="The Engineering department is on the third floor.",
        source_url="self-test:engineering",
        last_updated=0.0,
    )
    all_ok = True

    def _check(label: str, passed: bool) -> bool:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")
        return passed

    # 1. a 503 on the grounded call - the exact 2026-08-19 failure
    broken = _BrokenCloudGenerator("503 UNAVAILABLE")
    chain = ChainAnswerGenerator([("BrokenCloud", broken), local])
    answer = chain.generate_grounded_answer("where is the engineering department", [chunk])
    all_ok &= _check(
        f"a 503 on the cloud backend still produces a grounded answer: {answer!r}",
        bool(answer) and broken.grounded_calls == 1,
    )
    all_ok &= _check(
        "the answer came from the retrieved chunk, not from thin air",
        "third floor" in answer.lower() or "3rd floor" in answer.lower(),
    )

    # 2. a 429 on the general call - the exact 2026-08-17 failure
    broken = _BrokenCloudGenerator("429 RESOURCE_EXHAUSTED")
    chain = ChainAnswerGenerator([("BrokenCloud", broken), local])
    _, general = chain.classify_and_answer_general("tell me something interesting about libraries")
    all_ok &= _check(
        f"a 429 on the cloud backend still produces a general answer " f"({general[:60]!r}...)",
        bool(general) and broken.general_calls == 1,
    )

    # 3. an empty (non-raising) response must also fail over
    chain = ChainAnswerGenerator([("SilentCloud", _SilentCloudGenerator()), local])
    answer = chain.generate_grounded_answer("where is the engineering department", [chunk])
    all_ok &= _check(
        f"a blank cloud response fails over too, instead of being spoken: {answer!r}",
        bool(answer.strip()),
    )

    # 4. every backend down must RAISE, not invent - dialogue_manager.py's handle_question()
    #    owns what happens next (a live search, then the chunk verbatim), and a chain that
    #    quietly returned something here would take that decision away from it.
    chain = ChainAnswerGenerator(
        [
            ("BrokenCloud", _BrokenCloudGenerator("503")),
            ("BrokenLocal", _BrokenCloudGenerator("connection refused")),
        ]
    )
    try:
        chain.generate_grounded_answer("where is the engineering department", [chunk])
        raised = False
    except RuntimeError:
        raised = True
    all_ok &= _check("an exhausted chain raises rather than inventing an answer", raised)

    # 5. end to end through the real router, with the whole cloud side dead
    kb = KnowledgeBase()
    kb.upsert_many(list(SAMPLE_CHUNKS))
    dm = DialogueManager(
        kb, ChainAnswerGenerator([("BrokenCloud", _BrokenCloudGenerator("503")), local])
    )
    result = dm.handle_question("where is the nursing department")
    all_ok &= _check(
        f"end to end with no cloud: {result.intent.name} -> {result.answer!r}",
        "fifth floor" in result.answer.lower() or "5th floor" in result.answer.lower(),
    )

    # --- probing on arrival, so no visitor ever waits on a dead backend ----------------------
    # The behavioural fix for a measured 10s-per-question stall: check the cloud backend while
    # the visitor is still walking up, not when they ask.
    class _SlowDeadCloud:
        """A quota-exhausted cloud backend: fails, and takes the full timeout doing it."""

        inline_calls = 0

        def health_check(self):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        def generate_grounded_answer(self, question, chunks):
            _SlowDeadCloud.inline_calls += 1
            time.sleep(1.0)  # stands in for the real 10s API timeout
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        def classify_and_answer_general(self, question):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    class _InstantLocal:
        def generate_grounded_answer(self, question, chunks):
            return "The Engineering department is on the third floor."

        def classify_and_answer_general(self, question):
            return False, "sure"

    chain = ChainAnswerGenerator([("DeadCloud", _SlowDeadCloud()), ("Local", _InstantLocal())])
    chain.probe_now("visitor detected")
    for _ in range(50):  # the probe is on a thread; give it a moment to land
        if "DeadCloud" in chain._parked:
            break
        time.sleep(0.05)
    started = time.monotonic()
    answer = chain.generate_grounded_answer("where is engineering", [])
    elapsed = time.monotonic() - started
    all_ok &= _check(
        f"a visitor's first question skips the dead backend entirely ({elapsed * 1000:.0f}ms, "
        f"{_SlowDeadCloud.inline_calls} call(s) made on their question)",
        _SlowDeadCloud.inline_calls == 0 and elapsed < 0.5 and "third floor" in answer,
    )

    # ...and a healthy backend is completely unaffected by any of it.
    class _GoodCloud:
        def health_check(self):
            return None

        def generate_grounded_answer(self, question, chunks):
            return "cloud answer"

    healthy = ChainAnswerGenerator([("Cloud", _GoodCloud()), ("Local", _InstantLocal())])
    healthy.probe_now("visitor detected")
    for _ in range(50):
        if not healthy._probing:
            break
        time.sleep(0.05)
    all_ok &= _check(
        "a healthy cloud backend is still used, and probing never parks it",
        healthy.generate_grounded_answer("q", []) == "cloud answer" and not healthy._parked,
    )
    chain.close()
    healthy.close()

    print()
    if all_ok:
        print(
            "OK: a cloud backend that 503s, 429s, or returns nothing costs one retry against "
            "the local model instead of a wrong answer; a dead one is found while the visitor "
            "is still walking up, so their questions never wait on it; and a fully dead chain "
            "hands the decision back to DialogueManager rather than making one up."
        )
        return 0
    print("One or more checks failed - see above.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if "--self-test" in sys.argv:
        raise SystemExit(_run_chain_self_test())

    kb = KnowledgeBase()
    kb.upsert_many(list(SAMPLE_CHUNKS))
    print(
        f"Knowledge base: {len(kb)} chunk(s), embedded by {kb.embedding_model} "
        f"(relevance threshold {kb.similarity_threshold})."
    )

    # Says out loud who the robot currently thinks it is and where that came from (MySQL, the
    # offline cache, or the built-in seed) - the identity answers below are only as right as
    # that, and this REPL is where they get tried out.
    me = get_identity()
    print(
        f"Identity: {me.robot_name}, created by {me.creator_name} " f"(from {identity_source()})."
    )

    generator = _build_answer_generator()
    dm = DialogueManager(kb, generator)

    print("\nStage 05 live - type a question, Ctrl+C to quit.")
    print("Made-up sample knowledge base for this test:")
    for chunk in SAMPLE_CHUNKS:
        print(f"  - {chunk.text}")
    if isinstance(generator, UniversityOnlyAnswerGenerator):
        print(
            "\nTry a question about the above (grounded answer), and anything else (e.g. "
            "'what's the weather') to see the apology fallback - general chat is off in this "
            "mode.\n"
        )
    else:
        print(
            "\nTry a question about the above, something totally unrelated (weather, jokes, "
            "trivia), and a made-up university question (e.g. cafeteria menu) to see all three "
            "response types.\n"
        )

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            result = dm.handle_question(question)
            print(f"[{result.intent.name}] Robot: {result.answer}\n")
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
