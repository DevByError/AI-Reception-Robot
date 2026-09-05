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
  - ClaudeAnswerGenerator (Anthropic) - kept in the file, still fully working, but not checked
    by _build_answer_generator() by default (swapped for Gemini on 2026-08-17 per request).

WHY THE CHAIN EXISTS (2026-09-04): this project has twice been bitten by depending on a single
cloud backend - a Gemini 503 mid-answer (2026-08-19) and a 429 on the very first question of a
session (2026-08-17). Both times the robot fell all the way through to reading a raw scraped
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

import os
import sys
from pathlib import Path

# Loaded before GEMINI_API_KEY/GROQ_API_KEY are read in _build_answer_generator(), so an API key
# can live in the project-root .env alongside every other setting instead of having to be a real
# environment variable - see stage00_setup/project_env.py.
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from project_env import load_env_file  # noqa: E402

load_env_file()

from dialogue_manager import DialogueManager  # noqa: E402
from identity import get_identity, identity_source  # noqa: E402
from knowledge_base import KnowledgeBase, KnowledgeChunk  # noqa: E402

CLAUDE_MODEL = (
    "claude-haiku-4-5-20251001"  # fast/cheap tier - matches this stage's low-latency
)
# voice-interaction goal; swap to "claude-sonnet-5"
# if answer quality needs to be better and the extra
# latency/cost is acceptable
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = (
    "gemini-3.7-flash"  # confirmed current/GA model name via Google's own docs
)
# (ai.google.dev) as of 2026-08 - fast tier, matches this
# stage's low-latency voice-interaction goal same as
# CLAUDE_MODEL's choice above

# Deliberately two separate prompts/calls, not one free-form "answer this" call - they have
# opposite rules (never go beyond the given text vs. free to talk about anything), matching
# AnswerGenerator's two methods in dialogue_manager.py.

GROUNDED_SYSTEM_PROMPT = (
    "You are a friendly university front-desk robot receptionist. Every answer is spoken out "
    "loud through a speaker - there is no screen - so respond in plain spoken sentences only: "
    "no markdown, no bullet points, no headings, no asterisks. "
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
    "You are a friendly university front-desk robot receptionist. Every answer is spoken out "
    "loud through a speaker - there is no screen - so respond in plain spoken sentences only: "
    "no markdown, no bullet points, no headings, no asterisks.\n\n"
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

    def generate_grounded_answer(
        self, question: str, chunks: list[KnowledgeChunk]
    ) -> str:
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


class GeminiAnswerGenerator:
    """Implements the same AnswerGenerator protocol as MockAnswerGenerator
    (dialogue_manager.py) - a drop-in replacement, so DialogueManager itself never changes.
    Uses Google's Gemini API (google-genai SDK) instead of Anthropic's Claude - swapped in per
    request (2026-08-17) to use gemini-3.7-flash instead of Claude. ClaudeAnswerGenerator above
    is left in place (untouched, still fully working) rather than deleted, in case Claude is
    ever wanted again - _build_answer_generator() below just no longer checks for it first.
    """

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found - set the GEMINI_API_KEY environment variable (see "
                "README.md for how to get one from Google AI Studio)."
            )
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate_grounded_answer(
        self, question: str, chunks: list[KnowledgeChunk]
    ) -> str:
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
                max_output_tokens=300,
            ),
        )
        return response.text.strip()

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        from google.genai import types

        # Google Search grounding (2026-08-17, per request) - lets Gemini look something up for
        # real instead of only answering from whatever it happened to memorize during training,
        # which matters most for a smaller/less-documented institution like Lincoln University
        # College where the model's own training knowledge may be thin or stale. Deliberately
        # only enabled on this call, not generate_grounded_answer() above - the grounded path
        # must stay 100% restricted to the retrieved knowledge-base text, never go searching for
        # something else instead. GroqAnswerGenerator/ClaudeAnswerGenerator below don't have an
        # equivalent built-in search tool, so they rely on GENERAL_CLASSIFY_PROMPT's honesty
        # rule alone (no live verification) if you switch back to either of those.
        #
        # Search grounding is separately metered from plain generation (Google's own docs: a
        # monthly free-prompt quota, then billed per grounded request) - a real run
        # (2026-08-17) hit a 429 RESOURCE_EXHAUSTED on the very first question, well under any
        # published free quota, most likely because Google now requires billing to be set up on
        # an account before some features activate at all, even at $0 (a 2026 AI Studio change).
        # Rather than let that crash the whole interaction, try search first but fall back to a
        # plain (non-search) call if it fails for any reason - the robot still attempts to help
        # via the model's own knowledge instead of going silent because one quota-gated feature
        # wasn't available.
        base_config = {
            "system_instruction": GENERAL_CLASSIFY_PROMPT,
            "temperature": 0.7,  # higher - general chat can be more natural/varied
            "max_output_tokens": 300,
        }
        try:
            search_tool = types.Tool(google_search=types.GoogleSearch())
            response = self.client.models.generate_content(
                model=self.model,
                contents=question,
                config=types.GenerateContentConfig(**base_config, tools=[search_tool]),
            )
        except Exception as exc:
            print(f"[gemini] search-grounded call failed ({exc!r}) - retrying without search...")
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=question,
                    config=types.GenerateContentConfig(**base_config),
                )
            except Exception as exc2:
                print(f"[gemini] retry without search also failed ({exc2!r}) - giving up on "
                      f"this question.")
                # Same fail-safe direction as _parse_university_related() below: when truly
                # unable to tell, assume it might be university-related rather than risk an
                # unverified answer - the caller (dialogue_manager.py) uses FALLBACK_ANSWER for
                # a blank answer here, never a crash or silence.
                return True, ""

        raw = response.text.strip()
        return _parse_university_related(raw), _parse_answer(raw)


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
        self.client = Groq(api_key=key)
        self.model = model

    def generate_grounded_answer(
        self, question: str, chunks: list[KnowledgeChunk]
    ) -> str:
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
# (2026-08-19) where a scraped page-title chunk outranked the correct directory sentence for a
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

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.generators]

    def generate_grounded_answer(self, question: str, chunks: list[KnowledgeChunk]) -> str:
        errors = []
        for name, generator in self.generators:
            try:
                answer = generator.generate_grounded_answer(question, chunks)
            except Exception as exc:
                print(f"[chain] {name} failed on the grounded answer ({exc!r}) - trying the "
                      f"next backend.")
                errors.append(f"{name}: {exc!r}")
                continue
            if answer and answer.strip():
                if errors:
                    print(f"[chain] answered by {name} after {len(errors)} failure(s).")
                return answer
            print(f"[chain] {name} returned an empty grounded answer - trying the next backend.")
            errors.append(f"{name}: empty answer")
        raise RuntimeError(f"every answer backend failed ({'; '.join(errors)})")

    def classify_and_answer_general(self, question: str) -> tuple[bool, str]:
        for name, generator in self.generators:
            try:
                is_university_related, answer = generator.classify_and_answer_general(question)
            except Exception as exc:
                print(f"[chain] {name} failed on the general answer ({exc!r}) - trying the "
                      f"next backend.")
                continue
            if answer and answer.strip():
                return is_university_related, answer
            print(f"[chain] {name} had no general answer - trying the next backend.")
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

    def generate_grounded_answer(
        self, question: str, chunks: list[KnowledgeChunk]
    ) -> str:
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


def _build_cloud_generator():
    """Whichever cloud backend has a key, or None. Gemini first per request (2026-08-17) -
    ClaudeAnswerGenerator is still defined above and works fine if you want to switch back (add
    an ANTHROPIC_API_KEY check ahead of these)."""
    if os.environ.get("GEMINI_API_KEY"):
        return "Gemini", GeminiAnswerGenerator()
    if os.environ.get("GROQ_API_KEY"):
        return "Groq", GroqAnswerGenerator()
    return None


def _build_local_generator(warm: bool = True):
    """The local Ollama backend, or None if it isn't usable. Never raises - an absent local model
    is a normal, expected state (nobody has installed Ollama yet), not an error worth stopping
    for."""
    try:
        generator = OllamaAnswerGenerator()
    except Exception as exc:
        print(f"[ollama] local model unavailable: {exc}")
        return None

    if warm:
        # Pay the model load now, at startup, instead of on the first visitor's first question -
        # see ollama_client.warm_up() for why that difference is worth a few seconds here.
        try:
            import ollama_client

            elapsed = ollama_client.warm_up(generator.model)
            print(f"[ollama] {generator.model} warmed up in {elapsed:.1f}s.")
        except Exception as exc:
            # It answered is_available() a moment ago, so this is worth reporting but not worth
            # dropping the backend over - the first real question will simply pay the load.
            print(f"[ollama] warm-up failed ({exc!r}) - keeping the backend anyway.")
    return "Ollama", generator


def _build_answer_generator():
    if MODE == "university_only":
        print(
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
                "Install/pull it, or switch MODE back to \"auto\"."
            )
        print(f"Using the local model only: {local[1].model}.")
        return local[1]

    if MODE == "full":
        cloud = _build_cloud_generator()
        if cloud is None:
            raise RuntimeError(
                'MODE = "full" but no API key found - set GEMINI_API_KEY or GROQ_API_KEY (see '
                'README.md), or use MODE = "auto" to fall back to a local Ollama model.'
            )
        print(f"Using {cloud[0]} only (MODE = \"full\" - no local fallback).")
        return cloud[1]

    # --- MODE == "auto" ---------------------------------------------------------------------
    if MODE != "auto":
        print(f'Unknown MODE={MODE!r} - treating it as "auto".')

    backends = [b for b in (_build_cloud_generator(), _build_local_generator()) if b is not None]

    if not backends:
        print(
            "No LLM backend is available - no GEMINI_API_KEY/GROQ_API_KEY is set, and no local "
            "Ollama model is usable. Falling back to university-only answers: real knowledge-base "
            "retrieval still works and is still accurate, but general conversation is off and "
            "questions the knowledge base can't answer get the fixed apology. Install Ollama "
            "(README.md) to get a free local model with no API key."
        )
        return UniversityOnlyAnswerGenerator()

    if len(backends) == 1:
        name, generator = backends[0]
        note = ("no local Ollama fallback - if this API goes down mid-conversation the robot "
                "drops to knowledge-base answers only"
                if name != "Ollama" else
                "no cloud API key set, so this is also the fallback")
        print(f"Answer backend: {name} ({note}).")
        return generator

    chain = ChainAnswerGenerator(backends)
    print(f"Answer backends, in order: {' -> '.join(chain.names)}. Each question tries them in "
          f"turn, so an API failure costs a retry rather than a bad answer.")
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
        print("\nSKIPPED: this test needs a working local Ollama backend (see the message "
              "above, and README.md's 'Local LLM via Ollama').")
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
        f"a 429 on the cloud backend still produces a general answer "
        f"({general[:60]!r}...)",
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
    chain = ChainAnswerGenerator([
        ("BrokenCloud", _BrokenCloudGenerator("503")),
        ("BrokenLocal", _BrokenCloudGenerator("connection refused")),
    ])
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

    print()
    if all_ok:
        print("OK: a cloud backend that 503s, 429s, or returns nothing costs one retry against "
              "the local model instead of a wrong answer - and a fully dead chain hands the "
              "decision back to DialogueManager rather than making one up.")
        return 0
    print("One or more checks failed - see above.")
    return 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_run_chain_self_test())

    kb = KnowledgeBase()
    kb.upsert_many(list(SAMPLE_CHUNKS))
    print(f"Knowledge base: {len(kb)} chunk(s), embedded by {kb.embedding_model} "
          f"(relevance threshold {kb.similarity_threshold}).")

    # Says out loud who the robot currently thinks it is and where that came from (MySQL, the
    # offline cache, or the built-in seed) - the identity answers below are only as right as
    # that, and this REPL is where they get tried out.
    me = get_identity()
    print(f"Identity: {me.robot_name}, created by {me.creator_name} "
          f"(from {identity_source()}).")

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
