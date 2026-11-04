"""
Stage 05 (part 3) - what the robot REMEMBERS about a person, as opposed to what was said.

The robot has stored full conversation transcripts since 2026-09-04 (Stage 02's robot_db.py) and
has never once used them. A transcript is the wrong shape for recall: answering "what do I know
about this person" from raw utterances means replaying every line of every past visit through a
language model at the exact moment somebody is standing in front of the robot waiting to be
greeted. So this module does the work at the other end of the visit, when nobody is waiting:

  END of a visit    -> extract_facts()      one cheap call, a few short facts, stored
  START of a visit  -> build_recall_line()  pure string formatting, no model, no network

That second line is the entire point of the file. "Welcome back, Sara" says the robot identified
a face. "Hey Sara - did you get anywhere with the Master of Computer Science?" says it remembers
her, and nothing else in this project buys as much of that for as little work.

--- WHICH MODEL DOES THE EXTRACTING -----------------------------------------------------------
The LOCAL one first (Ollama), unlike every other generation path in this stage, which prefers
Gemini. Three reasons, all pointing the same way: this runs when the visitor has already walked
away, so its latency is invisible; it would otherwise spend paid/quota'd cloud calls on
bookkeeping rather than on answering someone; and if it fails there is no visitor to disappoint -
the robot simply remembers less. Cloud is the fallback here rather than the default, which is the
exact inverse of dialogue_manager_live.py's chain, and deliberately so.

--- PRIVACY ------------------------------------------------------------------------------------
These are free-text sentences about a real student, which makes them MORE sensitive than a face
embedding, not less: a vector is meaningless without the model that made it, a sentence is not.
Three rules, enforced here rather than trusted to callers:
  - Nothing is remembered about a visitor who did not consent to being remembered at all (they
    have no person_id, so there is nowhere to attach a fact - see main_live.py).
  - EXTRACTION_SYSTEM_PROMPT below explicitly refuses sensitive categories (health, religion,
    money troubles, grades, anything about another named person).
  - Facts cascade-delete with the person (robot_db.py's person_facts schema), so "forget me"
    really does.
See docs/PRIVACY.md.

Run: python visit_memory.py     (offline self-test - no model, no database needed)
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from project_env import load_env_file  # noqa: E402

load_env_file()

# Facts are tagged by kind on the way in, because the two kinds are used in completely different
# places and telling them apart afterwards with a heuristic would be guesswork:
#
#   INTEREST - what they were asking about, as a short noun phrase ("the Master of Computer
#              Science"). This is the only kind that can be dropped into a spoken recall
#              question, which is why the prompt insists on a phrase rather than a sentence:
#              "Did you get anywhere with the Master of Computer Science?" only works if the
#              stored text is a thing, not a statement about a thing.
#   DETAIL   - a short third-person statement worth knowing next time ("is in their final year").
#              Never spoken directly; it is context for a future answer.
INTEREST_PREFIX = "INTEREST: "
DETAIL_PREFIX = "DETAIL: "

MAX_FACTS_PER_VISIT = 3  # a visit that produces more than this is almost always the model
# padding rather than the visitor actually saying that much

EXTRACTION_SYSTEM_PROMPT = (
    "You read one short conversation between a university front-desk robot and a student, and "
    "write down only what is worth remembering for the NEXT time that student comes back.\n\n"
    "Output at most three lines. Each line is exactly one of:\n"
    "  INTEREST: <a short noun phrase for something they asked about>\n"
    "  DETAIL: <a short third-person statement about them>\n\n"
    "Rules:\n"
    "  - An INTEREST line must be a NOUN PHRASE, not a sentence: 'the Master of Computer "
    "Science', 'the January intake', 'scholarship options'. It will be read out later inside a "
    "question like 'did you get anywhere with ___', so it has to fit there. Keep the leading "
    "'the' wherever you would say it out loud - 'the Master of Computer Science', never "
    "'Master of Computer Science'.\n"
    "  - Write DETAIL lines in the third person: 'is in their final year', not 'you are in your "
    "final year'.\n"
    "  - Only write down something the student actually said. Never guess, never infer, and "
    "never write down anything the robot said.\n"
    "  - NEVER record: health or disability, religion, money problems, grades or exam results, "
    "immigration or visa status, anything about a person other than the student themselves, or "
    "anything they asked you to forget.\n"
    "  - Skip small talk, greetings and directions questions - 'where is the library' is not "
    "worth remembering.\n"
    "  - If nothing is worth remembering, output the single word NOTHING.\n\n"
    "Write nothing except those lines."
)

# How long before "welcome back" turns into "it's been a while". Two weeks is roughly the point
# where a student notices that the robot noticed - a shorter gap makes it sound like it is
# keeping a log, which is exactly the wrong feeling for something holding face data.
LONG_ABSENCE_S = 14 * 24 * 60 * 60
SAME_DAY_S = 8 * 60 * 60


def _strip_bullet(line: str) -> str:
    """Small models add '- ', '* ' or '1. ' to list lines no matter how the prompt is worded."""
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()


def parse_facts(raw: str) -> list[str]:
    """Turns a model's reply into tagged fact strings, dropping anything malformed.

    Deliberately strict: a line that does not carry one of the two known prefixes is discarded
    rather than stored as an untagged fact. An untagged fact could not be used for the recall
    line anyway (build_recall_line only speaks INTEREST facts), so keeping it would mean storing
    text about a real person that nothing will ever read - the worst of both privacy and
    usefulness.
    """
    facts: list[str] = []
    for raw_line in (raw or "").splitlines():
        line = _strip_bullet(raw_line)
        if not line or line.upper().startswith("NOTHING"):
            continue

        upper = line.upper()
        if upper.startswith("INTEREST:"):
            value = line.split(":", 1)[1].strip()
            prefix = INTEREST_PREFIX
        elif upper.startswith("DETAIL:"):
            value = line.split(":", 1)[1].strip()
            prefix = DETAIL_PREFIX
        else:
            continue

        value = value.strip(" .").strip()
        # A one-word "interest" ("programme", "university") is not specific enough to say back to
        # somebody, and an over-long one is a sentence that ignored the prompt.
        if len(value) < 3 or len(value) > 120:
            continue
        facts.append(f"{prefix}{value}")
        if len(facts) >= MAX_FACTS_PER_VISIT:
            break
    return facts


def format_transcript(turns: list[dict], max_turns: int = 20) -> str:
    """Renders robot_db.conversation_transcript() rows as plain dialogue for the extractor.

    Only the tail is kept: a visit long enough to overflow this is one where the early turns are
    greetings, and the recent ones are what the visitor actually came for.
    """
    lines = []
    for row in turns[-max_turns:]:
        speaker = "Student" if row.get("speaker") == "visitor" else "Robot"
        text = (row.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _default_completer():
    """Picks the model that will do the extracting - local first, cloud second, see the module
    docstring. Returns None if neither is usable, which is a normal outcome: the robot then
    simply remembers nothing from this visit and says nothing different next time.

    Imports are done inside the function on purpose. This module is imported by the orchestrator
    at startup, and neither Ollama's HTTP probe nor the Gemini SDK should be touched until a
    visit has actually ended.
    """
    try:
        import ollama_client

        if ollama_client.is_available() and ollama_client.has_model(ollama_client.CHAT_MODEL):

            def _complete(prompt: str) -> str:
                return ollama_client.chat(
                    prompt,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.1,  # this is extraction, not writing - near-deterministic
                )

            return _complete
    except Exception as exc:  # noqa: BLE001 - a broken local backend must not break the visit
        log.warning(f"[memory] local model unavailable for fact extraction ({exc!r}).")

    try:
        import os

        if os.environ.get("GEMINI_API_KEY"):
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            def _complete(prompt: str) -> str:
                from dialogue_manager_live import GEMINI_MODEL

                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=EXTRACTION_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=200,
                    ),
                )
                return response.text or ""

            return _complete
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[memory] cloud model unavailable for fact extraction ({exc!r}).")

    return None


def extract_facts(turns: list[dict], complete=None) -> list[str]:
    """Reads one finished visit and returns the few things worth remembering, tagged.

    Never raises. Every failure path - no model, a model that errors, a model that ignores the
    output format - returns an empty list, because the correct behaviour when the robot cannot
    work out what to remember is to remember nothing, not to interrupt the visit that just ended.
    """
    transcript = format_transcript(turns)
    if not transcript.strip():
        return []

    complete = complete or _default_completer()
    if complete is None:
        return []

    try:
        raw = complete(f"Conversation:\n{transcript}")
    except Exception as exc:  # noqa: BLE001 - see docstring
        log.warning(
            f"[memory] fact extraction failed ({exc!r}) - remembering nothing from this visit."
        )
        return []

    facts = parse_facts(raw)
    if facts:
        log.info(f"[memory] remembered {len(facts)} thing(s) from this visit.")
    return facts


def interests(facts: list[str]) -> list[str]:
    """The speakable half of what is stored - see INTEREST_PREFIX."""
    return [f[len(INTEREST_PREFIX) :] for f in facts if f.startswith(INTEREST_PREFIX)]


def details(facts: list[str]) -> list[str]:
    """The half that is context for an answer, never spoken as-is - see DETAIL_PREFIX."""
    return [f[len(DETAIL_PREFIX) :] for f in facts if f.startswith(DETAIL_PREFIX)]


# Deliberately templates rather than a model call. This runs while somebody is walking up to the
# robot, where every hundred milliseconds is visible, and the sentence is short enough that a
# language model adds risk (a hallucinated detail about a real person, spoken aloud, as the very
# first thing they hear) without adding quality.
_RECALL_TEMPLATES = (
    "Did you get anywhere with {interest}?",
    "How did you get on with {interest}?",
    "Last time you were asking about {interest} - any luck?",
    "Still looking into {interest}?",
)

_LONG_ABSENCE_OPENERS = (
    "It's been a while!",
    "Good to see you again!",
    "Long time no see!",
)

_SAME_DAY_OPENERS = (
    "Back again!",
    "Hello again!",
)


def build_recall_line(
    facts: list[str], last_seen: float | None = None, now: float | None = None, rng=None
) -> str:
    """The extra sentence appended to a returning visitor's greeting. "" when there is nothing
    worth saying, which is the common case and must stay silent rather than reaching for filler.

    Only ONE thing is ever recalled, even when several are stored. A robot that opens by listing
    three things it remembers about you is unsettling rather than friendly; a person mentions one.
    """
    import random

    rng = rng or random
    now = time.time() if now is None else now

    parts: list[str] = []
    if last_seen is not None:
        elapsed = now - last_seen
        if elapsed < SAME_DAY_S:
            parts.append(rng.choice(_SAME_DAY_OPENERS))
        elif elapsed > LONG_ABSENCE_S:
            parts.append(rng.choice(_LONG_ABSENCE_OPENERS))

    topics = interests(facts)
    if topics:
        parts.append(rng.choice(_RECALL_TEMPLATES).format(interest=topics[0]))

    return " ".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    # Offline throughout - no model, no database, no network. Everything below is the parsing and
    # formatting that decides what gets stored about a real person and what gets said back to
    # them, which is exactly the part that should not need a running LLM to verify.
    import random

    # --- parsing a well-behaved reply --------------------------------------------------------
    parsed = parse_facts(
        "INTEREST: the Master of Computer Science\n" "DETAIL: is in their final year of a diploma\n"
    )
    print(f"Parsed: {parsed}")
    assert parsed == [
        f"{INTEREST_PREFIX}the Master of Computer Science",
        f"{DETAIL_PREFIX}is in their final year of a diploma",
    ], parsed
    assert interests(parsed) == ["the Master of Computer Science"]
    assert details(parsed) == ["is in their final year of a diploma"]

    # --- parsing a badly-behaved one ---------------------------------------------------------
    # Small models bullet their lists, add commentary, and occasionally answer in prose. None of
    # that may become a stored "fact" about a real student.
    messy = parse_facts(
        "Here is what I found:\n"
        "- INTEREST: scholarship options\n"
        "  * DETAIL: asked about the January intake\n"
        "The student seemed happy.\n"
        "INTEREST: x\n"  # too short to say back to anybody
        "RANDOM: something untagged\n"
    )
    print(f"Parsed (messy): {messy}")
    assert messy == [
        f"{INTEREST_PREFIX}scholarship options",
        f"{DETAIL_PREFIX}asked about the January intake",
    ], messy

    assert parse_facts("NOTHING") == []
    assert parse_facts("") == []
    # The cap holds even when the model ignores it.
    assert len(parse_facts("\n".join(f"INTEREST: topic number {i}" for i in range(10)))) == 3

    # --- a visit that ends with no model available --------------------------------------------
    turns = [
        {"speaker": "robot", "text": "Hi there! How can I help you today?"},
        {"speaker": "visitor", "text": "How long is the Master of Computer Science?"},
        {"speaker": "robot", "text": "It's a two year programme."},
    ]
    assert (
        extract_facts(turns, complete=lambda _: (_ for _ in ()).throw(RuntimeError("down"))) == []
    )
    assert extract_facts([], complete=lambda _: "INTEREST: anything") == []
    print("Model failure and empty transcript both remember nothing - OK")

    assert "Student: How long is the Master of Computer Science?" in format_transcript(turns)

    # --- the spoken recall line ---------------------------------------------------------------
    rng = random.Random(0)
    now = time.time()
    facts = [f"{INTEREST_PREFIX}the Master of Computer Science"]

    same_day = build_recall_line(facts, last_seen=now - 3600, now=now, rng=rng)
    long_gap = build_recall_line(facts, last_seen=now - 40 * 24 * 3600, now=now, rng=rng)
    normal = build_recall_line(facts, last_seen=now - 3 * 24 * 3600, now=now, rng=rng)
    print(f"Same day : {same_day!r}")
    print(f"Long gap : {long_gap!r}")
    print(f"Few days : {normal!r}")

    assert "Master of Computer Science" in same_day
    assert any(opener in same_day for opener in _SAME_DAY_OPENERS)
    assert any(opener in long_gap for opener in _LONG_ABSENCE_OPENERS)
    # A few days is neither notable nor worth remarking on - just the recall, no opener.
    assert not any(o in normal for o in _SAME_DAY_OPENERS + _LONG_ABSENCE_OPENERS), normal

    # Nothing remembered, and nothing notable about the timing -> say nothing at all.
    assert build_recall_line([], last_seen=now - 3 * 24 * 3600, now=now, rng=rng) == ""
    # A DETAIL alone is never spoken - it is context for an answer, not a greeting.
    assert (
        build_recall_line(
            [f"{DETAIL_PREFIX}is in their final year"],
            last_seen=now - 3 * 24 * 3600,
            now=now,
            rng=rng,
        )
        == ""
    )
    # Only ever one thing recalled, however much is stored.
    many = [f"{INTEREST_PREFIX}topic {i}" for i in range(5)]
    line = build_recall_line(many, last_seen=None, now=now, rng=rng)
    assert sum(1 for i in range(5) if f"topic {i}" in line) == 1, line

    print(
        "\nOK: only well-formed, tagged facts are ever stored, a failing or missing model "
        "remembers nothing rather than breaking the visit, and the greeting recalls exactly "
        "one thing - or stays quiet."
    )
