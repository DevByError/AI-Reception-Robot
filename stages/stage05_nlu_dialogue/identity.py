"""
Stage 05 add-on - the robot's own identity: its name, and who built it.

Deliberately NOT routed through the knowledge base or an LLM. These facts aren't "content the
robot knows about" the way university facts are - they're fixed truths about the robot itself,
and the explicit requirement was that they work "even if he was offline". Everything below is a
plain string lookup plus rule-based phrase matching, so dialogue_manager.py checks this before
touching the knowledge base or any network call at all (see handle_question()) - no embeddings,
no API key, nothing that can fail or be slow.

--- WHERE THE VALUES COME FROM (changed 2026-09-04) -----------------------------------------
The facts below used to be module constants edited in this file. They now live in MySQL, in the
`robot_identity` table (Stage 02's robot_db.py owns the schema) - one field/value row each - so
renaming the robot or correcting an email is an UPDATE in Workbench, not a code change and a
redeploy. The robot's own name is data like everything else it knows.

That must not cost the offline guarantee above, so resolution falls through three sources, in
order, and identity_source() reports which one actually answered:

  1. MySQL                the source of truth. Read ONCE per process (see use_database) - never
                          per question, because a question must never wait on a round trip.
  2. identity_cache.json  written next to this file after every successful read. This is what
                          keeps a renamed robot correctly named during a MySQL outage: without
                          it, an outage would silently revert the robot to the seed values
                          below - a worse failure than being offline, because it still looks
                          like it is working.
  3. SEED_IDENTITY        last-resort built-in values, below. Also what a brand-new empty table
                          is seeded with on first connect, so a fresh install is never mute
                          about itself.

Nothing here raises: MySQL unreachable degrades to the cache, and no cache degrades to the seed
- the same graceful-degradation rule the rest of the robot follows.

--- CHANGING WHAT THE ROBOT SAYS ABOUT ITSELF -----------------------------------------------
    UPDATE robot_identity SET value = 'Nova' WHERE field = 'robot_name';
or, without touching SQL:
    python stages/stage05_nlu_dialogue/identity.py --set robot_name=Nova
and to see what it currently believes, and which of the three sources that came from:
    python stages/stage05_nlu_dialogue/identity.py --show
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# LUC_IDENTITY_CACHE moves it - the test suite points it at a scratch folder, so running the
# tests can never rewrite the real robot's offline copy of its name.
CACHE_FILE = Path(os.environ.get("LUC_IDENTITY_CACHE")
                  or Path(__file__).resolve().parent / "identity_cache.json")
_STAGE02_DIR = Path(__file__).resolve().parents[1] / "stage02_face_recognition"


@dataclass(frozen=True)
class Identity:
    """One field per row in `robot_identity`. Frozen because the name-derived phrase lists further
    down are cached against the profile they came from - a profile that could be mutated in place
    would leave those lists silently still describing the previous creator."""

    robot_name: str
    creator_name: str
    creator_email: str
    creator_nationality: str
    creator_country: str
    # Only used for "he's Malaysian" - every other answer says the creator's name outright. A
    # field rather than a literal in the f-string because it is a fact about a specific person,
    # and every other such fact now comes from the database.
    creator_pronoun: str = "he"

    def as_rows(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_rows(cls, rows: dict[str, str]) -> "Identity":
        """Builds a profile from whatever the table actually held, falling back per FIELD rather
        than per profile: a half-populated table (someone inserted `robot_name` by hand and
        nothing else) yields a working robot, not a KeyError. Unknown extra fields are ignored
        rather than rejected, so a row added for some future purpose can't stop the robot
        answering the questions it already knows how to answer."""
        known = {f: v for f, v in rows.items() if f in cls.__dataclass_fields__ and v}
        return replace(SEED_IDENTITY, **known)


# --- THE SEED VALUES -------------------------------------------------------------------------
# NOT what the robot says - what an empty `robot_identity` table gets populated with the first
# time it is connected to (and the last-resort fallback when there is neither a database nor a
# cache yet). Once a row exists, seeding never overwrites it, so editing these has no effect on
# a database that has already been seeded - change those values in MySQL, per the docstring.
SEED_IDENTITY = Identity(
    robot_name="Lina",
    creator_name="Alex Tan",
    creator_email="creator@example.com",
    creator_nationality="Malaysian",
    # Derived from the nationality, but kept as its own field since "he's from Malaysia" and "he's
    # Malaysian" are both natural things to say, and neither is mechanically derivable from the
    # other in general.
    creator_country="Malaysia",
    creator_pronoun="he",
)

_identity: Identity | None = None
_source: str = "not loaded"


# --- loading ----------------------------------------------------------------------------------


def _read_cache() -> dict[str, str] | None:
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_cache(rows: dict[str, str]) -> None:
    """Best effort by design - a read-only install directory is a reason to lose the offline
    fallback, not a reason to stop the robot from talking."""
    try:
        CACHE_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except OSError:
        pass


def _rows_from_db(db) -> dict[str, str] | None:
    """Reads the table, seeding any missing field from SEED_IDENTITY, and returns the rows.
    Returns None if the read itself failed, in which case the caller falls through to the
    cache."""
    try:
        rows = db.load_identity()
        # Seed from the last values this robot actually used (the cache mirrors the database)
        # before the built-in defaults. Added 2026-10-27 with the SQLite fallback: a brand-new
        # local database would otherwise rename a robot that has been "Nova" for months back to
        # the seed name the moment MySQL was unavailable.
        known = {**SEED_IDENTITY.as_rows(), **{k: v for k, v in (_read_cache() or {}).items()
                                               if k in SEED_IDENTITY.as_rows() and v}}
        missing = {f: v for f, v in known.items() if f not in rows}
        if missing:
            db.seed_identity_fields(missing)
            rows.update(missing)
        return rows
    except Exception as exc:  # a live query fails in ways a successful connect can't predict
        print(f"[identity] could not read robot_identity ({exc}) - falling back")
        return None


def _connect():
    """Opens a short-lived connection of our own. Only used when nobody handed us one: the
    assembled robot calls use_database() with the connection it already has (main_live.py), so
    this path is for running a Stage 05 module on its own."""
    if str(_STAGE02_DIR) not in sys.path:
        sys.path.append(str(_STAGE02_DIR))
    try:
        from robot_db import connect_or_none
    except ImportError:
        # mysql-connector-python is in stage02's requirements, not stage05's - a Stage 05-only
        # checkout legitimately doesn't have it, and identity questions still have to work there.
        return None
    return connect_or_none(verbose=False)


def use_database(db, *, verbose: bool = True) -> Identity:
    """Resolves the identity from an already-open RobotDatabase (or from cache/seed when `db` is
    None) and caches it for the rest of the process.

    This is what the orchestrator calls at startup, right after connect_or_none(): it reuses the
    connection the robot already has and - the point of doing it there - means the first visitor
    to ask "what's your name" is answered from memory instead of waiting on MySQL. Passing None
    is not a failure, it is the offline case: it marks the identity resolved from the cache, so
    no later question tries to connect mid-conversation either.
    """
    global _identity, _source

    rows = _rows_from_db(db) if db is not None else None
    if rows:
        _source = "database"
        _write_cache(rows)
    else:
        rows = _read_cache()
        _source = "identity_cache.json (MySQL unavailable)" if rows else "built-in seed values"

    _identity = Identity.from_rows(rows or {})
    if verbose:
        print(f"[identity] {_identity.robot_name}, created by {_identity.creator_name} "
              f"- from {_source}")
    return _identity


def get_identity(refresh: bool = False) -> Identity:
    """The active profile, loaded on first use. Every answer below goes through this."""
    if _identity is None or refresh:
        db = _connect()
        try:
            use_database(db, verbose=False)
        finally:
            if db is not None:
                db.close()
    return _identity


def set_identity(identity: Identity) -> None:
    """Overrides the active profile without touching MySQL - for tests, and for the self-test at
    the bottom of this file, which has to pass on a machine with no database."""
    global _identity, _source
    _identity = identity
    _source = "explicitly set"


def identity_source() -> str:
    """Which of the three sources answered. Printed at startup and by --show, so "the robot is
    using the wrong name" is one command to diagnose rather than a guess."""
    return _source


def save_identity(fields: dict[str, str], db=None) -> Identity:
    """Writes fields to MySQL and refreshes the active profile. Raises when there is no database
    - unlike reading, a write that silently did nothing would be a lie."""
    owned = db is None
    db = db or _connect()
    if db is None:
        raise RuntimeError(
            "cannot save identity: MySQL is unreachable. Check the settings in .env, and see "
            "stages/stage02_face_recognition/robot_db.py"
        )
    try:
        db.set_identity_fields(fields)
        return use_database(db, verbose=False)
    finally:
        if owned:
            db.close()


# ------------------------------------------------------------------------------------------
# Detection below. Same style as main_live.py's _parse_yes_no()/NEGATIVE_PHRASES - clean the
# text, then match short phrases as substrings, with single WORDS checked via a word-list
# membership test rather than substring containment. That distinction matters here more than it
# did there: a substring check for "his" would also match inside "this" ("what is THIS email
# for"), and a substring check for "dev" would match inside "device" or "development" - both
# real collision risks for a university robot where visa/admissions content plausibly mentions
# "nationality requirements" or a "mailing address". Word-list membership avoids both.


def _clean(text: str) -> str:
    cleaned = re.sub(r"[^a-z' ]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


# --- "who made you" ------------------------------------------------------------------------
# Bare nouns kept deliberately short: "creator" and "dev" are unlikely to occur in ordinary
# university content. "maker"/"designer"/"inventor"/"programmer" are NOT included bare - a real
# campus plausibly has a "makerspace"/"maker lab", and "designer" could appear describing a
# programme (e.g. "fashion designer") - those only count here when paired with "your" (see
# CREATOR_NOUN_PHRASES), which a facilities/programme question would never phrase that way.
CREATOR_BARE_TOKENS = {"creator", "dev"}

CREATOR_NOUN_PHRASES = [
    "your creator",
    "your maker",
    "your developer",
    "your dev",
    "your programmer",
    "your inventor",
    "your designer",
]

# Both tenses, since real speech uses either ("who made you" / "who's making you" is unlikely,
# but "who develops you" is a plausible present-tense phrasing). "you" must be the object -
# "who developed you" matches, "who developed the new curriculum" does not, so this can't
# misfire on a real university-history question.
CREATOR_VERB_PHRASES = [
    "who made you",
    "who make you",
    "who makes you",
    "made you",
    "who created you",
    "who create you",
    "who creates you",
    "created you",
    "who developed you",
    "who develop you",
    "who develops you",
    "developed you",
    "who built you",
    "who build you",
    "who builds you",
    "built you",
    "who designed you",
    "who design you",
    "who designs you",
    "designed you",
    "who programmed you",
    "who program you",
    "who programs you",
    "programmed you",
    "who coded you",
    "who code you",
    "who codes you",
    "coded you",
]

# --- "do you have a name" -------------------------------------------------------------------
NAME_PHRASES = [
    "your name",
    "have a name",
    "are you called",
    "what should i call you",
    "call you",
]
INTRO_PHRASES = [
    "who are you",
    "introduce yourself",
    "tell me about yourself",
    "what are you",
]

# --- "what is his email" ---------------------------------------------------------------------
EMAIL_TOKENS = {"email", "mail"}

# --- "where is he from" / "what's his nationality" -------------------------------------------
NATIONALITY_TOKEN = "nationality"
NATIONALITY_PHRASES = [
    "where is your creator from",
    "where's your creator from",
    "where is he from",
    "where's he from",
    "where does he come from",
    "what country is he from",
    "what country's he from",
]

# Shared "this is clearly about the creator" pronoun signal, used to let a bare "email"/
# "nationality" mention count even without the word "creator" itself present - e.g. the back
# half of a compound question ("...and what is his email") where "creator" was only said once,
# earlier in the same sentence. Word-list checked, not substring - see module docstring on why
# ("his" as a substring of "this" would otherwise misfire constantly).
CREATOR_PRONOUN_TOKENS = {"his", "him", "he", "their"}


# --- matching on the creator's own name -------------------------------------------------------
# "who is Alex" and "where is Tan from" have to keep working, which means the creator's name
# is part of the MATCHER, not only part of the answer - and it is now whatever the database says
# it is. So these two lists are derived from the loaded profile and memoized against it, instead
# of being written out by hand as they were while the name was a constant.
#
# Name parts shorter than three letters are dropped: a one- or two-letter fragment (an initial,
# or a particle like "al") is exactly the kind of token that would match inside an unrelated
# question, and the word-list membership test that protects "his"/"this" gives no protection
# against a real short word that happens to BE the fragment.
_MIN_NAME_TOKEN = 3
_derived_cache: tuple[Identity, frozenset, tuple] | None = None


def _derived(identity: Identity) -> tuple[frozenset, tuple]:
    """(name tokens, "where is <name> from" phrases) for this profile."""
    global _derived_cache
    if _derived_cache is not None and _derived_cache[0] == identity:
        return _derived_cache[1], _derived_cache[2]

    full_name = _clean(identity.creator_name)
    parts = [p for p in full_name.split() if len(p) >= _MIN_NAME_TOKEN]
    tokens = frozenset(parts)
    # Full name first so "where is alex tan from" matches as a whole phrase too, not only via
    # its parts - dict.fromkeys keeps the order and drops the duplicate for a one-word name.
    phrases = tuple(
        f"where is {p} from" for p in dict.fromkeys(([full_name] if full_name else []) + parts)
    )

    _derived_cache = (identity, tokens, phrases)
    return tokens, phrases


def _spell_out_email(email: str) -> str:
    """ "His email is creator@example.com" said aloud by TTS reads as one mangled glued
    word. Spelling the distinctive local part letter-by-letter (comma-paused, which Piper reads
    as a natural pause between letters) is how a person would actually read an email address out
    loud to someone taking it down - the domain half ("gmail.com") is common enough to just say
    normally rather than also spell out."""
    local, _, domain = email.partition("@")
    spelled_local = ", ".join(local)
    domain_spoken = domain.replace(".", " dot ")
    return f"{spelled_local}, at, {domain_spoken}"


def detect_identity_answer(question: str) -> str | None:
    """Returns a complete spoken answer if `question` is (or contains) one or more identity
    questions about the robot or its creator, or None if it isn't - in which case the caller
    should fall through to the normal knowledge-base/LLM path.

    Handles compound questions ("who created you and where is he from and what's his email")
    by detecting every sub-intent independently and answering all of them in one response,
    fixed order (name, creator, nationality, email) regardless of the order asked - simpler and
    just as clear spoken aloud, and avoids the real complexity of preserving asked-order across
    four independent boolean checks.
    """
    if not question:
        return None
    cleaned = _clean(question)
    if not cleaned:
        return None
    words = set(cleaned.split())

    # An attribute read on an already-loaded profile (see get_identity/use_database) - the
    # offline guarantee in the module docstring means this must never become a query per
    # question.
    identity = get_identity()
    creator_name_tokens, creator_from_phrases = _derived(identity)

    intro_intent = any(p in cleaned for p in INTRO_PHRASES)

    creator_intent = (
        bool(words & CREATOR_BARE_TOKENS)
        or bool(words & creator_name_tokens)
        or any(p in cleaned for p in CREATOR_NOUN_PHRASES)
        or any(p in cleaned for p in CREATOR_VERB_PHRASES)
        or intro_intent
    )
    name_intent = any(p in cleaned for p in NAME_PHRASES) or intro_intent
    email_intent = bool(words & EMAIL_TOKENS) and (
        creator_intent or bool(words & CREATOR_PRONOUN_TOKENS)
    )
    nationality_intent = (
        any(p in cleaned for p in NATIONALITY_PHRASES)
        or any(p in cleaned for p in creator_from_phrases)
        or (
            NATIONALITY_TOKEN in words
            and (creator_intent or bool(words & CREATOR_PRONOUN_TOKENS))
        )
    )

    if not (name_intent or creator_intent or nationality_intent or email_intent):
        return None

    fragments: list[str] = []
    if name_intent:
        fragments.append(f"my name is {identity.robot_name}.")
    if creator_intent:
        fragments.append(f"I was created by {identity.creator_name}.")
    if nationality_intent:
        fragments.append(
            f"{identity.creator_name} is from {identity.creator_country} - "
            f"{identity.creator_pronoun}'s {identity.creator_nationality}."
        )
    if email_intent:
        fragments.append(
            f"{identity.creator_name}'s email is {identity.creator_email}. Let me spell that "
            f"out for you: {_spell_out_email(identity.creator_email)}."
        )
    return " ".join(fragments)


if __name__ == "__main__":

    def _print_current() -> None:
        current = get_identity()
        print(f"Identity source: {identity_source()}")
        for field, value in current.as_rows().items():
            print(f"  {field:<20} {value}")

    if "--show" in sys.argv:
        _print_current()
        print(f"\nOffline cache: {CACHE_FILE} "
              f"({'present' if CACHE_FILE.exists() else 'not written yet'})")
        sys.exit(0)

    if "--set" in sys.argv:
        assignments: dict[str, str] = {}
        for argument in sys.argv[sys.argv.index("--set") + 1:]:
            field, sep, value = argument.partition("=")
            if not sep:
                print(f"Expected field=value, got {argument!r}")
                sys.exit(2)
            if field not in Identity.__dataclass_fields__:
                print(f"Unknown field {field!r}. Known fields: "
                      f"{', '.join(Identity.__dataclass_fields__)}")
                sys.exit(2)
            assignments[field] = value
        try:
            save_identity(assignments)
        except RuntimeError as exc:
            print(f"FAILED: {exc}")
            sys.exit(1)
        print(f"Wrote {', '.join(assignments)} to robot_identity.\n")
        _print_current()
        sys.exit(0)

    def _check(label: str, passed: bool) -> bool:
        print(f"  [{'OK' if passed else 'FAIL'}] {label}")
        return passed

    all_ok = True

    # Where the values are coming from on THIS machine, printed before the checks: a
    # "database"/"cache"/"seed" line is the first thing to look at if a case below fails.
    _print_current()
    print()

    # The checks are written against the ACTIVE profile rather than literal strings, so they
    # pass whether the values came from MySQL or the seed - that is the point of the change:
    # nothing outside the database decides what the robot's name is. The substituted profile at
    # the end then proves that independently.
    me = get_identity()
    email_prefix = ", ".join(me.creator_email.partition("@")[0][:5])

    # --- the exact questions from the spec, one at a time -----------------------------------
    cases = [
        ("do you have a name?", [me.robot_name]),
        ("who made you?", [me.creator_name]),
        ("who created you?", [me.creator_name]),
        ("who developed you?", [me.creator_name]),
        ("what is creator email?", [me.creator_name, me.creator_email, email_prefix]),
        ("what is creator mail?", [me.creator_email]),
        ("do your creator have an email?", [me.creator_email]),
        ("where is your creator from?", [me.creator_country]),
        ("what is your creator nationality?", [me.creator_nationality]),
        (f"what is {me.creator_name} nationality?", [me.creator_nationality]),
    ]
    for question, expected_substrings in cases:
        answer = detect_identity_answer(question)
        ok = answer is not None and all(s in answer for s in expected_substrings)
        all_ok &= _check(f"{question!r} -> {answer!r}", ok)

    # --- compound question, all four sub-intents in one utterance ---------------------------
    compound = (
        "who created you and where he is from and what is his email and what is his "
        "nationality"
    )
    answer = detect_identity_answer(compound)
    print(f"\nCompound: {compound!r}\n  -> {answer!r}")
    all_ok &= _check(
        "compound question answers all four parts",
        answer is not None
        and me.creator_name in answer
        and me.creator_country in answer
        and me.creator_nationality in answer
        and me.creator_email in answer,
    )

    # --- two-part and three-part compounds, not just the full four --------------------------
    all_ok &= _check(
        "'who is your creator and what is his nationality' answers both, not just one",
        (lambda a: a is not None and me.creator_name in a and me.creator_nationality in a)(
            detect_identity_answer("who is your creator and what is his nationality")
        ),
    )

    # --- must NOT fire on real university questions (false-positive regression) -------------
    negative_cases = [
        "what is the admissions office email",
        "where is the maker space located",
        "what are the nationality requirements for the student visa",
        "where is the library",
        "who is the head of the engineering department",
        "who developed the new curriculum",
        "what is the mailing address of the university",
    ]
    for question in negative_cases:
        answer = detect_identity_answer(question)
        all_ok &= _check(
            f"{question!r} correctly falls through (no identity match)", answer is None
        )

    # --- a completely different identity, proving nothing is hard-coded any more -------------
    # Substituted in memory (set_identity) rather than written to MySQL: this has to pass on a
    # machine with no database, and a self-test that renamed the real robot in a shared database
    # as a side effect of being run would be a genuinely bad way to find that out.
    print("\nSubstituting a different identity (what an UPDATE in robot_identity does):")
    set_identity(Identity(
        robot_name="Nova",
        creator_name="Mariam Haddad",
        creator_email="m.haddad@example.org",
        creator_nationality="Jordanian",
        creator_country="Jordan",
        creator_pronoun="she",
    ))
    substituted = detect_identity_answer("who are you and where is your creator from")
    print(f"  'who are you and where is your creator from' -> {substituted!r}")
    all_ok &= _check(
        "the substituted name, creator, country and pronoun are all what gets spoken",
        substituted is not None
        and all(s in substituted for s in ("Nova", "Mariam Haddad", "Jordan", "she's")),
    )
    all_ok &= _check(
        "the substituted creator's own name is matchable ('where is Haddad from')",
        (lambda a: a is not None and "Jordan" in a)(
            detect_identity_answer("where is Haddad from")
        ),
    )
    all_ok &= _check(
        "the previous creator's name no longer matches anything",
        detect_identity_answer("who is Alex") is None,
    )
    set_identity(me)

    print()
    if all_ok:
        print(
            "OK: identity questions (single and compound) are answered from the loaded profile, "
            "changing that profile changes every answer AND the matching, and real university "
            "questions that happen to share a word ('email', 'maker', 'nationality', "
            "'developed') are correctly left alone."
        )
    else:
        print(
            "One or more checks above failed - see the phrase/token lists above the failing "
            "case."
        )
