"""
Stage 02 - Face Recognition & Memory

Goal: recognize a previously stored face; register a new face only with explicit consent.
See docs/PRIVACY.md before changing any storage or retention logic.

This file is embedding-format agnostic: FaceMemoryStore takes an optional is_match function
so it works with the fake_embedding() placeholder below (Euclidean distance, used by this
file's own self-test) AND with real SFace embeddings from sface_embedder.py (cosine similarity,
used by face_memory_live.py) - same store/consent/retention logic either way.

Storage is a dict in memory PLUS an optional MySQL mirror (robot_db.py). Pass db= and the store
loads every enrolled person on startup and writes every change back, so faces survive a restart;
pass nothing and it behaves exactly as it always did, in memory only - which is what keeps this
file's own self-test below (and every mock-based test) runnable with no database at all.

The in-memory dict is not a cache to be optimized away later: recognition compares a live
embedding against every stored sample several times per second, and that has to be a plain
Python scan over arrays already in RAM. MySQL is the durable record, not the matching index.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable


RETENTION_DAYS = 120            # auto-expiry policy (PRIVACY.md): deleted if no visit within
                                 # this period

# reinforce() runs on every recognition tick - several times a second per visible person - so
# mirroring each one straight to MySQL would mean a delete-and-reinsert of somebody's whole
# embedding set a few times per second for as long as they stand there. The samples themselves
# are cheap to hold in memory and worthless if lost (the next visit re-collects equivalents), so
# the write is throttled: at most one persist per person per interval, and whatever the ring
# buffer holds at that moment is what gets written.
REINFORCE_PERSIST_INTERVAL_S = 60.0


MAX_EMBEDDINGS_PER_PERSON = 5  # cap on stored samples per person - see reinforce() below


def _evenly_spaced_subset(items: list, max_count: int) -> list:
    """Picks up to max_count items spread evenly across the whole list, instead of just the
    first N - used when register() gets more samples than MAX_EMBEDDINGS_PER_PERSON (e.g.
    observation-phase samples + a deliberate enrollment burst), so the kept samples still
    span the full capture instead of only the earliest moment."""
    if len(items) <= max_count:
        return items
    step = len(items) / max_count
    return [items[int(i * step)] for i in range(max_count)]


@dataclass
class FaceRecord:
    person_id: str
    name: str
    embeddings: list  # multiple samples per person (list[float] for the mock, list[np.ndarray]
                       # for real SFace) - matching against several samples taken at different
                       # angles/lighting is far more robust than a single frozen snapshot
    consent_given: bool
    first_seen: float
    last_seen: float
    visit_count: int = 1


def fake_embedding(face_crop) -> list[float]:
    """Placeholder used only by this file's own self-test below - see sface_embedder.py for
    the real implementation used by face_memory_live.py."""
    seed = float(face_crop)
    return [seed, seed * 0.5, seed * 0.25]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


EUCLIDEAN_MATCH_THRESHOLD = 0.6  # placeholder threshold, tuned for fake_embedding()'s scale only


# How much clear water the best candidate needs over the runner-up before the robot is willing
# to say a name out loud. In SFace cosine-similarity terms 0.05 is a small but real gap - two
# people landing inside it are not distinguishable from this one frame, and the honest answer is
# "I don't know you" rather than a coin toss between two students' identities. See
# FaceMemoryStore.find_match(); temporal voting in Stage 08 then gives borderline cases several
# more frames to resolve themselves before anything is said at all.
MATCH_AMBIGUITY_MARGIN = 0.05


def default_is_match(a: list[float], b: list[float]) -> bool:
    """Default matcher for the fake_embedding() placeholder - not meaningful for real
    embeddings, which should pass their own is_match (see sface_embedder.SFaceEmbedder)."""
    return _euclidean_distance(a, b) <= EUCLIDEAN_MATCH_THRESHOLD


class FaceMemoryStore:
    """A storage + recognition layer: an in-memory index for matching, optionally mirrored to
    MySQL for durability (see the module docstring for why it is both and not either).

    db: a robot_db.RobotDatabase, or None for the original memory-only behaviour. Every database
    write goes through _mirror(), which logs and continues on failure rather than raising - a
    student standing in front of the robot should still get recognized and answered when MySQL
    has gone away mid-session (Stage 09's graceful-degradation rule). The consequence is worth
    stating plainly: a failed write means that change is only in memory and will be lost on
    exit, which is why the warning is printed rather than swallowed."""

    def __init__(self, is_match: Callable[[object, object], bool] = default_is_match, db=None,
                 score: Callable[[object, object], float] | None = None,
                 ambiguity_margin: float = MATCH_AMBIGUITY_MARGIN):
        self._records: dict[str, FaceRecord] = {}
        self.is_match = is_match
        # Optional similarity function (SFaceEmbedder.similarity). With it, find_match() picks
        # the BEST candidate and refuses to guess between two close ones; without it, it keeps
        # the original first-match behaviour. See find_match().
        self.score = score
        self.ambiguity_margin = ambiguity_margin
        self.db = db
        self._last_persist: dict[str, float] = {}
        if db is not None:
            self.load_from_db()

    def load_from_db(self) -> int:
        """Rebuilds the in-memory index from MySQL. Returns how many people were loaded."""
        for row in self.db.load_people():
            self._records[row["person_id"]] = FaceRecord(
                person_id=row["person_id"],
                name=row["name"],
                embeddings=list(row["embeddings"]),
                consent_given=row["consent_given"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                visit_count=row["visit_count"],
            )
        return len(self._records)

    def __len__(self) -> int:
        """How many people the robot currently knows - so callers can report that after a
        startup load without reaching into _records."""
        return len(self._records)

    def _mirror(self, description: str, write) -> None:
        """Runs one database write, converting any failure into a warning."""
        if self.db is None:
            return
        try:
            write()
        except Exception as exc:
            print(f"[db] could not persist {description} ({exc!r}) - kept in memory only for "
                  f"this session.")

    def get(self, person_id: str) -> FaceRecord | None:
        """Looks somebody up by id when the caller already knows who they are - the recognition
        step has already happened and produced a person_id, and re-running find_match() against
        every stored embedding just to recover the record it came from would be absurd.

        Public for the same reason __len__ is: so callers don't reach into _records. Used by
        Stage 08 to read last_seen when building the greeting's recall line.
        """
        return self._records.get(person_id)

    def find_match(self, embedding) -> FaceRecord | None:
        """Who this face belongs to, or None.

        Matches against ANY of a record's stored samples - one match is enough, since a single
        failed comparison (bad angle/lighting on either side) shouldn't cost a recognition when
        another stored sample would have matched fine.

        BUT (fixed 2026-09-05) it returns the BEST match, not the first one found. The previous
        version walked the records in dictionary order and returned whoever cleared the threshold
        first, which is a misidentification waiting to happen and gets steadily more likely as
        more students enroll: if Ahmad was enrolled first and one of his samples scrapes 0.37,
        he is returned even when Sara scores 0.85 on the very same face. Greeting the wrong
        person by name is the worst failure this system has - it also attaches their whole
        conversation to somebody else's record.
        Falls back to the old first-match behaviour when no scoring function was supplied, so a
        caller that only has a boolean is_match still works.

        The margin is the second half of the fix. Two people scoring 0.62 and 0.60 is not a
        recognition, it is a coin toss - and answering "I don't know you" is always recoverable
        (the robot greets them generically, or asks their name), while answering with the wrong
        name is not.
        """
        if self.score is None:
            for rec in self._records.values():
                if any(self.is_match(embedding, sample) for sample in rec.embeddings):
                    return rec
            return None

        scored: list[tuple[float, FaceRecord]] = []
        for rec in self._records.values():
            if not rec.embeddings:
                continue
            best = max(self.score(embedding, sample) for sample in rec.embeddings)
            scored.append((best, rec))

        if not scored:
            return None

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_rec = scored[0]
        if not self.is_match(embedding, max(
            best_rec.embeddings, key=lambda s: self.score(embedding, s)
        )):
            return None  # nobody clears the threshold at all

        if len(scored) > 1:
            runner_up = scored[1][0]
            if best_score - runner_up < self.ambiguity_margin:
                print(f"[face] ambiguous match: {best_rec.name} {best_score:.3f} vs "
                      f"{scored[1][1].name} {runner_up:.3f} (need a {self.ambiguity_margin:.2f} "
                      f"gap) - treating as unknown rather than guessing.")
                return None

        return best_rec

    def register(self, name: str, embeddings, consent_given: bool) -> FaceRecord:
        """Only call this after getting explicit consent from the person (consent_given=True).

        embeddings: a list of one or more samples. Pass several (captured across a couple of
        seconds of natural head movement - see face_memory_live.py's enroll_new_face()) rather
        than a single frozen snapshot, or the very first re-recognition attempt at a slightly
        different angle will fail with nothing else to compare against."""
        if not consent_given:
            raise PermissionError("Cannot register a face without explicit consent - see "
                                   "PRIVACY.md")
        if not embeddings:
            raise ValueError("register() needs at least one embedding")
        now = time.time()
        rec = FaceRecord(
            person_id=str(uuid.uuid4()),
            name=name,
            embeddings=_evenly_spaced_subset(list(embeddings), MAX_EMBEDDINGS_PER_PERSON),
            consent_given=True,
            first_seen=now,
            last_seen=now,
        )
        self._records[rec.person_id] = rec
        self._last_persist[rec.person_id] = now
        self._mirror(
            f"the new record for {name!r}",
            lambda: self.db.insert_person(
                person_id=rec.person_id, name=rec.name, consent_given=True,
                first_seen=rec.first_seen, last_seen=rec.last_seen,
                visit_count=rec.visit_count, embeddings=rec.embeddings,
            ),
        )
        return rec

    def reinforce(self, rec: FaceRecord, embedding) -> None:
        """Adds another embedding sample to a known record (e.g. on a later successful
        recognition), improving future matching against natural angle/lighting variation.
        Caps at MAX_EMBEDDINGS_PER_PERSON, dropping the oldest sample - keeps recent
        appearance without growing forever.

        The mirrored write is throttled to once per REINFORCE_PERSIST_INTERVAL_S per person -
        see that constant for why."""
        rec.embeddings.append(embedding)
        if len(rec.embeddings) > MAX_EMBEDDINGS_PER_PERSON:
            rec.embeddings.pop(0)

        now = time.time()
        if now - self._last_persist.get(rec.person_id, 0.0) < REINFORCE_PERSIST_INTERVAL_S:
            return
        self._last_persist[rec.person_id] = now
        self._mirror(
            f"the refreshed face samples for {rec.name!r}",
            lambda: self.db.replace_embeddings(rec.person_id, rec.embeddings),
        )

    def mark_seen_again(self, rec: FaceRecord) -> None:
        rec.last_seen = time.time()
        rec.visit_count += 1
        self._mirror(
            f"visit #{rec.visit_count} for {rec.name!r}",
            lambda: self.db.touch_person(rec.person_id, rec.last_seen, rec.visit_count),
        )

    def forget(self, person_id: str) -> bool:
        """Implements the right to be forgotten ("forget me") - PRIVACY.md."""
        removed = self._records.pop(person_id, None)
        self._last_persist.pop(person_id, None)
        if removed is not None:
            self._mirror(
                f"the deletion of {removed.name!r}",
                lambda: self.db.delete_person(person_id),
            )
        return removed is not None

    def purge_expired(self, now: float | None = None) -> list[str]:
        """Deletes records past RETENTION_DAYS without a visit - called on a periodic
        schedule.

        Works off the in-memory records rather than issuing a single DELETE ... WHERE last_seen
        query, so memory and MySQL are guaranteed to expire exactly the same people. Anyone in
        the database but not in memory (added by another process since startup) is simply left
        for the next run's load, rather than being silently deleted by a sweep that never
        actually looked at them."""
        now = now or time.time()
        expired = [
            pid for pid, rec in self._records.items()
            if (now - rec.last_seen) > RETENTION_DAYS * 86400
        ]
        for pid in expired:
            del self._records[pid]
            self._last_persist.pop(pid, None)
            self._mirror(
                f"the retention expiry of {pid}",
                lambda pid=pid: self.db.delete_person(pid),
            )
        return expired


class FaceRecognitionFlow:
    """Ties together: detect face -> match -> (known: update visit) or (new: ask for name and
    consent)."""

    def __init__(self, store: FaceMemoryStore, embed_fn: Callable = fake_embedding):
        """
        embed_fn: turns raw input into an embedding. Defaults to fake_embedding() for this
        file's own self-test. face_memory_live.py passes embed_fn=lambda x: x instead, since
        it already computes the real SFace embedding before calling handle_face().
        """
        self.store = store
        self.embed_fn = embed_fn

    def handle_face(self, face_crop, ask_consent_and_name) -> tuple[str, FaceRecord | None]:
        """
        ask_consent_and_name: a function (replaced by the real voice interaction from
        Stage 04/05/06) returning (consent: bool, name: str | None).

        Returns (status, record):
          status = "known" | "registered" | "declined"
        """
        embedding = self.embed_fn(face_crop)
        match = self.store.find_match(embedding)
        if match:
            self.store.mark_seen_again(match)
            return "known", match

        consent, name = ask_consent_and_name()
        if not consent:
            return "declined", None

        rec = self.store.register(name=name, embeddings=[embedding], consent_given=True)
        return "registered", rec


if __name__ == "__main__":
    store = FaceMemoryStore()
    flow = FaceRecognitionFlow(store)

    # Simulate: the same "face" (crop=42) visits twice, and a second face (crop=99) declines
    # to register
    def consent_yes():
        return True, "Ahmad"

    def consent_no():
        return False, None

    status1, rec1 = flow.handle_face(face_crop=42, ask_consent_and_name=consent_yes)
    print(f"First visit (new face): status={status1}, name={rec1.name if rec1 else None}")

    status2, rec2 = flow.handle_face(face_crop=42, ask_consent_and_name=consent_yes)
    print(f"Second visit (same face): status={status2}, "
          f"visit_count={rec2.visit_count if rec2 else None}")

    status3, rec3 = flow.handle_face(face_crop=99, ask_consent_and_name=consent_no)
    print(f"Third face declined consent: status={status3}, record={rec3}")

    assert status1 == "registered"
    assert status2 == "known" and rec2.visit_count == 2
    assert status3 == "declined"
    print("\nOK: the base flow (register / recognize / decline) works correctly.")

    # --- the MySQL mirror, without a MySQL --------------------------------------------------
    # A stand-in that records the calls it receives instead of executing them, so the
    # write-through logic (what gets persisted, when, and what happens when a write fails) is
    # testable here with no database, no camera and no network - the same reason this file's
    # matching self-test above uses fake_embedding(). robot_db.py's own --self-test covers the
    # other half: that these calls do the right thing against a real server.
    class _FakeDB:
        def __init__(self, people=None, fail=False):
            self.calls: list[str] = []
            self._people = people or []
            self.fail = fail

        def _record(self, call):
            self.calls.append(call)
            if self.fail:
                raise RuntimeError("simulated database outage")

        def load_people(self):
            return self._people

        def insert_person(self, person_id, name, **kwargs):
            self._record(f"insert:{name}")

        def replace_embeddings(self, person_id, embeddings):
            self._record(f"replace_embeddings:{len(embeddings)}")

        def touch_person(self, person_id, last_seen, visit_count):
            self._record(f"touch:{visit_count}")

        def delete_person(self, person_id):
            self._record("delete")
            return True

    print("\nPersistence mirror:")

    # 1. a store handed existing rows comes up already knowing those people
    now = time.time()
    preloaded = _FakeDB(people=[{
        "person_id": "abc-123", "name": "Sara", "consent_given": True,
        "first_seen": now, "last_seen": now, "visit_count": 7,
        "embeddings": [fake_embedding(7)],
    }])
    restored = FaceMemoryStore(db=preloaded)
    assert len(restored) == 1
    match = restored.find_match(fake_embedding(7))
    assert match is not None and match.name == "Sara" and match.visit_count == 7
    print("  [OK] a stored person is recognized straight after startup, with their visit count")

    # 2. every kind of change is written through
    db = _FakeDB()
    store = FaceMemoryStore(db=db)
    rec = store.register(name="Ahmad", embeddings=[fake_embedding(1)], consent_given=True)
    store.mark_seen_again(rec)
    store.forget(rec.person_id)
    assert db.calls == ["insert:Ahmad", "touch:2", "delete"], db.calls
    print(f"  [OK] register / mark_seen_again / forget all reach the database: {db.calls}")

    # 3. reinforce() is throttled - it runs several times a second per visible person
    db = _FakeDB()
    store = FaceMemoryStore(db=db)
    rec = store.register(name="Ahmad", embeddings=[fake_embedding(1)], consent_given=True)
    for i in range(20):
        store.reinforce(rec, fake_embedding(1))
    assert db.calls == ["insert:Ahmad"], f"reinforce() wrote to the database: {db.calls}"
    store._last_persist[rec.person_id] = 0.0  # pretend the interval has elapsed
    store.reinforce(rec, fake_embedding(1))
    assert db.calls[-1].startswith("replace_embeddings:"), db.calls
    assert len(rec.embeddings) == MAX_EMBEDDINGS_PER_PERSON
    print(f"  [OK] 20 reinforcements wrote nothing; the next one after the interval wrote "
          f"once: {db.calls}")

    # 4. a database outage must not stop the robot recognizing people
    failing = _FakeDB(fail=True)
    store = FaceMemoryStore(db=failing)
    rec = store.register(name="Ahmad", embeddings=[fake_embedding(1)], consent_given=True)
    assert store.find_match(fake_embedding(1)) is not None, \
        "a failed database write lost the person from memory too"
    print("  [OK] a failing database warns but keeps working in memory (see the line above)")

    print("\nOK: persistence mirroring behaves correctly (loading, write-through, throttling, "
          "and degrading safely when the database is down).")

    # --- best match, not first match (2026-09-05) -----------------------------------------
    # The bug this replaces: find_match() walked the records in insertion order and returned
    # whoever cleared the threshold FIRST. With one person enrolled that is indistinguishable
    # from correct; with several it silently becomes "greet whoever enrolled earliest and is
    # vaguely similar", which is the worst failure a face system has.
    print("\n== Best-match selection ==")

    def cosine_like(a, b):
        """Stand-in scorer with the shape of SFaceEmbedder.similarity - higher is more alike."""
        return 1.0 - _euclidean_distance(a, b)

    def cosine_like_match(a, b):
        return cosine_like(a, b) >= 0.5

    store = FaceMemoryStore(is_match=cosine_like_match, score=cosine_like)
    # Enrolled FIRST, and only a mediocre match for the probe below.
    early = store.register("Ahmad", [[0.60, 0.0, 0.0]], consent_given=True)
    # Enrolled second, and a near-perfect match.
    late = store.register("Sara", [[0.99, 0.0, 0.0]], consent_given=True)

    probe = [1.0, 0.0, 0.0]
    print(f"  probe scores: Ahmad={cosine_like(probe, early.embeddings[0]):.2f}, "
          f"Sara={cosine_like(probe, late.embeddings[0]):.2f}")
    best = store.find_match(probe)
    print(f"  find_match -> {best.name if best else None}")
    assert best is not None and best.name == "Sara", (
        f"returned {best.name if best else None} - the first person to clear the threshold won "
        f"over the far better match, which is the misidentification bug this replaces"
    )

    # Two people who are genuinely too close to call must produce NO answer rather than a guess.
    # "I don't know you" is recoverable; the wrong name is not.
    ambiguous = FaceMemoryStore(is_match=cosine_like_match, score=cosine_like)
    ambiguous.register("Twin A", [[0.80, 0.0, 0.0]], consent_given=True)
    ambiguous.register("Twin B", [[0.81, 0.0, 0.0]], consent_given=True)
    tie = ambiguous.find_match([0.805, 0.0, 0.0])
    print(f"  two near-identical candidates -> {tie.name if tie else None}")
    assert tie is None, f"guessed {tie.name!r} between two indistinguishable people"

    # A clear winner must still be returned even when somebody else is ALSO above threshold -
    # the margin must not make the recognizer uselessly timid. (Note the two Twins above stay
    # unresolvable at any probe, which is correct: nothing separates them by more than 0.01.)
    crowded = FaceMemoryStore(is_match=cosine_like_match, score=cosine_like)
    crowded.register("Distant", [[0.60, 0.0, 0.0]], consent_given=True)
    crowded.register("Obvious", [[0.99, 0.0, 0.0]], consent_given=True)
    clear = crowded.find_match([1.0, 0.0, 0.0])
    print(f"  clear winner among several above threshold -> {clear.name if clear else None}")
    assert clear is not None and clear.name == "Obvious", clear

    # And a face nobody matches is still unknown.
    assert crowded.find_match([5.0, 5.0, 5.0]) is None

    # Without a scorer, the original first-match behaviour is preserved for callers that only
    # have a boolean matcher (this file's own mock, and anything built against the old API).
    legacy = FaceMemoryStore(is_match=cosine_like_match)
    legacy.register("First", [[0.60, 0.0, 0.0]], consent_given=True)
    legacy.register("Second", [[0.99, 0.0, 0.0]], consent_given=True)
    assert legacy.find_match(probe).name == "First"

    print("  [OK] best match wins, ambiguous pairs return nobody, legacy callers unchanged")
    print("\nOK: recognition picks the best candidate rather than the first, and refuses to "
          "choose between two people it cannot actually tell apart.")
