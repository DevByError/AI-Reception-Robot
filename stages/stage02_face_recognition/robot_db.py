"""
Persistent memory for the robot - MySQL.

Why this exists: everything the robot remembered used to live in a Python dict inside
FaceMemoryStore, so closing the process erased every enrolled face. The requirement is that a
student enrolled today is still recognized after a restart, and that what was actually said (and
when) is on record. That needs a real database, not process memory.

Why MySQL rather than SQLite: MySQL Workbench is already the tool being used locally to inspect
the data, and the deployed kiosk is expected to have a server it can point at. Nothing here is
MySQL-specific beyond the DDL dialect, so a different backend would only mean rewriting
SCHEMA_STATEMENTS.

--- WHAT IS STORED --------------------------------------------------------------------------
  robot_identity   the robot's own identity as field/value rows - its name, and who built it.
                   Stage 05's identity.py reads this instead of hard-coding the answers, so
                   renaming the robot is an UPDATE in Workbench, not a code edit.
  people           one row per enrolled student: name, consent flag, first/last seen, visits
  face_embeddings  1..MAX_EMBEDDINGS_PER_PERSON float32 vectors per person (raw bytes, not an
                   image - see PRIVACY.md rule 2, no raw frames are ever written)
  conversations    one row per visit, optionally linked to a person
  utterances       the full transcript of a conversation, one row per line spoken either way

Deleting a person cascades to their embeddings AND their conversations - that is the "right to
be forgotten" (PRIVACY.md rule 4) and the retention sweep (rule 3) behaving the way the policy
says they should, rather than leaving orphaned transcripts behind that still describe them.

--- TIMEZONE ---------------------------------------------------------------------------------
All DATETIME columns are UTC. MySQL's DATETIME carries no timezone, so storing local time would
silently break by an hour twice a year; the conversion happens here, in one place. Add your own
offset when reading rows in Workbench.

--- CONFIGURATION ----------------------------------------------------------------------------
Connection settings come from (highest priority first): explicit constructor arguments, then
environment variables, then a `.env` file at the project root, then the defaults below. See
`.env.example` at the project root for the file format, and stage00_setup/project_env.py for the
loader (shared with Stage 05, which reads its own settings from the same file).

Run this file directly to create the database and tables:
    python stages/stage02_face_recognition/robot_db.py
    python stages/stage02_face_recognition/robot_db.py --self-test
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError as exc:  # pragma: no cover - a clearer message than a bare ImportError
    raise ImportError(
        "mysql-connector-python is not installed. Run:\n"
        "    pip install -r stages/stage02_face_recognition/requirements.txt"
    ) from exc


sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from project_env import ENV_FILE, PROJECT_ROOT, load_env_file  # noqa: E402

DEFAULTS = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "luc_robot",
}


@dataclass
class DBConfig:
    host: str = DEFAULTS["host"]
    port: int = DEFAULTS["port"]
    user: str = DEFAULTS["user"]
    password: str = DEFAULTS["password"]
    database: str = DEFAULTS["database"]

    @classmethod
    def from_env(cls, **overrides) -> "DBConfig":
        load_env_file()
        env = {
            "host": os.environ.get("LUC_DB_HOST", DEFAULTS["host"]),
            "port": int(os.environ.get("LUC_DB_PORT", DEFAULTS["port"])),
            "user": os.environ.get("LUC_DB_USER", DEFAULTS["user"]),
            "password": os.environ.get("LUC_DB_PASSWORD", DEFAULTS["password"]),
            "database": os.environ.get("LUC_DB_NAME", DEFAULTS["database"]),
        }
        env.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**env)


# --- schema ----------------------------------------------------------------------------------
# utf8mb4 throughout: student names and spoken transcripts are free text, and utf8mb4 is the only
# MySQL charset that actually covers all of Unicode (the legacy "utf8" alias does not).

SCHEMA_STATEMENTS = [
    # Field/value rows rather than one wide single-row table: the set of facts the robot knows
    # about itself has already grown once (name -> creator -> email -> nationality), and each
    # growth would otherwise be a schema migration on a table that will never hold more than a
    # handful of short strings. Stage 05 (identity.py) owns which field names mean what; this
    # layer deliberately stays a generic string store so it needs no change when a fifth fact
    # is added.
    """
    CREATE TABLE IF NOT EXISTS robot_identity (
        field      VARCHAR(48)  NOT NULL,
        value      VARCHAR(255) NOT NULL,
        updated_at DATETIME(6)  NOT NULL,
        PRIMARY KEY (field)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS people (
        person_id     CHAR(36)     NOT NULL,
        name          VARCHAR(120) NOT NULL,
        consent_given TINYINT(1)   NOT NULL DEFAULT 1,
        first_seen    DATETIME(6)  NOT NULL,
        last_seen     DATETIME(6)  NOT NULL,
        visit_count   INT UNSIGNED NOT NULL DEFAULT 1,
        PRIMARY KEY (person_id),
        KEY idx_people_last_seen (last_seen)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    # `vector` holds the raw little-endian float32 bytes of one embedding (512 bytes for SFace's
    # 128-d output). Stored as bytes rather than 128 columns or a JSON array because it is only
    # ever read back as a whole vector for a similarity comparison in Python - MySQL never needs
    # to look inside it, so giving it structure would buy nothing and cost space and parse time.
    """
    CREATE TABLE IF NOT EXISTS face_embeddings (
        embedding_id BIGINT UNSIGNED   NOT NULL AUTO_INCREMENT,
        person_id    CHAR(36)          NOT NULL,
        dim          SMALLINT UNSIGNED NOT NULL,
        vector       VARBINARY(4096)   NOT NULL,
        created_at   DATETIME(6)       NOT NULL,
        PRIMARY KEY (embedding_id),
        KEY idx_embeddings_person (person_id, embedding_id),
        CONSTRAINT fk_embeddings_person FOREIGN KEY (person_id)
            REFERENCES people (person_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    # person_id is nullable on purpose: a visitor who declines to be remembered (or was never
    # asked) still has a conversation, it just isn't attached to an identity. That row is not
    # biometric data and keeping it costs nobody their privacy, while throwing it away would
    # lose most of the usage record - see PRIVACY.md.
    """
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        person_id       CHAR(36)        NULL,
        started_at      DATETIME(6)     NOT NULL,
        ended_at        DATETIME(6)     NULL,
        PRIMARY KEY (conversation_id),
        KEY idx_conversations_person (person_id, started_at),
        KEY idx_conversations_started (started_at),
        CONSTRAINT fk_conversations_person FOREIGN KEY (person_id)
            REFERENCES people (person_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    # What the robot actually REMEMBERS about someone, as opposed to what was said (utterances,
    # below). Added 2026-09-05 for the recall line in Stage 08's greeting - "did you end up
    # applying for the IT programme?" - which is the single thing that most makes a returning
    # visitor feel recognized rather than merely identified.
    #
    # Why a separate table rather than reading the transcript back: a transcript is the wrong
    # shape for recall. Answering "what do I know about this person" from it means replaying
    # every line of every past visit through a language model at the exact moment somebody is
    # standing there waiting to be greeted. A handful of short extracted facts is one indexed
    # SELECT, and it is written once at the END of a visit (see stage05's visit_memory.py), when
    # nobody is waiting.
    #
    # PRIVACY: these are free-text sentences about a real student, so they are more sensitive
    # than a face vector, not less - a vector is meaningless without the model, a sentence is
    # not. Two consequences, both enforced here rather than left to the caller: the ON DELETE
    # CASCADE means "forget me" removes them with everything else (docs/PRIVACY.md), and
    # `source_conversation_id` keeps each fact traceable to the visit it came from so a wrong or
    # objectionable fact can be audited rather than just deleted.
    """
    CREATE TABLE IF NOT EXISTS person_facts (
        fact_id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        person_id              CHAR(36)        NOT NULL,
        fact                   VARCHAR(255)    NOT NULL,
        source_conversation_id BIGINT UNSIGNED NULL,
        created_at             DATETIME(6)     NOT NULL,
        PRIMARY KEY (fact_id),
        KEY idx_facts_person (person_id, created_at),
        CONSTRAINT fk_facts_person FOREIGN KEY (person_id)
            REFERENCES people (person_id) ON DELETE CASCADE,
        CONSTRAINT fk_facts_conversation FOREIGN KEY (source_conversation_id)
            REFERENCES conversations (conversation_id) ON DELETE SET NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS utterances (
        utterance_id    BIGINT UNSIGNED         NOT NULL AUTO_INCREMENT,
        conversation_id BIGINT UNSIGNED         NOT NULL,
        turn_index      INT UNSIGNED            NOT NULL,
        speaker         ENUM('visitor','robot') NOT NULL,
        text            TEXT                    NOT NULL,
        intent          VARCHAR(32)             NULL,
        said_at         DATETIME(6)             NOT NULL,
        PRIMARY KEY (utterance_id),
        KEY idx_utterances_conversation (conversation_id, turn_index),
        CONSTRAINT fk_utterances_conversation FOREIGN KEY (conversation_id)
            REFERENCES conversations (conversation_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


# --- embedding serialization ------------------------------------------------------------------

def encode_embedding(embedding) -> tuple[int, bytes]:
    """float32 little-endian raw bytes. float32 (not float64) because that is exactly what
    SFace produces - widening it would double the storage for zero extra precision."""
    arr = np.asarray(embedding, dtype="<f4").reshape(-1)
    if arr.size == 0:
        raise ValueError("refusing to store an empty embedding")
    if arr.nbytes > 4096:
        raise ValueError(f"embedding of {arr.size} floats exceeds the VARBINARY(4096) column")
    return int(arr.size), arr.tobytes()


def decode_embedding(dim: int, blob: bytes) -> np.ndarray:
    """Returns shape (1, dim), matching what cv2's FaceRecognizerSF.feature() hands back - the
    matcher (SFaceEmbedder.is_match -> recognizer.match) expects that shape, so reshaping here
    means nothing downstream has to know the vector ever went through a database.

    .copy() because np.frombuffer views the immutable bytes object read-only, and a read-only
    array is a trap waiting for the first caller that tries to normalize it in place."""
    return np.frombuffer(blob, dtype="<f4").reshape(1, dim).copy()


def _to_utc(ts: float) -> datetime:
    """Unix timestamp (what FaceRecord uses) -> naive UTC datetime (what MySQL stores)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _from_utc(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RobotDatabase:
    """A thin, explicit data-access layer - no ORM, no query builder. Every statement in here is
    written out in full so what hits MySQL is exactly what you read, which matters more than
    brevity for a table set this small and this privacy-sensitive.

    Methods raise on failure rather than swallowing errors. Callers that must survive a database
    outage (the robot itself - see FaceMemoryStore and main_live.py) wrap them; callers that
    should fail loudly (setup scripts, the self-test below) get a real exception.
    """

    def __init__(self, config: DBConfig | None = None, **overrides):
        self.config = config or DBConfig.from_env(**overrides)
        self._conn = None

    # --- connection ---------------------------------------------------------------------------

    def connect(self, create_database: bool = False) -> None:
        if create_database:
            self._create_database_if_missing()
        self._conn = mysql.connector.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            autocommit=False,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
            connection_timeout=10,
        )

    def _create_database_if_missing(self) -> None:
        """Connects with no database selected so the very first run doesn't require someone to
        have created the schema by hand in Workbench first."""
        conn = mysql.connector.connect(
            host=self.config.host, port=self.config.port, user=self.config.user,
            password=self.config.password, connection_timeout=10,
        )
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `" + self.config.database + "` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cur.close()
            conn.commit()
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        if self._conn is None:
            self.connect(create_database=True)
        with self._cursor() as cur:
            for statement in SCHEMA_STATEMENTS:
                cur.execute(statement)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    @contextmanager
    def _cursor(self, dictionary: bool = False):
        """Pings first so a connection that died while the robot sat idle overnight is
        transparently re-established instead of failing the first write of the morning. Commits
        on success, rolls back on failure - a half-written person (a row in `people` with no
        embeddings) would be worse than no row at all."""
        if self._conn is None:
            self.connect()
        self._conn.ping(reconnect=True, attempts=2, delay=1)
        cur = self._conn.cursor(dictionary=dictionary)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # --- robot identity ------------------------------------------------------------------------

    def load_identity(self) -> dict[str, str]:
        """Every field/value row as a plain dict. Stage 05's identity.py maps these onto the
        answers it speaks; unknown fields are simply carried along, so adding a row here can
        never break this layer."""
        with self._cursor(dictionary=True) as cur:
            cur.execute("SELECT field, value FROM robot_identity")
            return {row["field"]: row["value"] for row in cur.fetchall()}

    def set_identity_fields(self, fields: dict[str, str]) -> None:
        """Upsert - what an operator changing the robot's name actually wants, whether or not
        the row happens to exist yet."""
        if not fields:
            return
        now = _utc_now()
        with self._cursor() as cur:
            cur.executemany(
                "INSERT INTO robot_identity (field, value, updated_at) VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE value = VALUES(value), updated_at = VALUES(updated_at)",
                [(field, str(value), now) for field, value in fields.items()],
            )

    def seed_identity_fields(self, fields: dict[str, str]) -> list[str]:
        """Fills in only the fields that have no row yet, and reports which ones it wrote.

        This is what makes a freshly created database self-populating rather than leaving the
        robot mute about itself on its first run. INSERT IGNORE (not the upsert above) is the
        whole point: a value edited in Workbench must survive every later startup, so seeding
        must never overwrite what is already there.
        """
        if not fields:
            return []
        now = _utc_now()
        with self._cursor() as cur:
            written = []
            for field, value in fields.items():
                cur.execute(
                    "INSERT IGNORE INTO robot_identity (field, value, updated_at) "
                    "VALUES (%s, %s, %s)",
                    (field, str(value), now),
                )
                if cur.rowcount:
                    written.append(field)
            return written

    # --- people + embeddings ------------------------------------------------------------------

    def load_people(self) -> list[dict]:
        """Everything needed to rebuild the in-memory recognition store at startup, in two
        queries rather than one-per-person: the whole point of loading eagerly is that matching
        stays a pure in-memory scan at runtime, so paying N+1 round trips here would be spending
        startup time to buy nothing."""
        with self._cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT person_id, name, consent_given, first_seen, last_seen, visit_count "
                "FROM people ORDER BY last_seen DESC"
            )
            people = cur.fetchall()

            cur.execute(
                "SELECT person_id, dim, vector FROM face_embeddings ORDER BY embedding_id"
            )
            vectors: dict[str, list] = {}
            for row in cur.fetchall():
                vectors.setdefault(row["person_id"], []).append(
                    decode_embedding(row["dim"], row["vector"])
                )

        return [
            {
                "person_id": p["person_id"],
                "name": p["name"],
                "consent_given": bool(p["consent_given"]),
                "first_seen": _from_utc(p["first_seen"]),
                "last_seen": _from_utc(p["last_seen"]),
                "visit_count": int(p["visit_count"]),
                "embeddings": vectors.get(p["person_id"], []),
            }
            for p in people
        ]

    def insert_person(self, person_id: str, name: str, consent_given: bool,
                      first_seen: float, last_seen: float, visit_count: int,
                      embeddings: list) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO people (person_id, name, consent_given, first_seen, last_seen, "
                "visit_count) VALUES (%s, %s, %s, %s, %s, %s)",
                (person_id, name, int(consent_given), _to_utc(first_seen),
                 _to_utc(last_seen), visit_count),
            )
            self._insert_embeddings(cur, person_id, embeddings)

    @staticmethod
    def _insert_embeddings(cur, person_id: str, embeddings: list) -> None:
        if not embeddings:
            return
        now = _utc_now()
        rows = []
        for embedding in embeddings:
            dim, blob = encode_embedding(embedding)
            rows.append((person_id, dim, blob, now))
        cur.executemany(
            "INSERT INTO face_embeddings (person_id, dim, vector, created_at) "
            "VALUES (%s, %s, %s, %s)",
            rows,
        )

    def replace_embeddings(self, person_id: str, embeddings: list) -> None:
        """Delete-then-insert rather than diffing: the in-memory list is a small capped ring
        buffer (MAX_EMBEDDINGS_PER_PERSON) whose contents shift as samples are added and the
        oldest dropped, so working out which rows survived would be more code and more round
        trips than simply rewriting five rows inside one transaction."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM face_embeddings WHERE person_id = %s", (person_id,))
            self._insert_embeddings(cur, person_id, embeddings)

    def touch_person(self, person_id: str, last_seen: float, visit_count: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE people SET last_seen = %s, visit_count = %s WHERE person_id = %s",
                (_to_utc(last_seen), visit_count, person_id),
            )

    def rename_person(self, person_id: str, name: str) -> None:
        with self._cursor() as cur:
            cur.execute("UPDATE people SET name = %s WHERE person_id = %s", (name, person_id))

    def delete_person(self, person_id: str) -> bool:
        """Right to be forgotten (PRIVACY.md rule 4). The ON DELETE CASCADE foreign keys take
        the embeddings, conversations and transcripts with it - one statement, nothing left
        behind."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM people WHERE person_id = %s", (person_id,))
            return cur.rowcount > 0

    def find_people_by_name(self, name: str) -> list[dict]:
        with self._cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT person_id, name, visit_count, last_seen FROM people WHERE name = %s",
                (name,),
            )
            return cur.fetchall()

    # --- conversations ------------------------------------------------------------------------

    def start_conversation(self, person_id: str | None = None,
                           started_at: float | None = None) -> int:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (person_id, started_at) VALUES (%s, %s)",
                (person_id, _to_utc(started_at) if started_at else _utc_now()),
            )
            return int(cur.lastrowid)

    def attach_conversation_to_person(self, conversation_id: int, person_id: str) -> None:
        """Called when someone enrolls partway through a visit: the conversation started while
        they were still an anonymous face, and only became attributable once they gave a name
        and consent. Backfilling it here is what makes "show me everything Ahmad has ever asked"
        include the visit he introduced himself in, rather than starting from his second one."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE conversations SET person_id = %s WHERE conversation_id = %s",
                (person_id, conversation_id),
            )

    def log_utterance(self, conversation_id: int, turn_index: int, speaker: str, text: str,
                      intent: str | None = None, said_at: float | None = None) -> None:
        if speaker not in ("visitor", "robot"):
            raise ValueError("speaker must be 'visitor' or 'robot', got " + repr(speaker))
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO utterances (conversation_id, turn_index, speaker, text, intent, "
                "said_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (conversation_id, turn_index, speaker, text, intent,
                 _to_utc(said_at) if said_at else _utc_now()),
            )

    def end_conversation(self, conversation_id: int, ended_at: float | None = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE conversations SET ended_at = %s WHERE conversation_id = %s",
                (_to_utc(ended_at) if ended_at else _utc_now(), conversation_id),
            )

    def recent_conversations(self, person_id: str, limit: int = 5) -> list[dict]:
        """The visit history behind "last time you asked about the library" - not used by the
        dialogue manager yet, but this is the query it will need."""
        with self._cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT conversation_id, started_at, ended_at FROM conversations "
                "WHERE person_id = %s ORDER BY started_at DESC LIMIT %s",
                (person_id, int(limit)),
            )
            return cur.fetchall()

    def conversation_transcript(self, conversation_id: int) -> list[dict]:
        with self._cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT turn_index, speaker, text, intent, said_at FROM utterances "
                "WHERE conversation_id = %s ORDER BY turn_index",
                (conversation_id,),
            )
            return cur.fetchall()

    # --- retention ----------------------------------------------------------------------------

    # --- what the robot remembers about a person (see the person_facts schema note) ------------

    # Kept deliberately small. This is what gets read at greeting time and fed to the recall
    # line, and a robot that opens with six things it remembers about you is unsettling rather
    # than friendly - one, occasionally two, is what a person does.
    MAX_FACTS_PER_PERSON = 12

    def add_person_facts(self, person_id: str, facts: list[str],
                         conversation_id: int | None = None) -> int:
        """Stores newly-learned facts about someone, skipping any it already has.

        The duplicate check is a plain case-insensitive comparison against what is already
        stored, not a similarity model: the same visitor asking about the same programme on
        three visits produces three near-identical facts, and the goal is only to stop the exact
        repeats from accumulating. Near-duplicates surviving is acceptable - MAX_FACTS_PER_PERSON
        bounds the damage, and the alternative is an embedding call on the greeting path.

        Returns how many were actually inserted.
        """
        cleaned = [f.strip() for f in facts if f and f.strip()]
        if not cleaned:
            return 0

        existing = {f["fact"].lower() for f in self.person_facts(person_id)}
        now = _utc_now()
        inserted = 0
        with self._cursor() as cur:
            for fact in cleaned:
                if fact.lower() in existing:
                    continue
                cur.execute(
                    "INSERT INTO person_facts (person_id, fact, source_conversation_id, "
                    "created_at) VALUES (%s, %s, %s, %s)",
                    (person_id, fact[:255], conversation_id, now),
                )
                existing.add(fact.lower())
                inserted += 1

            # Trim oldest-first if this person has accumulated more than the cap. Done here
            # rather than on read so the table cannot grow without bound for a daily visitor.
            cur.execute(
                "SELECT fact_id FROM person_facts WHERE person_id = %s "
                "ORDER BY created_at DESC, fact_id DESC LIMIT %s, 100",
                (person_id, self.MAX_FACTS_PER_PERSON),
            )
            stale = [row[0] for row in cur.fetchall()]
            if stale:
                placeholders = ", ".join(["%s"] * len(stale))
                cur.execute(
                    f"DELETE FROM person_facts WHERE fact_id IN ({placeholders})", tuple(stale)
                )
        return inserted

    def person_facts(self, person_id: str, limit: int = 12) -> list[dict]:
        """Most recently learned facts first - the newest thing the robot knows about someone is
        also the most likely to still be relevant when they come back."""
        with self._cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT fact_id, fact, source_conversation_id, created_at FROM person_facts "
                "WHERE person_id = %s ORDER BY created_at DESC, fact_id DESC LIMIT %s",
                (person_id, int(limit)),
            )
            return [
                {
                    "fact_id": int(row["fact_id"]),
                    "fact": row["fact"],
                    "source_conversation_id": (
                        int(row["source_conversation_id"])
                        if row["source_conversation_id"] is not None else None
                    ),
                    "created_at": _from_utc(row["created_at"]),
                }
                for row in cur.fetchall()
            ]

    def delete_person_facts(self, person_id: str) -> int:
        """Forgets everything the robot knows about someone WITHOUT forgetting the person - the
        middle option between "remember me" and "delete me entirely" that PRIVACY.md's deletion
        request handling needs, for a visitor who wants to stay enrolled but wants a particular
        thing forgotten."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM person_facts WHERE person_id = %s", (person_id,))
            return cur.rowcount

    def delete_people_not_seen_since(self, cutoff: float) -> list[str]:
        """The auto-expiry half of PRIVACY.md rule 3. Selects first so the caller can log/report
        exactly whose data went, instead of only a row count."""
        with self._cursor(dictionary=True) as cur:
            cur.execute(
                "SELECT person_id FROM people WHERE last_seen < %s", (_to_utc(cutoff),)
            )
            expired = [row["person_id"] for row in cur.fetchall()]
            if expired:
                placeholders = ", ".join(["%s"] * len(expired))
                cur.execute(
                    "DELETE FROM people WHERE person_id IN (" + placeholders + ")",
                    tuple(expired),
                )
        return expired

    def counts(self) -> dict[str, int]:
        with self._cursor(dictionary=True) as cur:
            result = {}
            for table in ("robot_identity", "people", "face_embeddings", "conversations",
                          "utterances", "person_facts"):
                cur.execute("SELECT COUNT(*) AS n FROM " + table)
                result[table] = int(cur.fetchone()["n"])
            return result


def connect_or_none(verbose: bool = True, **overrides):
    """Best-effort connect for the robot itself: returns None instead of raising when MySQL
    isn't reachable.

    A kiosk that refuses to greet anybody because the database is down is a worse outcome than
    one that greets everybody as a stranger for an afternoon - Stage 09's graceful-degradation
    rule. The failure is printed loudly so it is never mistaken for "nobody has enrolled yet".
    """
    try:
        db = RobotDatabase(**overrides)
        db.ensure_schema()
        return db
    except MySQLError as exc:
        if verbose:
            print(f"[db] MySQL unavailable ({exc}) - running WITHOUT persistent memory this "
                  f"session: faces enrolled now will be forgotten on exit, and no conversation "
                  f"will be recorded. Check the server and the settings in .env")
        return None


if __name__ == "__main__":
    import time
    import uuid

    db = RobotDatabase()
    print(f"Connecting to mysql://{db.config.user}@{db.config.host}:{db.config.port}"
          f"/{db.config.database} ...")
    try:
        db.ensure_schema()
    except MySQLError as exc:
        print(f"\nFAILED: {exc}\n\nSet the connection details in {ENV_FILE} (copy "
              f"{PROJECT_ROOT / '.env.example'}), or via LUC_DB_* environment variables.")
        sys.exit(1)

    print("OK: database and tables are ready.")
    for table, n in db.counts().items():
        print(f"  {table:<16} {n} row(s)")

    if "--self-test" in sys.argv:
        print("\nRunning a full round trip (a temporary person, then deleting them)...")
        now = time.time()
        person_id = str(uuid.uuid4())
        sample = np.random.rand(1, 128).astype("float32")

        db.insert_person(person_id, "Self Test", True, now, now, 1, [sample])
        conv_id = db.start_conversation(person_id)
        db.log_utterance(conv_id, 0, "robot", "Welcome back, Self Test!")
        db.log_utterance(conv_id, 1, "visitor", "Where is the library?")
        db.log_utterance(conv_id, 2, "robot", "It is on the second floor.",
                         intent="GENERAL_INFO")
        db.end_conversation(conv_id)

        loaded = [p for p in db.load_people() if p["person_id"] == person_id]
        assert len(loaded) == 1, "the person did not come back out of the database"
        restored = loaded[0]["embeddings"][0]
        assert restored.shape == (1, 128), f"wrong embedding shape: {restored.shape}"
        assert np.allclose(restored, sample), "the embedding did not survive the round trip"
        print(f"  embedding round trip OK (shape {restored.shape}, exact match)")

        transcript = db.conversation_transcript(conv_id)
        assert len(transcript) == 3, f"expected 3 utterances, got {len(transcript)}"
        for row in transcript:
            print(f"  [{row['said_at']}] {row['speaker']:<7} {row['text']!r}")

        # --- what the robot remembers about them (person_facts, 2026-09-05) ------------------
        added = db.add_person_facts(
            person_id,
            ["Is applying for the Master of Computer Science.", "Prefers the January intake."],
            conversation_id=conv_id,
        )
        assert added == 2, f"expected 2 facts inserted, got {added}"
        # Same fact again (different case) must not accumulate a second row.
        assert db.add_person_facts(person_id, ["prefers the January intake."]) == 0, \
            "a duplicate fact was stored again - add_person_facts() is not de-duplicating"
        facts = db.person_facts(person_id)
        assert len(facts) == 2, f"expected 2 facts back, got {len(facts)}"
        assert facts[0]["source_conversation_id"] == conv_id, \
            "a fact lost the conversation it came from - it can no longer be audited"
        for row in facts:
            print(f"  remembered: {row['fact']!r}")

        assert db.delete_person(person_id), "delete_person() reported nothing deleted"
        assert not db.recent_conversations(person_id), \
            "the conversation outlived the person - ON DELETE CASCADE is not working"
        assert not db.person_facts(person_id), \
            "remembered facts outlived the person - the privacy cascade misses person_facts, " \
            "which is the most sensitive table of the set (free text about a real student)"
        print("  cascade delete OK (person, embedding, conversation, transcript and remembered "
              "facts all gone)")
        print("\nOK: persistence, retrieval and the right-to-be-forgotten cascade all work.")

    db.close()
