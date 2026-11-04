"""The SQLite memory backend (stage02/robot_db.py): same tables, same cascades, same methods as
MySQL, so a robot without a MySQL server still remembers faces and conversations."""

from __future__ import annotations

import time

import numpy as np
import pytest

robot_db = pytest.importorskip("robot_db")


@pytest.fixture
def db(tmp_path):
    database = robot_db.SQLiteRobotDatabase(tmp_path / "memory.db")
    database.ensure_schema()
    yield database
    database.close()


def test_full_round_trip_and_cascade(db):
    robot_db.self_test_round_trip(db)  # the same check MySQL's --self-test runs
    assert db.counts()["people"] == 0


def test_identity_upsert_and_seed(db):
    db.set_identity_fields({"name": "Nova", "creator": "LUC"})
    db.set_identity_fields({"name": "Nova 2"})
    assert db.load_identity() == {"name": "Nova 2", "creator": "LUC"}
    assert db.seed_identity_fields({"name": "ignored", "email": "a@b.c"}) == ["email"]
    assert db.load_identity()["name"] == "Nova 2"


def test_survives_a_restart(tmp_path):
    path = tmp_path / "memory.db"
    first = robot_db.SQLiteRobotDatabase(path)
    first.ensure_schema()
    now = time.time()
    first.insert_person("p1", "Aisha", True, now, now, 1, [np.ones((1, 128), dtype="float32")])
    first.close()

    second = robot_db.SQLiteRobotDatabase(path)
    second.ensure_schema()
    people = second.load_people()
    assert [p["name"] for p in people] == ["Aisha"]
    assert abs(people[0]["last_seen"] - now) < 1e-3
    assert people[0]["embeddings"][0].shape == (1, 128)
    second.close()


def test_retention_sweep_and_fact_cap(db):
    now = time.time()
    db.insert_person("old", "Old", True, now - 400 * 86400, now - 400 * 86400, 1, [])
    db.insert_person("new", "New", True, now, now, 1, [])
    assert db.delete_people_not_seen_since(now - 365 * 86400) == ["old"]
    facts = [f"fact {i}" for i in range(20)]
    db.add_person_facts("new", facts)
    assert len(db.person_facts("new", limit=100)) == db.MAX_FACTS_PER_PERSON


def test_face_store_on_sqlite(db):
    from face_memory import FaceMemoryStore

    store = FaceMemoryStore(db=db)
    rec = store.register("Ravi", [np.random.rand(1, 128).astype("float32")], consent_given=True)
    again = FaceMemoryStore(db=db)
    assert again.load_from_db() == 1
    assert again.get(rec.person_id).name == "Ravi"


def test_connect_or_none_falls_back_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("LUC_DB_BACKEND", "auto")
    monkeypatch.setenv("LUC_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("LUC_DB_PORT", "1")  # nothing listens there
    monkeypatch.setenv("LUC_DB_SQLITE_PATH", str(tmp_path / "fallback.db"))
    db = robot_db.connect_or_none(verbose=False)
    assert isinstance(db, robot_db.SQLiteRobotDatabase)
    assert (tmp_path / "fallback.db").exists()
    db.close()


def test_mysql_only_mode_still_returns_none(monkeypatch):
    monkeypatch.setenv("LUC_DB_BACKEND", "mysql")
    monkeypatch.setenv("LUC_DB_PORT", "1")
    assert robot_db.connect_or_none(verbose=False) is None
