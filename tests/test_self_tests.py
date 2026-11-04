"""Runs every module's built-in self-test (the `--self-test` / `__main__` checks that need no
hardware) under pytest, each in its own process so none of them can leak state into another.

They were written to be run by hand; nothing ran them together, so a change in one stage could
quietly break another stage's checks. Now `python -m pytest tests` covers them too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "stages"

CASES = {
    "orchestrator": ["stages/stage08_orchestrator/main_live.py", "--self-test"],
    "conversation_state": ["stages/stage08_orchestrator/conversation_state.py"],
    "identity": ["stages/stage05_nlu_dialogue/identity.py"],
    "face_memory": ["stages/stage02_face_recognition/face_memory.py"],
    "sface_embedder": ["stages/stage02_face_recognition/sface_embedder.py"],
    "database_sqlite": ["stages/stage02_face_recognition/robot_db.py", "--self-test", "--sqlite"],
    "listener": [
        "-c",
        "import sys; sys.path.insert(0, 'stages/stage04_speech_to_text'); "
        "import stt_listener_live as s; s.run_self_tests()",
    ],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_module_self_test(name, tmp_path):
    env = dict(os.environ)
    env["LUC_DB_BACKEND"] = "sqlite"
    env["LUC_DB_SQLITE_PATH"] = str(tmp_path / "robot.db")  # never the real data/ folder
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, *CASES[name]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    if proc.returncode != 0 and "SKIPPED" in proc.stdout:
        pytest.skip(tail)
    assert proc.returncode == 0, tail
