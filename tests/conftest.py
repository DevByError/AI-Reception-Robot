"""Shared pytest setup: every stage folder is importable, the way main_live.py arranges it.

Run all tests from the project root:
    python -m pytest tests -q
Tests that need an optional model file skip themselves (with the reason) when it is missing -
run stages/stage00_setup/download_models.py to fetch them.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "stages"
for folder in sorted(STAGES.glob("stage0*")):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

# Tests must never reach a real cloud API. Set to empty rather than removed: the project's .env
# loader only fills in variables that are not set at all, so a removed key would come back.
for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ[key] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Tests must never touch the robot's real memory either: not the MySQL database named in .env
# (where the enrolled faces and transcripts live), not data/robot_memory.db, and not the robot's
# identity_cache.json. Everything goes to a throwaway folder. The tests are written for a robot
# called "Nova", whatever the real one is called.
_SCRATCH = Path(tempfile.mkdtemp(prefix="ai-reception-robot-tests-"))
os.environ["LUC_DB_BACKEND"] = "sqlite"
os.environ["LUC_DB_SQLITE_PATH"] = str(_SCRATCH / "robot_memory.db")
os.environ["LUC_IDENTITY_CACHE"] = str(_SCRATCH / "identity_cache.json")
(_SCRATCH / "identity_cache.json").write_text(json.dumps({"robot_name": "Nova"}),
                                              encoding="utf-8")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: whole-robot simulation in a fake room (slow; -m 'not e2e' skips it)")
