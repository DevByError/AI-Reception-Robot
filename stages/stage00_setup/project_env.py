"""
Stage 00 - the project's one .env reader.

Several stages need settings that shouldn't be hard-coded in source: MySQL credentials (Stage
02), the Ollama host and models and which embedding backend to use (Stage 05), an optional cloud
LLM API key (Stage 05). They all come from the same place - a gitignored `.env` at the project
root, or real environment variables - so the loader lives here rather than being reimplemented
per stage and drifting.

It lives in stage00_setup because that is where "things every stage needs before it can run"
already belongs, and because this module must import nothing heavier than the standard library:
it runs before numpy, torch, mysql-connector or requests are touched, and every one of those has
already been a startup problem in this project at some point.

Precedence, highest first:
  1. a real environment variable (this is what a deployed kiosk should use)
  2. the project-root .env file
  3. whatever default the reading module names

Rule 1 is the one that matters: the file NEVER overwrites something already set, so a deployment
can override any single value without editing a file that's shared with local testing.

Usage - call it once, at the top of any module that reads os.environ, BEFORE it reads:

    from project_env import load_env_file
    load_env_file()
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

_loaded = False


def load_env_file(path: Path | None = None, force: bool = False) -> bool:
    """Reads a simple KEY=VALUE file into os.environ without overwriting anything already set.
    Returns whether a file was actually read.

    Deliberately hand-rolled rather than pulling in python-dotenv: this is fifteen lines and one
    fewer dependency to pin, for a file that only ever holds a handful of settings.

    Runs at most once per process unless force=True - it's called from several module tops, and
    re-reading the same file on every import would be pure waste.
    """
    global _loaded
    if _loaded and not force and path is None:
        return False

    path = Path(path) if path is not None else ENV_FILE
    if path is None or not path.exists():
        _loaded = True
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Strip one layer of matching quotes - a password pasted from a password manager often
        # arrives wrapped in them, and MySQL would otherwise reject it with a confusing
        # "access denied" that looks like a wrong password rather than a quoting problem.
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value

    _loaded = True
    return True


def add_to_path() -> None:
    """Makes this module importable from any stage folder. A module that wants it does:

        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
        from project_env import load_env_file

    which is the same sys.path pattern main_live.py already uses to reach across stages.
    """
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.append(here)


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f".env file:    {ENV_FILE} ({'found' if ENV_FILE.exists() else 'not present'})")
    read = load_env_file()
    print(f"Loaded: {read}")

    # Names only - never the values. This file's whole reason for existing is that some of them
    # are secrets, and a helper that prints a password to the terminal on request would undo
    # that in one command.
    known = [
        "LUC_DB_HOST",
        "LUC_DB_PORT",
        "LUC_DB_USER",
        "LUC_DB_PASSWORD",
        "LUC_DB_NAME",
        "LUC_OLLAMA_HOST",
        "LUC_OLLAMA_CHAT_MODEL",
        "LUC_OLLAMA_EMBED_MODEL",
        "LUC_EMBEDDING_BACKEND",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    print("\nSettings this project reads (value hidden - only whether it is set):")
    for name in known:
        print(f"  {name:<24} {'set' if os.environ.get(name) else '-'}")
