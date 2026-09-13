"""
Stage 05 - a local LLM and a local embedding model, both served by Ollama on this machine.

Why: every other answer path in this stage depends on somebody else's server staying up and a
free-tier quota not running out. Gemini has already produced real failures in this project - a
503 mid-answer (2026-08-19) and a 429 on the very first question of a session (2026-08-17) - and
each one degraded the robot to reading a raw scraped page-title fragment aloud. A kiosk standing
in a university lobby cannot depend on that. Ollama runs the model on this machine: no API key,
no quota, no per-request cost, and it keeps working with the network unplugged.

Two separate jobs, both served here so neither caller has to reimplement the HTTP details:
  - chat()  -> answer generation, the local counterpart to Gemini/Groq
               (see OllamaAnswerGenerator in dialogue_manager_live.py)
  - embed() -> the vectors behind RAG retrieval
               (see OllamaEmbeddingBackend in knowledge_base.py)

--- NO NEW PYTHON DEPENDENCY -----------------------------------------------------------------
Deliberately plain HTTP through `requests` (already required by Stage 07's scraper) rather than
the official `ollama` package. The two endpoints used here are a handful of lines each, and the
package would add a pin to keep in sync for no capability this file doesn't already have.

--- CONFIGURATION ----------------------------------------------------------------------------
Environment variables (all optional, defaults below):
    LUC_OLLAMA_HOST         default http://127.0.0.1:11434
    LUC_OLLAMA_CHAT_MODEL   default llama3.2:3b
    LUC_OLLAMA_EMBED_MODEL  default nomic-embed-text

Setup and the reasoning behind those two model choices are in README.md ("Local LLM via
Ollama"). Run this file directly to check the server, the models, and the real latency:
    python stages/stage05_nlu_dialogue/ollama_client.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

# Loaded before any os.environ read below, so the project-root .env can set these the same way
# it sets the database credentials - see stage00_setup/project_env.py.
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from project_env import load_env_file  # noqa: E402

load_env_file()

OLLAMA_HOST = os.environ.get("LUC_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# llama3.2:3b (~2GB at Q4) rather than an 8B: this machine's GTX 1650 has 4GB of VRAM, and a
# model that doesn't fit gets partially offloaded to the CPU, which is the difference between
# answering in about a second and answering in fifteen. A voice robot that takes fifteen seconds
# to reply has failed at the task even if the sentence it eventually says is better. Override
# with LUC_OLLAMA_CHAT_MODEL on a machine with more VRAM - qwen2.5:7b and llama3.1:8b are both
# drop-in and noticeably stronger.
CHAT_MODEL = os.environ.get("LUC_OLLAMA_CHAT_MODEL", "llama3.2:3b")

# nomic-embed-text: 768-d, ~274MB, and it wants a task prefix on every input (see embed()) -
# retrieval quality drops measurably without it, which is the kind of detail that silently costs
# accuracy rather than failing loudly.
EMBED_MODEL = os.environ.get("LUC_OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Long enough for a first-token wait on a cold model, short enough that a wedged server doesn't
# hold a visitor in silence forever. The load itself is what's slow (several seconds of reading
# weights off disk into VRAM), which is why warm_up() exists - pay it at startup, not while
# somebody is standing there waiting.
CHAT_TIMEOUT_S = 60.0
EMBED_TIMEOUT_S = 120.0   # a bulk startup embed of thousands of chunks goes through one call
PROBE_TIMEOUT_S = 2.0     # is_available() must never be the thing that delays startup

# The FIRST call after a fresh `ollama pull` is in a different league from every later one: the
# weights aren't in the OS page cache yet, so 2GB gets read off disk before the model can even
# start, and Ollama serializes requests per model so anything queued behind it waits too. Hit
# for real on 2026-09-04 - a 60s timeout was not enough for that one call, while the very next
# warm request answered in 0.9s. A separate, generous limit here means the one-time cost doesn't
# look like a broken backend.
WARM_UP_TIMEOUT_S = 300.0

# How long Ollama keeps the model in VRAM after a request. The default is 5 minutes, which is
# shorter than a quiet stretch in a university lobby - so without this the first visitor after
# every lull would wait through a full model load before hearing anything back.
KEEP_ALIVE = "30m"

# Cap on generated length. These prompts ask for a short spoken answer; the cap is a backstop
# against a model that ignores that and monologues at somebody waiting to hear a floor number.
MAX_TOKENS = 250


class OllamaError(RuntimeError):
    """Raised for anything that went wrong talking to Ollama - server down, model not pulled,
    malformed response. Callers catch this to fall through to the next backend rather than
    letting one local failure end the conversation."""


def _url(path: str) -> str:
    return f"{OLLAMA_HOST}{path}"


def is_available(timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """Is there an Ollama server answering right now? Used at startup to decide whether the
    local backend goes into the chain at all. Never raises - "no" is a normal answer."""
    try:
        response = requests.get(_url("/api/tags"), timeout=timeout_s)
        return response.status_code == 200
    except requests.RequestException:
        return False


def installed_models(timeout_s: float = PROBE_TIMEOUT_S) -> list[str]:
    """Model names currently pulled on this machine, e.g. ['llama3.2:3b', 'nomic-embed-text'].
    Returns [] if the server isn't reachable."""
    try:
        response = requests.get(_url("/api/tags"), timeout=timeout_s)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return []


def has_model(name: str) -> bool:
    """Whether `name` is pulled. Tolerates the ':latest' suffix Ollama adds to an unqualified
    tag - `ollama pull nomic-embed-text` lands as 'nomic-embed-text:latest', so a plain equality
    check against the configured name would wrongly report it missing."""
    wanted = name if ":" in name else f"{name}:latest"
    return any(m == name or m == wanted for m in installed_models())


def chat(prompt: str, system: str | None = None, temperature: float = 0.2,
         model: str = CHAT_MODEL, max_tokens: int = MAX_TOKENS,
         timeout_s: float = CHAT_TIMEOUT_S) -> str:
    """One non-streaming chat completion. Returns the assistant's text.

    Still here, and still the right call for everything that needs a COMPLETE answer before it
    can act - classify_and_answer_general() has to parse a two-line format out of the response,
    which is not something you can do halfway through. chat_stream() below is the variant for
    the grounded path, where Stage 06 can start speaking sentence one while sentence two is
    still being generated.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        response = requests.post(_url("/api/chat"), json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise OllamaError(f"chat request to {OLLAMA_HOST} failed: {exc}") from exc
    except ValueError as exc:
        raise OllamaError(f"Ollama returned a non-JSON response: {exc}") from exc

    content = data.get("message", {}).get("content", "")
    if not content.strip():
        raise OllamaError(f"model {model!r} returned an empty response")
    return content.strip()


def chat_stream(prompt: str, system: str | None = None, temperature: float = 0.2,
                model: str = CHAT_MODEL, max_tokens: int = MAX_TOKENS,
                timeout_s: float = CHAT_TIMEOUT_S):
    """Same call as chat(), yielding the answer in pieces as the model produces them.

    Added 2026-09-05 for the streaming answer path: Stage 06's LiveTTSSpeaker.speak_stream()
    starts speaking as soon as the FIRST SENTENCE exists, so a local model that takes two
    seconds to finish a three-sentence answer now starts talking after roughly the first one.
    The note on chat() above ("streaming would only mean reassembling them here") was true while
    the TTS could only accept a finished string; it can accept a live stream now, so the
    reassembly happens at the sentence boundary in the speaker instead.

    Ollama streams newline-delimited JSON objects, one per token-ish chunk. A malformed line is
    skipped rather than raising: losing one fragment mid-sentence is a smaller failure than
    aborting an answer the visitor is already hearing.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    produced_anything = False
    try:
        with requests.post(
            _url("/api/chat"), json=payload, timeout=timeout_s, stream=True
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                piece = data.get("message", {}).get("content", "")
                if piece:
                    produced_anything = True
                    yield piece
                if data.get("done"):
                    break
    except requests.RequestException as exc:
        raise OllamaError(f"streaming chat request to {OLLAMA_HOST} failed: {exc}") from exc

    if not produced_anything:
        raise OllamaError(f"model {model!r} returned an empty streamed response")


def embed(texts: list[str], is_query: bool = False, model: str = EMBED_MODEL,
          timeout_s: float = EMBED_TIMEOUT_S) -> list[list[float]]:
    """Embeds one or more texts. Returns raw (un-normalized) vectors - the caller normalizes,
    see OllamaEmbeddingBackend in knowledge_base.py.

    is_query distinguishes the two sides of a retrieval, which matters more than it looks:
    nomic-embed-text is trained with task prefixes, and a stored chunk ("search_document: ...")
    and a spoken question ("search_query: ...") are meant to be encoded differently. Skipping
    the prefixes doesn't fail - it just quietly retrieves worse, which is the kind of bug this
    project has been bitten by before (see knowledge_base.py's module docstring on word-overlap
    scoring). Other embedding models ignore an unfamiliar prefix, so this stays harmless if
    LUC_OLLAMA_EMBED_MODEL is pointed elsewhere.
    """
    if not texts:
        return []
    prefix = "search_query: " if is_query else "search_document: "
    payload = {
        "model": model,
        "input": [prefix + t for t in texts],
        "keep_alive": KEEP_ALIVE,
    }
    try:
        response = requests.post(_url("/api/embed"), json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except requests.HTTPError as exc:
        # A 400 here is nearly always too many inputs in one request rather than anything wrong
        # with the text: past roughly 128, Ollama's model runner subprocess dies and the error
        # that surfaces is a confusing "connection refused" to an internal port, which reads
        # like the server is down when it is actually up and the request was simply too big.
        # See OllamaEmbeddingBackend.MAX_EMBED_BATCH in knowledge_base.py.
        detail = ""
        try:
            detail = response.json().get("error", "")
        except ValueError:
            detail = response.text[:200]
        hint = (f" - {len(texts)} inputs in one request is likely too many; batch them smaller"
                if len(texts) > 64 else "")
        raise OllamaError(
            f"embed request to {OLLAMA_HOST} failed: {exc}{hint}"
            + (f" (server said: {detail})" if detail else "")
        ) from exc
    except requests.RequestException as exc:
        raise OllamaError(f"embed request to {OLLAMA_HOST} failed: {exc}") from exc
    except ValueError as exc:
        raise OllamaError(f"Ollama returned a non-JSON response: {exc}") from exc

    vectors = data.get("embeddings")
    if not vectors or len(vectors) != len(texts):
        raise OllamaError(
            f"expected {len(texts)} embedding(s) from {model!r}, got "
            f"{len(vectors) if vectors else 0} - is that model pulled? "
            f"Run: ollama pull {model}"
        )
    return vectors


def warm_up(model: str = CHAT_MODEL) -> float:
    """Loads the chat model into VRAM before anybody is waiting on it, and returns how long that
    took. Called once at startup (see dialogue_manager_live.py's builder).

    Without this, the model load lands on the first real question of the session - the robot
    stands silent for several seconds mid-conversation while weights stream off disk, which
    reads to a visitor as "it's broken", not as "it's loading". Failures are the caller's to
    interpret; this only reports the cost.
    """
    started = time.monotonic()
    chat("Say OK.", system="Reply with exactly: OK", temperature=0.0, model=model,
         max_tokens=5, timeout_s=WARM_UP_TIMEOUT_S)
    return time.monotonic() - started


def describe() -> str:
    """One line for a startup log: what's running and what's pulled."""
    if not is_available():
        return f"Ollama: not reachable at {OLLAMA_HOST}"
    return (f"Ollama at {OLLAMA_HOST}: chat={CHAT_MODEL} "
            f"({'ready' if has_model(CHAT_MODEL) else 'NOT PULLED'}), "
            f"embed={EMBED_MODEL} "
            f"({'ready' if has_model(EMBED_MODEL) else 'NOT PULLED'})")


if __name__ == "__main__":
    import sys

    print(f"Checking Ollama at {OLLAMA_HOST} ...")
    if not is_available():
        print(
            "\nFAILED: nothing is answering there.\n\n"
            "  1. Install it from https://ollama.com/download (or: winget install Ollama.Ollama)\n"
            "  2. It runs as a background service once installed - check with: ollama list\n"
            "  3. Pull the two models this project uses:\n"
            f"       ollama pull {CHAT_MODEL}\n"
            f"       ollama pull {EMBED_MODEL}\n\n"
            "The robot still runs without any of this - it just loses the local fallback and "
            "depends entirely on the Gemini API being up. See README.md."
        )
        sys.exit(1)

    print(f"OK: server is up. Models pulled: {installed_models()}")

    missing = [m for m in (CHAT_MODEL, EMBED_MODEL) if not has_model(m)]
    if missing:
        print("\nMissing model(s) - pull them before the robot can use this backend:")
        for name in missing:
            print(f"    ollama pull {name}")
        sys.exit(1)

    print(f"\nWarming up {CHAT_MODEL} (loads it into VRAM)...")
    load_s = warm_up()
    print(f"  loaded and answered in {load_s:.1f}s (this is the cost the robot pays at startup, "
          f"not per question)")

    print("\nTiming a realistic grounded answer...")
    started = time.monotonic()
    answer = chat(
        "Information:\n- The Engineering department is on the third floor.\n\n"
        "Question: where is the engineering department?",
        system="You are a university reception robot. Every answer is spoken aloud, so reply in "
               "one plain spoken sentence. Answer using ONLY the information given.",
    )
    elapsed = time.monotonic() - started
    print(f"  [{elapsed:.1f}s] {answer!r}")
    if elapsed > 3.0:
        print(f"  NOTE: {elapsed:.1f}s is slower than the under-3s target in STAGES.md's success "
              f"metrics. A smaller chat model (LUC_OLLAMA_CHAT_MODEL) is the lever here - most "
              f"real questions hit the knowledge base's is_direct_answer fast path and never "
              f"reach a model at all, so this cost applies to the harder questions only.")

    print(f"\nTiming embeddings with {EMBED_MODEL}...")
    started = time.monotonic()
    doc_vectors = embed(["The Engineering department is on the third floor.",
                          "The library is on the ground floor."])
    query_vector = embed(["where is the engineering department"], is_query=True)
    elapsed = time.monotonic() - started
    print(f"  {len(doc_vectors)} document vector(s) + 1 query vector in {elapsed * 1000:.0f}ms, "
          f"{len(doc_vectors[0])} dimensions each")

    # Proves the vectors actually separate a relevant chunk from an irrelevant one - a backend
    # that returns well-formed but meaningless numbers would pass every check above.
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return dot / norm if norm else 0.0

    relevant = _cosine(query_vector[0], doc_vectors[0])
    irrelevant = _cosine(query_vector[0], doc_vectors[1])
    print(f"  'where is the engineering department' vs. the engineering chunk: {relevant:.3f}")
    print(f"                                        vs. the library chunk:     {irrelevant:.3f}")
    if relevant > irrelevant:
        print("\nOK: chat and embeddings both work, and retrieval actually discriminates.")
    else:
        print("\nWARNING: the irrelevant chunk scored at least as high as the relevant one - "
              "something is wrong with the embedding model or the task prefixes.")
        sys.exit(1)
