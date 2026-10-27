"""
Stage 05 (part 1) - Knowledge Base (Vector Store)

Real implementation: a local sentence-embedding model + cosine similarity, replacing the
word-overlap keyword scoring this file carried as a TODO since Stage 05 first went live. Runs
fully offline once the model is cached after the first run (no API key, no network needed at
runtime) - consistent with the project's no-API-dependency default (see dialogue_manager_live.py).

TWO EMBEDDING BACKENDS, same interface (see EmbeddingBackend below), chosen by
LUC_EMBEDDING_BACKEND:
  - "sentence-transformers" (default): all-MiniLM-L6-v2, 384-d, in-process. Its threshold is
    measured, its cache format is what's already on disk, and it needs nothing running.
  - "ollama": nomic-embed-text, 768-d, over HTTP to a local Ollama server. A stronger model on
    paraphrased questions, which is what a spoken question usually is - a student says "what
    floor do I go to for nursing", not the directory's own wording.

Switching is an ACCURACY choice and it is EXPENSIVE. Measured on this machine (2026-09-04) on
realistic 400-character chunks:

                              per chunk    Stage 07's ~14,800 chunks    one spoken query
    all-MiniLM (in-process)      ~12 ms              ~3 minutes                 ~31 ms
    nomic-embed-text (Ollama)   ~430 ms            ~106 minutes                ~125 ms

That is 35x, and it is not an HTTP overhead story - it is a bigger model doing more work, on a
4GB GPU it has to share with the chat model. Which is why sentence-transformers stays the
DEFAULT: it is already offline, already fast, and already measured. Turn Ollama embeddings on
only if real usage shows retrieval picking the wrong chunk, and expect the corpus rebuild to
take the better part of two hours.

Note this is the opposite of the answer-generation story (see dialogue_manager_live.py), where
the local model exists for AVAILABILITY - there is nothing to fail over to here, because the
default backend already runs in-process with no server and no network.

Whichever backend is active, its name is stamped into the cache and checked on load, so vectors
from one model can never be silently compared against another's - see save()/load().

Why this matters in practice, not just in theory: with the fuller building directory now in
`dialogue_manager_live.py`'s SAMPLE_CHUNKS, two chunks - "the men's mosque is on the first floor"
and "the women's mosque is on the second floor" - differ by exactly one content word. Word-overlap
scoring couldn't tell them apart (both scored equally on "mosque"/"floor"/etc.), so "where is the
mosque for women" silently returned the men's mosque - wrong answer, sounding perfectly confident.
A real embedding model represents "men's" and "women's" as different concepts, not just different
tokens, and gets this right - see this file's own self-test below, and evaluate_retrieval.py for
a fuller, tunable check.

THRESHOLD (SIMILARITY_THRESHOLDS below) is the one number here that must not be eyeballed. Every
other stage's routing depends on a contract: search() returns [] when nothing is actually
relevant, so handle_question() (dialogue_manager.py) falls through to "does this even sound
university-related?" instead of grounding an answer in a barely-related chunk - this is the same
anti-hallucination guarantee the project has enforced from the start. Cosine similarity is never
exactly 0, so without a threshold, search() would always return its closest chunk regardless of
actual relevance, silently breaking that guarantee. The value below is a reasonable starting point
for all-MiniLM-L6-v2 on short factual sentences, but "reasonable starting point" is exactly the
kind of number Stage 04 (WER) and Stage 06 (clarity) taught this project not to trust without
measuring - run evaluate_retrieval.py after install and retune the threshold against real
relevant/irrelevant query pairs before trusting this in a live conversation. A threshold is
measured per embedding model and is meaningless across models, which is why
SIMILARITY_THRESHOLDS is a table rather than one number: 0.35 was measured for MiniLM and says
nothing about where nomic-embed-text's relevant/irrelevant score clusters actually fall.

TODO (real implementation later):
  - replace storage with FAISS/Chroma/pgvector once the corpus is large enough that brute-force
    cosine similarity over every chunk (fine for the current few dozen facts) stops being fast
    enough - Stage 07's scraper will grow this corpus over time.
"""

from __future__ import annotations

import os

# Must be set BEFORE numpy/torch/sentence-transformers are imported below - works around a real
# crash hit on Windows (2026-08-17, exit code -1073740791 = 0xC0000409
# STATUS_STACK_BUFFER_OVERRUN): torch and scikit-learn (a sentence-transformers dependency) each
# bundle their own copy of the Intel OpenMP runtime (libiomp5md.dll); when both get loaded into
# the same process, OpenMP treats the duplicate as fatal and aborts the whole process with no
# Python-level traceback at all - it happened silently, right as the embedding model ran its
# first real computation. This is a well-known, widely-documented issue with ML stacks on
# Windows, not a bug in this project's own code - KMP_DUPLICATE_LIB_OK=TRUE tells OpenMP to
# tolerate the duplicate instead of aborting (the standard, broadly-used workaround). Set here
# (not just as a one-off environment variable) so it applies automatically no matter which
# script imports this module - dialogue_manager_live.py, main_live.py,
# stage07_data_pipeline/indexer.py, evaluate_retrieval.py, etc.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Loaded before LUC_EMBEDDING_BACKEND is read below, so the project-root .env can select the
# backend the same way it supplies the database credentials - see stage00_setup/project_env.py.
sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from project_env import load_env_file  # noqa: E402

load_env_file()

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "numpy is required - run: pip install -r stages/stage05_nlu_dialogue/requirements.txt"
    ) from exc

# Small (~80MB), fast on CPU, a standard default for short-sentence retrieval - good latency fit
# for a live voice interaction. First use downloads it from Hugging Face and caches it locally;
# every run after that is fully offline.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# "sentence-transformers" | "ollama" - see the module docstring for the tradeoff. An environment
# variable rather than a constant edited in the file, because it has to be switchable from the
# same place the rest of the deployment config lives (the project-root .env, read by
# robot_db.load_env_file()) without touching source.
#
# "auto" (default since 2026-10-27): the ONNX export of the SAME all-MiniLM-L6-v2 model when its
# files are in models/ (no PyTorch needed - see OnnxMiniLMBackend), otherwise sentence-transformers.
EMBEDDING_BACKEND = os.environ.get("LUC_EMBEDDING_BACKEND", "auto")

# Minimum cosine similarity for a chunk to count as "actually relevant" to a query - see the
# module docstring above for why this exists and why it needs real measurement, not eyeballing.
#
# Keyed by embedding model, because a threshold measured for one model says nothing about
# another: different models put their relevant/irrelevant score clusters in different places, so
# reusing MiniLM's 0.35 for a 768-d model would be exactly the eyeballing this project keeps
# warning against. An unmeasured model is loudly flagged rather than silently defaulted - see
# threshold_for().
SIMILARITY_THRESHOLDS = {
    # Measured via evaluate_retrieval.py on the real building directory (2026-08-12): 11/11
    # positive queries correctly matched, 0/5 negative (out-of-scope) queries falsely matched.
    "all-MiniLM-L6-v2": 0.35,
    # Measured via evaluate_retrieval.py (2026-09-04): 11/11 positive queries correctly matched,
    # 0/5 negatives falsely matched, with a clean band from 0.650 to 0.700 - real margin either
    # side rather than a knife-edge.
    #
    # The distance between this and MiniLM's 0.35 is exactly why this is a table and not a
    # constant. nomic-embed-text scores an UNRELATED query around 0.4-0.64 - higher than MiniLM
    # scores a genuinely relevant one - and a relevant query around 0.7-0.89. Reusing 0.35 here
    # would match literally everything, silently turning off the anti-hallucination guarantee
    # that search() returning [] is supposed to provide.
    "ollama:nomic-embed-text": 0.65,
}

# Only used for a model with no measured entry above, and only alongside a warning. Deliberately
# on the permissive side: a too-high threshold rejects real matches and makes the robot say "I
# don't have that information" about facts it holds, which is harder to notice in testing than
# the opposite.
UNMEASURED_THRESHOLD_FALLBACK = 0.35

_warned_about_threshold: set[str] = set()


def threshold_for(model_name: str) -> float:
    """The measured threshold for an embedding model, or the fallback plus a one-time warning."""
    if model_name in SIMILARITY_THRESHOLDS:
        return SIMILARITY_THRESHOLDS[model_name]
    if model_name not in _warned_about_threshold:
        _warned_about_threshold.add(model_name)
        print(f"[kb] no measured SIMILARITY_THRESHOLD for embedding model {model_name!r} - "
              f"using {UNMEASURED_THRESHOLD_FALLBACK} as a placeholder. Run "
              f"evaluate_retrieval.py and add the measured value to SIMILARITY_THRESHOLDS in "
              f"knowledge_base.py before trusting this in a live conversation.")
    return UNMEASURED_THRESHOLD_FALLBACK


# --- embedding backends ---------------------------------------------------------------------
# Both expose the same three things: .name (stamped into the cache), encode_documents() and
# encode_query(). Two methods rather than one because some models - nomic-embed-text among them -
# encode a stored document and a search query differently, and collapsing that distinction costs
# retrieval accuracy silently. Backends that don't care simply ignore it.
#
# Both return UNIT-NORMALIZED vectors, so score_all()'s plain dot product IS cosine similarity
# and nothing downstream has to re-normalize per search.


def _duration(seconds: float) -> str:
    """Human-readable, so a small corpus doesn't get reported as "about 0 minutes"."""
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def _normalize(matrix: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)   # guard against a zero vector, not expected


class SentenceTransformerBackend:
    """The original in-process backend: all-MiniLM-L6-v2 via sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for the default embedding backend - run: "
                "pip install -r stages/stage05_nlu_dialogue/requirements.txt"
            ) from exc
        print(f"Loading embedding model '{model_name}' (one-time download on first run)...")
        self._model = SentenceTransformer(model_name)
        # No prefix and no "ollama:" qualifier - keeping this string exactly as it was means
        # caches written before the backend split still load instead of triggering a full
        # rescrape-and-re-embed.
        self.name = model_name

    def encode_documents(self, texts: list[str], batch_size: int = 256) -> "np.ndarray":
        return self._model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
        )

    def encode_query(self, text: str) -> "np.ndarray":
        return self._model.encode(text, normalize_embeddings=True)


class OnnxMiniLMBackend:
    """all-MiniLM-L6-v2 run through onnxruntime instead of PyTorch (added 2026-10-27).

    Same model, same tokenizer, same mean pooling and normalization as sentence-transformers -
    measured identical on 300 real chunks from kb_cache.npz (cosine 1.00000, min and mean). That
    is why it keeps the plain name "all-MiniLM-L6-v2": the existing cache and the measured 0.35
    similarity threshold stay valid, and switching backends costs no rebuild.

    What it saves: PyTorch is ~2GB of the install and most of the startup time, and it was
    loaded only to embed one short question per turn. onnxruntime + tokenizers is ~50MB, and a
    query embeds in ~6ms on a laptop CPU.
    """

    MAX_TOKENS = 256  # sentence-transformers' max_seq_length for this model

    def __init__(self, model_dir: Path | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime and tokenizers are required for the ONNX embedding backend - run: "
                "pip install onnxruntime tokenizers"
            ) from exc
        from model_paths import MINILM_DIR

        model_dir = Path(model_dir) if model_dir else MINILM_DIR
        model_file = model_dir / "model.onnx"
        tokenizer_file = model_dir / "tokenizer.json"
        if not model_file.exists() or not tokenizer_file.exists():
            raise FileNotFoundError(
                f"ONNX MiniLM not found in {model_dir} - run "
                f"python stages/stage00_setup/download_models.py"
            )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=self.MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self._session = ort.InferenceSession(
            str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self.name = EMBEDDING_MODEL_NAME
        print(f"Using ONNX embeddings: {self.name} (no PyTorch)")

    def _encode(self, texts: list[str]) -> "np.ndarray":
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]
        summed = (hidden * mask[..., None]).sum(axis=1)
        pooled = summed / np.maximum(mask.sum(axis=1, keepdims=True), 1e-9)
        return _normalize(pooled).astype(np.float32)

    def encode_documents(self, texts: list[str], batch_size: int = 64) -> "np.ndarray":
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        parts = [self._encode(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(parts)

    def encode_query(self, text: str) -> "np.ndarray":
        return self._encode([text])[0]


class OllamaEmbeddingBackend:
    """Embeddings from a local Ollama server (see ollama_client.py).

    Batches are chunked rather than sent as one enormous request: Stage 07's real scrape produces
    ~14,800 chunks, and a single HTTP body holding all of them is both a memory spike and an
    all-or-nothing request where one timeout loses the entire startup's work.
    """

    def __init__(self, model_name: str | None = None):
        import ollama_client

        self._client = ollama_client
        self._model = model_name or ollama_client.EMBED_MODEL
        if not ollama_client.is_available():
            raise RuntimeError(
                f"LUC_EMBEDDING_BACKEND=ollama but no Ollama server is answering at "
                f"{ollama_client.OLLAMA_HOST}. Start it, or unset the variable to use the "
                f"in-process sentence-transformers backend instead."
            )
        if not ollama_client.has_model(self._model):
            raise RuntimeError(
                f"Ollama is running but {self._model!r} is not pulled. Run: "
                f"ollama pull {self._model}"
            )
        # Qualified so an Ollama-built cache can never be mistaken for a sentence-transformers
        # one, even if both ever used the same underlying model name.
        self.name = f"ollama:{self._model}"
        print(f"Using Ollama embeddings: {self.name}")

    # Hard cap on how many texts go into ONE /api/embed request, regardless of what the caller
    # asks for. Found the hard way (2026-09-04): upsert_many()'s batch_size defaults to 256 -
    # a sensible number for an in-process transformer, where a "batch" is just a bigger tensor -
    # and passing that straight through killed Ollama's model runner subprocess mid-scrape:
    #
    #   400 {"error":"Post \"http://127.0.0.1:64843/tokenize\": ... actively refused it"}
    #
    # (that port is the runner Ollama spawns; the connection was refused because the runner had
    # just died). Measured on this machine: 128 inputs per request is fine, 256 kills it,
    # independent of how long each text is. 64 keeps real margin under that.
    #
    # This is why the cap lives HERE rather than being left to the caller: batch size means two
    # different things on the two sides of this interface. For sentence-transformers it is a
    # throughput knob with no failure mode; for an HTTP call into a separate process it is a
    # request-size limit that crashes the server when exceeded. The backend is the only place
    # that knows which one it is.
    MAX_EMBED_BATCH = 64

    # Above this many documents in one call, print an upfront warning with an ETA instead of
    # just going quiet. Measured on this machine (2026-09-04, GTX 1650) on realistic 400-char
    # chunks - the size indexer.chunk_text() actually produces:
    #
    #     all-MiniLM-L6-v2, in-process:  ~12 ms/chunk   -> ~3 minutes for Stage 07's ~14,800
    #     nomic-embed-text, via Ollama: ~430 ms/chunk   -> ~106 minutes for the same corpus
    #
    # A 35x difference, and a startup that looks frozen for an hour and a half is indistinguish-
    # able from a hang unless it says what it's doing. (An earlier version of this comment
    # claimed ~15 ms/chunk for Ollama - that was benchmarked on 45-character toy strings, which
    # is not what the scraper produces. Short strings are genuinely that fast; real ones are not.)
    LARGE_CORPUS_WARN = 500

    def encode_documents(self, texts: list[str], batch_size: int = MAX_EMBED_BATCH
                         ) -> "np.ndarray":
        batch_size = max(1, min(batch_size, self.MAX_EMBED_BATCH))
        show_progress = len(texts) > batch_size

        if len(texts) >= self.LARGE_CORPUS_WARN:
            # Rough order-of-magnitude estimate from the per-chunk rates in the comment above;
            # the live progress line below reports the real ETA once it has actual timings.
            print(f"[kb] embedding {len(texts)} chunks through Ollama - roughly "
                  f"{_duration(len(texts) * 0.43)} on this hardware, against about "
                  f"{_duration(len(texts) * 0.012)} for the in-process sentence-transformers "
                  f"backend. Unset LUC_EMBEDDING_BACKEND to switch back; see "
                  f"stages/stage05_nlu_dialogue/README.md for the tradeoff.")

        started = time.monotonic()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(self._client.embed(batch, is_query=False, model=self._model))
            if show_progress:
                done = min(start + batch_size, len(texts))
                elapsed = time.monotonic() - started
                remaining = (elapsed / done) * (len(texts) - done)
                print(f"\r[kb] embedding via Ollama: {done}/{len(texts)} "
                      f"({elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left)   ",
                      end="", flush=True)
        if show_progress:
            print()
        return _normalize(np.asarray(vectors, dtype="float32"))

    def encode_query(self, text: str) -> "np.ndarray":
        vector = self._client.embed([text], is_query=True, model=self._model)[0]
        return _normalize(np.asarray(vector, dtype="float32"))


def build_embedding_backend(backend: str | None = None):
    """Picks a backend by name, falling back to sentence-transformers with a warning if Ollama
    was asked for but isn't usable - the same graceful-degradation rule the rest of the project
    follows. A robot that won't start because a local server is down is worse than one running
    on its second-choice embedding model."""
    backend = backend or EMBEDDING_BACKEND
    if backend in ("auto", "onnx"):
        try:
            return OnnxMiniLMBackend()
        except Exception as exc:
            if backend == "onnx":
                print(f"[kb] {exc}\n[kb] falling back to the sentence-transformers backend.")
            return SentenceTransformerBackend()
    if backend == "ollama":
        try:
            return OllamaEmbeddingBackend()
        except Exception as exc:
            print(f"[kb] {exc}\n[kb] falling back to the sentence-transformers backend.")
            return SentenceTransformerBackend()
    if backend != "sentence-transformers":
        print(f"[kb] unknown LUC_EMBEDDING_BACKEND={backend!r} - using sentence-transformers.")
    return SentenceTransformerBackend()


@dataclass
class KnowledgeChunk:
    chunk_id: str
    text: str
    source_url: str
    last_updated: float
    # Added 2026-08-19 after a real regression: once Stage 07's real scraper filled the KB with
    # 14,000+ real lincoln.edu.my chunks alongside the small hand-curated building directory
    # (SAMPLE_CHUNKS in dialogue_manager_live.py), "Where is the Engineering Department?" started
    # matching a scraped page-title chunk ("Civil Engineering Department" - no location info at
    # all) ahead of the correct, deliberately terse directory sentence ("The Engineering
    # department is on the third floor.") - real content mentioning a term densely (course/dept
    # listings, news pages) can out-score a short authoritative sentence on plain cosine
    # similarity, even though the directory sentence is the one actually meant to answer a
    # wayfinding question. `boost` lets a small set of known-authoritative chunks (right now:
    # just the facilities directory) be preferred without touching SIMILARITY_THRESHOLD itself or
    # trying to solve general re-ranking - see score_all() below and SAMPLE_CHUNKS' own comment
    # for the reasoning behind the specific value chosen.
    boost: float = 0.0

    # "This chunk's text is ALREADY a complete, natural, spoken-ready answer - speaking it
    # verbatim is correct, no language model needed to rephrase it." Added 2026-08-19 as the
    # single biggest per-answer latency win available (explicit request: the robot should reply
    # about as fast as a person would).
    #
    # Only true for the hand-written facilities directory (SAMPLE_CHUNKS in
    # dialogue_manager_live.py), whose entries are deliberately written as finished sentences:
    # "The Engineering department is on the third floor." Sending that to an LLM and asking it to
    # answer "where is the engineering department" from it costs 1-3 real seconds (much more when
    # Gemini 503s and the fallback chain runs) to produce a sentence essentially identical to the
    # one we already had.
    #
    # Deliberately NOT true for Stage 07's scraped chunks (indexer.py leaves this at the default)
    # - those are raw fragments of real web pages: navigation text, half-sentences, page titles
    # with no verb. Reading one aloud verbatim is exactly the failure seen live on 2026-08-19
    # ("Civil Engineering Department" spoken as a complete answer). Those genuinely need a model
    # to turn into a sentence, so they still take the normal grounded-generation path.
    is_direct_answer: bool = False


class KnowledgeBase:
    def __init__(self, backend=None):
        """backend: an embedding backend (see build_embedding_backend). Defaults to whatever
        LUC_EMBEDDING_BACKEND selects, so callers that don't care - which is all of them - keep
        working unchanged."""
        self._backend = backend or build_embedding_backend()
        self._model_name = self._backend.name  # stamped into save()'s metadata - see load()
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._embeddings: dict[str, "np.ndarray"] = {}  # chunk_id -> unit-normalized embedding

    @property
    def embedding_model(self) -> str:
        return self._model_name

    @property
    def similarity_threshold(self) -> float:
        """The minimum score search() requires, for THIS instance's embedding model. Read per
        call rather than frozen at construction so a newly-measured value added to
        SIMILARITY_THRESHOLDS takes effect without anything having to be rebuilt."""
        return threshold_for(self._model_name)

    def upsert(self, chunk: KnowledgeChunk) -> None:
        """Called from Stage 07 when content from the university website is added/updated.

        One chunk at a time - fine for the handful of curated directory entries, but use
        upsert_many() for anything bulk (see why there)."""
        self._chunks[chunk.chunk_id] = chunk
        # encode_documents(), not _embed() - _embed() is the QUERY side, and for a model with
        # asymmetric document/query encoding (nomic-embed-text) storing a chunk as if it were a
        # question would quietly degrade every search that chunk ever takes part in.
        self._embeddings[chunk.chunk_id] = self._backend.encode_documents([chunk.text])[0]

    def upsert_many(self, chunks: list[KnowledgeChunk], batch_size: int = 256) -> None:
        """Bulk version of upsert() - embeds every chunk in batched model calls instead of one
        call per chunk.

        batch_size is a HINT, not a command: a backend is free to cap it, and the Ollama one
        does (see OllamaEmbeddingBackend.MAX_EMBED_BATCH - 256 in a single HTTP request kills
        its model runner). Callers shouldn't have to know which backend is active to pick a
        safe number.

        Added 2026-08-19 for startup latency: Stage 07's real scrape produces ~14,800 chunks, and
        feeding those through upsert() meant ~14,800 separate SentenceTransformer.encode() calls,
        each with its own Python/tensor setup overhead and each processing a single short string.
        Transformer models are heavily throughput-optimized for batches - encoding 256 short
        strings in one call is far closer in wall-clock cost to encoding one than to encoding 256
        - so batching here is a large, purely mechanical speedup with identical output (same
        model, same normalization; embeddings are per-string and don't depend on what else is in
        the batch).
        """
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self._backend.encode_documents(texts, batch_size=batch_size)
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._embeddings[chunk.chunk_id] = vector

    def delete(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)
        self._embeddings.pop(chunk_id, None)

    def __len__(self) -> int:
        return len(self._chunks)

    # --- persistence ---------------------------------------------------------------------
    # Added 2026-08-19. Before this, every single launch of main_live.py re-scraped 87
    # lincoln.edu.my pages (rate-limited, ~a minute of network) and then re-embedded all ~14,800
    # resulting chunks from scratch, purely to rebuild a knowledge base identical to the previous
    # run's. Saving it means a normal startup reloads precomputed vectors from disk instead.
    #
    # Format: one .npz holding the embedding matrix plus a JSON sidecar of chunk metadata, in
    # matching row order. No new dependency (numpy is already required), and both halves are
    # written together so they can't drift apart.

    @staticmethod
    def _meta_path(path: Path) -> Path:
        """Sidecar path for a cache file: kb_cache.npz -> kb_cache.meta.json.

        Deliberately built with with_name(stem + ...) rather than with_suffix(".meta.json") -
        with_suffix's handling of multi-dot suffixes is a fussy corner of pathlib, and save() and
        load() must agree on this path exactly. If they ever disagreed the failure would be
        silent and confusing: every save would appear to succeed and every load would report a
        missing cache, so the robot would just quietly rescrape forever. One shared function
        means they cannot drift apart."""
        return path.with_name(path.stem + ".meta.json")

    def save(self, path: Path) -> None:
        path = Path(path)
        chunk_ids = list(self._chunks.keys())
        if not chunk_ids:
            # Refuse to write an empty cache. Without this, a scrape that came back with nothing
            # (network down, site unreachable, every seed URL 404ing) would persist "the
            # university has no information" to disk, and load() would then serve that empty
            # result for the whole KB_CACHE_MAX_AGE_S window - turning one bad network moment
            # into a robot that answers nothing for a day. Failing to cache just means the next
            # run scrapes again, which is exactly what should happen.
            print("[kb] nothing to cache (0 chunks) - not writing a cache file.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.vstack([self._embeddings[cid] for cid in chunk_ids])
        meta = {
            # Stamped so a cache built by a DIFFERENT embedding model is never silently reused -
            # vectors from another model aren't just less accurate, they're meaningless to
            # compare against, and the resulting nonsense retrieval would be very hard to trace
            # back to a stale file on disk.
            "embedding_model": self._model_name,
            "saved_at": time.time(),
            "chunks": [asdict(self._chunks[cid]) for cid in chunk_ids],
        }
        np.savez_compressed(path, embeddings=matrix)
        self._meta_path(path).write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: Path, max_age_s: float | None = None) -> bool:
        """Repopulates this knowledge base from a save() file. Returns True if it actually
        loaded, False if the cache is missing, stale, or was built by a different embedding
        model - in which case the caller should rebuild from source. Never raises on a corrupt
        or unreadable cache: a bad cache file should cost a rescrape, not crash the robot."""
        path = Path(path)
        meta_path = self._meta_path(path)
        if not path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("embedding_model") != self._model_name:
                print(f"[kb] cache was built with '{meta.get('embedding_model')}', now using "
                      f"'{self._model_name}' - ignoring it and rebuilding.")
                return False
            age_s = time.time() - float(meta.get("saved_at", 0))
            if max_age_s is not None and age_s > max_age_s:
                print(f"[kb] cache is {age_s / 3600:.1f}h old (limit {max_age_s / 3600:.1f}h) - "
                      f"refreshing from the website instead.")
                return False

            matrix = np.load(path)["embeddings"]
            chunk_dicts = meta["chunks"]
            if len(chunk_dicts) != len(matrix):
                print("[kb] cache metadata and embeddings disagree on length - ignoring it.")
                return False
            if not chunk_dicts:
                # Belt-and-braces against the empty-cache trap save() now refuses to create -
                # an older or hand-made empty cache file would otherwise "load successfully"
                # with zero chunks and suppress the rescrape that would fix it.
                print("[kb] cache is empty - rebuilding from source instead.")
                return False

            for chunk_dict, vector in zip(chunk_dicts, matrix):
                chunk = KnowledgeChunk(**chunk_dict)
                self._chunks[chunk.chunk_id] = chunk
                self._embeddings[chunk.chunk_id] = vector
            print(f"[kb] loaded {len(chunk_dicts)} cached chunk(s) from disk "
                  f"({age_s / 3600:.1f}h old) - skipping the website scrape.")
            return True
        except Exception as exc:
            print(f"[kb] could not read the cache ({exc!r}) - rebuilding from source instead.")
            return False

    def _embed(self, text: str) -> "np.ndarray":
        """One QUERY embedding. Backends return unit vectors, so the plain dot product in
        score_all() below IS the cosine similarity - cheaper than re-normalizing per search.

        Note this is the query side specifically: upsert()/upsert_many() go through
        encode_documents() instead, which some models encode differently on purpose (see the
        backend section above)."""
        return self._backend.encode_query(text)

    def score_all_detailed(self, query: str) -> list[tuple[float, float, KnowledgeChunk]]:
        """(raw_similarity, ranking_score, chunk) for every chunk, sorted by ranking_score
        descending, with no threshold filtering. The two numbers do two different jobs and are
        deliberately kept apart:

          raw_similarity - "is this chunk actually about the question?" This is what the
                           relevance threshold compares against.
          ranking_score  - raw + the chunk's own `boost`. This is what decides ORDER among
                           chunks that already cleared the bar.

        Keeping boost out of the threshold comparison (changed 2026-09-04) fixes a real
        contradiction: KnowledgeChunk.boost documents itself as "a ranking nudge only, not a
        substitute for real relevance", but folding it into the thresholded score meant a
        boosted chunk needed less genuine similarity than an unboosted one to be considered
        relevant at all - exactly the substitution the docstring rules out.

        It also broke threshold measurement in a way that only became visible with a second
        embedding model. evaluate_retrieval.py measures against the boosted directory chunks, so
        any threshold it found silently included DIRECTORY_BOOST - and that threshold was then
        applied to Stage 07's ~14,800 scraped chunks, which have no boost at all and therefore
        had to clear a bar 0.15 higher than the one that was actually measured.
        """
        if not self._chunks:
            return []
        query_vec = self._embed(query)
        scored = [
            (
                float(np.dot(query_vec, self._embeddings[chunk_id])),
                float(np.dot(query_vec, self._embeddings[chunk_id])) + chunk.boost,
                chunk,
            )
            for chunk_id, chunk in self._chunks.items()
        ]
        scored.sort(key=lambda triple: triple[1], reverse=True)
        return scored

    def score_all(self, query: str) -> list[tuple[float, KnowledgeChunk]]:
        """(ranking_score, chunk) for every chunk, sorted descending, no filtering - the
        ranking view of score_all_detailed(), kept for callers that only care about order."""
        return [(ranking, chunk) for _, ranking, chunk in self.score_all_detailed(query)]

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """Chunks that are genuinely relevant to the query, best first.

        Filters on raw similarity and orders by the boosted score - see score_all_detailed()
        for why those are two different numbers."""
        threshold = self.similarity_threshold
        relevant = [c for raw, _, c in self.score_all_detailed(query) if raw >= threshold]
        return relevant[:top_k]


def _check(label: str, passed: bool) -> bool:
    status = "OK" if passed else "WARNING (see evaluate_retrieval.py / retune the threshold)"
    print(f"  [{status}] {label}")
    return passed


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"Embedding model: {kb.embedding_model} (threshold {kb.similarity_threshold})")
    kb.upsert(KnowledgeChunk(
        chunk_id="dept-cs",
        text="The Computer Science department is located on the 4th floor.",
        source_url="https://university.example.edu/departments/cs",
        last_updated=0.0,
    ))
    kb.upsert(KnowledgeChunk(
        chunk_id="mosque-men",
        text="The men's mosque is on the first floor.",
        source_url="https://university.example.edu/campus/mosque-men",
        last_updated=0.0,
    ))
    kb.upsert(KnowledgeChunk(
        chunk_id="mosque-women",
        text="The women's mosque is on the second floor.",
        source_url="https://university.example.edu/campus/mosque-women",
        last_updated=0.0,
    ))
    kb.upsert(KnowledgeChunk(
        chunk_id="event-orientation",
        text="Freshman orientation event starts next Monday at the main auditorium.",
        source_url="https://university.example.edu/events",
        last_updated=0.0,
    ))

    results = kb.search("where is the computer science department")
    print("\nSearch results for 'where is the computer science department':")
    for r in results:
        print(f"  - {r.chunk_id}: {r.text}")

    # The exact case the old word-overlap scoring got wrong (see module docstring) - real
    # embeddings need to actually distinguish "men's" from "women's", not just token-match
    # "mosque"/"floor".
    women_results = kb.search("where is the mosque for women")
    print(f"\n'where is the mosque for women' -> {[r.chunk_id for r in women_results]}")

    no_match = kb.search("what is the cafeteria menu today")
    print(f"\nOut-of-scope question 'what is the cafeteria menu today' -> {no_match}")

    print()
    all_ok = True
    all_ok &= _check(
        "CS query's top result is dept-cs", bool(results) and results[0].chunk_id == "dept-cs"
    )
    all_ok &= _check(
        "'mosque for women' top result is mosque-women, not mosque-men",
        bool(women_results) and women_results[0].chunk_id == "mosque-women",
    )
    all_ok &= _check("out-of-scope question returns no match", no_match == [])

    # --- save/load round trip (2026-08-19) --------------------------------------------------
    # The cache exists to skip a ~1 minute scrape-and-embed on every launch, which means a subtly
    # wrong cache would be served silently for up to 24h - worth proving it actually round-trips
    # rather than assuming it. Checks the parts that would fail quietly: that every field
    # survives (including the newer boost/is_direct_answer flags, which change ranking and
    # whether the LLM is called at all), and that reloaded embeddings still produce the same
    # search result rather than merely loading without error.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "kb_cache.npz"
        kb.upsert(KnowledgeChunk(
            chunk_id="flagged-example",
            text="The Engineering department is on the third floor.",
            source_url="facilities-directory:engineering",
            last_updated=123.0,
            boost=0.15,
            is_direct_answer=True,
        ))
        kb.save(cache_path)

        # Reuses the same backend object - a fresh KnowledgeBase() would reload the model
        # (or reopen the Ollama connection) for no benefit; what's being tested is that
        # load() repopulates empty chunk/embedding dicts, not that a model can be loaded
        # twice.
        reloaded = KnowledgeBase(backend=kb._backend)
        loaded_ok = reloaded.load(cache_path)
        all_ok &= _check("cache reports a successful load", loaded_ok)
        all_ok &= _check(
            "every chunk survived the round trip", len(reloaded) == len(kb)
        )

        original = kb._chunks["flagged-example"]
        restored = reloaded._chunks.get("flagged-example")
        all_ok &= _check(
            "boost / is_direct_answer / metadata all survived",
            restored is not None
            and restored.text == original.text
            and restored.source_url == original.source_url
            and restored.last_updated == original.last_updated
            and restored.boost == original.boost
            and restored.is_direct_answer == original.is_direct_answer,
        )
        reloaded_results = reloaded.search("where is the computer science department")
        all_ok &= _check(
            "reloaded embeddings still return the same top result",
            bool(reloaded_results) and reloaded_results[0].chunk_id == "dept-cs",
        )

        # A cache written by a different embedding model must be rejected, not silently used -
        # vectors from another model aren't merely less accurate, they're not comparable at all,
        # and the resulting nonsense retrieval would be very hard to trace back to a stale file.
        mismatched = KnowledgeBase(backend=kb._backend)
        mismatched._model_name = "some-other-embedding-model"
        all_ok &= _check(
            "a cache from a different embedding model is refused",
            mismatched.load(cache_path) is False,
        )

    if all_ok:
        print(
            "\nOK: semantic search finds the correct result even when chunks share nearly all "
            "their words, stays empty when nothing is actually relevant, and survives a "
            "save/load round trip with its ranking flags intact."
        )
    else:
        print(
            f"\nOne or more checks above didn't hold with the current threshold "
            f"({kb.similarity_threshold}, for embedding model {kb.embedding_model!r}) - run "
            "evaluate_retrieval.py for a fuller measurement and "
            "a suggested value before trusting this in a live conversation."
        )
