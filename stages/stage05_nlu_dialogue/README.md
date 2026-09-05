# Stage 05 — Understanding & Dialogue (NLU + RAG)

## Files here

- `identity.py` — the robot's own identity (its name, and who built it) — "what's your name",
  "who made you", "what's your creator's email", including compound questions asking several of
  these at once. Deliberately checked before the knowledge base or any LLM call (see
  `dialogue_manager.py`'s `handle_question()`) — fully offline, plain rule-based string matching,
  no embeddings or network involved. The values themselves are **not** in the source any more
  (2026-09-04): they come from MySQL, table `robot_identity` (schema in Stage 02's `robot_db.py`),
  read once at startup and cached to `identity_cache.json` so an outage can't revert the robot to
  an old name. To change what it says:

  ```powershell
  python stages\stage05_nlu_dialogue\identity.py --show
  python stages\stage05_nlu_dialogue\identity.py --set robot_name=Nova creator_name="Someone Else"
  ```

  or `UPDATE robot_identity SET value = 'Nova' WHERE field = 'robot_name';` in Workbench. The
  creator's name is matched on as well as spoken ("where is Tan from"), so a rename updates the
  question matching too — nothing about the robot's identity is spelled out in code.
- `knowledge_base.py` — Stage 05 part 1: real semantic search — a local sentence-embedding model
  + cosine similarity, not a hardware/API-dependent piece, so there's no separate mock/live split
  here — both `dialogue_manager.py`'s self-test and `dialogue_manager_live.py`'s REPL use the real
  thing. Two interchangeable embedding backends (`all-MiniLM-L6-v2` in-process by default, or
  `nomic-embed-text` via Ollama — see "Ollama for the RAG vectors too" below). Replaces the
  word-overlap keyword scoring this file carried as a documented TODO since Stage 05 first went
  live — see "Real search, not keyword matching" below for why that mattered in practice, not
  just in theory.
- `ollama_client.py` — the local LLM and local embedding model, over plain HTTP to an Ollama
  server on this machine. No API key, no quota, no network. Run it directly to check the server,
  the models, and the real latency on your hardware.
- `evaluate_retrieval.py` — measures and tunes `knowledge_base.py`'s `SIMILARITY_THRESHOLDS`
  against real query/chunk pairs, instead of trusting a guessed number — same discipline as
  Stage 04's `evaluate_wer.py` and Stage 06's `evaluate_clarity.py`.
- `dialogue_manager.py` — Stage 05 part 2: intent routing + the `AnswerGenerator` interface.
  Updated (not just a hardware swap this time — a real design addition): added the
  `GENERAL_CHAT` intent and a `MockAnswerGenerator` for its own self-test. No network needed at
  runtime (the answer generation itself is mocked), but its self-test now needs
  `sentence-transformers` installed, since it imports the real `knowledge_base.py`.
- `dialogue_manager_live.py` — the real version: drop-in `AnswerGenerator`s —
  `GeminiAnswerGenerator` (Google, the strongest option because it can search the live web),
  `GroqAnswerGenerator`, `OllamaAnswerGenerator` (local, free, always available),
  `UniversityOnlyAnswerGenerator` (no LLM at all), `ClaudeAnswerGenerator` (Anthropic — kept
  working but not checked by default), and `ChainAnswerGenerator`, which tries them in order so
  an API outage costs a retry instead of a wrong answer. `MODE` picks the arrangement; the
  default `"auto"` assembles whatever is actually usable at startup. Tested with a console REPL,
  and with `--self-test` for the failover itself:

  ```powershell
  python stages\stage05_nlu_dialogue\dialogue_manager_live.py --self-test
  ```

  That one deliberately breaks the cloud backend the exact ways it has broken for real (503, 429,
  and an empty response) and checks the local model picks each of them up.

## Real search, not keyword matching

The old scoring counted shared words between the query and each chunk. That broke the moment two
chunks were genuinely similar in wording: `"the men's mosque is on the first floor"` and
`"the women's mosque is on the second floor"` share almost every word, so word-overlap scored
them identically — "where is the mosque for women" silently returned the men's mosque, sounding
just as confident as a correct answer would have. A local embedding model
(`all-MiniLM-L6-v2` via `sentence-transformers`) represents `"men's"` and `"women's"` as different
concepts rather than different tokens, and gets this right — see the assertions in
`knowledge_base.py`'s own self-test.

**The relevance threshold is the one number in that file that must not be eyeballed** — and there
is one per embedding model (`SIMILARITY_THRESHOLDS`), because a value measured for one model is
meaningless for another. Every other
stage's routing depends on a contract: `search()` returns `[]` when nothing is actually relevant,
so `handle_question()` falls through to "does this even sound university-related?" instead of
grounding an answer in a barely-related chunk — the same anti-hallucination guarantee this
project has enforced from the start. Cosine similarity is never exactly 0, so without a
threshold, `search()` would always return its closest chunk regardless of real relevance. Run

```powershell
python stages\stage05_nlu_dialogue\evaluate_retrieval.py
```

after installing, to measure the current threshold against real relevant/irrelevant query pairs
and get a suggested value — update that model's entry in `SIMILARITY_THRESHOLDS` (`knowledge_base.py`) if it differs
from what's currently set.

## Local LLM via Ollama — the answer to "what if the API is down"

This is not a hypothetical. The cloud backend has failed twice in real testing here: a Gemini
**503** in the middle of an answer (2026-08-19) and a **429** on the very first question of a
session (2026-08-17). Both times the robot fell through to reading a raw scraped page-title
fragment aloud as if it were an answer. A kiosk in a university lobby can't depend on somebody
else's free tier staying up.

So the robot now runs a **fallback chain**: a cloud API first when it works, a model running on
this machine when it doesn't, and knowledge-base-only answers if neither is available. Ollama is
free, needs no API key, has no quota, and works with the network unplugged.

### Install (one time)

```powershell
winget install Ollama.Ollama          # or download from https://ollama.com/download
ollama pull llama3.2:3b               # the chat model  (~2GB)
ollama pull nomic-embed-text          # the embedding model (~274MB, optional - see below)
```

Ollama runs as a background service once installed. Check everything at once:

```powershell
python stages\stage05_nlu_dialogue\ollama_client.py
```

That prints whether the server is up, which models are pulled, how long the model takes to warm
up, and the real latency of a grounded answer and an embedding — measured on your machine rather
than assumed.

### Why `llama3.2:3b` and not something bigger

This machine has a **GTX 1650 with 4GB of VRAM**. A 7–8B model at Q4 is around 4.7GB, so it
doesn't fit and Ollama offloads part of it to the CPU — the difference between answering in about
a second and answering in fifteen. A voice robot that takes fifteen seconds to reply has failed
at the task even if the sentence it eventually produces is better. `llama3.2:3b` fits in VRAM
with room to spare.

On a machine with more VRAM, `qwen2.5:7b` and `llama3.1:8b` are drop-in and noticeably stronger —
no code change, just:

```powershell
$env:LUC_OLLAMA_CHAT_MODEL="qwen2.5:7b"
```

or the same line in the project-root `.env`.

Ollama stores models under `C:\Users\<you>\.ollama` by default. Set `OLLAMA_MODELS` to a path on
another drive if C: is tight.

### Mode

`dialogue_manager_live.py`'s `MODE` constant:

```python
MODE = "auto"             # default - use whatever is available, cloud first, local second
MODE = "full"             # cloud API only, no local fallback (for testing the cloud path alone)
MODE = "ollama_only"      # local only, never call out (what the robot does offline)
MODE = "university_only"  # no LLM at all - knowledge-base text verbatim, general chat off
```

In `auto`, startup prints exactly which backends are in the chain and in what order, and every
failover during a conversation is logged with the backend's name — a chain that silently
downgraded would hide the very API problem you'd want to know about.

`UniversityOnlyAnswerGenerator`, `GeminiAnswerGenerator`, `GroqAnswerGenerator`,
`OllamaAnswerGenerator`, `ClaudeAnswerGenerator` and `ChainAnswerGenerator` all implement the
same `AnswerGenerator` protocol, so `DialogueManager` is untouched by any of this.

### Ollama for the RAG vectors too (optional)

`knowledge_base.py` can take its embeddings from Ollama instead of the in-process
sentence-transformers model:

```powershell
$env:LUC_EMBEDDING_BACKEND="ollama"
```

**Be clear about what this buys, and what it costs.** It is an **accuracy** change, not a speed
one — and the cost is far larger than it first looks. `nomic-embed-text` (768-d, trained with
separate document/query encodings, which this code uses) should retrieve the right chunk more
often on paraphrased questions, and a spoken question is almost always a paraphrase — a student
says "what floor do I go to for nursing", not the directory's own wording.

Measured on this machine (2026-09-04), on identical 400-character chunks — the size
`indexer.chunk_text()` actually produces:

| | all-MiniLM-L6-v2 | ollama:nomic-embed-text |
|---|---|---|
| Dimensions | 384 | 768 |
| Runs | in-process, no server | HTTP to Ollama, sharing the 4GB GPU with the chat model |
| One spoken query | **31 ms** | **125 ms** |
| Per chunk, bulk | **12 ms** | **430 ms** |
| Stage 07's ~14,800 chunks | **~3 minutes** | **~106 minutes** |
| Relevance threshold | 0.35 | 0.65 |
| Test set result | 11/11 relevant, 0/5 false | 11/11 relevant, 0/5 false |

That last row is the one that matters: **both models score perfectly on the current test set.**
It's small and hand-written, so this shows neither is broken — not that either is better. Nothing
measured so far justifies paying 35× for the switch.

So **leave it off** unless real usage shows retrieval picking the wrong chunk for a question the
knowledge base clearly can answer. If you do turn it on, expect the rebuild to take the better
part of two hours; the backend prints an upfront estimate and a live progress line with an ETA,
so an hour of silence doesn't look like a hang.

> An earlier version of this table claimed ~15 ms/chunk for Ollama. That was benchmarked on
> 45-character toy strings rather than real chunks — short strings genuinely are that fast, real
> ones are not. The numbers above are measured on the real thing.

Two things to know before turning it on:

- **The threshold is per model and is not transferable.** Look at the two numbers above: 0.35 vs
  0.65. `nomic-embed-text` scores a completely *unrelated* query around 0.40–0.64 — higher than
  MiniLM scores a genuinely relevant one. Running it on MiniLM's 0.35 would match everything and
  silently switch off the guarantee that `search()` returns `[]` when nothing is relevant.
  `SIMILARITY_THRESHOLDS` is keyed by model, and an unmeasured model warns on every startup
  rather than borrowing another model's number.
- **The knowledge-base cache is invalidated.** Vectors from two different models are not
  comparable, so the cache is stamped with the model name and refused on mismatch — switching
  costs one full re-scrape and re-embed.

### A bug this turned up: `boost` was doing two jobs

Adding a second embedding model exposed a real problem in the existing code. `KnowledgeChunk.boost`
describes itself as "a ranking nudge only, not a substitute for real relevance", but `search()`
was thresholding the *boosted* score — so a boosted directory chunk needed 0.15 less genuine
similarity than a scraped chunk to count as relevant at all, which is exactly the substitution
the docstring rules out.

It also corrupted the measurement: `evaluate_retrieval.py` measures against the boosted directory
chunks, so every threshold it ever produced had `DIRECTORY_BOOST` baked in — and that number was
then applied to Stage 07's ~14,800 *unboosted* scraped chunks, which had to clear a bar 0.15
higher than the one actually measured.

Fixed 2026-09-04: `search()` now filters on raw similarity and orders by the boosted score, and
`evaluate_retrieval.py` measures the same way. Reassuringly, re-measuring MiniLM this way returns
**0.35** — exactly the value measured back on 2026-08-12, before the boost existed.

## Why a new intent

The project requirement is that the robot can talk about anything, not just the university — but
university facts must still never be hallucinated. Those two goals need different rules, so
`handle_question()` branches three ways based on what the knowledge base finds:

```
question
   │
   ▼
identity check (identity.py) ──▶ IDENTITY if it's about the robot itself - offline, no KB, no LLM
   │ (no match)
   ▼
knowledge base search
   │
   ├─ found something ──▶ GENERAL_INFO
   │                       ("where is X" questions land here too - the building-directory chunks
   │                       already state the floor, so a floor-only answer comes straight out of
   │                       this same grounded path. No separate wayfinding/directions stage.)
   │                        │
   │                        ├─ top chunk is_direct_answer ──▶ spoken VERBATIM, no LLM call at all
   │                        │                                  (the curated directory entries are
   │                        │                                   already finished sentences - see
   │                        │                                   below)
   │                        └─ otherwise ──▶ three-tier fallback (2026-08-19):
   │                             1. generate_grounded_answer(chunks) - answer strictly from the
   │                                retrieved text (the normal, best case)
   │                             2. failed, OR succeeded but honestly said it doesn't have the
   │                                info ──▶ classify_and_answer_general() - a LIVE search
   │                                instead of trusting a possibly-incomplete scraped fragment
   │                             3. both empty ──▶ speak the matched chunk verbatim, so the
   │                                robot still says something real rather than going silent
   │
   └─ found nothing ──▶ ask the model: does this even sound like a university question?
                            │
                            ├─ yes ──▶ UNKNOWN: try to answer anyway (own knowledge, or a live
                            │           Google Search for GeminiAnswerGenerator) - flagged as
                            │           unverified with a pointer to lincoln.edu.my rather than
                            │           stated as certain. Only falls back to the fixed apology
                            │           if the generator truly has nothing (never a silent
                            │           response either way).
                            └─ no  ──▶ GENERAL_CHAT: answered openly, no grounding constraint
```

Changed 2026-08-19: the GENERAL_INFO path's fallback-on-failure moved from `AnswerGenerator` (each
generator silently swallowing its own errors and returning the raw top chunk) to
`DialogueManager.handle_question()` owning the whole chain. Real bug this fixes: a Gemini 503 on
`generate_grounded_answer()` used to mean the robot spoke `chunks[0].text` verbatim, and for a raw
scraped fragment (not one of the curated `is_direct_answer` sentences) that can be a page title
with no actual information — "where is Lincoln University College located" got answered "Lincoln
University College". Per explicit request, an answer that isn't available locally should come
from a live API search rather than a guess at reading a fragment aloud - see
`_looks_like_no_info()`/`handle_question()` in `dialogue_manager.py`.

Changed 2026-08-17: the `UNKNOWN` branch used to always return the fixed apology, full stop -
real testing showed that too restrictive (a question naming the university directly, e.g. asking
its ranking, got refused outright even though the LLM could plausibly help, especially with
search). The "never confidently state an unverified university-specific fact" guarantee now
lives in `GENERAL_CLASSIFY_PROMPT` and in `dialogue_manager.py`'s fallback-only-when-blank
handling, not in a router rule that refused to even try. The strictly-grounded `GENERAL_INFO`
path above is untouched either way — still 100% restricted to the retrieved knowledge-base text,
never a search/guess substitute for it.

The grounded path and the open-chat path are deliberately two separate calls with two separate
system prompts (`AnswerGenerator.generate_grounded_answer` vs. `.classify_and_answer_general`),
not one free-form "answer this" call — they have opposite rules, and keeping them separate is
what makes the "never hallucinate university facts" guarantee actually hold.

## Getting an API key (optional — the local model works without one)

A cloud key is worth having: Gemini can search the live web, which a local model can't, and that
matters most for questions about a smaller institution the model barely saw in training. But it
is no longer required for anything — with no key, `MODE = "auto"` runs on Ollama alone.

`dialogue_manager_live.py` auto-detects which one you've set and puts it at the head of the chain,
so you only need one (Gemini is checked first if both are set — see `_build_answer_generator()`).
The key can also go in the project-root `.env` alongside every other setting, instead of being a
real environment variable.

**Google (Gemini)** — default backend as of 2026-08-17 (`GeminiAnswerGenerator`, model
`gemini-3.7-flash`). Google AI Studio's free tier doesn't need a card to get a key, but which
models are actually included in the free tier (and the rate limits) does change over time — check
[aistudio.google.com](https://aistudio.google.com) for gemini-3.7-flash's current status if you
hit a billing/quota error:

1. Go to [aistudio.google.com](https://aistudio.google.com) and sign in with any Google account.
2. Create an API key ("Get API key").
3. Set it as an environment variable:

```powershell
$env:GEMINI_API_KEY = "your-key-here"         # current session only
setx GEMINI_API_KEY "your-key-here"           # persists across future sessions
```

**Groq** — an ongoing daily free quota (no card needed), but the API Keys page in the console
only appears after you've verified your email from the signup confirmation message:

1. Create a free account at [console.groq.com](https://console.groq.com) and verify your email.
2. Create an API key at [console.groq.com/keys](https://console.groq.com/keys).
3. Set it as an environment variable:

```powershell
$env:GROQ_API_KEY = "your-key-here"           # current session only
setx GROQ_API_KEY "your-key-here"             # persists across future sessions
```

**Anthropic (Claude)** — `ClaudeAnswerGenerator` still works, but isn't checked by default
anymore (see `_build_answer_generator()`); new accounts get a small one-time free credit (~$5,
no card needed), but it needs a payment method added before the API will actually use it, and it
runs out eventually (not an ongoing daily quota). Only bother with this if you specifically want
Claude back over Gemini:

1. Create a free account at [platform.claude.com](https://platform.claude.com).
2. Create an API key from your account/organization settings.
3. Set `ANTHROPIC_API_KEY` the same way as above, and add a check for it ahead of
   `GEMINI_API_KEY` in `_build_answer_generator()`.

Either way: never hardcode a key into a file or commit it anywhere. Free-tier terms/limits can
change on any provider — check your account dashboard if you hit errors.

## Install (one time, venv already active)

```powershell
pip install -r stages\stage05_nlu_dialogue\requirements.txt
```

Reuses `torch` already installed from Stage 01's requirements.txt (same shared venv). The first
time `knowledge_base.py` runs (mock or live), it downloads the ~80MB embedding model from Hugging
Face and caches it locally — needs internet once, fully offline after that.

## Run

Mock regression test (no API key, but needs the embedding model installed/cached — see above):

```powershell
python stages\stage05_nlu_dialogue\dialogue_manager.py
```

Real live REPL — no API key needed in the current default mode:

```powershell
python stages\stage05_nlu_dialogue\dialogue_manager_live.py
```

It preloads a made-up building directory (see `SAMPLE_CHUNKS`, floor-only, no room numbers)
purely so the grounded path has something real to retrieve — Stage 07's actual scraper doesn't
exist yet. In the default
`university_only` mode, try a question about those facts (grounded answer, quoted verbatim from
the knowledge base) and anything else (apology fallback — general chat is off). Once `MODE` is
flipped to `"full"` (needs `GEMINI_API_KEY` or `GROQ_API_KEY` set), a totally unrelated
question (weather, jokes, trivia) shows `GENERAL_CHAT`, and a university-specific question the
knowledge base doesn't cover (e.g. "what's Lincoln's ranking") shows the `UNKNOWN` path — with
Gemini, that's now a real attempted (search-grounded) answer with a caveat pointing to
lincoln.edu.my, not an automatic apology — see "Why a new intent" above.

## Known limitations (by design, for now)

- The `SIMILARITY_THRESHOLDS` entries in `knowledge_base.py` were measured against a small, hand-written test
  set in `evaluate_retrieval.py` — real usage will surface query phrasings that set doesn't cover.
  If search starts missing things it should find (or grounding answers in the wrong chunk), add
  the failing query to `TEST_CASES` and re-run the sweep rather than hand-tuning the number.
- Brute-force cosine similarity over every chunk — fine for the current few dozen facts, but
  won't stay fast forever as Stage 07's scraper grows the corpus (documented TODO in
  `knowledge_base.py` to move to FAISS/Chroma/pgvector once that matters).
- No dedicated NLU/entity extraction — every question goes through the same knowledge-base
  search → grounded-or-general routing described above. That's a deliberate simplification (see
  "Why a new intent"), not a gap to fill later — there's no separate directions/wayfinding intent
  by design, since a floor number is all the project wants the robot to give out.
- The `UNIVERSITY_RELATED: yes/no` classification is a single LLM call parsing a specific text
  format — if the model doesn't follow the format, `_parse_university_related()` (shared by both
  backends) fails safe toward the apology path (defaults to `True`) rather than risk an
  ungrounded guess at a missing university fact. Worth watching in testing; a structured output
  mode could replace this text-parsing approach later if it proves unreliable.
- Not yet linked to Stage 04 (STT) or Stage 06 (TTS) — tested here as typed text in, spoken-style
  text out. Stage 08 (Orchestrator) wires the full voice loop together.
- `generate_grounded_answer()`/`classify_and_answer_general()` are both single blocking API
  calls (no streaming) — fine for this stage's own testing, but Stage 06 (TTS) will want to
  think about streaming the answer so speech can start before the full response is generated,
  same tradeoff already noted in Stage 06's own file.
