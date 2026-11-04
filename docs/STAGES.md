# Staged Build Plan

Every stage: **one clear goal** + **built independently of real hardware (mock data)** + **a
specific pass/fail test** before moving to the next stage. The actual skeleton code for each
stage lives in `stages/stageXX_.../`.

---

## Stage 00 — Environment and hardware setup
**Goal:** prepare the device (Mini PC/Jetson), camera, microphone, speaker, and Python
environment.
**Test:** a single script confirms the camera returns frames, the mic records audio, and the
speaker plays a test tone.
**Output:** `requirements.txt`, `test_hardware.py` prints pass/fail for each device.

## Stage 01 — Presence Detection
**Goal:** detect when a person/group enters the camera's field of view, and estimate proximity
(far / close enough to interact).
**Suggested tech:** a lightweight person detector (YOLOv8n or MediaPipe) fast enough for an edge
device.
**Test:** run the camera (or a test video) and print "person detected" with a headcount the
moment someone enters.
**Success criterion:** under 500ms from a person appearing to detection.

## Stage 02 — Face Recognition & Memory
**Goal:** recognize a previously stored face; register a new face only with explicit consent
from the person.
**Built with:** OpenCV's YuNet (detection) + SFace (128-d embeddings) — see
`stages/stage02_face_recognition/sface_embedder.py` for why those rather than the suggested
InsightFace (licensing) or `face_recognition`/dlib (compiler toolchain).
**Storage:** MySQL (`robot_db.py`), holding (name, face_embedding, consent_flag, first_seen,
last_seen, visit_count) plus the conversation transcripts from Stage 08. MySQL rather than the
originally-suggested SQLite because it's what's already installed and inspectable via Workbench
here, and a kiosk deployment can point at a server; matching itself is still an in-memory scan,
so no vector index (FAISS) is needed at this scale.
**Test:** enroll a face with `face_memory_live.py`, quit, run it again -> it recognizes you
without asking, and `SELECT * FROM people` in Workbench shows the row.
**Privacy note:** see `PRIVACY.md` — without explicit consent, no embedding is ever stored.

## Stage 03 — Proactive Greeting
**Goal:** a simple state machine: `IDLE -> DETECTED -> GREETING -> LISTENING`. For a group, one
shared group greeting. If a known face is in the group, add a personal touch to that same
greeting.
**Test:** simulate a person entering (mock event) -> TTS is called automatically without any user
command.

## Stage 04 — Listening (Speech-to-Text)
**Goal:** after the greeting, record audio and convert it to text quickly, with Voice Activity
Detection (knows when the person finished speaking) — English only in this first stage.
**Suggested tech:** Whisper (local, streaming) or a cloud STT streaming service.
**Test:** run a sample audio file -> compare the resulting text against the expected text
(acceptable WER).

## Stage 05 — Understanding & Dialogue (NLU + RAG)
**Goal:** classify intent (answerable from the knowledge base / general question / not
understood) + search the knowledge base (vector search) + generate an answer via an LLM grounded
strictly in the retrieved content. Deliberately no separate directions/wayfinding intent — a
"where is X" question is answered straight from the knowledge base, floor-only (e.g. "the
Engineering department is on the third floor"), no step-by-step route. A floor number is enough
to save the student time, and a dedicated wayfinding stage wasn't worth the added surface area.
**Test:** a set of sample test questions -> correct intent classification + an answer containing
the right information, and an out-of-scope question -> the specific apology response, not a
hallucinated answer.
**Not dependent on one LLM:** answer generation is a fallback chain — cloud (Gemini/Groq) first
because it can search the live web, then a local Ollama model, then the retrieved chunk spoken
verbatim. Added 2026-09-04 after two real cloud failures (a 503 mid-answer, a 429 on a session's
first question) each ended with the robot reading a raw page-title fragment aloud. Retrieval has
a local option too (`LUC_EMBEDDING_BACKEND=ollama`), but that one is an accuracy choice, not an
availability one — the default embedding model already runs offline in-process.

## Stage 06 — Speaking (Text-to-Speech)
**Goal:** convert the answer text to natural speech with the lowest possible latency (streaming
TTS).
**Test:** measure Time-To-First-Audio for a medium-length sentence, target < 1-2 seconds.

## Stage 07 — Continuous University Data Pipeline
**Goal:** a periodically scheduled scraper that crawls public university pages, extracts
structured data, diffs it against the previous snapshot, and automatically updates the knowledge
base (embeddings) on any change — no manual work required.
**Test:** run the crawler once manually -> get structured JSON output; then modify a test page
locally -> confirm the diff is detected and the index is updated automatically on the next run.

## Stage 08 — Orchestrator
**Goal:** tie all the stages together in one loop (event loop / state machine) that manages:
detection, greeting, listening, understanding, answering, speaking, and handling interruptions
(a new person arrives mid-interaction, more than one voice at once).
**Test:** a full end-to-end scenario with test people (or recorded audio/video files) from entry
to exit.

## Stage 09 — Testing, Monitoring, and Deployment
**Goal:** log events (without storing raw biometric data beyond the retention policy), handle
errors with fallback phrases ("Sorry, could you repeat that?"), measure latency per stage, run as
a service that starts automatically on boot with a watchdog that restarts it on crash.
**Test:** a stress test (rapid back-to-back interactions) + simulate a module crash and confirm
automatic recovery.

---

## Stage dependencies (build order)

```
00 -> 01 -> 02 ┐
             ├-> 03 -> 04 -> 05 -> 06 -> 08 -> 09
      07 ─────────────────^ (feeds knowledge into 05, can run in parallel from the start)
```

Stage 07 can be built in parallel with 01-06 since it's independent (a standalone crawler), but
Stage 08 (full integration) needs all the stages before it working independently first.
