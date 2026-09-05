# Stage 08 — Orchestrator

Two files here:

- `main.py` — the pure-Mock skeleton: every module (presence, face, STT, dialogue, TTS) is the
  fake version from each stage. Proves the wiring/loop logic (detect → recognize → greet →
  listen → understand → speak) works, with no hardware needed. Keep this working as a regression
  test of the orchestration logic itself.
- `main_live.py` — the real version: every Mock swapped for its stage's real implementation. A
  real camera + real face recognition feed a real greeting, a real microphone question gets a
  real grounded answer from the knowledge base, spoken back with real streaming TTS.

## Install (one time, venv already active)

Every upstream stage's requirements must already be installed (this stage reuses their models
and code directly), plus this stage's own (which just documents that dependency chain):

```powershell
pip install -r stages\stage01_presence_detection\requirements.txt
pip install -r stages\stage02_face_recognition\requirements.txt
pip install -r stages\stage03_proactive_greeting\requirements.txt
pip install -r stages\stage04_speech_to_text\requirements.txt
pip install -r stages\stage05_nlu_dialogue\requirements.txt
pip install -r stages\stage06_text_to_speech\requirements.txt
pip install -r stages\stage07_data_pipeline\requirements.txt
```

Models that must already be downloaded (see each stage's own README):
Stage 02's YuNet/SFace ONNX files, Stage 06's Piper voice. Stage 04's faster-whisper model and
Stage 05's sentence-transformers model both download automatically on first run.

## Where the robot's answers actually come from

Three separate sources feed answers, and it's worth knowing which is which when testing:

- **The robot's own identity** (`identity.py`, Stage 05) - "what's your name", "who made you",
  "what's your creator's email/nationality", including compound versions asking several of these
  at once. Answered offline, before the knowledge base or any LLM is even touched - see
  `identity.py`'s module docstring. The values come from MySQL (`robot_identity`), loaded once
  in `RobotApp.__init__` on the connection the robot already has — so change the robot's name,
  creator, email or nationality with
  `python stages\stage05_nlu_dialogue\identity.py --set robot_name=Nova` (or an UPDATE in
  Workbench), not by editing code. Whisper's vocabulary bias is built from the same values, so a
  rename also changes what the robot listens *for*.
- **Hardcoded floor directory** (`dialogue_manager_live.py`'s `SAMPLE_CHUNKS`) - which floor a
  department/service/mosque is on. Manually maintained, not scraped - see its `source_url`
  comment for why. Ask e.g. *"where is the library"* or *"where is the engineering
  department"* to test this path - the answer will be a short floor-only sentence.
- **Real lincoln.edu.my content** (Stage 07's scraper - see `main_live.py` `__init__`) - actual
  programme pages (duration, intakes, entry requirements, career outcomes, etc.) and school
  overview pages. Ask e.g. *"how long is the PhD in Information Technology programme"*, *"what
  are the entry requirements for the Master of Computer Science"*, or *"tell me about the School
  of AI Computing and Multimedia"* to test this path - a floor-directory-only question can't
  answer these, so a correct, specific answer confirms the real scraped content is actually
  loaded and being retrieved, not the hardcoded chunks.

If a real-content question instead gets the fixed apology, check the startup log for "Indexed N
real chunk(s) from ... lincoln.edu.my page(s)" - if N is 0 or the fetch failed, the scrape
itself didn't succeed (network issue, a changed URL, etc.), not a retrieval/answering bug.

### The two sources compete, and that's handled deliberately

The scraped corpus is ~14,800 chunks; the floor directory is 10. On raw similarity a keyword-dense
scraped page title ("Civil Engineering Department") can outrank the actual answer ("The
Engineering department is on the third floor") for a plain *"where is the engineering
department"* — which happened for real on 2026-08-19. Two fields on `KnowledgeChunk` (see
`knowledge_base.py`) resolve it:

- `boost` — the curated directory entries get `DIRECTORY_BOOST`, so they win wayfinding questions
  against merely keyword-adjacent scraped text. `evaluate_retrieval.py` now includes adversarial
  chunks reproducing the real failure, so this can't regress silently.
- `is_direct_answer` — those same entries are already complete spoken sentences, so they're
  spoken verbatim with **no LLM call at all**. That removes 1–3 seconds from the robot's most
  common question type, and removes the LLM as a failure point for it (a Gemini 503 on this path
  is what made the robot read a raw page title aloud).

Scraped chunks deliberately get neither flag — raw web fragments still need a model to turn into
a sentence, and when that model call fails or genuinely can't answer, `DialogueManager` now falls
through to a **live search** before ever falling back to reading that raw fragment verbatim (see
Stage 05's README, "Why a new intent") — the fix for the flip side of the same 2026-08-19 bug: a
Gemini 503 used to mean the robot spoke a bare scraped title as if it were the answer.

### Startup is cached

The first run scrapes 87 pages and embeds ~14,800 chunks (about a minute). That result is cached
to `kb_cache.npz` (gitignored) and reused for 24 hours, so later launches start almost
immediately. The cache self-invalidates if it's older than `KB_CACHE_MAX_AGE_S` or was built by a
different embedding model. To force a fresh scrape now:

```powershell
python stages\stage08_orchestrator\main_live.py --refresh
```

## Run the live robot

```powershell
python stages\stage08_orchestrator\main_live.py
```

Walk into frame → the same presence/recognition/greeting flow as Stage 03's `greeting_live.py`
(checking... while recognition gets a fair chance, then a personal or generic greeting) → once
the state reaches `LISTENING`, the robot is genuinely listening on the microphone (same
VAD/preroll logic as Stage 04's own live demo) for up to `LISTEN_TIMEOUT_S` (8s). Ask a real
question — if the knowledge base has something relevant, you get a grounded answer; if not,
either the fixed apology (sounds university-related but no data) or open conversation, depending
on Stage 05's own `MODE` setting in `dialogue_manager_live.py`. After each answer it asks "do you
have any other questions?" — you can either say yes, or just ask your next question straight away
(both work; up to `MAX_QUESTIONS_PER_VISIT`). Say "no, that's all" / "that's it" / "I'm good" to
wrap up.

The camera keeps running the whole time — it no longer freezes while the robot listens, thinks,
or speaks (see "Threading" below).

### Who is it talking to?

With more than one person in frame, the robot picks whoever's **mouth is moving** as the person
speaking to it, and outlines them in green with "talking to me" in the preview. This is
`mouth_activity.py` in Stage 02 (mediapipe face mesh → lip landmarks → how much the mouth-open
ratio *varies* over the last ~half second; movement, not openness, is what separates talking from
a resting face).

It matters for more than politeness: the "can I remember you?" flow now asks **only** the person
who was speaking. Before, it looped over everyone in view, asking each in turn — so a name spoken
by one person could be saved against a different person's face.

`SPEAKING_ACTIVITY_THRESHOLD` in that file is a starting value, not a measured one — it depends on
your camera. Calibrate it by running the module directly, which prints live scores:

```powershell
python stages\stage02_face_recognition\mouth_activity.py
```

Sit still and note the score, then talk and note it again; put the threshold in the gap. If
mediapipe isn't installed the robot still runs — it just can't tell two speakers apart, and says
so at startup.

Only once you've said you're done does the robot ask (by voice, if you weren't recognized this
visit) whether to remember you — say "yes" and then your name when asked, or "no thanks" to skip
it. This ordering (all your questions first, *then* the remember-me ask) came from a real test
run where asking too early cut the visitor's conversation short.
This is the real replacement for `face_memory_live.py`'s keyboard-based consent prompt, and for
`greeting_live.py`'s 'r'-to-register testing shortcut — both of those stay as they are (useful on
their own for testing those stages in isolation), this is just where the real voice-driven
version actually lives.

## Threading

Perception (camera, person tracking, face recognition, mouth activity) runs continuously on the
main thread. Each visit — greeting, listening, thinking, speaking — runs on its own short-lived
thread (`_run_visit`).

This replaced the original single-threaded design on 2026-08-19. Previously the video feed froze
for the entire conversation, so the robot was blind between "hello" and "goodbye": it kept talking
to people who had already walked away, and the preview window stopped responding.

The split is deliberate about which side gets what — OpenCV's capture and window stay on the main
thread (the portable, supported way to use them), and the blocking audio/LLM work moves off it.
Shared state between the two is small and explicitly locked (`_state_lock`); the lock is only ever
held for short dictionary operations, never across a blocking call, so it can't stall the camera.

## What's still a deliberate simplification (see main_live.py's own docstring for the reasoning)

- **Consent yes/no and name parsing is heuristic**, not real NLU - `_parse_yes_no()` /
  `_parse_name_from_speech()` in `main_live.py`. Good enough for direct answers, defaults to "no"
  (never registers) on anything ambiguous, per `docs/PRIVACY.md`'s explicit-consent requirement.
- **Active speaker detection is visual only.** It measures lip motion and does not correlate it
  against the audio waveform, so someone silently mouthing words (or chewing) near the robot can
  be picked as the speaker. The stronger audio-visual version needs synchronized audio/video
  timestamps the current capture doesn't honestly provide - see `mouth_activity.py`'s docstring.
- **One visit at a time.** A second group arriving mid-conversation waits rather than being
  greeted in parallel - correct for a single mouth and one speaker, but the robot can look like
  it's ignoring new arrivals while busy.
- **Speed/accuracy is tuned toward speed** as of 2026-08-19, per an explicit call: Whisper
  `small.en` with greedy decoding rather than the more accurate, multiple-seconds-slower
  `medium.en`. On the Jetson target (which has a GPU) that tradeoff is worth revisiting - measure
  with Stage 04's `evaluate_wer.py` rather than guessing.
