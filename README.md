  # University AI Reception Robot

A stationary (non-mobile) robot that sees students, proactively greets them by voice, listens to
their questions, understands them, and replies fully by voice (no info screen, no text — 100%
voice interaction). It recognizes returning students (with their consent) and tells them which
floor a department or service is on (floor-only — no step-by-step directions; a floor number is
enough). University data (departments, staff, events...) is pulled automatically and periodically
from the university's public website, with no manual intervention.

## The concept (confirmed with the user)

- **Hardware**: a fixed point (camera + microphone + speaker + Mini PC/Jetson), no locomotion.
  The interface is either an animated robot face on a small screen or a physical animatronic
  head — that decision isn't final yet, so the "expression/face" layer is designed as a
  swappable module (one interface, two possible implementations).
- **No info screen, no visual maps**: every answer and every direction must be fully spoken and
  clear verbally.
- **Language**: we start with **English only** (STT/NLU/TTS). Arabic and Chinese come later —
  the language layer is designed to be extensible (no rebuild needed to add a language).
- **Location answers**: floor-only (e.g. "the Engineering department is on the third floor") —
  no step-by-step turn-by-turn directions and no separate wayfinding stage; a floor number is
  enough to save the student time.
- **Multiple people**: if more than one student approaches at once, greet everyone as one group
  ("Hi everyone...").
- **Remembering names**: the first time it sees a new face, it asks the student directly and
  requests explicit consent before storing their name + face embedding.
- **Remembering across restarts**: names, face embeddings and full conversation transcripts
  (with timestamps) are stored in MySQL, so closing and reopening the robot doesn't make it
  forget anyone. Setup is one file to copy and one command — see "Set up the database" in
  `stages/stage02_face_recognition/README.md`, and `docs/PRIVACY.md` for what's stored and how
  a deletion request is handled.
- **Unanswerable questions**: if outside the knowledge base, it apologizes, suggests the official
  website as an alternative source, and mentions the nearest human front-desk office.
- **Never dependent on one API**: answers come from a fallback chain — a cloud LLM (Gemini) when
  it's working, a free local model running on the robot itself (Ollama) when it isn't, and the
  knowledge base's own text if neither is available. A cloud outage costs a retry instead of a
  wrong answer. See `stages/stage05_nlu_dialogue/README.md` ("Local LLM via Ollama").
- **Build approach**: built in sequential stages, each stage built and tested independently
  before moving to the next.
- **Data source**: periodic web scraping of the public university website (no API exists yet) +
  a local database for faces/internal maps.

## Where to start

Read the files in this order:

1. `docs/ARCHITECTURE.md` — the full picture of the system workflow and the data workflow.
2. `docs/STAGES.md` — the detailed build plan across 10 stages (0 → 9), each with a goal and a
   test.
3. `docs/PRIVACY.md` — the policy for handling face data and privacy (legally and ethically
   important).
4. The `stages/` folder — a runnable code skeleton for each stage, ordered `stage00_...` through
   `stage09_...`.

## Installing and running

Full step-by-step instructions — from creating the virtual environment through downloading the
models, setting up the database, and launching the robot — are in **[INSTALL.md](INSTALL.md)**.

The short version, on a fresh clone:

```powershell
# Windows
.\stages\stage00_setup\setup.ps1        # creates .venv and installs the base packages
.\.venv\Scripts\Activate.ps1            # activate - needed in EVERY new terminal
pip install -r requirements.txt          # every stage's dependencies
python stages\stage08_orchestrator\main_live.py
```

```bash
# Linux/macOS/Jetson
bash stages/stage00_setup/setup.sh
source .venv/bin/activate
pip install -r requirements.txt
python stages/stage08_orchestrator/main_live.py
```

The model weights, the `.env` credentials file and the knowledge-base cache are all gitignored
and are **not** in a fresh clone — INSTALL.md steps 4 to 6 fetch each of them with a single
command.

## Trying out any stage

Each stage is at least one Python file with mock data built into its self-test at the bottom, so
you can run it and confirm the logic works even before the real hardware is connected (activate
the venv at the start of every new terminal session):

```bash
python stages/stage01_presence_detection/presence_detector.py
```

The `*_live.py` file beside it is that same stage against real hardware.

## Short runtime flow

```
IDLE → person/group approaches (camera) → proactive group greeting (TTS)
     → face recognition (known: personalize greeting / new: ask for name with consent)
     → listening (STT) → intent understanding + knowledge search (RAG)
        → [found in knowledge base]   → grounded answer (department/service location is
                                         floor-only - "the X department is on the Yth floor" -
                                         no step-by-step directions, no screen/map)
        → [general conversation]      → open answer, not grounded (any topic)
        → [outside knowledge base]    → apology + alternative source + human referral
     → speak the answer (TTS) → back to listening or IDLE after a timeout
```

Separately, there is an independent background flow: a periodic scheduler re-crawls the
university's public website, detects changes, and automatically updates the robot's knowledge
base — no manual work required.
