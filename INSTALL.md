# Installing AI-Reception-Robot

From an empty machine to a talking robot. Every step is one-time unless it says otherwise.

Two things to know before you start:

- **One virtual environment at the project root** (`.venv/`), shared by all ten stages — not one
  per stage. Stage 08 imports from every other stage at once, so they have to live together.
- **Nothing large is in this repository.** Model weights (~170MB), the knowledge-base cache and
  your `.env` are all gitignored on purpose. Steps 4–6 below fetch them; each is a single
  command.

---

## 0. What you need first

| | Version / note |
|---|---|
| **Python** | 3.10 (developed and tested on 3.10.11). 3.11–3.12 should work; **3.13+ is not supported yet** — `mediapipe==0.10.18` publishes no wheel for it. |
| **Git** | any recent version |
| **MySQL Server** | 8.x — only needed for persistent memory (step 5). The robot runs without it, it just forgets everyone on exit. |
| **Ollama** | optional but strongly recommended (step 6) — the local LLM the robot falls back to when the cloud API is down. |
| **Hardware** | a webcam, a microphone and a speaker. You can install and run most stages' self-tests without them. |
| **Disk** | ~6GB: ~3.5GB of Python packages (torch dominates), ~170MB of models, ~2GB if you pull the Ollama chat model. |

Check Python before anything else — if this prints 3.13 or higher, install 3.10 alongside it and
use that interpreter's full path in step 2:

```powershell
python --version
```

---

## 1. Clone the repository

```powershell
git clone https://github.com/DevByError/robot_LUC.git
cd robot_LUC
```

Everything below is run **from the project root** (the folder containing `README.md`), unless a
step says otherwise. Two steps genuinely need a different working directory and both say so.

---

## 2. Create the virtual environment

There is a ready-made script per platform. It creates `.venv/`, upgrades pip, and installs the
Stage 00 base packages:

```powershell
# Windows (PowerShell)
.\stages\stage00_setup\setup.ps1
```

```bash
# Linux / macOS / Jetson
bash stages/stage00_setup/setup.sh
```

<details>
<summary>Or do it by hand (same thing, four commands)</summary>

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r stages\stage00_setup\requirements.txt
```

```bash
# Linux / macOS / Jetson
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r stages/stage00_setup/requirements.txt
```

</details>

If PowerShell refuses to run the script (`running scripts is disabled on this system`), allow
local scripts for your own user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Activate it — every new terminal, forever

The setup script does **not** leave the venv active in your shell. You have to activate it at the
start of every session, before running anything in this project:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS / Jetson
source .venv/bin/activate
```

You'll know it worked when your prompt starts with `(.venv)`. Leave it with `deactivate`.

> Forgetting this is the single most common cause of a `ModuleNotFoundError` in this project. If
> an import fails, check your prompt for `(.venv)` first.

---

## 3. Install the Python dependencies

With the venv **active**:

```powershell
pip install -r requirements.txt
```

That is every stage in one go (~3.5GB, several minutes — torch is most of it). The per-stage
files are still there if you would rather install stage by stage as you work through the project:

```powershell
pip install -r stages\stage01_presence_detection\requirements.txt
pip install -r stages\stage02_face_recognition\requirements.txt
# ... and so on through stage07
```

Each stage's `requirements.txt` explains *why* each pin is what it is (licensing, wheel
availability, past version conflicts). The root `requirements.txt` is the same set flattened.

**On a small or older computer, use the lite set instead** (no PyTorch, about 2GB less):

```powershell
pip install -r requirements-lite.txt
```

It has everything the robot uses by default. It leaves out only the fallbacks: YOLO person
detection (`PRESENCE_BACKEND=yolo`), sentence-transformers (the ONNX copy of the same model is
used instead), speaker diarization and the Stage 03 placeholder voice.

**NVIDIA GPU (for example the GTX 1650):** Whisper uses it automatically once NVIDIA's CUDA
libraries are installed (about 1.3GB, no CUDA Toolkit needed). Without them it runs on the CPU,
more slowly, and the startup log says why.

```powershell
pip install -r requirements-gpu.txt
```

Startup then prints `[stt] Whisper is running on the GPU.` The versions in that file are pinned
on purpose: the cuDNN version has to match the one Whisper's engine ships with.

### The opencv wrinkle

`opencv-contrib-python` is a strict superset of `opencv-python`, and the two overwrite each
other's files if both are installed. `requirements.txt` lists only the contrib build — but
ultralytics declares plain `opencv-python` as its own dependency, so pip installs it anyway.
After the install above, remove it:

```powershell
pip uninstall -y opencv-python
```

`pip check` will then report that ultralytics is missing a requirement. That warning is expected
and safe to ignore: `opencv-contrib-python` provides the exact same `cv2` module ultralytics
imports, pip just cannot tell that from package names alone.

### Verify the install

```powershell
python stages\stage00_setup\test_hardware.py
```

It prints pass/fail for the camera, microphone and speaker. Developing on a machine with no
webcam plugged in? Failures there are fine at this point — what matters is that the imports do
not crash.

If the speaker test is silent, or you have several audio devices:

```powershell
python stages\stage00_setup\list_audio_devices.py
python stages\stage00_setup\test_speaker_device.py
```

---

## 4. Download the model weights

**The new models, in one command** (Kokoro voice, streaming speech recognizer, end-of-turn
model, ONNX sentence embeddings, wake word; about 370MB, checksums verified):

```powershell
python stages\stage00_setup\download_models.py
```

`--list` shows what is present. Anything missing falls back to the older component, and the
startup log says which. The rest of this step covers the models that were already there.

None of these are in the repository — they are large binaries with a one-command re-download, so
they are gitignored. Around 170MB here, plus ~500MB more the first time Whisper and the
sentence-transformer run.

**Face detection + recognition** (YuNet ~338KB, SFace ~37MB). This one **must** run from the
stage folder — it saves next to itself:

```powershell
cd stages\stage02_face_recognition
python download_models.py
cd ..\..
```

**The Piper voice** (~63MB, US English, female). Also from its own stage folder:

```powershell
cd stages\stage06_text_to_speech
python -m piper.download_voices en_US-amy-medium
cd ..\..
```

**Person detection** (YOLOv8n, ~6.5MB) — nothing to do. ultralytics downloads it automatically
the first time you run a stage that needs it, into whatever directory you launched from.

**Speech-to-text** (faster-whisper `small.en`, ~500MB) — also automatic, on first run, cached in
your user profile afterwards. Expect the first run of Stage 04 or Stage 08 to pause while it
downloads.

**Sentence embeddings** (~80MB from Hugging Face) — same story, downloaded and cached on the
first Stage 05 or Stage 08 run.

---

## 5. Set up the database (persistent memory)

This is what lets the robot still recognize a student after being closed and reopened. **Skip it
and the robot still runs** — it says so at startup and forgets every face and conversation on
exit.

You do *not* create the database or the tables by hand; `robot_db.py` creates both.

1. **Copy the credentials template and fill in your MySQL password:**

   ```powershell
   copy .env.example .env
   notepad .env
   ```

   ```bash
   cp .env.example .env
   nano .env
   ```

   Set at minimum `LUC_DB_USER` and `LUC_DB_PASSWORD`. `.env` is gitignored — it holds a real
   password and must never be committed. `.env.example` is the committed template; keep it in
   sync when you add a setting.

   On the deployed kiosk, set the same `LUC_DB_*` names as **real environment variables** instead
   of shipping the file. A real environment variable always wins over `.env`.

2. **Create the schema and check the connection:**

   ```powershell
   python stages\stage02_face_recognition\robot_db.py
   ```

   It prints the row count of each table. Add `--self-test` for a full round trip — a temporary
   person is written, read back, given a conversation, then deleted, confirming the privacy
   cascade removes their transcript along with them:

   ```powershell
   python stages\stage02_face_recognition\robot_db.py --self-test
   ```

3. **Check which settings the project can actually see** (prints names only, never values):

   ```powershell
   python stages\stage00_setup\project_env.py
   ```

---

## 6. Install Ollama — the offline fallback brain

Also optional, also strongly recommended. Without it, a cloud API outage means the robot answers
straight from the knowledge base with no LLM at all. Ollama is free, needs no API key, has no
quota, and works with the network unplugged.

```powershell
winget install Ollama.Ollama     # or download from https://ollama.com/download
ollama pull llama3.2:3b          # the chat model (~2GB)
```

`nomic-embed-text` (~274MB) is only needed if you deliberately switch `LUC_EMBEDDING_BACKEND` to
`ollama`. Do not — the default in-process sentence-transformers backend measured ~35× faster at
building the knowledge base, for identical retrieval quality.

Ollama runs as a background service once installed. Check it end to end:

```powershell
python stages\stage05_nlu_dialogue\ollama_client.py
```

That prints whether the server is up, which models are pulled, warm-up time, and real answer and
embedding latency measured on *your* machine.

---

## 7. (Optional) Add a cloud LLM key

With a key set, the robot tries the cloud first and falls back to Ollama when it fails. With no
key, Ollama is the only LLM. Uncomment one of these in `.env`:

```ini
GEMINI_API_KEY=...   # free tier, no card needed - aistudio.google.com
GROQ_API_KEY=...     # free daily quota, no card needed
```

`ANTHROPIC_API_KEY` also works, but Claude is no longer the default pick — see
`stages/stage05_nlu_dialogue/dialogue_manager_live.py`.

The Groq key can also do the robot's hearing. Add `STT_CLOUD=groq` and each finished question is
transcribed by Groq's Whisper large-v3, a much bigger model than the one on the laptop, while the
local Whisper keeps doing the live partials and stays ready as the backup:

```ini
GROQ_API_KEY=...
STT_CLOUD=groq
```

At startup the log says `[stt] Groq whisper-large-v3 is ready` (or why not), and each question
shows `[stt] Groq (whisper-large-v3, 640ms)`. A recording of every question is then sent to Groq —
see `docs/PRIVACY.md`.

---

## 8. Run it

Activate the venv first (step 2), then:

```powershell
python stages\stage08_orchestrator\main_live.py
```

The first launch scrapes lincoln.edu.my and embeds ~14,800 chunks — that takes a few minutes. The
result is cached to disk with a 24-hour expiry, so every launch after that starts quickly. Force
a fresh scrape any time with:

```powershell
python stages\stage08_orchestrator\main_live.py --refresh
```

Walk into frame and look at it for a second; it greets you. Ask a question out loud and it
answers by voice. You can talk over it to interrupt. Say "no, that's all" to end the visit.

The startup log prints a short report of what it found: which voice, which speech recognizer,
whether echo cancellation is on, and the measured barge-in threshold. Every setting is described
in `.env.example` (all optional). For a robot with no monitor, set `HEADLESS=true`. To start it
automatically at boot, see `stages/stage09_deployment/README.md`.

### Run the tests

```powershell
python -m pytest tests -m "not e2e"     # ~1 minute, no hardware needed
python -m pytest tests                   # + the two simulated end-to-end visits (~3 minutes)
```

The end-to-end tests need the models from step 4. They play a whole visit through a simulated
camera, microphone and speaker: greeting, a question, an interruption, a bystander, the wake word
and the goodbye. See `docs/RESULTS.md` for the measured numbers.

### Trying a single stage on its own

Every stage's main module has a self-test with mock data at the bottom, so you can confirm the
logic works before the hardware is connected:

```powershell
python stages\stage01_presence_detection\presence_detector.py
```

The `*_live.py` file in each stage is the real hardware version of that stage:

```powershell
python stages\stage01_presence_detection\presence_detector_live.py
python stages\stage02_face_recognition\face_memory_live.py
python stages\stage03_proactive_greeting\greeting_live.py
python stages\stage04_speech_to_text\stt_listener_live.py
python stages\stage05_nlu_dialogue\dialogue_manager_live.py
python stages\stage06_text_to_speech\tts_speaker_live.py
```

---

## 9. Freeze the exact environment (before deploying)

Once everything works on the dev machine, snapshot every version actually installed — including
transitive dependencies — so the kiosk device can be reproduced exactly:

```powershell
pip freeze > stages\stage00_setup\requirements.lock.txt
```

That lock file is gitignored: it is machine-specific (it records `torch==2.12.0+cpu`,
Windows-only packages, and so on). Copy it to the target device by hand and install with
`pip install -r requirements.lock.txt` there. `requirements.txt` stays the readable, editable,
cross-platform list.

---

## Troubleshooting

**`ModuleNotFoundError` on anything** — the venv is not active. Look for `(.venv)` in your prompt.

**`running scripts is disabled on this system`** — see the `Set-ExecutionPolicy` line in step 2.

**`cv2` errors, or "recursion is detected during loading of cv2"** — you have both opencv builds
installed. Run `pip uninstall -y opencv-python opencv-contrib-python`, then
`pip install opencv-contrib-python==4.9.0.80`.

**pip wants to build a wheel from source, or asks for a C compiler** — you are most likely on
Python 3.13+, where several of these pins have no prebuilt wheel. Use Python 3.10.

**MySQL "access denied" with a password you are sure is right** — check for stray quotes or
trailing spaces in `.env`. The loader strips one layer of matching quotes, but nothing else.

**The robot answers, but generically, and never quotes real university content** — the scrape
failed or the cache is stale. Re-run with `--refresh` and watch the startup log for
`Indexed N chunks`.

**The robot answers the wrong person in a group** — `mediapipe` is not installed. It degrades
rather than crashing, so this shows up as behaviour, not as an error. Re-run step 3.

**Piper says the voice file is missing** — the download in step 4 landed in the wrong folder. It
must be inside `stages/stage06_text_to_speech/`.

**The robot interrupts itself, or you have to shout to interrupt it** — look at the startup line
`[barge-in] this room returns the robot's own voice at RMS ...`. With echo cancellation on, it
should be around 0.002-0.005. A much higher value means the microphone is very close to the
speaker or the speaker distorts. Move them apart, lower the volume, or use a speakerphone with
its own echo cancellation. `INTERRUPTION_ENABLED=false` switches barge-in off.

**`[aec] echo cancellation unavailable`** — `pip install livekit==1.0.0 "protobuf<5"`. Without it
the robot still works, but barge-in falls back to "talk louder than the speaker".

**The wrong microphone or speaker is used** — run `python stages\stage00_setup\list_audio_devices.py`
and set `MIC_DEVICE` / `SPEAKER_DEVICE` in `.env` to part of the device name.

**`[audio] input overflow`** — the computer is too busy to read the microphone in time. Set
`AUDIO_CAPTURE_PROCESS=true` (reads it in a separate process) and/or lower `VISION_FPS`.

**The voice is Piper, not Kokoro** — the log says why: either the model is not downloaded (step 4)
or this computer synthesizes Kokoro less than 2x faster than real time. `TTS_BACKEND=kokoro` forces it.

---

## Licensing note before you ship this

Two dependencies carry copyleft terms that matter if this robot is ever distributed as
closed-source software to the university:

- **ultralytics** (Stage 01) — AGPL-3.0
- **Piper** (Stage 06) — GPL-3.0-or-later, and each voice model has its own separate licence

Fine for prototyping and internal use. Get legal review before a production deployment. Nearly
every other dependency here was chosen partly to avoid this problem — see the per-stage
`requirements.txt` comments for the reasoning behind each one.
