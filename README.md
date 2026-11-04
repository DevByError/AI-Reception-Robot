# AI-Reception-Robot

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A stationary reception robot for a university front desk. It notices people who stop in front of
it, greets them by voice, listens to their questions and answers by voice. It can remember
returning visitors (only with their consent) and answers from content scraped from the
university's public website.

## Overview

The robot is a fixed point at a reception desk: a camera, a microphone, a speaker and a computer,
with an optional screen that shows an animated face. Every interaction is spoken. There is no
information screen and no map, so location answers are kept to the floor ("The Engineering
department is on the third floor").

The project is built as ten stages (`stages/stage00_setup` to `stages/stage09_deployment`). Each
stage has a self-test with mock data, so most of it can be developed and tested without hardware.
`stages/stage08_orchestrator/main_live.py` wires the stages together into the running robot.

English is the only supported language for now. The language is passed through every layer so
others can be added later.

## Features

- Greets people who stop and look at the robot, not people walking past. Groups get one greeting.
- Face recognition with YuNet and SFace. A new visitor is asked before their name and face
  embedding are stored.
- Persistent memory in MySQL (or a local SQLite file if MySQL is not available): enrolled people,
  conversation transcripts and short notes from earlier visits.
- Speech-to-text with faster-whisper on CPU or NVIDIA GPU, an optional streaming Zipformer
  recognizer, and optional Groq Whisper large-v3 for finished questions.
- End-of-turn detection (smart-turn model) so the robot answers soon after a finished question
  and waits through pauses.
- Barge-in: WebRTC echo cancellation removes the robot's own voice from the microphone, so a
  visitor can interrupt it.
- Answers from a fallback chain: cloud LLM (Groq or Gemini), then a local LLM through Ollama,
  then the retrieved knowledge-base text spoken directly.
- Retrieval over scraped website content using MiniLM sentence embeddings (ONNX, no PyTorch
  needed), cached on disk and refreshed in the background.
- Text-to-speech with Kokoro, falling back to Piper, spoken sentence by sentence as the answer is
  generated.
- Privacy controls: consent before enrolment, automatic deletion after a retention period, and
  deletion on request. See [docs/PRIVACY.md](docs/PRIVACY.md).

## Hardware

| Component | Model | Purpose |
|---|---|---|
| Main computer | Windows laptop for development; NVIDIA Jetson planned for deployment (model TBD) | Runs every stage |
| GPU (optional) | NVIDIA GTX 1650 4 GB (development laptop) | Runs Whisper with CUDA; everything else runs on the CPU |
| Camera | USB webcam (model TBD), device index 0, captured at 1280x720 | Presence detection, face recognition, lip movement |
| Microphone | Realtek microphone array (development laptop); USB microphone or array for deployment (model TBD) | Speech input; select with `MIC_DEVICE` |
| Speaker | TBD | Speech output; select with `SPEAKER_DEVICE` |
| Display | Small screen (model TBD), optional | Animated face (`FACE_DISPLAY`, `FACE_FULLSCREEN`); use `HEADLESS=true` without one |
| Animatronic head | Planned | Alternative to the on-screen face; not supported in code yet |
| Enclosure, power, other sensors | TBD | Not part of the software yet |

Minimum and recommended specs:

| | CPU-only mode | GPU mode |
|---|---|---|
| Install | `requirements-lite.txt` | `requirements.txt` or `-lite`, plus `requirements-gpu.txt` |
| CPU | 4-core x86-64 (the measurements in [docs/RESULTS.md](docs/RESULTS.md) use one) | 4 cores or more |
| GPU | None; Whisper runs on the CPU, or use `STT_BACKEND=sherpa` / `STT_CLOUD=groq` | NVIDIA GPU with CUDA 12 support and 4 GB VRAM or more |
| RAM | 8 GB suggested (not measured) | 16 GB suggested (not measured) |
| Disk | About 3 GB | About 6 GB |

## Software requirements

- Python 3.10 (3.11 and 3.12 should work; 3.13 is not supported because `mediapipe==0.10.18` has
  no wheel for it)
- Windows 10/11 or Linux (including Jetson). macOS is untested.
- Optional: MySQL Server 8.x for persistent memory
- Optional: [Ollama](https://ollama.com) with `llama3.2:3b` as the offline language model
- Optional: a Groq or Gemini API key for the cloud language model

## Architecture

```
camera -> face detection / recognition -> speech-to-text -> language model -> text-to-speech -> speaker
```

In more detail:

1. The camera loop finds faces (YuNet), recognizes known people (SFace) and checks whether
   someone has stopped and is facing the robot.
2. The robot greets the visitor and listens. Echo cancellation and a voice-activity gate clean
   the microphone signal; an end-of-turn model decides when the question is finished.
3. Whisper (local or Groq) turns the question into text.
4. The dialogue manager answers identity questions directly, otherwise searches the knowledge
   base and asks the language model chain for a grounded answer.
5. The answer is spoken sentence by sentence with Kokoro or Piper.

A background task re-scrapes the university website, re-embeds changed pages and swaps in the new
knowledge base without interrupting a conversation. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for the full flow.

## Installation

The short version is below. [docs/SETUP.md](docs/SETUP.md) has every step in detail, including
the database and Ollama.

```bash
git clone https://github.com/DevByError/AI-Reception-Robot.git
cd AI-Reception-Robot

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install the dependencies (pick one)
pip install -r requirements-lite.txt   # recommended: no PyTorch
pip install -r requirements.txt        # everything, including PyTorch-based fallbacks

# Optional: Whisper on an NVIDIA GPU
pip install -r requirements-gpu.txt

# Download the model weights (not stored in git)
python stages/stage00_setup/download_models.py
cd stages/stage02_face_recognition && python download_models.py && cd ../..
cd stages/stage06_text_to_speech && python -m piper.download_voices en_US-amy-medium && cd ../..

# Create your settings file
cp .env.example .env               # Windows: copy .env.example .env
```

The requirement files:

| File | Purpose |
|---|---|
| `requirements-lite.txt` | Everything the robot uses by default, without PyTorch. Use this on most machines. |
| `requirements.txt` | The lite set plus PyTorch-based fallbacks: YOLO person detection, sentence-transformers, speaker diarization and the Stage 03 placeholder voice. |
| `requirements-gpu.txt` | NVIDIA cuBLAS and cuDNN for Whisper on the GPU. Install on top of one of the above. |
| `requirements-dev.txt` | pytest, black and ruff. |
| `stages/*/requirements.txt` | Per-stage subsets, with notes on why each version is pinned. |

Model weights are downloaded, never committed. `download_models.py` fetches Kokoro, the streaming
Zipformer recognizer, the smart-turn model, the MiniLM ONNX embeddings and the openWakeWord models
into `models/` (about 370 MB, with checksums). The face models go into
`stages/stage02_face_recognition/` and the Piper fallback voice into
`stages/stage06_text_to_speech/`. Whisper and YOLOv8n download themselves on first use.

## Configuration

All settings are read from environment variables or from a `.env` file in the project root. A
real environment variable always takes precedence over `.env`. Every setting is listed with its
default in [.env.example](.env.example).

The ones most people change:

| Variable | Purpose |
|---|---|
| `LUC_DB_HOST`, `LUC_DB_USER`, `LUC_DB_PASSWORD`, `LUC_DB_NAME` | MySQL connection for persistent memory |
| `LUC_DB_BACKEND` | `auto` (MySQL, else SQLite), `mysql` or `sqlite` |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | Optional cloud language models |
| `ANSWER_BACKENDS` | Order of the answer chain, default `groq,gemini,ollama` |
| `STT_DEVICE`, `WHISPER_MODEL` | Whisper on `auto`, `cuda` or `cpu`, and the model size |
| `STT_CLOUD` | `groq` to transcribe finished questions with Groq Whisper |
| `MIC_DEVICE`, `SPEAKER_DEVICE` | Part of the audio device name to use |
| `HEADLESS`, `FACE_DISPLAY`, `FACE_FULLSCREEN` | Screen behaviour |

Run `python stages/stage00_setup/realtime_config.py` to print every setting and where its value
came from. `python stages/stage00_setup/project_env.py` shows which credentials are set without
printing them.

The robot's name and creator details are stored in the `robot_identity` table and can be changed
with `python stages/stage05_nlu_dialogue/identity.py --set robot_name=Nova`.

## Usage

Start the robot:

```bash
python stages/stage08_orchestrator/main_live.py
```

The first launch scrapes the university website and builds the knowledge base, which takes a few
minutes. The result is cached for 24 hours. Use `--refresh` to force a new scrape.

Stand in front of the camera and look at it; the robot greets you. Ask a question out loud. You
can talk over the robot to interrupt it. Say "no, that's all" to end the visit.

Check the hardware:

```bash
python stages/stage00_setup/test_hardware.py
python stages/stage00_setup/list_audio_devices.py
```

Run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -m "not e2e"     # about a minute, no hardware needed
python -m pytest                  # includes two simulated end-to-end visits
```

Tests that need a model file or `tests/data/face.jpg` skip themselves when it is missing.

To start the robot automatically, see [stages/stage09_deployment/README.md](stages/stage09_deployment/README.md)
(`start_robot.bat` on Windows, a systemd user service on Linux).

## Project structure

```
AI-Reception-Robot/
  docs/                         Architecture, setup guide, privacy policy, measurements
  stages/
    stage00_setup/              Environment, settings, model downloader, hardware checks
    stage01_presence_detection/ Person and face presence (YuNet, YOLOv8n fallback)
    stage02_face_recognition/   SFace recognition, consent flow, MySQL/SQLite memory
    stage03_proactive_greeting/ Greeting state machine
    stage04_speech_to_text/     Microphone, echo cancellation, Whisper, end-of-turn detection
    stage05_nlu_dialogue/       Knowledge base, answer chain, robot identity, visit memory
    stage06_text_to_speech/     Kokoro and Piper voices, audio output
    stage07_data_pipeline/      Website scraper and indexer
    stage08_orchestrator/       main_live.py, the running robot
    stage09_deployment/         Startup scripts and latency logging
  tests/                        pytest suite and simulated end-to-end visits
  .env.example                  Every setting, with defaults
  requirements*.txt             Dependency sets (see Installation)
```

## Troubleshooting

**`ModuleNotFoundError`**: the virtual environment is not active. Your prompt should start with
`(.venv)`.

**PowerShell says running scripts is disabled**: run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

**OpenCV conflict (`cv2` errors, "recursion is detected during loading of cv2")**: ultralytics
installs `opencv-python` next to `opencv-contrib-python`. Run `pip uninstall -y opencv-python`.
If that does not fix it, uninstall both and reinstall `opencv-contrib-python==4.9.0.80`. The
`pip check` warning about ultralytics afterwards is expected.

**pip tries to build packages from source**: you are probably on Python 3.13. Use Python 3.10.

**Whisper runs on the CPU although there is a GPU**: install `requirements-gpu.txt`. The startup
log prints `[stt] Whisper is running on the GPU.` when it works.

**MySQL "access denied"**: check `.env` for stray quotes or trailing spaces in the password. The
robot still runs without MySQL and uses SQLite instead.

**Generic answers with no university content**: the scrape failed or the cache is stale. Run with
`--refresh` and look for `Indexed N chunks` in the log.

**The robot interrupts itself, or you have to shout to interrupt it**: check the startup line
`[barge-in] this room returns the robot's own voice at RMS ...`. With echo cancellation it should
be around 0.002 to 0.005. Move the microphone away from the speaker or lower the volume.
`INTERRUPTION_ENABLED=false` turns barge-in off.

**Wrong microphone or speaker**: run `list_audio_devices.py` and set `MIC_DEVICE` or
`SPEAKER_DEVICE` to part of the device name.

**`[audio] input overflow`**: set `AUDIO_CAPTURE_PROCESS=true` or lower `VISION_FPS`.

**The voice is Piper instead of Kokoro**: the Kokoro model is missing, or the computer synthesizes
it less than twice as fast as real time. The startup log says which. `TTS_BACKEND=kokoro` forces
it.

## Roadmap

- Choose and document the deployment hardware (computer, camera, microphone, speaker, enclosure).
- Make the camera index configurable instead of fixed at 0.
- Animatronic head as an alternative to the on-screen face.
- Arabic and Chinese speech and answers.
- Rotating log files and alerts when a stage is slow.
- A vector index if the knowledge base grows beyond what brute-force search handles.

## License

Released under the [MIT License](LICENSE).

Some dependencies use copyleft licences: ultralytics (AGPL-3.0, only used for the YOLO fallback
in `requirements.txt`) and Piper (GPL-3.0-or-later). Check their terms before distributing a
build that includes them.

## Acknowledgements

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for the YuNet and SFace models
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and OpenAI Whisper
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for the streaming Zipformer recognizer
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- [Piper](https://github.com/OHF-Voice/piper1-gpl)
- [Smart Turn](https://github.com/pipecat-ai/smart-turn) for end-of-turn detection
- [openWakeWord](https://github.com/dscripka/openWakeWord)
- [LiveKit](https://github.com/livekit/python-sdks) for the WebRTC echo canceller
- [sentence-transformers](https://www.sbert.net) and all-MiniLM-L6-v2
- [MediaPipe](https://developers.google.com/mediapipe) for lip movement detection
- [Ollama](https://ollama.com), Groq and Google Gemini for the language models
