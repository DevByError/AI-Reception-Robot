# Stage 00 — Setup + Python Virtual Environment (venv)

## Why venv?

So this project's libraries (pinned versions) never conflict with any other Python project on
the same machine, and don't depend on device-wide installed libraries. Anyone running this
project — a dev machine or the real kiosk device — gets **the exact same versions**.

**One venv at the project root** (`E:\AI-Reception-Robot\.venv`), and you add each stage's libraries to it
as you reach that stage — not a separate venv per stage, since later modules (Stage 08) need to
import from every stage together.

## Creating the venv (one time)

### Windows (PowerShell) — the current dev machine

```powershell
cd E:\AI-Reception-Robot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r stages\stage00_setup\requirements.txt
```

Or just run the ready-made script: `.\stages\stage00_setup\setup.ps1`

### Linux / Jetson — the actual deployment device at the kiosk

```bash
cd ~/AI-Reception-Robot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r stages/stage00_setup/requirements.txt
```

Or just run the ready-made script: `bash stages/stage00_setup/setup.sh`

## Every time you open a new terminal

You need to activate the venv before running anything in the project:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
```
```bash
# Linux/macOS
source .venv/bin/activate
```

You'll know it's active when you see `(.venv)` at the start of the prompt. To exit: `deactivate`.

## Adding a new stage's libraries

Every time you move to a stage with extra libraries (e.g. Stage 02 needs face recognition,
Stage 04 needs Whisper), we do the same thing: add a `requirements.txt` in that stage's folder,
and install it into the same active venv:

```bash
pip install -r stages/stage02_face_recognition/requirements.txt
```

That way each stage only adds its own dependency once you actually reach it — you don't install
everything on day one.

## Freezing the actual installed versions (optional but recommended)

Once you've installed everything and confirmed it works, take an exact snapshot of everything
actually installed (including sub-dependencies) as a final reference:

```bash
pip freeze > stages/stage00_setup/requirements.lock.txt
```

Use this file (`requirements.lock.txt`) to reproduce the exact same environment on the real kiosk
device later, while `requirements.txt` stays readable and manually editable.

## Verifying everything works

After activating and installing:

```bash
python stages/stage00_setup/test_hardware.py
```

Checks the camera/mic/speaker and prints pass/fail for each one. If you don't have the hardware
connected yet (developing on a laptop, say), expect fail results for the camera/mic — that's
fine at this stage; what matters is that the imports don't crash.
