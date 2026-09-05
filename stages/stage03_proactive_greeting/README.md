# Stage 03 — Proactive Greeting

Two files here:

- `greeting_state_machine.py` — the mock-based version, keep this working as a regression test
  of the IDLE → DETECTED → GREETING → LISTENING logic (solo/known/group cases). No hardware
  needed.
- `greeting_live.py` — the real version: wires Stage 01's live YOLOv8n person tracking and
  Stage 02's live YuNet+SFace face recognition into the same state machine, and speaks the
  greeting out loud instead of printing it.

## Install (one time, venv already active)

Stage 01 and Stage 02's requirements must already be installed (this stage reuses their models
directly), plus this stage's own:

```powershell
pip install -r stages\stage01_presence_detection\requirements.txt
pip install -r stages\stage02_face_recognition\requirements.txt
pip install -r stages\stage03_proactive_greeting\requirements.txt
```

Stage 02's YuNet/SFace ONNX models must already be downloaded — see
`stages/stage02_face_recognition/README.md` if you haven't run `download_models.py` yet.

## Run the live demo

```powershell
python stages\stage03_proactive_greeting\greeting_live.py
```

Walk into frame: once your presence is confirmed (a few consecutive frames, same debouncing as
Stage 01), the robot does **not** greet immediately — the overlay shows `checking...` on your
box while it spends up to `RECOGNITION_WAIT_S` (2.5s) trying to actually recognize your face
first (several recognition ticks, a bit of natural head movement helps, same idea as Stage 02's
enrollment/observation windows). Only after that — either a confident match, or the wait running
out — does it actually greet you and the console shows the state changing
`DETECTED → GREETING → LISTENING`. Since you're not in the face memory store yet on a first run,
that first greeting is generic ("Hi there!..."). After `LISTENING_TIMEOUT_S` (6s — a temporary
stand-in for the real "conversation is over" signal that will come from Stage 04/05), it
returns to `IDLE`.

Two people in frame at once get one shared group greeting. If either of them is already known,
their name is folded into that same group greeting rather than greeting each person separately.

## Registering a face for personalization testing

Stage 02's own full consent/enrollment flow is already tested independently in
`face_memory_live.py` — this stage doesn't re-implement it. Instead, for quick testing here:
press **'r'** while your face is the only unresolved one in frame, type a name, and it's stored
immediately (single embedding sample, no multi-angle capture). Step out and back in — the next
greeting is personalized ("Welcome back, `<name>`!..."). This is a testing shortcut, not the
real consent UX (see `docs/PRIVACY.md`); the real flow stays keyboard-based here too until
Stage 04 (STT) and Stage 06 (TTS) exist to make it a real conversation, at which point Stage 08
(Orchestrator) is what actually owns triggering enrollment mid-interaction.

## Known limitations (by design, for now)

- `RECOGNITION_WAIT_S` (2.5s) is a placeholder, same as Stage 02's similarly-named constants —
  tune it against real testing. Too short and known people still get misgreeted as strangers
  (the original bug this fixed); too long and a genuine stranger stands there in silence longer
  than necessary before the generic greeting fires.
- The greeting's voice (`LiveTTSSpeaker`, pyttsx3) is a temporary stand-in, not the real Stage 06
  engine — it's not streaming and blocks the frame loop while speaking.
- Returning to `IDLE` after `LISTENING` is a fixed timeout, not a real "the person is done
  talking" signal — that only becomes possible once Stage 04/05 are wired in (Stage 08).
- If a second person is confirmed as "entered" while the robot is already mid-interaction with
  someone else, they're silently not greeted until the state returns to `IDLE` — this is the
  same "handled later by Stage 08" limitation called out in `greeting_state_machine.py`, not
  fixed here.
- Face-to-person linking assumes a detected face's center falls inside the YOLO person box that
  owns it — reliable for solo/small-group testing, but two people standing very close together
  could have a face misattributed to the wrong tracked person.
