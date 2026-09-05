# Stage 02 — Face Recognition & Memory

- `face_memory.py` — the store: matching, consent, retention. Runs standalone against mock
  embeddings with no camera and no database, so it stays a regression test of that logic.
- `sface_embedder.py` — real face detection (YuNet) + embeddings (SFace), both Apache-2.0,
  loaded through OpenCV. No dlib/CMake compiler needed, no non-commercial model license risk
  (unlike InsightFace's default models).
- `robot_db.py` — **persistent memory**: MySQL storage for enrolled people, their face
  embeddings, and the conversation transcripts recorded by Stage 08. This is what makes the
  robot still know you after it's been closed and reopened.
- `face_memory_live.py` — the real version: your webcam + real face recognition + a
  keyboard-based consent flow (temporary stand-in for Stage 04/06's real voice interaction).
- `mouth_activity.py` — active speaker detection (which visible person is the one talking).

## Install (one time, venv already active)

`opencv-contrib-python` is a strict superset of `opencv-python` — installing both causes file
conflicts, so remove the old one first:

```powershell
pip uninstall opencv-python -y
pip install -r stages\stage02_face_recognition\requirements.txt
```

## Download the models (one time)

```powershell
cd stages\stage02_face_recognition
python download_models.py
```

Downloads two ONNX files (~338KB detector + ~37MB recognizer) from opencv/opencv_zoo, verified
against known checksums. They're saved in this folder and gitignored (see root `.gitignore`).

## Set up the database (one time)

The robot needs a running MySQL server — locally that's the one behind MySQL Workbench. You do
**not** need to create the database or the tables by hand; `robot_db.py` creates both.

1. Copy the credentials template and fill in your MySQL password:

   ```powershell
   copy .env.example .env      # from the project root
   notepad .env
   ```

   `.env` is gitignored — it holds a real password, so it must never be committed. On the
   deployed kiosk, set the same `LUC_DB_*` names as real environment variables instead of
   shipping the file; a real environment variable always wins over `.env`.

2. Create the schema and check the connection:

   ```powershell
   python stages\stage02_face_recognition\robot_db.py
   ```

   It prints the row count of each table. Add `--self-test` to also run a full round trip — a
   temporary person is written, read back (checking the embedding survives byte-for-byte), given
   a conversation, and then deleted to confirm the privacy cascade actually removes their
   transcript along with them:

   ```powershell
   python stages\stage02_face_recognition\robot_db.py --self-test
   ```

Refresh the schema list in Workbench afterwards and you'll see `luc_robot` with five tables:
`robot_identity`, `people`, `face_embeddings`, `conversations`, `utterances`. `robot_identity` is
what the robot says about itself (Stage 05's `identity.py` reads it) — the others are visitor
data. See `docs/PRIVACY.md` for what's in
them, why `conversations.person_id` is allowed to be NULL, and the queries for handling a
"forget me" request.

**If MySQL isn't reachable**, nothing crashes — the robot prints a loud warning at startup and
runs with memory-only storage for that session, exactly as it behaved before this existed.
That's deliberate (a kiosk that won't greet anybody because a database is down is the worse
failure), so read the startup output rather than assuming persistence is on.

## Run the live demo

```powershell
python face_memory_live.py
```

Stand in frame — first time, it'll ask in the terminal whether to remember you. Say yes, give
your name, then **slowly turn your head left, then right, then center** for about 2.5 seconds
while it captures several samples (shown as a live counter in the preview window) instead of
one frozen snapshot. Step out and back in afterward: it should recognize you and print
`visit #2` without asking again. A second, different face gets asked separately. Say no to
consent and nothing is stored at all (see `docs/PRIVACY.md`).

Then **quit with `q` and run it again**. It should print `Loaded 1 known person(s) from MySQL`
and recognize you immediately, with no consent prompt — that's the whole point of the database,
and it's the one test worth actually doing rather than assuming.

Every later recognition also adds one more sample to that person's bank (capped at
`MAX_EMBEDDINGS_PER_PERSON` in `face_memory.py`, oldest dropped first), so accuracy keeps
improving the more they're seen, not just at enrollment.

## Known limitations (by design, for now)

- Consent/name collection is keyboard-based, not voice — real voice interaction comes with
  Stage 04 (STT) + Stage 06 (TTS) in Stage 08's full integration.
- Recognition runs every `RECOGNIZE_EVERY_N_FRAMES` frames (default 5), not every frame — face
  embedding is more expensive than simple detection, and a few checks per second is enough.
- Not yet linked to Stage 01's person tracking (`track_id`). Right now face identity is
  determined purely by face similarity, independent of the tracker. Stage 08 will need to
  decide how these two signals combine (e.g., re-check face identity periodically per tracked
  person, rather than per frame).
- Enrollment's `_closest_detection()` heuristic (nearest box center) assumes the person
  enrolling stays the closest face to their own last position - fine for one-on-one testing,
  but could latch onto the wrong person in a crowded multi-person scene. Worth revisiting once
  this is tied to Stage 01's tracker.

## Licensing note

YuNet + SFace are both Apache-2.0 (confirmed directly against opencv_zoo's `LICENSE` file for
these models) — safe for closed-source/commercial use, unlike InsightFace's default
`buffalo_l`-family models, which are non-commercial-only without a separate paid license.
