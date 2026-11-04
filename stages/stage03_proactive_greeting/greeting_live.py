"""
Stage 03 (live) - the real Stage 01 (YOLOv8n person tracking) and Stage 02 (YuNet+SFace face
recognition) feeding real Person events into the GreetingStateMachine from
greeting_state_machine.py, speaking the greeting out loud instead of printing it.

Stage 04 (STT) and Stage 05 (dialogue) aren't wired in yet, so there's no real way to know when
the person is actually done talking - LISTENING_TIMEOUT_S below is a temporary stand-in that
just returns to IDLE after a fixed pause, same spirit as face_memory_live.py using keyboard
input as a stand-in for real voice consent. Both stand-ins go away once Stage 08 wires in the
real thing.

What this intentionally does NOT do: register brand-new faces through the full consent flow
that face_memory_live.py already tests on its own (that's Stage 02's job, already covered).
Here, an unrecognized face just gets a generic greeting like any other "unknown" case - but for
convenience while testing personalization live, press 'r' to quickly register the person
currently in frame (single-sample, keyboard name prompt) so the next entrance is personalized.

Install first:
  pip install -r stages\\stage01_presence_detection\\requirements.txt
  pip install -r stages\\stage02_face_recognition\\requirements.txt
  pip install -r stages\\stage03_proactive_greeting\\requirements.txt
Models: stage02's YuNet/SFace ONNX files must already be downloaded (see stage02's README).

Run: python stages/stage03_proactive_greeting/greeting_live.py
Press 'r' to register the (single) unresolved face currently in frame, 'q' to quit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

THIS_DIR = Path(__file__).resolve().parent
STAGE01_DIR = THIS_DIR.parent / "stage01_presence_detection"
STAGE02_DIR = THIS_DIR.parent / "stage02_face_recognition"
for p in (STAGE01_DIR, STAGE02_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# YOLO (ultralytics, which pulls in PyTorch) is imported only by this file's own demo below:
# stage08 imports match_faces_to_people() from here, and the robot's default face-based presence
# must not need PyTorch installed.
from presence_detector_live import (  # noqa: E402
    PerTrackPresenceTracker,
    detect_people,
)
from face_memory import FaceMemoryStore  # noqa: E402
from sface_embedder import SFaceEmbedder  # noqa: E402

from greeting_state_machine import (  # noqa: E402
    ExpressionController,
    GreetingStateMachine,
    Person,
    RobotState,
)

# How often to run face recognition - same reasoning as stage02: embeddings are more expensive
# than person detection/tracking, a few checks per second is enough.
RECOGNIZE_EVERY_N_FRAMES = 5

# How long to hold off greeting a newly-arrived person while recognition is still checking who
# they are, before giving up and treating them as a stranger. Without this, the greeting fires
# on the very first "entered" event - often before even one recognition tick has run - so a
# returning person gets the generic "Hi there!" instead of "Welcome back". This mirrors stage02's
# OBSERVATION_DURATION_S/ENROLLMENT_DURATION_S (~2.5-3s): give recognition several ticks, across
# a bit of natural head movement, before deciding known vs. unknown.
RECOGNITION_WAIT_S = 2.5

# Temporary stand-in for "the conversation is over" (real signal comes from Stage 04/05 later):
# just return to IDLE this long after entering LISTENING.
LISTENING_TIMEOUT_S = 6.0


class LiveTTSSpeaker:
    """Real audio stand-in for Stage 06 (which doesn't exist yet) - same .speak(text) interface
    as greeting_state_machine.TTSStub, so it's a drop-in replacement. Uses pyttsx3 (offline,
    cross-platform - SAPI5 on Windows) instead of a real streaming TTS engine; blocks the frame
    loop while speaking, which is fine for this console prototype (same tradeoff
    face_memory_live.py accepts for keyboard-based consent)."""

    def __init__(self):
        import pyttsx3

        self.engine = pyttsx3.init()
        # One-time warm-up: on Windows, SAPI5 needs a moment to actually open the audio device,
        # and if that happens lazily on the first real speak() call, the opening word(s) get
        # clipped. Speaking something throwaway here at startup, while nobody's listening yet,
        # avoids wasting that clip on the first real greeting.
        self.engine.say(" ")
        self.engine.runAndWait()

    def speak(self, text: str) -> None:
        print(f'[TTS] "{text}"')
        # Same clipping issue can resurface after any idle gap between greetings (SAPI5 seems to
        # partially release the audio device when idle), not just at startup - so queue a short
        # silent primer immediately before the real text every time, in the same runAndWait()
        # flush, instead of only once at __init__.
        self.engine.say(" ")
        self.engine.say(text)
        self.engine.runAndWait()


def _face_center_in_bbox(face_bbox, person_bbox) -> bool:
    """True if the face detection's center point falls inside the person's bounding box - used
    to link a frame-wide face detection to the YOLO-tracked person it belongs to."""
    fx, fy, fw, fh = face_bbox
    face_center = (fx + fw / 2, fy + fh / 2)
    px, py, pw, ph = person_bbox
    return (px <= face_center[0] <= px + pw) and (py <= face_center[1] <= py + ph)


def match_faces_to_people(people, face_detections):
    """Returns {track_id: FaceDetection} for every tracked person whose bbox contains a
    detected face's center. If several people bboxes could claim the same face (unlikely once
    people are reasonably spread out), the first match wins - fine for solo/small-group
    testing."""
    matches = {}
    for det in face_detections:
        for p in people:
            if _face_center_in_bbox(det.bbox, p.bbox):
                matches[p.track_id] = det
                break
    return matches


def draw_overlay(
    frame,
    people,
    resolved_names: dict[int, str | None],
    pending_ids: set[int],
    state: RobotState,
):
    for p in people:
        x, y, w, h = p.bbox
        name = resolved_names.get(p.track_id)
        if name:
            label = f"#{p.track_id} {name}"
        elif p.track_id in pending_ids:
            label = f"#{p.track_id} checking..."
        else:
            label = f"#{p.track_id} unknown"
        color = (0, 200, 0) if name else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    cv2.putText(
        frame,
        f"state: {state.name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2,
    )
    cv2.putText(
        frame,
        "press 'r' to register the face in frame, 'q' to quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 200, 0),
        1,
    )
    return frame


if __name__ == "__main__":
    print("Loading YOLOv8n + YuNet + SFace models...")
    # Point at stage01's copy explicitly (reuses its already-downloaded weights, and downloads
    # there if missing) instead of a bare "yolov8n.pt", which depends on the current working
    # directory when this script is launched from the project root.
    from ultralytics import YOLO

    yolo = YOLO(str(STAGE01_DIR / "yolov8n.pt"))
    embedder = SFaceEmbedder()
    store = FaceMemoryStore(is_match=embedder.is_match, score=embedder.similarity)

    presence = PerTrackPresenceTracker()
    sm = GreetingStateMachine(ExpressionController(), LiveTTSSpeaker())

    resolved_names: dict[int, str | None] = {}  # track_id -> known name, once matched
    last_embedding_by_track: dict[int, object] = (
        {}
    )  # track_id -> most recent face embedding
    pending_since: dict[int, float] = (
        {}
    )  # track_id -> when presence confirmed them, while
    # still waiting on recognition to resolve (or
    # time out) before they're actually greeted
    ready_to_greet: set[int] = set()  # track_ids finalized (known or genuinely
    # unknown) and waiting for a greeting to fire
    listening_since: float | None = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index=0")

    print(
        "Live greeting running. Press 'r' to register the face in frame, 'q' to quit."
    )
    frame_count = 0
    last_people: list = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame - stopping.")
                break
            frame_count += 1

            people = detect_people(yolo, frame)
            seen_ids = {p.track_id for p in people}
            entered_ids, left_ids = presence.update(seen_ids)
            if left_ids:
                print(f"[event] left - ids={left_ids}")
                for tid in left_ids:
                    resolved_names.pop(tid, None)
                    last_embedding_by_track.pop(tid, None)
                    pending_since.pop(tid, None)
                    ready_to_greet.discard(tid)

            # Presence just confirmed these people, but don't greet yet - start the "let me
            # check first" clock instead, so recognition gets a fair chance before we decide
            # known vs. unknown (see RECOGNITION_WAIT_S above).
            for tid in entered_ids:
                pending_since[tid] = time.monotonic()
            if entered_ids:
                print(
                    f"[event] presence confirmed - ids={entered_ids} - checking face before "
                    f"greeting..."
                )

            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0 and people:
                face_detections = embedder.detect(frame)
                matches = match_faces_to_people(people, face_detections)
                for tid, det in matches.items():
                    last_embedding_by_track[tid] = det.embedding
                    match = store.find_match(det.embedding)
                    if match:
                        store.reinforce(match, det.embedding)
                        resolved_names[tid] = match.name
                        if tid in pending_since:
                            # confidently recognized - no need to keep waiting, greet as known
                            pending_since.pop(tid)
                            ready_to_greet.add(tid)

            # Anyone who's been pending too long without a confident match: stop waiting and
            # treat them as a genuine stranger rather than holding the greeting forever.
            now = time.monotonic()
            for tid in [
                t
                for t, since in pending_since.items()
                if now - since >= RECOGNITION_WAIT_S
            ]:
                pending_since.pop(tid)
                ready_to_greet.add(tid)

            # Drop anyone who became "ready" but then walked off before a greeting fired.
            ready_to_greet &= seen_ids

            if ready_to_greet and sm.state == RobotState.IDLE:
                arriving = [
                    Person(track_id=tid, known_name=resolved_names.get(tid))
                    for tid in ready_to_greet
                ]
                print(
                    f"[event] ready to greet - ids={sorted(ready_to_greet)}, "
                    f"names={[p.known_name for p in arriving]}"
                )
                sm.on_people_entered(arriving)
                ready_to_greet.clear()
                if sm.state == RobotState.LISTENING:
                    listening_since = time.monotonic()

            if sm.state == RobotState.LISTENING and listening_since is not None:
                if time.monotonic() - listening_since >= LISTENING_TIMEOUT_S:
                    sm.on_interaction_timeout_or_done()
                    listening_since = None

            last_people = people
            frame = draw_overlay(
                frame, last_people, resolved_names, set(pending_since), sm.state
            )
            cv2.imshow("Stage 03 - live proactive greeting (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                unresolved = [tid for tid in seen_ids if not resolved_names.get(tid)]
                if len(unresolved) == 0:
                    print("[robot] No unresolved face in frame right now.")
                elif len(unresolved) > 1:
                    print(
                        f"[robot] More than one unresolved face in frame ({unresolved}) - "
                        f"step out so only one is visible, then try again."
                    )
                else:
                    tid = unresolved[0]
                    embedding = last_embedding_by_track.get(tid)
                    if embedding is None:
                        print(
                            "[robot] Haven't gotten a clear look at that face yet - wait a "
                            "moment and try again."
                        )
                    else:
                        name = input("[robot] What's your name? ").strip() or "Guest"
                        store.register(
                            name=name, embeddings=[embedding], consent_given=True
                        )
                        resolved_names[tid] = name
                        print(
                            f"[robot] Got it, {name} - I'll greet you personally next time "
                            f"you walk in."
                        )
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
