"""
Stage 02 (live) - real face recognition with consent flow, using the actual webcam.

Stage 04 (speech-to-text) and Stage 06 (text-to-speech) aren't wired in yet, so consent and
name collection here use the keyboard (input()) as a temporary stand-in for the real voice
interaction - this is purely for testing the face memory logic against real faces. The video
feed pauses while waiting for keyboard input; that's expected for this console prototype and
goes away once Stage 08 wires in real STT/TTS instead of typed input.

Registration captures several embedding samples across ~2.5 seconds while prompting for
natural head movement, instead of a single frozen snapshot - a single sample is fragile
against normal angle/lighting variation, causing the very next recognition attempt to fail
with nothing else to compare against. See enroll_new_face() below.

Enrollments persist to MySQL (robot_db.py), so the real test of this demo is to enroll, quit,
and run it again - it should greet you by name without asking. Without a reachable database it
falls back to the original memory-only behaviour and prints a warning saying so.

Install first: see README.md in this folder (includes the one-time model download AND the
one-time database setup).
Run: python face_memory_live.py
Press 'q' in the preview window to quit.
"""

from __future__ import annotations

import logging
import sys
import time

import cv2
from face_memory import FaceMemoryStore
from robot_db import connect_or_none
from sface_embedder import SFaceEmbedder

# Face embedding + matching is more expensive than simple detection - only run it a few times
# per second, not on every single frame, and reuse the last result for drawing in between.
RECOGNIZE_EVERY_N_FRAMES = 5

# Presence debouncing per recognized person, in recognition ticks (not frames - see
# RECOGNIZE_EVERY_N_FRAMES). Without this, standing in frame for 20 seconds looks like 60+
# separate "visits" - one per recognition tick - instead of one continuous visit.
ARRIVAL_CONFIRM_TICKS = 2  # consecutive recognized ticks before counting a new visit
DEPARTURE_TIMEOUT_TICKS = 6  # consecutive missed ticks before considering them gone

# Enrollment burst: how long to capture samples for, and how many we're happy with.
ENROLLMENT_DURATION_S = 2.5
ENROLLMENT_MIN_SAMPLES = 3  # fall back gracefully if fewer were captured (e.g. bad lighting)

# How long to silently watch an unrecognized face before concluding it's genuinely new and
# asking about it. Without this, the very first failed match (a bad angle, or someone briefly
# covering their face with a hand) looks identical to "never seen this person before" and
# triggers the consent prompt immediately - too eager.
OBSERVATION_DURATION_S = 3.0
# Tolerate this many consecutive recognition ticks with no unmatched face seen at all (person
# turned away, hand over face, brief detector miss) without resetting the observation clock -
# only give up and forget this candidate if the gap runs longer than that.
OBSERVATION_GRACE_TICKS = 3


class UnknownFaceObserver:
    """Silently watches an unrecognized face for OBSERVATION_DURATION_S seconds before
    concluding it's genuinely new. Tolerates brief gaps (a hand over the face, a quick turn)
    without resetting the clock, so a momentary miss doesn't get treated as a brand-new
    person. Assumes at most one unresolved unknown face is being watched at a time - fine for
    solo testing, see README.md's known limitations for the multi-person caveat."""

    def __init__(self):
        self.start_time: float | None = None
        self.embeddings: list = []
        self.missed_ticks = 0

    def observe_unmatched(self, embedding) -> list | None:
        """Call once per tick for a detection that failed to match the store and hasn't
        already been asked about. Returns the accumulated samples once the observation window
        has elapsed (time to ask about them), or None while still watching."""
        if self.start_time is None:
            self.start_time = time.monotonic()
        self.embeddings.append(embedding)
        self.missed_ticks = 0

        if time.monotonic() - self.start_time >= OBSERVATION_DURATION_S:
            samples = self.embeddings
            self.reset()
            return samples
        return None

    def tick_nothing_unresolved(self) -> None:
        """Call once per tick where no still-unresolved unknown face was seen at all (matched,
        no detection, or already asked) - advances the grace counter instead of resetting
        immediately, and gives up only if the gap drags on."""
        if self.start_time is None:
            return
        self.missed_ticks += 1
        if self.missed_ticks > OBSERVATION_GRACE_TICKS:
            self.reset()

    def reset(self) -> None:
        self.start_time = None
        self.embeddings = []
        self.missed_ticks = 0


class PersonPresenceTracker:
    """Turns repeated per-tick recognitions of the same person into a single debounced
    "arrived" event, mirroring Stage 01's PerTrackPresenceTracker but keyed by recognized
    identity (person_id) instead of a raw tracker id."""

    def __init__(self):
        self._present_streak: dict[str, int] = {}
        self._absent_ticks: dict[str, int] = {}
        self.currently_present: set[str] = set()

    def on_recognized(self, person_id: str) -> bool:
        """Call once per recognition tick a person is matched. Returns True only on the tick
        that confirms a new arrival (i.e., call mark_seen_again() then, not on every tick)."""
        self._absent_ticks[person_id] = 0
        self._present_streak[person_id] = self._present_streak.get(person_id, 0) + 1

        if person_id in self.currently_present:
            return False
        if self._present_streak[person_id] >= ARRIVAL_CONFIRM_TICKS:
            self.currently_present.add(person_id)
            return True
        return False

    def tick_all_others_absent(self, seen_person_ids: set[str]) -> None:
        """Call once per recognition tick with the set of person_ids matched THIS tick, so
        everyone else's absence counter advances and stale presence eventually clears."""
        for person_id in list(self.currently_present):
            if person_id in seen_person_ids:
                continue
            self._absent_ticks[person_id] = self._absent_ticks.get(person_id, 0) + 1
            self._present_streak[person_id] = 0
            if self._absent_ticks[person_id] >= DEPARTURE_TIMEOUT_TICKS:
                self.currently_present.discard(person_id)


def ask_consent() -> bool:
    """Stand-in for the real voice-based consent flow (Stage 04/05/06)."""
    answer = (
        input(
            "\n[robot] I don't recognize you. Would you like me to remember you for "
            "next time? (y/n): "
        )
        .strip()
        .lower()
    )
    return answer == "y"


def ask_name() -> str:
    return input("[robot] Great! What's your name? ").strip() or "Guest"


def _closest_detection(detections, anchor_bbox):
    """Picks the detection whose box center is nearest the anchor - keeps enrollment locked
    onto the same physical person across frames even if someone else is also in frame."""
    ax, ay, aw, ah = anchor_bbox
    anchor_center = (ax + aw / 2, ay + ah / 2)

    def dist(det):
        x, y, w, h = det.bbox
        center = (x + w / 2, y + h / 2)
        return (center[0] - anchor_center[0]) ** 2 + (center[1] - anchor_center[1]) ** 2

    return min(detections, key=dist) if detections else None


def enroll_new_face(cap, embedder: SFaceEmbedder, anchor_bbox, initial_embedding) -> list:
    """Captures several embedding samples over ENROLLMENT_DURATION_S seconds, prompting for
    natural head movement, so the stored face isn't fragile to a single angle/lighting
    snapshot. Shows a live preview with a capture counter so it doesn't look frozen. Falls
    back to just the initial embedding if too few samples were captured."""
    print(
        "[robot] Please slowly turn your head left, then right, then look straight at me "
        "for a couple of seconds..."
    )

    samples = [initial_embedding]
    start = time.monotonic()

    while time.monotonic() - start < ENROLLMENT_DURATION_S:
        ok, frame = cap.read()
        if not ok:
            break

        detections = embedder.detect(frame)
        closest = _closest_detection(detections, anchor_bbox)
        if closest is not None:
            samples.append(closest.embedding)
            anchor_bbox = closest.bbox  # follow natural movement during the capture window
            x, y, w, h = closest.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

        remaining = ENROLLMENT_DURATION_S - (time.monotonic() - start)
        cv2.putText(
            frame,
            f"Capturing... {remaining:.1f}s ({len(samples)} samples)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )
        cv2.imshow("Stage 02 - live face recognition (press q to quit)", frame)
        cv2.waitKey(1)

    if len(samples) < ENROLLMENT_MIN_SAMPLES:
        print(
            f"[robot] Only got {len(samples)} good sample(s) - that's a bit thin, but "
            f"I'll remember what I have and learn more as we talk."
        )
    else:
        print(f"[robot] Got it - {len(samples)} samples captured.")

    return samples


def draw_faces(frame, detections, statuses: dict[int, str]):
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        status = statuses.get(i, "")
        color = (0, 200, 0) if status.startswith("known") else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, status, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    print("Loading YuNet + SFace models...")
    embedder = SFaceEmbedder()

    # Persistent across runs (robot_db.py). This demo is the quickest way to prove that: enroll,
    # quit with 'q', run it again, and it should recognize you without asking. Without a
    # reachable MySQL it falls back to the old memory-only behaviour and says so.
    db = connect_or_none()
    store = FaceMemoryStore(is_match=embedder.is_match, score=embedder.similarity, db=db)
    if db is not None:
        print(f"Loaded {len(store)} known person(s) from MySQL.")

    # Embeddings already asked-about this run (yes or no) - avoids re-prompting every
    # recognition tick for the same still-unresolved/declined face while they're in view.
    asked_embeddings: list = []
    presence = PersonPresenceTracker()
    observer = UnknownFaceObserver()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index=0")

    print("Live face recognition running. Press 'q' in the preview window to quit.")
    frame_count = 0
    last_detections: list = []
    last_statuses: dict[int, str] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame - stopping.")
                break

            frame_count += 1

            if frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
                detections = embedder.detect(frame)
                statuses: dict[int, str] = {}
                seen_person_ids: set[str] = set()
                unresolved_embedding = None  # at most one candidate observed at a time
                unresolved_bbox = None

                for i, det in enumerate(detections):
                    match = store.find_match(det.embedding)
                    if match:
                        seen_person_ids.add(match.person_id)
                        store.reinforce(match, det.embedding)  # widen the sample bank further
                        is_new_arrival = presence.on_recognized(match.person_id)
                        if is_new_arrival:
                            store.mark_seen_again(match)
                            print(f"[event] {match.name} arrived - visit #{match.visit_count}")
                        statuses[i] = f"known: {match.name} (visit #{match.visit_count})"
                        continue

                    already_asked = any(
                        embedder.is_match(det.embedding, e) for e in asked_embeddings
                    )
                    if already_asked:
                        statuses[i] = "unknown (already asked)"
                        continue

                    # Still unresolved and not yet asked about - hand to the observer instead
                    # of asking immediately (see UnknownFaceObserver docstring for why).
                    unresolved_embedding = det.embedding
                    unresolved_bbox = det.bbox
                    statuses[i] = "checking..."

                if unresolved_embedding is not None:
                    observed_samples = observer.observe_unmatched(unresolved_embedding)
                else:
                    observer.tick_nothing_unresolved()
                    observed_samples = None

                if observed_samples is not None:
                    # Observation window elapsed and it's STILL unmatched - genuinely new.
                    asked_embeddings.append(unresolved_embedding)
                    if not ask_consent():
                        print("[robot] No problem, I won't remember this visit.")
                    else:
                        name = ask_name()
                        burst_samples = enroll_new_face(
                            cap, embedder, unresolved_bbox, unresolved_embedding
                        )
                        # combine what we passively observed over the last 3s with the
                        # deliberate turn-your-head burst, for the widest angle coverage
                        all_samples = observed_samples + burst_samples
                        store.register(name=name, embeddings=all_samples, consent_given=True)
                        print(f"[robot] Nice to meet you, {name}! I'll remember you next time.")

                presence.tick_all_others_absent(seen_person_ids)
                last_detections, last_statuses = detections, statuses

            frame = draw_faces(frame, last_detections, last_statuses)
            cv2.imshow("Stage 02 - live face recognition (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if db is not None:
            db.close()
