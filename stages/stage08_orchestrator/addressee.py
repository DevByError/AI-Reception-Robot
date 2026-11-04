"""
Stage 08 - was that sentence meant for the robot?

The gap this closes: once a visit started, EVERYTHING the microphone heard was treated as a
question. Two friends talking to each other in front of the robot got answered; so did someone
on the phone behind the visitor. Siri and Alexa solve this with a wake word. A receptionist
solves it by looking: you are talking to me if you are facing me and your lips are moving while
I hear you.

The robot can do both:

  1. Lip evidence. The camera loop records every face's mouth activity (stage02's
     mouth_activity.py) with a timestamp in LipHistory. When an utterance ends, the orchestrator
     asks: during the seconds that speech was heard, was any face that was LOOKING AT the robot
     also MOVING ITS LIPS? That is a time-aligned audio-visual check - the earlier code only
     ever asked "whose mouth is moving right now", with no link to when the audio happened.

  2. The wake word. Saying the robot's name ("Nova, where is the library?" / "Hey Nova ...")
     always counts, even from off camera or while looking away. find_wake_word() also strips the
     name off so the rest is answered as the question.

decide() combines them, with one safety valve: after max_ignored utterances in a row have been
ignored, the next one is accepted anyway, so a camera that misjudges lips can make the robot
slow to respond but never deaf.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class FaceSample:
    face_id: int
    speaking: bool
    facing: bool


@dataclass
class FaceSummary:
    face_id: int
    frames: int
    speaking_frames: int
    facing_frames: int

    @property
    def speaking_fraction(self) -> float:
        return self.speaking_frames / self.frames if self.frames else 0.0

    @property
    def facing_fraction(self) -> float:
        return self.facing_frames / self.frames if self.frames else 0.0


class LipHistory:
    """A few seconds of per-frame mouth activity, timestamped. Written by the camera thread,
    read by the conversation thread - hence the lock."""

    def __init__(self, seconds: float = 30.0):
        self.seconds = seconds
        self._frames: deque[tuple[float, list[FaceSample]]] = deque()
        self._lock = threading.Lock()

    def record(self, samples: list[FaceSample], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._frames.append((now, list(samples)))
            while self._frames and now - self._frames[0][0] > self.seconds:
                self._frames.popleft()

    def summarize(self, start: float, end: float) -> tuple[int, list[FaceSummary]]:
        """(frames recorded in the window, per-face summaries)."""
        with self._lock:
            frames = [(t, s) for t, s in self._frames if start <= t <= end]
        by_face: dict[int, FaceSummary] = {}
        for _t, samples in frames:
            for s in samples:
                summary = by_face.setdefault(s.face_id, FaceSummary(s.face_id, 0, 0, 0))
                summary.frames += 1
                summary.speaking_frames += int(s.speaking)
                summary.facing_frames += int(s.facing)
        return len(frames), list(by_face.values())


# --- the wake word -----------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()


def wake_phrases(robot_name: str, extra: str = "") -> list[str]:
    """The robot's name, "hey <name>", "ok <name>", plus any WAKE_WORDS from .env."""
    phrases = []
    name = _clean(robot_name or "")
    if name:
        phrases += [f"hey {name}", f"hi {name}", f"ok {name}", f"okay {name}", name]
    phrases += [_clean(p) for p in (extra or "").split(",") if _clean(p)]
    # longest first, so "hey nova" is stripped whole rather than leaving "hey"
    return sorted(set(phrases), key=len, reverse=True)


def find_wake_word(text: str | None, phrases: list[str]) -> tuple[bool, str]:
    """(found, the rest of the sentence with the wake phrase removed).

    Matches whole words only ("Nova" must not match "innovation"), anywhere in the sentence -
    people say "where's the library, Nova?" as often as "Nova, where's the library?".
    """
    if not text or not phrases:
        return False, text or ""
    cleaned = _clean(text)
    for phrase in phrases:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        if re.search(pattern, cleaned):
            # Remove it from the ORIGINAL text (keeps capitals/punctuation for the transcript).
            loose = r"\b" + r"[\s,.!?]*".join(re.escape(w) for w in phrase.split()) + r"\b[\s,.!?]*"
            rest = re.sub(loose, " ", text, count=1, flags=re.IGNORECASE)
            rest = re.sub(r"\s+", " ", rest).strip(" ,.!")
            return True, rest
    return False, text


# --- the decision ---------------------------------------------------------------------------------

@dataclass
class Decision:
    addressed: bool
    reason: str
    question: str          # the text to answer (wake word removed)


class AddresseeJudge:
    """Holds the policy settings and the ignored-in-a-row counter for one robot."""

    def __init__(self, mode: str = "auto", min_lip_fraction: float = 0.25,
                 min_facing_fraction: float = 0.5, max_ignored: int = 2,
                 wake: list[str] | None = None):
        self.mode = mode
        self.min_lip_fraction = min_lip_fraction
        self.min_facing_fraction = min_facing_fraction
        self.max_ignored = max_ignored
        self.wake = wake or []
        self.ignored_in_a_row = 0

    def reset(self) -> None:
        self.ignored_in_a_row = 0

    def decide(self, text: str | None, frames: int, faces: list[FaceSummary]) -> Decision:
        decision = self._decide(text, frames, faces)
        if decision.addressed:
            self.ignored_in_a_row = 0
        else:
            self.ignored_in_a_row += 1
            if self.ignored_in_a_row > self.max_ignored:
                self.ignored_in_a_row = 0
                return Decision(True, f"accepted anyway after {self.max_ignored} ignored in a "
                                      f"row ({decision.reason})", decision.question)
        return decision

    def _decide(self, text, frames, faces) -> Decision:
        found, rest = find_wake_word(text, self.wake)
        question = rest if found else (text or "")
        if self.mode == "off":
            return Decision(True, "addressee check off", question)
        if found:
            return Decision(True, "wake word", question)

        visible = [f for f in faces if f.frames > 0]
        if frames == 0 or not visible:
            if self.mode == "strict":
                return Decision(False, "no face evidence and no wake word (strict mode)",
                                question)
            return Decision(True, "no face evidence - answering", question)

        facing = [f for f in visible if f.facing_fraction >= self.min_facing_fraction]
        talking_to_me = [f for f in facing if f.speaking_fraction >= self.min_lip_fraction]
        if talking_to_me:
            best = max(talking_to_me, key=lambda f: f.speaking_fraction)
            return Decision(True, f"face {best.face_id} facing the robot with lips moving "
                                  f"{best.speaking_fraction:.0%} of the time", question)

        # One person, looking at the robot, a little lip movement: the lip detector is weakest
        # exactly here (small face, soft speaker), and a robot that ignores the only person in
        # front of it is the worst failure - so a lower bar applies.
        if (len(visible) == 1 and facing
                and facing[0].speaking_fraction >= self.min_lip_fraction / 3):
            return Decision(True, "only visitor, facing the robot, some lip movement", question)

        talking_elsewhere = [f for f in visible if f.speaking_fraction >= self.min_lip_fraction]
        if talking_elsewhere:
            return Decision(False, "the person talking was not facing the robot - probably "
                                   "talking to someone else", question)
        return Decision(False, "nobody facing the robot was moving their lips - the voice came "
                               "from someone else or off camera", question)
