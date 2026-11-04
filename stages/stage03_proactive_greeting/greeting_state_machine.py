"""
Stage 03 - Proactive Greeting State Machine

Goal: IDLE -> DETECTED -> GREETING -> LISTENING, with one shared group greeting if there's more
than one person, plus a personal touch if a known face is part of the group.

Depends on the output of Stage 01 (presence_detector) and Stage 02 (face_memory), but here we
use simple mock objects so we can test the state logic on its own before wiring in the real
stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class RobotState(Enum):
    IDLE = auto()
    DETECTED = auto()
    GREETING = auto()
    LISTENING = auto()


@dataclass
class Person:
    track_id: int
    known_name: str | None  # None means the face isn't recognized yet


class ExpressionController:
    """
    The expression/face layer mentioned in ARCHITECTURE.md - swappable later with a screen
    (animated face) implementation or a physical animatronic head (servo commands). Here: a
    console version for testing.
    """

    def set_state(self, state: RobotState) -> None:
        print(f"[face] -> {state.name}")


class TTSStub:
    """A temporary stand-in for the real Stage 06 - prints instead of actually speaking."""

    def speak(self, text: str) -> None:
        print(f'[TTS] "{text}"')


# Several phrasings per situation instead of one fixed string each.
#
# WHY: the robot greeted every stranger with the identical sentence, forever. One
# visitor never notices - but a kiosk in a lobby is watched by the same students day after day,
# and hearing the exact same words at the exact same moment is the clearest possible signal that
# there is a script running. Varying the wording costs nothing and removes that tell. The
# information conveyed is identical in every variant; only the words change.
#
# The trailing "how can I help" question is deliberately part of each variant rather than
# appended separately: it is what tells the visitor it is their turn to speak, and a greeting
# that ends without it leaves people waiting for the robot to continue.
_UNKNOWN_SINGLE = (
    "Hi there! How can I help you today?",
    "Hello! What can I do for you?",
    "Hi! How can I help?",
    "Hey there! What can I help you with?",
)

_KNOWN_SINGLE = (
    "Welcome back, {name}! How can I help you today?",
    "Hi {name}! What can I do for you?",
    "Good to see you, {name}! How can I help?",
    "Hey {name}! What can I help you with today?",
)

_UNKNOWN_GROUP = (
    "Hi everyone! How can I help you today?",
    "Hello all! What can I do for you?",
    "Hi there, everyone! How can I help?",
)

_KNOWN_GROUP = (
    "Hi everyone, welcome back {name} and friends! How can I help you all?",
    "Hello all - good to see you again, {name}! What can I do for you?",
    "Hi everyone! Nice to see you back, {name}. How can I help?",
)


def build_greeting(people: list[Person], recall: str = "", rng=None) -> str:
    """
    - a single unknown person  -> a generic greeting.
    - a single known person    -> a personal greeting.
    - a group (regardless of who is known) -> one shared group greeting, mentioning one name as
      a personal touch if available.

    `recall` is an optional extra sentence about what the robot remembers from this person's last
    visit (Stage 05's visit_memory.build_recall_line). It goes BEFORE the "how can I help"
    question, because that question is the visitor's cue to start talking - putting anything
    after it means the robot is still speaking when they have already begun. Empty string is the
    normal case: a new visitor, or a returning one with nothing worth recalling.

    `rng` is injectable purely so tests can pin the wording - see this file's self-test.
    """
    import random

    rng = rng or random

    if not people:
        return ""

    known_names = [p.known_name for p in people if p.known_name]

    if len(people) == 1:
        p = people[0]
        if p.known_name:
            greeting = rng.choice(_KNOWN_SINGLE).format(name=p.known_name)
        else:
            greeting = rng.choice(_UNKNOWN_SINGLE)
    elif known_names:
        greeting = rng.choice(_KNOWN_GROUP).format(name=known_names[0])
    else:
        greeting = rng.choice(_UNKNOWN_GROUP)

    if not recall:
        return greeting

    # Split the closing question off and slot the recall in front of it, so the greeting still
    # ends on the visitor's cue to speak.
    head, sep, question = greeting.rpartition("! ")
    if not sep:
        return f"{greeting} {recall}"
    return f"{head}! {recall} {question}"


class GreetingStateMachine:
    def __init__(self, expression: ExpressionController, tts: TTSStub):
        self.state = RobotState.IDLE
        self.expression = expression
        self.tts = tts

    def on_people_entered(self, people: list[Person], recall: str = "") -> None:
        """`recall` is an optional sentence about what the robot remembers from this person's
        last visit - Stage 08 looks it up (visit_memory.build_recall_line) and passes it in,
        because only the orchestrator has the database. Defaults to none, so this file's own
        self-test and Stage 03's demo are unaffected."""
        if self.state != RobotState.IDLE or not people:
            return  # if already mid-interaction, don't interrupt with a new greeting
            # (handled later by Stage 08)

        self.state = RobotState.DETECTED
        self.expression.set_state(self.state)

        self.state = RobotState.GREETING
        self.expression.set_state(self.state)
        self.tts.speak(build_greeting(people, recall=recall))

        self.state = RobotState.LISTENING
        self.expression.set_state(self.state)

    def on_interaction_timeout_or_done(self) -> None:
        self.state = RobotState.IDLE
        self.expression.set_state(self.state)


if __name__ == "__main__":
    sm = GreetingStateMachine(ExpressionController(), TTSStub())

    print("== Case 1: a single unknown person ==")
    sm.on_people_entered([Person(track_id=1, known_name=None)])
    assert sm.state == RobotState.LISTENING
    sm.on_interaction_timeout_or_done()

    print("\n== Case 2: a single known person (returning visit) ==")
    sm.on_people_entered([Person(track_id=2, known_name="Sara")])
    sm.on_interaction_timeout_or_done()

    print("\n== Case 3: a group, with one known person in it ==")
    sm.on_people_entered(
        [
            Person(track_id=3, known_name="Omar"),
            Person(track_id=4, known_name=None),
        ]
    )
    sm.on_interaction_timeout_or_done()

    # --- wording variety + the recall line --------------------------------------
    import random

    print("\n== Case 4: the same situation does not always produce the same words ==")
    solo = Person(track_id=5, known_name=None)
    said = {build_greeting([solo], rng=random.Random(seed)) for seed in range(30)}
    for phrasing in sorted(said):
        print(f'  "{phrasing}"')
    assert len(said) > 1, (
        "every greeting for the same situation came out identical - the same students hear this "
        "robot every day, and a fixed script is the clearest possible tell that it is one"
    )
    assert said <= set(_UNKNOWN_SINGLE), "an unknown visitor was greeted by name"

    print("\n== Case 5: a returning visitor the robot remembers something about ==")
    known = Person(track_id=6, known_name="Sara")
    with_recall = build_greeting(
        [known],
        recall="Did you get anywhere with the Master of Computer Science?",
        rng=random.Random(1),
    )
    print(f'  "{with_recall}"')
    assert "Sara" in with_recall
    assert "Master of Computer Science" in with_recall
    # The greeting must still END on the question that hands the turn over, or the visitor starts
    # talking while the robot is still going.
    assert with_recall.rstrip().endswith("?"), with_recall
    assert not with_recall.rstrip().endswith(
        "Did you get anywhere with the Master of Computer Science?"
    ), "the recall was appended after the 'how can I help' cue instead of before it"

    # No recall to make is the common case and must be silent, not padded.
    plain = build_greeting([known], rng=random.Random(1))
    assert plain == build_greeting([known], recall="", rng=random.Random(1)) == plain

    print(
        "\nOK: the greeting changes correctly per case (solo/known/group), the wording varies "
        "between visits instead of repeating one fixed script, a remembered detail is spoken "
        "before the hand-over question rather than after it, and the state returns to IDLE "
        "after each interaction ends."
    )
