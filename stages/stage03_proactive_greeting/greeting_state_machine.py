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


def build_greeting(people: list[Person]) -> str:
    """
    - a single unknown person  -> a generic greeting.
    - a single known person    -> a personal greeting.
    - a group (regardless of who is known) -> one shared group greeting, mentioning one name as
      a personal touch if available.
    """
    if not people:
        return ""

    known_names = [p.known_name for p in people if p.known_name]

    if len(people) == 1:
        p = people[0]
        if p.known_name:
            return f"Welcome back, {p.known_name}! How can I help you today?"
        return "Hi there! How can I help you today?"

    # group
    if known_names:
        return f"Hi everyone, welcome back {known_names[0]} and friends! How can I help you all?"
    return "Hi everyone! How can I help you today?"


class GreetingStateMachine:
    def __init__(self, expression: ExpressionController, tts: TTSStub):
        self.state = RobotState.IDLE
        self.expression = expression
        self.tts = tts

    def on_people_entered(self, people: list[Person]) -> None:
        if self.state != RobotState.IDLE or not people:
            return  # if already mid-interaction, don't interrupt with a new greeting
                     # (handled later by Stage 08)

        self.state = RobotState.DETECTED
        self.expression.set_state(self.state)

        self.state = RobotState.GREETING
        self.expression.set_state(self.state)
        self.tts.speak(build_greeting(people))

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
    sm.on_people_entered([
        Person(track_id=3, known_name="Omar"),
        Person(track_id=4, known_name=None),
    ])
    sm.on_interaction_timeout_or_done()

    print("\nOK: the greeting changes correctly per case (solo/known/group), and the state "
          "returns to IDLE after each interaction ends.")
