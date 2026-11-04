"""
Stage 08 - the conversation's own state machine.

WHAT THIS IS FOR
----------------
The robot has had a state machine since Stage 03 (greeting_state_machine.py: IDLE -> DETECTED ->
GREETING -> LISTENING). That one models the PERCEPTION side - somebody walked up, greet them -
and it is still the thing that drives the face/expression controller. It is untouched and still
owns that job.

What has never existed is a model of the CONVERSATION side: whether the visitor is mid-sentence
or has finished, whether the robot is thinking or speaking, and whether it was just interrupted.
Those distinctions were real - the code made them, every one of them - but they lived as local
variables inside listen_for_one_utterance() and _handle_conversation(), so nothing could observe
them and a misbehaving turn ("why did it answer me there?") could only be debugged by reading
audio timings out of a log that never said what the robot thought was happening.

This module gives those distinctions a name, one transition table, and one log line each.

DELIBERATELY OBSERVATIONAL
--------------------------
This machine does not GATE anything. No audio decision, no dialogue decision and no playback
decision asks it for permission; it is told what happened and records it. That is the whole
reason it is safe to add to a working robot: a bug here can produce a wrong log line, and cannot
produce a wrong answer, a dropped turn or a deadlock.

If a transition arrives that the table does not allow, it is taken anyway and reported as
UNEXPECTED. The alternative - refusing it - would mean the state machine and the real
conversation quietly disagreeing from that point on, which is strictly worse than a noisy log:
the state would be wrong AND would not say so.

Run `python conversation_state.py` for the self-test (no hardware, no models).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[1] / "stage00_setup"))
from realtime_config import RT  # noqa: E402


class ConvState(Enum):
    """Every state the conversation can be in.

    IDLE          nobody is being served; the microphone is open but nothing is expected
    LISTENING     waiting for the visitor to start - silence here is normal, not a turn end
    USER_SPEAKING VAD has confirmed continuous speech and audio is being recorded
    USER_PAUSED   a pause inside a turn. NOT an end of turn: this is the state the whole
                  adaptive endpointer exists to resolve, and the one a naive "silence means
                  finished" assistant gets wrong
    USER_FINISHED the endpointer has committed - the turn is over and the transcript is final
    THINKING      retrieval and generation are running; nothing is audible yet
    AI_SPEAKING   Piper audio is actually reaching the speaker
    INTERRUPTED   the visitor talked over the robot and playback was cut
    """

    IDLE = auto()
    LISTENING = auto()
    USER_SPEAKING = auto()
    USER_PAUSED = auto()
    USER_FINISHED = auto()
    THINKING = auto()
    AI_SPEAKING = auto()
    INTERRUPTED = auto()


# The allowed edges. Written out in full rather than derived, because the value of this table is
# that a reader can see the whole conversation shape in one screen - including the two loops
# that matter:
#
#   USER_SPEAKING <-> USER_PAUSED   somebody thinking mid-sentence, going round as many times as
#                                   they need. Only USER_PAUSED -> USER_FINISHED leaves it, and
#                                   only the endpointer may take that edge.
#   AI_SPEAKING -> INTERRUPTED -> LISTENING   barge-in. INTERRUPTED never goes to IDLE: somebody
#                                   who just talked over the robot is, by definition, still there
#                                   and still talking.
_ALLOWED: dict[ConvState, frozenset[ConvState]] = {
    ConvState.IDLE: frozenset({ConvState.LISTENING, ConvState.AI_SPEAKING}),
    # AI_SPEAKING from IDLE is the proactive greeting: the robot speaks first, before anybody
    # has said anything to it. That edge is the whole premise of Stage 03.
    ConvState.LISTENING: frozenset(
        {
            ConvState.USER_SPEAKING,
            ConvState.IDLE,
            ConvState.THINKING,
            ConvState.AI_SPEAKING,
        }
    ),
    # LISTENING -> THINKING is a listen that timed out or came back empty and is being handled;
    # LISTENING -> AI_SPEAKING is the robot re-prompting ("anything else?") without having heard
    # anything.
    ConvState.USER_SPEAKING: frozenset(
        {
            ConvState.USER_PAUSED,
            ConvState.USER_FINISHED,
            ConvState.LISTENING,
            ConvState.IDLE,
        }
    ),
    # USER_SPEAKING -> LISTENING is the too-short clip being discarded (a cough): speech was
    # detected, then rejected, and the robot goes back to waiting without a turn having happened.
    ConvState.USER_PAUSED: frozenset(
        {
            ConvState.USER_SPEAKING,
            ConvState.USER_FINISHED,
            ConvState.LISTENING,
            ConvState.IDLE,
        }
    ),
    ConvState.USER_FINISHED: frozenset(
        {
            ConvState.THINKING,
            ConvState.LISTENING,
            ConvState.IDLE,
            ConvState.AI_SPEAKING,
        }
    ),
    ConvState.THINKING: frozenset(
        {
            ConvState.AI_SPEAKING,
            ConvState.LISTENING,
            ConvState.IDLE,
        }
    ),
    ConvState.AI_SPEAKING: frozenset(
        {
            ConvState.INTERRUPTED,
            ConvState.LISTENING,
            ConvState.IDLE,
            ConvState.THINKING,
        }
    ),
    ConvState.INTERRUPTED: frozenset(
        {
            ConvState.LISTENING,
            ConvState.USER_SPEAKING,
            ConvState.IDLE,
        }
    ),
}


@dataclass
class TurnStats:
    """What one visitor turn cost, in the only terms that matter to somebody standing there.

    Every field is filled from a real transition, not estimated. The one people actually care
    about is `to_first_word_ms` - the gap between the visitor stopping and the robot starting -
    because that is the silence they experience. Total answer length is not that number: the
    robot is still generating while it talks.
    """

    started_at: float = field(default_factory=time.monotonic)
    speech_started_at: float | None = None
    finished_speaking_at: float | None = None
    thinking_started_at: float | None = None
    first_word_at: float | None = None
    interrupted_at: float | None = None
    pauses: int = 0

    @property
    def speech_duration_ms(self) -> float | None:
        if self.speech_started_at is None or self.finished_speaking_at is None:
            return None
        return (self.finished_speaking_at - self.speech_started_at) * 1000

    @property
    def to_first_word_ms(self) -> float | None:
        """Visitor stops talking -> robot's first audible word. The latency target."""
        if self.finished_speaking_at is None or self.first_word_at is None:
            return None
        return (self.first_word_at - self.finished_speaking_at) * 1000

    def summary(self) -> str:
        bits = []
        if self.speech_duration_ms is not None:
            bits.append(f"they spoke for {self.speech_duration_ms:.0f}ms")
        if self.pauses:
            bits.append(f"{self.pauses} mid-turn pause(s) waited out")
        if self.to_first_word_ms is not None:
            bits.append(f"answer began {self.to_first_word_ms:.0f}ms after they stopped")
        if self.interrupted_at is not None:
            bits.append("interrupted")
        return ", ".join(bits) if bits else "nothing measured"


class ConversationStateMachine:
    """Tracks one conversation. Thread-safe, because it is written from three threads.

    Which threads, and why that is not negotiable: the visit thread drives listening and
    speaking; BargeInMonitor runs on its own thread and is the ONLY thing that can report an
    interruption; the camera thread ends a visit when the visitor walks away. All three set
    state, so every read and write is under one lock and `state` is a snapshot, never a field
    somebody can catch mid-update.
    """

    def __init__(self, on_change=None, debug: bool | None = None):
        self._lock = threading.RLock()
        self._state = ConvState.IDLE
        self._changed_at = time.monotonic()
        self._history: list[tuple[ConvState, float]] = [(ConvState.IDLE, self._changed_at)]
        self._on_change = on_change
        self._debug = RT.REALTIME_DEBUG if debug is None else debug
        self.turn = TurnStats()
        self.unexpected_transitions = 0

    @property
    def state(self) -> ConvState:
        with self._lock:
            return self._state

    def is_in(self, *states: ConvState) -> bool:
        with self._lock:
            return self._state in states

    def time_in_state_ms(self) -> float:
        with self._lock:
            return (time.monotonic() - self._changed_at) * 1000

    def to(self, new_state: ConvState, note: str = "") -> ConvState:
        """Moves to `new_state` and logs it. Returns the state actually now held.

        A transition to the state already held is a no-op and is NOT logged: the audio loop
        reports "still speaking" on every 30ms frame, and honouring each one would bury the
        real transitions under thirty lines a second.
        """
        with self._lock:
            previous = self._state
            if new_state is previous:
                return previous

            allowed = new_state in _ALLOWED.get(previous, frozenset())
            if not allowed:
                self.unexpected_transitions += 1

            now = time.monotonic()
            held_ms = (now - self._changed_at) * 1000
            self._state = new_state
            self._changed_at = now
            self._history.append((new_state, now))
            if len(self._history) > 200:
                # Bounded: a long visit must not grow this without limit, and nothing reads
                # further back than the last few transitions anyway.
                del self._history[:-200]

            self._record_timing(previous, new_state, now)

            if self._debug:
                tail = f" ({note})" if note else ""
                flag = "" if allowed else "  << UNEXPECTED - not in the transition table"
                log.info(
                    f"[STATE] {previous.name} -> {new_state.name}"
                    f"  after {held_ms:.0f}ms{tail}{flag}"
                )

        # Outside the lock deliberately: a callback is caller code (it draws the robot's face)
        # and must never be able to deadlock the audio threads by calling back in.
        if self._on_change is not None:
            try:
                self._on_change(previous, new_state)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    f"[STATE] state-change callback failed ({exc!r}) - ignoring; the "
                    f"conversation itself is unaffected."
                )
        return new_state

    def _record_timing(self, previous: ConvState, new_state: ConvState, now: float) -> None:
        """Fills TurnStats from the transitions themselves, so no caller has to remember to
        time anything by hand. Called with the lock held."""
        if new_state is ConvState.USER_SPEAKING and previous is not ConvState.USER_PAUSED:
            self.turn = TurnStats(started_at=now, speech_started_at=now)
        elif new_state is ConvState.USER_PAUSED:
            self.turn.pauses += 1
        elif new_state is ConvState.USER_FINISHED:
            self.turn.finished_speaking_at = now
        elif new_state is ConvState.THINKING:
            self.turn.thinking_started_at = now
        elif new_state is ConvState.AI_SPEAKING:
            if self.turn.first_word_at is None:
                self.turn.first_word_at = now
        elif new_state is ConvState.INTERRUPTED:
            self.turn.interrupted_at = now

    def recent(self, count: int = 8) -> str:
        """The last few transitions as one arrow-separated line - what to print when something
        went wrong and the question is "how did it get here?"."""
        with self._lock:
            return " -> ".join(s.name for s, _ in self._history[-count:])

    def reset(self, note: str = "visit ended") -> None:
        """Back to IDLE between visits, from wherever it was.

        Unconditional on purpose: a visit can end from any state at all (someone walks off
        mid-answer), and a reset that could be refused would leave the next visitor's
        conversation starting from the last one's leftovers.
        """
        with self._lock:
            if self._state is not ConvState.IDLE:
                self.to(ConvState.IDLE, note)
            self.turn = TurnStats()


def _self_test() -> None:
    """Offline checks - no microphone, no models, no database."""
    print("--- conversation state machine self-test ---\n")

    seen: list[tuple[str, str]] = []
    sm = ConversationStateMachine(on_change=lambda a, b: seen.append((a.name, b.name)), debug=True)
    assert sm.state is ConvState.IDLE

    # --- the ordinary turn, including a mid-sentence pause -----------------------------------
    # "Where is the computer science... department?" - the exact case from the brief. The pause
    # must NOT produce USER_FINISHED, and the machine must come back to USER_SPEAKING.
    sm.to(ConvState.LISTENING, "visitor greeted")
    sm.to(ConvState.USER_SPEAKING, "VAD confirmed speech")
    sm.to(ConvState.USER_PAUSED, "550ms of silence")
    assert sm.state is ConvState.USER_PAUSED
    sm.to(ConvState.USER_SPEAKING, "they carried on - 'department?'")
    assert sm.turn.pauses == 1, "the mid-turn pause should have been counted, not ended the turn"
    sm.to(ConvState.USER_PAUSED, "silence again")
    sm.to(ConvState.USER_FINISHED, "endpointer committed")
    sm.to(ConvState.THINKING, "retrieval + generation")
    sm.to(ConvState.AI_SPEAKING, "first Piper chunk")
    sm.to(ConvState.LISTENING, "answer finished")
    assert sm.unexpected_transitions == 0, "the ordinary turn used only allowed edges"
    assert sm.turn.speech_duration_ms is not None
    assert sm.turn.to_first_word_ms is not None
    print(f"\nordinary turn: {sm.turn.summary()}")
    print(f"path: {sm.recent(12)}\n")

    # A repeated transition must be silent and must not count as a pause.
    pauses_before = sm.turn.pauses
    sm.to(ConvState.LISTENING, "should be a no-op")
    assert sm.turn.pauses == pauses_before

    # --- barge-in ------------------------------------------------------------------------------
    sm2 = ConversationStateMachine(debug=True)
    sm2.to(ConvState.LISTENING)
    sm2.to(ConvState.USER_SPEAKING)
    sm2.to(ConvState.USER_FINISHED)
    sm2.to(ConvState.THINKING)
    sm2.to(ConvState.AI_SPEAKING, "'The Computer Science department is located on...'")
    sm2.to(ConvState.INTERRUPTED, "'Wait, I mean the AI department'")
    sm2.to(ConvState.LISTENING, "straight back to listening, no re-prompt")
    assert sm2.unexpected_transitions == 0, "barge-in is an ordinary, expected path"
    assert sm2.turn.interrupted_at is not None
    print(f"\nbarge-in path: {sm2.recent(12)}")

    # --- an illegal edge is taken anyway, and says so ------------------------------------------
    sm3 = ConversationStateMachine(debug=True)
    sm3.to(ConvState.LISTENING)
    print("\n(the next line SHOULD be flagged UNEXPECTED - that is what is being tested)")
    sm3.to(ConvState.INTERRUPTED, "cannot be interrupted while listening")
    assert sm3.state is ConvState.INTERRUPTED, (
        "an unexpected transition must still be TAKEN - refusing it would leave the machine "
        "describing a conversation that is not happening"
    )
    assert sm3.unexpected_transitions == 1

    # --- reset works from any state ------------------------------------------------------------
    sm3.reset()
    assert sm3.state is ConvState.IDLE
    assert sm3.turn.pauses == 0

    # --- a failing callback must not break the conversation ------------------------------------
    def _explode(_a, _b):
        raise RuntimeError("the face renderer fell over")

    sm4 = ConversationStateMachine(on_change=_explode, debug=False)
    sm4.to(ConvState.LISTENING)
    assert sm4.state is ConvState.LISTENING, "a broken callback must not stop the transition"

    # --- thread safety --------------------------------------------------------------------------
    # Three threads hammering transitions, mirroring the visit/barge-in/camera threads. The
    # assertion is only that it stays internally consistent and never deadlocks.
    sm5 = ConversationStateMachine(debug=False)
    ring = [
        ConvState.LISTENING,
        ConvState.USER_SPEAKING,
        ConvState.USER_PAUSED,
        ConvState.USER_FINISHED,
        ConvState.THINKING,
        ConvState.AI_SPEAKING,
    ]

    def _churn():
        for _ in range(300):
            for state in ring:
                sm5.to(state)

    threads = [threading.Thread(target=_churn) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads), "state machine deadlocked under concurrency"
    assert sm5.state in ring
    assert len(sm5._history) <= 200, "history must stay bounded over a long visit"
    print(
        f"\nconcurrency: survived 3 threads x 1800 transitions, "
        f"history bounded at {len(sm5._history)}"
    )

    print("\nOK: all conversation state machine checks passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    _self_test()
