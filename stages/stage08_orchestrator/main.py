"""
Stage 08 - Orchestrator

Goal: tie all the stages (01 -> 06) together in one loop that manages: presence detection, face
recognition, greeting, listening, understanding, and speaking. This is the first file that runs
the "full interaction" start to finish - the stages before this one were each tested in
isolation. Stage 07 (data pipeline) runs on its own background schedule, not as part of this
per-visit loop - it feeds the knowledge base that Stage 05 reads from here.

Note: this is still a skeleton - the listening/detection here is all Mock (same principle as
every earlier stage), but the wiring between modules is real and ready for each Mock to be
replaced with a real implementation stage by stage, without changing the loop logic itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

STAGES_DIR = Path(__file__).resolve().parent.parent
for stage_folder in [
    "stage01_presence_detection",
    "stage02_face_recognition",
    "stage03_proactive_greeting",
    "stage04_speech_to_text",
    "stage05_nlu_dialogue",
    "stage06_text_to_speech",
]:
    sys.path.append(str(STAGES_DIR / stage_folder))

from presence_detector import (
    DetectedPerson,
    MockFrameSource,
    PresenceDetector,
)  # noqa: E402
from face_memory import FaceMemoryStore, FaceRecognitionFlow  # noqa: E402
from greeting_state_machine import (  # noqa: E402
    ExpressionController,
    GreetingStateMachine,
    Person,
    RobotState,
)
from stt_listener import MockSTTEngine, STTListener, VoiceActivityDetector  # noqa: E402
from knowledge_base import KnowledgeBase, KnowledgeChunk  # noqa: E402
from dialogue_manager import DialogueManager, MockAnswerGenerator  # noqa: E402
from tts_speaker import MockTTSEngine, TTSSpeaker  # noqa: E402


class RobotOrchestrator:
    """
    The main loop. In the real implementation (Stage 09) this runs inside a persistent event
    loop (while True) connected to a real camera/mic, with interruption handling (a new person
    entering mid-interaction) - not implemented here, to keep this first integration pass
    simple.
    """

    def __init__(self):
        self.face_store = FaceMemoryStore()
        self.face_flow = FaceRecognitionFlow(self.face_store)
        self.expression = ExpressionController()
        self.tts_speaker = TTSSpeaker(MockTTSEngine())
        self.greeting_sm = GreetingStateMachine(
            self.expression, _TTSAdapter(self.tts_speaker)
        )
        self.stt = STTListener(MockSTTEngine(), VoiceActivityDetector(), locale="en-US")

        self.kb = KnowledgeBase()
        self.kb.upsert(
            KnowledgeChunk(
                chunk_id="dept-cs",
                text="The Computer Science department is on the 4th floor.",
                source_url="https://university.example.edu/departments/cs",
                last_updated=0.0,
            )
        )
        # MockAnswerGenerator - same "everything mocked, wiring real" principle as the rest of
        # this file (see class docstring). dialogue_manager_live.py has the real generators for
        # when this orchestrator itself gets a real (Stage 09) upgrade.
        self.dialogue = DialogueManager(self.kb, MockAnswerGenerator())

    def run_visit(
        self,
        faces: list[int],
        question_words: list[bytes],
        ask_consent_and_name=lambda: (True, "Guest"),
    ) -> str:
        """
        Runs one full visit end-to-end:
          1. people enter (Stage 01, mocked)
          2. recognize each face (Stage 02)
          3. group greeting (Stage 03)
          4. hear the question (Stage 04)
          5. understand + answer (Stage 05)
          6. speak the answer (Stage 06)
        Returns the final answer text (for test verification).
        """
        # 1) presence detection
        detector = PresenceDetector(
            MockFrameSource(
                [
                    [
                        DetectedPerson(
                            track_id=fid,
                            bbox=(0, 0, 100, 200),
                            distance_estimate="near",
                        )
                        for fid in faces
                    ]
                ]
            )
        )
        step = detector.step()

        # 2) face recognition
        people: list[Person] = []
        for p in step["entered"]:
            status, rec = self.face_flow.handle_face(p.track_id, ask_consent_and_name)
            name = rec.name if rec and status in ("known", "registered") else None
            people.append(Person(track_id=p.track_id, known_name=name))

        # 3) greeting
        self.greeting_sm.on_people_entered(people)
        assert self.greeting_sm.state == RobotState.LISTENING

        # 4) listening (assume enough silence right away, to keep the test simple)
        question = self.stt.listen(question_words, trailing_silence_ms=1000)

        # 5) understanding + answer
        result = self.dialogue.handle_question(question or "")

        # 6) speaking
        self.tts_speaker.speak(result.answer)

        self.greeting_sm.on_interaction_timeout_or_done()
        return result.answer


class _TTSAdapter:
    """Adapts TTSSpeaker (Stage 06) to the simple interface expected by
    GreetingStateMachine."""

    def __init__(self, speaker: TTSSpeaker):
        self.speaker = speaker

    def speak(self, text: str) -> None:
        self.speaker.speak(text)


if __name__ == "__main__":
    orchestrator = RobotOrchestrator()

    print("== Visit 1: a new student asks about the CS department ==")
    answer1 = orchestrator.run_visit(
        faces=[1],
        question_words=[
            b"where",
            b"is",
            b"the",
            b"computer",
            b"science",
            b"department",
        ],
    )
    print(f"\nFinal answer: {answer1!r}")
    assert "4th floor" in answer1

    print("\n== Visit 2: the same student returns (should be recognized) ==")
    answer2 = orchestrator.run_visit(
        faces=[1],
        question_words=[
            b"where",
            b"is",
            b"the",
            b"computer",
            b"science",
            b"department",
        ],
    )
    rec = list(orchestrator.face_store._records.values())[0]
    print(f"Visit count for face 1: {rec.visit_count} (expected 2)")
    assert rec.visit_count == 2

    print(
        "\nOK: full end-to-end integration succeeded: detect -> recognize -> greet -> listen "
        "-> understand -> speak."
    )
