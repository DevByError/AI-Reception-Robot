"""
Stage 09 - Logging & Monitoring

Goal: log every interaction (without raw biometric data - see docs/PRIVACY.md) for quality
purposes, and measure latency per stage to know which part of the real system is slow.

TODO (real implementation later):
  - send logs to a rotating file or a centralized monitoring system.
  - add alerting if any stage's latency exceeds a threshold (e.g. STT > 3s).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class InteractionLog:
    """
    One interaction's log record - stores only what's needed for quality purposes, no raw
    images/face embeddings. person_id here is an internal identifier (uuid), not a name or
    direct identifying data in the log itself.
    """
    interaction_id: str
    person_ids_seen: list[str]
    question_text: str | None
    answer_given: str
    intent: str
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    had_error: bool = False


class LatencyTracker:
    """A simple tool to measure the time each stage takes within one interaction."""

    def __init__(self):
        self.measurements: dict[str, float] = {}

    @contextmanager
    def measure(self, stage_name: str):
        start = time.monotonic()
        try:
            yield
        finally:
            self.measurements[stage_name] = (time.monotonic() - start) * 1000


if __name__ == "__main__":
    tracker = LatencyTracker()

    with tracker.measure("stt"):
        time.sleep(0.01)  # simulate processing time
    with tracker.measure("dialogue"):
        time.sleep(0.02)
    with tracker.measure("tts"):
        time.sleep(0.01)

    log = InteractionLog(
        interaction_id="test-1",
        person_ids_seen=["anon-123"],
        question_text="Where is the computer science department?",
        answer_given="The Computer Science department is on the 4th floor.",
        intent="GENERAL_INFO",
        stage_latencies_ms=tracker.measurements,
    )

    print("Interaction log:")
    print(f"  intent={log.intent}")
    print(f"  latencies_ms={ {k: round(v, 1) for k, v in log.stage_latencies_ms.items()} }")

    total = sum(log.stage_latencies_ms.values())
    print(f"  total_ms={total:.1f}")

    assert set(log.stage_latencies_ms.keys()) == {"stt", "dialogue", "tts"}
    print("\nOK: measuring per-stage latency + building an interaction log without raw "
          "biometric data works correctly.")
