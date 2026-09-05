# Stage 09 — Testing, Monitoring, and Deployment

This is the last stage, and it's more operational than code-heavy. It covers:

## 1) Running as a persistent service
- Linux: a `systemd` unit that runs `stages/stage08_orchestrator/main.py` on boot.
- Watchdog: if the process crashes, restart it automatically (`Restart=always` in systemd is
  enough as a starting point).

## 2) Graceful degradation
Every module must fail safely without taking down the whole robot:
- STT didn't understand? → "Sorry, could you repeat that?" instead of crashing the program.
- Camera disconnected temporarily? → go back to IDLE, retry the camera connection every few
  seconds.
- Cloud LLM/TTS service slow or down? → a pre-written local fallback answer.

## 3) Stress testing
Run `stage08_orchestrator/main.py` in a loop calling `run_visit()` dozens of times back-to-back
with different people/questions, and confirm there's no memory leak or response-time
degradation over time.

## 4) Final success metrics (before the robot is considered "ready")
- Time from a person entering to the start of the greeting: < 1 second.
- Time from the end of a question to the start of the spoken answer: < 2-3 seconds.
- Recognition accuracy for enrolled faces: > 95% (measured manually against a real test sample).
- University data updates reflected in the knowledge base within the scheduled interval (e.g.
  every 6 hours), with no manual intervention.

See `logging_config.py` for the per-stage latency measurement tool used to evaluate these
metrics.
