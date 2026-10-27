# Stage 09 — Testing, Monitoring, and Deployment

This is the last stage, and it's more operational than code-heavy. It covers:

## 1) Running as a persistent service
Both start the robot with its face full screen and no camera debug window, and restart it within
5 seconds if it crashes (a clean exit, Ctrl+C, is not restarted):

- **Windows**: `start_robot.bat`. Put a shortcut to it in the Startup folder (Win+R,
  `shell:startup`) so it starts when the robot's computer logs in.
- **Linux / Jetson**: `luc-robot.service`, a systemd user service. Install steps are at the top
  of the file; the log is `journalctl --user -u luc-robot -f`.

For a robot with no screen at all, set `HEADLESS=true` (no windows; stop it with Ctrl+C or by
stopping the service).

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
