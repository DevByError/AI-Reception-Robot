# Measured results

Every number here was measured, not estimated. They come from a 4-core Linux machine with no
GPU, which is slower than a laptop with a GTX 1650, so the timings are pessimistic for that
laptop. Each section names the test or script that produced it, so you can re-run it on the
robot's own computer and put your numbers in the report.

What could NOT be measured here, and needs a run on the real robot: a real microphone, camera
and speaker in a real room, Windows audio drivers, the cloud language models (Groq/Gemini), Ollama,
and Whisper on a GPU.

## 1. Echo cancellation (the robot not hearing itself)

WebRTC AEC3 through `livekit` (stage04 `audio_io.py`). "ERLE" is how much quieter the robot's
own voice is after cancellation (higher is better; 30 dB = 1000x less energy).

| Test | Result |
|---|---|
| Recorded speech, simulated room, delay 40 / 120 / 250 / 400 ms, after convergence | 54 / 54 / 65 / 64 dB |
| Same, first third (still converging) | 40 / 42 / 31 / 19 dB |
| Resampling the reference with soxr vs the fixed linear resampler (40 ms lead) | 13 dB vs 45 dB |
| A visitor's voice after the robot stops | passes through at 0.6-1.4x its level |

`tests/test_audio_io.py`.

**Why the robot keeps one speaker stream open.** The first end-to-end runs opened a new
speaker stream per sentence. Each sentence then started a few milliseconds earlier or later
relative to the microphone. That shift breaks the canceller's learned filter, and on some
sentences the "cleaned" audio was LOUDER than the raw echo for half a second, so the robot
interrupted itself on "my name is Nova". The same recorded session was replayed offline both ways:

| Robot sentence (95th-percentile level after cancellation, x1000) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| new stream per sentence | 2 | 58 | 2 | (visitor talking) | 5 | 40 | 2 | 53 | 18 |
| one continuous stream | 3 | 3 | 2 | (visitor talking) | 2 | 2 | 3 | 3 | 3 |

The noise floor is 2. In the end-to-end test, the startup echo measurement dropped from 0.031
(raw echo, cancellation ineffective) to 0.0021, so barge-in needs only 0.008, which is a normal
speaking voice.

A speaker that stalls in the middle of a sentence (synthesis falling behind): echo left in the
loudest 5% of frames, raw 0.025, sample counting alone 0.0015-0.0056, with the clock-anchored
reference 0.0003-0.0005 (`test_echo_stays_cancelled_across_pauses_and_stalls`).

## 2. Voice (text to speech)

Real-time factor (RTF) = synthesis time / audio length. Below 1 is faster than real time.

| Voice | RTF (0.9 s / 2.6 s / 8 s of speech) |
|---|---|
| Kokoro fp32 | 0.42 / 0.34 / 0.34 |
| **Kokoro fp16 (used)** | 0.44 / 0.30 / 0.25 |
| Kokoro int8 | 0.95 / 1.13 / 1.09 (the int8 export is slow on CPU) |
| Piper (amy, the old voice) | 0.11 / 0.04 / 0.04 |

At robot startup, with the camera and recognizer loading, Kokoro measured 3.2x faster than real
time. `TTS_BACKEND=auto` uses Kokoro only if it is at least 2x faster than real time on the
computer it runs on, and Piper otherwise.

Fixed lines (greetings, "anything else?", consent questions) are synthesized in the background
after startup and cached, so they start instantly. The background synthesis pauses while
someone is being served and during the echo measurement.

## 3. Speech recognition

Seven local recordings (five clean, two noisy). Recordings are not published; make your own with
`stages/stage04_speech_to_text/record_sample.py` and score them with `evaluate_wer.py`.
Whisper was run as the sherpa-onnx export of small.en:

| Recognizer | Word error rate | Whole clip, CPU | Wait after the speaker stops |
|---|---|---|---|
| Whisper small.en | 18.5 % | 606-1631 ms | the whole clip (it starts after the end) |
| Zipformer streaming (sherpa-onnx, int8) | 40.7 % | 178-625 ms | 49-145 ms (already decoded while talking) |
| Zipformer + the robot's name as a hotword | 40.7 % | | 49-142 ms |

The clean clips are nearly perfect with both; the gap is the two noisy clips (Zipformer:
"THE LOW WARMTH I AM BUT ALONE" for "hello word, I am Alex") and proper nouns ("LINCOLN
UNIVERSITY COLLEACH"). The hotword makes "Nova" at the start of a sentence come out right, which
is what the wake word needs. **Recommendation: keep `STT_BACKEND=whisper` on the GTX 1650
laptop** (Whisper on the GPU is both fast and accurate) and use `sherpa` only on a computer
without a GPU or a noisy microphone.

## 4. Turn-taking

Smart-turn v3.2 (8 MB, CPU) decides whether the visitor has finished:

- Clearly finished questions: probability 0.97-0.98 after 270 ms of silence. The robot ends the
  turn there instead of waiting the old fixed 900 ms.
- Decision time 40-230 ms; the slowest is the first call after startup.
- A pause in the middle of a sentence scores low (0.35 in the listener's self-test), so the
  robot waits up to the longer limit instead. Together with the "transcript ends mid-thought"
  check ("where is the...") it keeps listening and may say a quiet "mm-hm".

## 5. Seeing who is there

| Step | Before | After |
|---|---|---|
| Finding people | YOLOv8n ~170 ms/frame (camera loop ~5 fps) | YuNet face detector ~45 ms at 640x480 |
| Face recognition per face | full-frame re-detection 55-168 ms | crop around the known face 14-35 ms |
| Robot face on screen | (none) | 2.5 ms per frame |

The camera-loop rate is capped by `VISION_FPS` (default 10). The engagement gate measures its
1-second dwell in seconds, not frames, so it works at 1-2 fps too (`tests/test_perception.py`).

## 6. Wake word (openWakeWord, stock "hey jarvis" model)

Three different voices x three sentences: detected in all 3 "Hey Jarvis, ..." sentences (score
1.00), and no false detection in the 6 sentences without it. CPU cost: 2.3-4.1% of real time.
In the end-to-end test, the robot was woken from off camera with a score of 0.77 and answered
the question that followed.

## 7. The whole robot, end to end (`tests/test_end_to_end.py`)

A simulated room: a camera showing a face photo, a microphone that hears the visitor plus the
robot's own echo (60 ms late, 35% level) plus noise, and a speaker that behaves like a sound
card. Everything else is the real code and real models: detection, recognition, engagement,
the addressee check, Zipformer, smart-turn, Kokoro, echo cancellation, barge-in, SQLite memory
and openWakeWord. The language model is a stand-in that waits 1.4 s before its first word.

The script: a visitor walks up, asks about engineering programmes, interrupts the answer with
"Wait, where is the library?", a bystander off camera says something about football, the
visitor asks "Nova, what is your name?", says "No thanks, that's all" and declines to be
remembered. A second test calls the robot with the wake word from off camera.

| Moment | Measured |
|---|---|
| Visitor stops in front of the robot -> greeting starts | 2.6 s (1 s engagement dwell + the recognition wait) |
| End of a question the LLM must answer -> first sound | 0.8-1.6 s ("Mm, okay" filler), answer 3.7-4.1 s (1.4 s is the stand-in LLM) |
| End of "where is the library" -> first word of the answer | 1.5-1.9 s |
| End of "what is your name" -> first word | 1.1-1.3 s |
| Visitor talks over the answer -> robot stops | after 330 ms of their speech, and the interrupting question is answered |
| Bystander off camera | ignored (no lips moving on the visible face) |

All latencies are measured from the moment the visitor's voice stopped, which includes the
end-of-turn wait and the speech recognition. That is the wait the visitor actually experiences.
