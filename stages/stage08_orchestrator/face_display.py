"""
Stage 08 - the robot's face on a screen (the Expression Layer). Added 2026-10-27.

ARCHITECTURE.md always had an expression layer; until now it was a print() ("[face] ->
LISTENING"). People talk to a face much more naturally than to a box: they look at it, they
wait for it, and they can SEE whether it heard them. This draws one with OpenCV - no extra
dependency, no GPU - in its own window (FACE_DISPLAY, FACE_FULLSCREEN for the robot's screen).

What it shows, from the conversation state (ConvState) and the voice level:

  idle            eyes open, slow breathing, a blink every few seconds
  listening       eyes a little wider, a soft ring that pulses: "I'm listening"
  user speaking   ring brighter, eyes follow the visitor
  thinking        eyes glance up and to the side, three dots
  speaking        mouth opens and closes with the robot's own voice (AudioPlayer.level_at)
  interrupted     a quick widen of the eyes, then listening

`look_x` (-1..1) turns the eyes toward where the visitor is standing in the camera image, which
is a big part of "it is talking to ME".

Drawing happens on the main (camera) thread - OpenCV windows must be driven from one thread.

Run: python face_display.py     (cycles through every state; q to quit)
"""

from __future__ import annotations

import math
import random
import time

import cv2
import numpy as np

BG = (24, 18, 14)            # BGR
EYE = (245, 225, 170)
ACCENT = (230, 170, 60)
MOUTH = (220, 200, 160)


class FaceDisplay:
    def __init__(self, width: int = 800, height: int = 480, window: str = "Nova",
                 fullscreen: bool = False):
        self.width, self.height = width, height
        self.window = window
        self.fullscreen = fullscreen
        self._opened = False
        self._next_blink = time.monotonic() + random.uniform(2.5, 5.0)
        self._blink_until = 0.0
        self._mouth = 0.0           # smoothed opening, 0..1
        self._look = 0.0            # smoothed gaze, -1..1
        self._state_since = time.monotonic()
        self._state = "IDLE"

    # -- drawing -----------------------------------------------------------------------------------
    def render(self, state: str, level: float = 0.0, look_x: float = 0.0,
               now: float | None = None) -> np.ndarray:
        """The face as a BGR image. `state` is a ConvState name (or IDLE/GREETING)."""
        now = time.monotonic() if now is None else now
        if state != self._state:
            self._state, self._state_since = state, now
        in_state = now - self._state_since
        w, h = self.width, self.height
        img = np.full((h, w, 3), BG, dtype=np.uint8)

        # Gaze: follow the visitor; while thinking, glance up and to the side.
        target_x = max(-1.0, min(1.0, look_x))
        target_y = 0.0
        if state == "THINKING":
            target_x, target_y = 0.55, -0.6
        self._look += (target_x - self._look) * 0.25
        gx = int(self._look * w * 0.05)
        gy = int(target_y * h * 0.05)

        # Blinks.
        if now >= self._next_blink:
            self._blink_until = now + 0.12
            self._next_blink = now + random.uniform(2.5, 6.0)
        blinking = now < self._blink_until

        # Eye size by state.
        eye_w, eye_h = int(w * 0.085), int(h * 0.20)
        if state in ("LISTENING", "USER_SPEAKING", "USER_PAUSED"):
            eye_h = int(eye_h * 1.08)
        if state == "INTERRUPTED" and in_state < 0.4:
            eye_h = int(eye_h * 1.25)
        if state == "IDLE":
            eye_h = int(eye_h * (0.97 + 0.03 * math.sin(now * 1.3)))   # breathing
        if blinking:
            eye_h = max(4, int(eye_h * 0.08))

        cy = int(h * 0.42) + gy
        for side in (-1, 1):
            cx = w // 2 + side * int(w * 0.17) + gx
            cv2.ellipse(img, (cx, cy), (eye_w, eye_h), 0, 0, 360, EYE, -1, cv2.LINE_AA)
            if not blinking:   # a highlight makes the eyes read as eyes, not ovals
                cv2.circle(img, (cx - eye_w // 3, cy - eye_h // 3), max(3, eye_w // 5),
                           (255, 255, 255), -1, cv2.LINE_AA)

        # Mouth: follows the voice level while speaking, a calm line otherwise.
        target = min(1.0, level * 1.6) if level > 0.02 else 0.0
        self._mouth += (target - self._mouth) * (0.6 if target > self._mouth else 0.35)
        mx, my = w // 2 + gx // 2, int(h * 0.74)
        mouth_w = int(w * 0.11)
        if self._mouth > 0.05:
            cv2.ellipse(img, (mx, my), (mouth_w, max(3, int(h * 0.09 * self._mouth))), 0, 0,
                        360, MOUTH, -1, cv2.LINE_AA)
        else:
            curve = 10 if state not in ("THINKING", "USER_SPEAKING") else 4
            cv2.ellipse(img, (mx, my - curve), (mouth_w, curve), 0, 20, 160, MOUTH, 4,
                        cv2.LINE_AA)

        # Listening ring and thinking dots.
        if state in ("LISTENING", "USER_SPEAKING", "USER_PAUSED", "INTERRUPTED"):
            pulse = 0.5 + 0.5 * math.sin(now * (6.0 if state == "USER_SPEAKING" else 2.5))
            axes = (int(w * 0.40 + pulse * 6), int(h * 0.46 + pulse * 4))
            strength = 0.35 + 0.4 * pulse if state == "USER_SPEAKING" else 0.15 + 0.2 * pulse
            ring = img.copy()
            cv2.ellipse(ring, (w // 2, h // 2), axes, 0, 0, 360, ACCENT, 6, cv2.LINE_AA)
            img = cv2.addWeighted(ring, strength, img, 1 - strength, 0)
        if state == "THINKING":
            for i in range(3):
                phase = (now * 3 - i * 0.5) % 3
                r = 7 + int(4 * max(0.0, math.sin(phase)))
                cv2.circle(img, (w // 2 - 30 + i * 30, int(h * 0.90)), r, ACCENT, -1,
                           cv2.LINE_AA)
        return img

    def mouth_opening(self) -> float:
        return self._mouth

    # -- the window ----------------------------------------------------------------------------------
    def show(self, state: str, level: float = 0.0, look_x: float = 0.0) -> None:
        if not self._opened:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(self.window, self.width, self.height)
            self._opened = True
        cv2.imshow(self.window, self.render(state, level, look_x))

    def close(self) -> None:
        if self._opened:
            try:
                cv2.destroyWindow(self.window)
            except cv2.error:
                pass
            self._opened = False


def look_x_for(bbox, frame_width: int) -> float:
    """Where a face is, left-right, as -1..1 from the ROBOT's point of view. The camera image is
    not mirrored, so somebody on the camera's left is on the robot's right."""
    if not bbox or frame_width <= 0:
        return 0.0
    x, _y, w, _h = bbox
    centre = (x + w / 2.0) / frame_width
    return max(-1.0, min(1.0, (0.5 - centre) * 2.0))


if __name__ == "__main__":
    face = FaceDisplay()
    states = ["IDLE", "LISTENING", "USER_SPEAKING", "THINKING", "AI_SPEAKING", "INTERRUPTED"]
    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        state = states[int(t / 3) % len(states)]
        level = abs(math.sin(t * 9)) * 0.5 if state == "AI_SPEAKING" else 0.0
        face.show(state, level, look_x=math.sin(t * 0.7))
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
    face.close()
