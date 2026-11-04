"""
Stage 08 - who is actually coming to talk to the robot.

Before this, anybody the camera saw for ~2.5 seconds was greeted - including people walking
past, people queueing behind somebody else, and people standing across the lobby. A real
receptionist (and Alexa's "is somebody addressing me" logic) waits for three signals:

  near    - close enough to talk to: face width >= ENGAGE_MIN_FACE_FRACTION of the frame width
  facing  - head turned towards the robot: nose roughly between the eyes (face_yaw)
  staying - both have held for ENGAGE_DWELL_S, so a glance while walking past does not count

Everything here is plain arithmetic on the YuNet landmarks that face detection already
produces, so it costs nothing per frame and is fully testable without a camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def face_yaw(landmarks) -> float | None:
    """How far the head is turned, from YuNet's 5 landmarks.

    YuNet order: right eye, left eye, nose tip, right mouth corner, left mouth corner. Returns
    (nose x - midpoint between the eyes x) / distance between the eyes: 0 when looking straight
    at the camera, about +/-0.5 at a 45 degree turn, larger in profile. None when the landmarks
    are missing or degenerate.
    """
    if not landmarks or len(landmarks) < 3:
        return None
    (rx, ry), (lx, ly), (nx, _ny) = landmarks[0], landmarks[1], landmarks[2]
    eye_dist = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    if eye_dist < 2:
        return None
    return (nx - (rx + lx) / 2.0) / eye_dist


def is_near(bbox, frame_width: int, min_fraction: float) -> bool:
    if not bbox or frame_width <= 0:
        return False
    return bbox[2] >= min_fraction * frame_width


def is_facing(landmarks, max_yaw: float) -> bool:
    yaw = face_yaw(landmarks)
    return yaw is not None and abs(yaw) <= max_yaw


@dataclass
class _State:
    since: float | None = None  # when near+facing started holding
    last_ok: float | None = None  # last time near+facing was observed
    bad_since: float | None = None  # start of the current run of NOT near+facing observations


class EngagementTracker:
    """Per track: has this person been near and facing the robot for long enough?

    Tolerates short dropouts (GRACE_S) - a blink, a nod, one missed detection - without
    restarting the dwell clock; only a run of "not facing" observations longer than that resets
    it. Gaps with NO observation at all (a slow frame) never reset anything, so the gate still
    works when the camera loop drops to a few frames per second under load.
    """

    GRACE_S = 0.5
    STALE_S = 2.0  # an engagement older than this with no fresh "ok" observation has lapsed

    def __init__(self, min_face_fraction: float, max_yaw: float, dwell_s: float):
        self.min_face_fraction = min_face_fraction
        self.max_yaw = max_yaw
        self.dwell_s = dwell_s
        self._states: dict[int, _State] = {}

    def observe(
        self, track_id: int, bbox, landmarks, frame_width: int, now: float | None = None
    ) -> bool:
        """Feed one observation of a tracked face. Returns whether this track is engaged now."""
        now = time.monotonic() if now is None else now
        state = self._states.setdefault(track_id, _State())
        ok = is_near(bbox, frame_width, self.min_face_fraction) and is_facing(
            landmarks, self.max_yaw
        )
        if ok:
            if state.since is None:
                state.since = now
            state.bad_since = None
            state.last_ok = now
        else:
            if state.bad_since is None:
                state.bad_since = now
            if now - state.bad_since > self.GRACE_S:
                state.since = None
        return self.is_engaged(track_id, now)

    def is_engaged(self, track_id: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        state = self._states.get(track_id)
        if state is None or state.since is None or state.last_ok is None:
            return False
        if now - state.last_ok > self.STALE_S:
            return False
        if state.bad_since is not None and now - state.bad_since > self.GRACE_S:
            return False
        return now - state.since >= self.dwell_s

    def facing_now(self, track_id: int, now: float | None = None) -> bool:
        """Near and facing at the latest observation (no dwell requirement)."""
        now = time.monotonic() if now is None else now
        state = self._states.get(track_id)
        return bool(
            state
            and state.bad_since is None
            and state.last_ok is not None
            and now - state.last_ok <= self.STALE_S
        )

    def forget(self, track_id: int) -> None:
        self._states.pop(track_id, None)
