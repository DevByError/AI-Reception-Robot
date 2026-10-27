"""Stage 01/08: face-based presence, the engagement gate, and the addressee check."""

from pathlib import Path

import numpy as np
import pytest

from addressee import AddresseeJudge, FaceSample, LipHistory, find_wake_word, wake_phrases
from engagement import EngagementTracker, face_yaw, is_facing, is_near
from face_presence import FaceTracker

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FACE = ROOT / "tests" / "data" / "face.jpg"

STRAIGHT = ((100, 100), (140, 100), (120, 125), (105, 145), (135, 145))
TURNED = ((100, 100), (140, 100), (142, 125), (105, 145), (135, 145))


def test_yaw_is_zero_when_looking_straight_and_large_when_turned():
    assert face_yaw(STRAIGHT) == pytest.approx(0.0)
    assert face_yaw(TURNED) == pytest.approx(0.55)
    assert face_yaw(()) is None
    assert is_facing(STRAIGHT, 0.35) and not is_facing(TURNED, 0.35)


def test_near_is_relative_to_frame_width():
    assert is_near((0, 0, 70, 90), 1280, 0.05)
    assert not is_near((0, 0, 40, 50), 1280, 0.05)


def test_engagement_needs_dwell_and_survives_a_blink():
    e = EngagementTracker(min_face_fraction=0.05, max_yaw=0.35, dwell_s=1.0)
    box = (500, 300, 100, 120)
    assert not e.observe(1, box, STRAIGHT, 1280, now=0.0)
    assert not e.observe(1, box, STRAIGHT, 1280, now=0.6)
    assert not e.observe(1, box, TURNED, 1280, now=0.8)      # a glance away inside the grace
    assert e.observe(1, box, STRAIGHT, 1280, now=1.05)
    # a passer-by: far away, never engaged however long they are visible
    for t in range(0, 30):
        assert not e.observe(2, (10, 10, 30, 40), STRAIGHT, 1280, now=t / 10)
    # someone walking past side-on
    for t in range(0, 30):
        assert not e.observe(3, box, TURNED, 1280, now=t / 10)


def test_engagement_restarts_after_turning_away_for_long():
    e = EngagementTracker(0.05, 0.35, 1.0)
    box = (500, 300, 100, 120)
    e.observe(1, box, STRAIGHT, 1280, now=0.0)
    e.observe(1, box, TURNED, 1280, now=0.9)
    e.observe(1, box, TURNED, 1280, now=1.6)                 # away for > grace
    assert not e.observe(1, box, STRAIGHT, 1280, now=1.7)
    assert e.observe(1, box, STRAIGHT, 1280, now=2.8)


def test_face_tracker_keeps_ids_through_small_moves_and_misses():
    t = FaceTracker()
    a = t.update([(100, 100, 80, 100), (600, 100, 80, 100)])
    b = t.update([(110, 105, 80, 100), (590, 100, 80, 100)])
    assert a == b
    t.update([(115, 105, 80, 100)])                          # second face missed one frame
    c = t.update([(118, 105, 80, 100), (595, 100, 80, 100)])
    assert c == a
    d = t.update([(118, 105, 80, 100), (1000, 400, 80, 100)])
    assert d[0] == a[0] and d[1] not in a                   # a new person gets a new id


@pytest.mark.skipif(not SAMPLE_FACE.exists(), reason="tests/data/face.jpg missing")
def test_yunet_presence_finds_a_real_face():
    import cv2
    from face_presence import FacePresenceDetector

    img = cv2.imread(str(SAMPLE_FACE))
    canvas = np.full((720, 1280, 3), 90, np.uint8)
    canvas[160:560, 440:840] = cv2.resize(img, (400, 400))
    people = FacePresenceDetector().detect_people(canvas)
    assert len(people) == 1 and people[0].distance_estimate == "near"
    assert abs(face_yaw(people[0].landmarks)) < 0.4
    mirrored = FacePresenceDetector().detect_people(canvas[:, ::-1].copy())
    assert face_yaw(people[0].landmarks) * face_yaw(mirrored[0].landmarks) < 0


def test_wake_word_found_and_removed():
    phrases = wake_phrases("Nova", "excuse me")
    assert find_wake_word("Hey Nova, where is the library?", phrases) == (True, "where is the library?")
    assert find_wake_word("where's the library, Nova?", phrases) == (True, "where's the library")
    assert find_wake_word("Excuse me where is the bursary", phrases)[0]
    assert find_wake_word("I love innovation", phrases) == (False, "I love innovation")
    assert find_wake_word("nova", phrases) == (True, "")


def _history(samples_per_frame, start=0.0, step=0.1):
    h = LipHistory()
    for i, samples in enumerate(samples_per_frame):
        h.record(samples, now=start + i * step)
    return h


def test_addressee_answers_the_person_facing_and_talking():
    h = _history([[FaceSample(1, True, True), FaceSample(2, False, True)]] * 20)
    frames, faces = h.summarize(0.0, 2.0)
    d = AddresseeJudge(wake=wake_phrases("Nova")).decide("where is the library", frames, faces)
    assert d.addressed and "face 1" in d.reason


def test_addressee_ignores_friends_talking_to_each_other():
    # two people, the talker is turned towards their friend
    h = _history([[FaceSample(1, True, False), FaceSample(2, False, False)]] * 20)
    judge = AddresseeJudge(wake=wake_phrases("Nova"))
    d = judge.decide("did you finish the assignment", *h.summarize(0, 2))
    assert not d.addressed and "not facing" in d.reason
    # ...unless they use the robot's name
    d = judge.decide("Nova, did you see where the lab is", *h.summarize(0, 2))
    assert d.addressed and d.question == "did you see where the lab is"


def test_addressee_ignores_off_camera_voices_but_never_goes_deaf():
    h = _history([[FaceSample(1, False, True), FaceSample(2, False, True)]] * 20)
    judge = AddresseeJudge(max_ignored=2)
    results = [judge.decide("blah", *h.summarize(0, 2)).addressed for _ in range(3)]
    assert results == [False, False, True]


def test_addressee_without_camera_evidence():
    assert AddresseeJudge(mode="auto").decide("hello", 0, []).addressed
    assert not AddresseeJudge(mode="strict").decide("hello", 0, []).addressed
    assert AddresseeJudge(mode="strict", wake=["nova"]).decide("nova hello", 0, []).addressed
    assert AddresseeJudge(mode="off").decide("x", 5, []).addressed


def test_only_visitor_facing_with_weak_lip_signal_is_answered():
    frames = [[FaceSample(1, i % 10 == 0, True)] for i in range(20)]   # lips seen moving 10%
    d = AddresseeJudge().decide("where is the library", *_history(frames).summarize(0, 2))
    assert d.addressed


@pytest.mark.parametrize("fps", [1.0, 2.0, 5.0])
def test_engagement_works_when_the_camera_loop_is_slow(fps):
    """On a small computer the camera loop can drop to a frame or two a second. The dwell is
    measured in seconds, not frames, and a single bad frame must not restart it."""
    e = EngagementTracker(0.05, 0.35, dwell_s=1.0)
    box = (500, 300, 100, 120)
    step = 1.0 / fps
    t, engaged_at = 0.0, None
    while t < 4.0 and engaged_at is None:
        landmarks = TURNED if abs(t - step) < 1e-9 else STRAIGHT   # one glance away
        if e.observe(1, box, landmarks, 1280, now=t):
            engaged_at = t
        t += step
    assert engaged_at is not None and engaged_at <= 1.0 + 2 * step
