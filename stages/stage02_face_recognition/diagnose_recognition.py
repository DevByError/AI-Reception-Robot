"""
Stage 02 - why is a known face not being recognized?

"The robot doesn't remember me" has about six different causes that all look identical from
in front of the camera, and the running robot cannot tell you which one it is: it just greets
you as a stranger. This walks the whole chain in order and says where it actually breaks.

  1. Is MySQL reachable at all?                  - no database, no memory between runs
  2. Is anybody enrolled, with embeddings?       - a person row with no samples matches nothing
  3. Are the STORED samples self-consistent?     - samples that don't match each other were
                                                   captured badly and can never match you either
  4. Does the camera see a face?                 - YuNet finding nothing is a lighting/framing
                                                   problem, not a recognition one
  5. Does that face pass the quality gate?       - too small/blurry faces are SKIPPED before
                                                   matching (sface_embedder.py's is_usable), so
                                                   this looks exactly like "not recognized"
  6. Does it match anybody once it gets there?   - the only step that is actually recognition

Step 5 is the one worth knowing about, because it is silent in normal operation and it is
distance-dependent: at 640x480 a face needs to be MIN_FACE_WIDTH_PX wide to be matched at all
and MIN_ENROLL_FACE_WIDTH_PX to be stored, which is a real limit on how far away you can stand.
This prints the numbers so that's a measurement instead of a guess.

Nothing here writes to the database - it only reads, so it is safe to run against the real one.

Run:  python diagnose_recognition.py
      python diagnose_recognition.py --seconds 15          (watch for longer)
      python diagnose_recognition.py --width 1280 --height 720   (try a higher capture size)
Press 'q' in the preview window to stop early.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

import robot_db
from face_memory import FaceMemoryStore
from sface_embedder import (
    COSINE_MATCH_THRESHOLD,
    DETECTION_SCORE_THRESHOLD,
    MIN_BLUR_VARIANCE,
    MIN_ENROLL_BLUR_VARIANCE,
    MIN_ENROLL_DETECTION_SCORE,
    MIN_ENROLL_FACE_WIDTH_PX,
    MIN_FACE_WIDTH_PX,
    SFaceEmbedder,
)


def _rule(title: str) -> None:
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


def _summary(name: str, values: list[float], fmt: str = "8.2f") -> str:
    a = np.asarray(values, dtype=float)
    return (f"  {name:22s} min={a.min():{fmt}}  median={np.median(a):{fmt}}  "
            f"max={a.max():{fmt}}")


def check_database() -> tuple[object | None, list[dict]]:
    """Steps 1-3: the stored side of recognition, with no camera involved."""
    _rule("1. MySQL")
    db = robot_db.connect_or_none(verbose=True)
    if db is None:
        print("\n  VERDICT: no database. Enrolled faces are not being loaded at all, and "
              "anybody\n  registered in a session is forgotten when it exits. Fix the "
              "connection first\n  (.env in the project root) - nothing below this can work "
              "without it.")
        return None, []

    print(f"  row counts: {db.counts()}")

    _rule("2. Who is enrolled")
    people = db.load_people()
    if not people:
        print("  Nobody is enrolled. The robot has never been given a face to remember, so "
              "being\n  greeted as a stranger is correct behaviour - say yes when it asks to "
              "remember you.")
        return db, []

    for p in people:
        last = time.strftime("%Y-%m-%d %H:%M", time.localtime(p["last_seen"]))
        print(f"  {p['name']!r}: {len(p['embeddings'])} sample(s), {p['visit_count']} visit(s), "
              f"last seen {last}, consent={p['consent_given']}")
        if not p["embeddings"]:
            print("     ^ NO SAMPLES - this person can never be recognized. Re-enroll them.")

    _rule("3. Are the stored samples any good?")
    # Samples of ONE person must match EACH OTHER comfortably. If they do not, they were
    # captured across bad frames (motion blur, a half-turned head, a dark room) and no live
    # frame will match them either - the recognizer is being asked to match you against
    # something that is barely a face. This is the check that distinguishes "recognition is
    # broken" from "what was stored was never usable".
    embedder = SFaceEmbedder()
    for p in people:
        samples = p["embeddings"]
        if len(samples) < 2:
            print(f"  {p['name']!r}: only {len(samples)} sample - nothing to cross-check. One "
                  f"sample is\n     fragile; it has to match you at whatever angle you happen "
                  f"to arrive at.")
            continue
        sims = [
            embedder.similarity(samples[i], samples[j])
            for i in range(len(samples)) for j in range(i + 1, len(samples))
        ]
        worst = min(sims)
        print(f"  {p['name']!r}: samples agree with each other "
              f"min={worst:.3f} mean={np.mean(sims):.3f} (threshold {COSINE_MATCH_THRESHOLD})")
        if worst < COSINE_MATCH_THRESHOLD:
            print("     ^ SOME STORED SAMPLES DO NOT EVEN MATCH EACH OTHER. They were captured "
                  "badly.\n       Delete this person and re-enroll them in good light, close "
                  "to the camera.")
    return db, people


def watch_camera(db, people: list[dict], seconds: float, size: tuple[int, int] | None) -> None:
    """Steps 4-6: what the camera actually delivers, and what happens to it."""
    _rule("4-6. What the camera sees")
    embedder = SFaceEmbedder()
    # The same store the orchestrator builds, so find_match() here behaves exactly as it does
    # in the running robot - including the best-match selection and the ambiguity margin.
    store = FaceMemoryStore(
        is_match=embedder.is_match, score=embedder.similarity, db=db
    )

    cap = cv2.VideoCapture(0)
    if size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        print("  Could not open the camera (index 0). Nothing else can be measured.")
        return

    frames = detections = usable = enrollable = matched = 0
    widths: list[float] = []
    sharpnesses: list[float] = []
    scores: list[float] = []
    best_per_person: dict[str, list[float]] = {p["name"]: [] for p in people}
    matched_names: dict[str, int] = {}

    ok, probe = cap.read()
    if not ok:
        print("  The camera opened but returned no frames.")
        cap.release()
        return
    print(f"  capturing {probe.shape[1]}x{probe.shape[0]} for {seconds:.0f}s - "
          f"stand where you normally stand when you talk to the robot.\n")

    started = time.time()
    while time.time() - started < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        for det in embedder.detect(frame):
            detections += 1
            widths.append(det.width)
            sharpnesses.append(det.sharpness)
            scores.append(det.score)
            usable += det.is_usable
            enrollable += det.is_enrollable

            for p in people:
                if p["embeddings"]:
                    best_per_person[p["name"]].append(
                        max(embedder.similarity(det.embedding, s) for s in p["embeddings"])
                    )

            # The gate the orchestrator applies before it will even look somebody up. A face
            # that fails here is never compared against anybody, which on screen is
            # indistinguishable from being compared and not matching.
            if not det.is_usable:
                colour, label = (0, 165, 255), det.quality_reason()
            else:
                match = store.find_match(det.embedding)
                if match is None:
                    colour, label = (200, 200, 200), "usable, but matches nobody"
                else:
                    matched += 1
                    matched_names[match.name] = matched_names.get(match.name, 0) + 1
                    colour, label = (0, 220, 0), f"MATCH: {match.name}"

            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{det.width}px sharp {det.sharpness:.0f} conf {det.score:.2f}",
                        (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

        cv2.imshow("face recognition diagnostic (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    _rule("Results")
    print(f"  frames read            {frames}")
    print(f"  faces detected         {detections}")
    print(f"  passed the quality gate{usable:>6}   (good enough to MATCH against)")
    print(f"  good enough to enroll  {enrollable:>6}   (good enough to STORE as a sample)")
    print(f"  actually matched       {matched:>6}")

    if detections:
        print()
        print(_summary("face width (px)", widths))
        print(_summary("sharpness", sharpnesses))
        print(_summary("detector confidence", scores))
        print(f"\n  gates: match needs width>={MIN_FACE_WIDTH_PX} sharp>={MIN_BLUR_VARIANCE:.0f} "
              f"conf>={DETECTION_SCORE_THRESHOLD}")
        print(f"         enroll needs width>={MIN_ENROLL_FACE_WIDTH_PX} "
              f"sharp>={MIN_ENROLL_BLUR_VARIANCE:.0f} conf>={MIN_ENROLL_DETECTION_SCORE}")
        for name, sims in best_per_person.items():
            if sims:
                print(_summary(f"similarity to {name}", sims, "8.3f"))

    _verdict(detections, usable, matched, widths, sharpnesses, people, matched_names)


def _verdict(detections, usable, matched, widths, sharpnesses, people, matched_names) -> None:
    """Names the single most likely blocker, rather than leaving the numbers to be interpreted."""
    _rule("Verdict")
    if not detections:
        print("  No face was detected at all, so recognition never even started. That is a\n"
              "  framing or lighting problem: check the camera is pointed at you, that you "
              "are lit\n  from the front rather than backlit by a window, and that nothing is "
              "covering the lens.")
        print("\n  NOTE: YuNet will also report confident 'faces' on background clutter - if "
              "the\n  numbers above look like a face when nobody is in frame, that is what "
              "you are\n  seeing, and the quality gate is what stops the robot matching "
              "against it.")
        return

    if not usable:
        reasons = []
        if np.median(widths) < MIN_FACE_WIDTH_PX:
            reasons.append(
                f"TOO FAR AWAY. Your face measures about {np.median(widths):.0f}px wide and the "
                f"matcher\n     needs {MIN_FACE_WIDTH_PX}px (enrolling needs "
                f"{MIN_ENROLL_FACE_WIDTH_PX}px). Stand closer, or re-run with\n     "
                f"--width 1280 --height 720: at double the capture size the same distance "
                f"gives\n     double the face pixels."
            )
        if np.median(sharpnesses) < MIN_BLUR_VARIANCE:
            reasons.append(
                f"TOO BLURRY. Sharpness is about {np.median(sharpnesses):.0f} and the matcher "
                f"needs {MIN_BLUR_VARIANCE:.0f}.\n     Usually too little light (the webcam "
                f"drops its shutter speed and every small\n     movement smears), or motion - "
                f"hold still and turn a light on."
            )
        print("  Every face was REJECTED BEFORE MATCHING, so the robot never looked you up.\n"
              "  This is the cause that looks exactly like a broken database but is not:")
        for r in reasons or ["the quality gate rejected them - see the numbers above."]:
            print(f"   - {r}")
        return

    if matched:
        who = ", ".join(f"{n} x{c}" for n, c in matched_names.items())
        rate = matched / usable
        print(f"  Recognition WORKS - {matched}/{usable} usable faces matched ({who}).")
        if rate < 0.6:
            print(f"\n  But only {rate:.0%} of usable frames matched, which is marginal. The "
                  f"orchestrator\n  needs IDENTITY_VOTE_REQUIRED separate recognitions to agree "
                  f"before it will use a\n  name, so at this hit rate it will sometimes greet "
                  f"you as a stranger anyway.\n  Re-enrolling in better light would raise it.")
        return

    print("  Faces passed the quality gate but matched NOBODY. Recognition ran and said no.")
    if people:
        print("  Either the person in front of the camera is not one of the enrolled people, or\n"
              "  the stored samples no longer look like them (a new beard, glasses, very\n"
              "  different lighting from the day they enrolled). Re-enrolling is the fix.\n"
              "  Check the 'similarity to ...' numbers above: close to but under "
              f"{COSINE_MATCH_THRESHOLD} means the\n  samples are stale; near zero means it is "
              "a genuinely different face.")


def calibrate(db, people: list[dict], seconds: float, size: tuple[int, int] | None) -> None:
    """Measures what sharpness is actually good enough ON THIS CAMERA, for THIS person.

    MIN_BLUR_VARIANCE exists because an embedding taken from a smeared crop is an embedding of
    mush, and matching (or worse, enrolling) from one is how somebody gets recognized as the
    wrong person. That reasoning is sound. What it does NOT establish is that 40 is the right
    number here - it was picked once, and a threshold that rejects every frame a camera can
    produce is not protecting anything, it just switches recognition off.

    So: gather real detections of a real face, and for each one record both its sharpness and
    how well its embedding actually matches the enrolled samples. If similarity holds up at
    sharpness 25, then 25 is usable evidence and the gate is simply set too high for this
    camera. If it collapses, the gate is right and the room needs more light.

    This is the same question the enrolled samples already answer implicitly: they were captured
    before the gate existed, with no sharpness requirement at all, and they match each other
    perfectly well.
    """
    if not people:
        print("\nNobody is enrolled, so there is nothing to measure similarity against.")
        return

    _rule("Threshold calibration")
    embedder = SFaceEmbedder()
    cap = cv2.VideoCapture(0)
    if size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if not cap.isOpened():
        print("  Could not open the camera.")
        return

    ok, probe = cap.read()
    if not ok:
        print("  The camera opened but returned no frames.")
        cap.release()
        return
    print(f"  capturing {probe.shape[1]}x{probe.shape[0]} for {seconds:.0f}s.")
    print("  Look at the camera and move the way you normally would - turn your head, lean in\n"
          "  and back. The point is to collect BOTH sharp and blurry frames of the same face.\n")

    samples: list[tuple[float, float, int]] = []  # (sharpness, best similarity, width)
    started = time.time()
    while time.time() - started < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        for det in embedder.detect(frame):
            if det.width < MIN_FACE_WIDTH_PX:
                continue  # size is a separate gate; do not mix the two questions
            best = max(
                max(embedder.similarity(det.embedding, s) for s in p["embeddings"])
                for p in people if p["embeddings"]
            )
            samples.append((det.sharpness, best, det.width))
        cv2.imshow("calibration (press q to stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < 10:
        print(f"  Only {len(samples)} usable detection(s) - not enough to conclude anything.\n"
              f"  Make sure your face is in frame and at least {MIN_FACE_WIDTH_PX}px wide.")
        return

    print(f"  {len(samples)} detections.  Match threshold is {COSINE_MATCH_THRESHOLD}.\n")
    print(f"  {'sharpness band':>18} | {'n':>4} | {'median sim':>10} | {'% that would match':>18}")
    print(f"  {'-' * 18}-+-{'-' * 4}-+-{'-' * 10}-+-{'-' * 18}")

    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, 1e9)]
    recommended = None
    for lo, hi in bands:
        in_band = [(sh, sim) for sh, sim, _ in samples if lo <= sh < hi]
        if not in_band:
            continue
        sims = np.array([sim for _, sim in in_band])
        hit_rate = float((sims >= COSINE_MATCH_THRESHOLD).mean())
        label = f"{lo:.0f}-{hi:.0f}" if hi < 1e9 else f"{lo:.0f}+"
        print(f"  {label:>18} | {len(in_band):>4} | {np.median(sims):>10.3f} | "
              f"{hit_rate * 100:>17.0f}%")
        # The lowest band where the great majority of frames still match is where the real
        # floor is: below it the embedding has stopped carrying the face.
        if hit_rate >= 0.8 and recommended is None:
            recommended = lo

    print()
    if recommended is None:
        print("  NO sharpness band matched reliably. That is a genuine image-quality problem,\n"
              "  not a threshold that is set too high - more light is the fix, and lowering\n"
              "  MIN_BLUR_VARIANCE would only let bad embeddings through.")
    elif recommended >= MIN_BLUR_VARIANCE:
        print(f"  Matching only becomes reliable at sharpness {recommended:.0f}+, which is at or\n"
              f"  above the current MIN_BLUR_VARIANCE of {MIN_BLUR_VARIANCE:.0f}. The gate is set\n"
              f"  correctly - the room needs more light.")
    else:
        print(f"  Frames from sharpness {recommended:.0f} upward already match reliably, while\n"
              f"  MIN_BLUR_VARIANCE is {MIN_BLUR_VARIANCE:.0f} - so usable frames are being thrown away.\n"
              f"  Lowering MIN_BLUR_VARIANCE in sface_embedder.py to about {recommended:.0f} would let\n"
              f"  this camera recognize you, on this evidence.\n\n"
              f"  Leave MIN_ENROLL_BLUR_VARIANCE ({MIN_ENROLL_BLUR_VARIANCE:.0f}) alone unless it also\n"
              f"  blocks everything: a bad frame used for MATCHING costs one bad moment, while a bad\n"
              f"  frame STORED as a reference sample degrades every future comparison permanently.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="how long to watch the camera (default 10)")
    parser.add_argument("--width", type=int, help="capture width to request")
    parser.add_argument("--height", type=int, help="capture height to request")
    parser.add_argument("--no-camera", action="store_true",
                        help="only run the database checks (steps 1-3)")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure what sharpness is actually good enough on this camera, "
                             "instead of trusting MIN_BLUR_VARIANCE")
    args = parser.parse_args()

    # Default to the SAME capture size the orchestrator asks for (main_live.py's
    # CAPTURE_WIDTH/HEIGHT). Measuring at the camera's 640x480 default would answer a question
    # nobody is asking: face size and sharpness both change with resolution, so a threshold
    # calibrated at one is meaningless at the other. Not imported from stage08 - this file is
    # deliberately runnable on its own - so keep the two in step by hand if you change them.
    size = (1280, 720)
    if args.width and args.height:
        size = (args.width, args.height)
    elif args.width or args.height:
        parser.error("--width and --height must be given together")

    db, people = check_database()
    try:
        if args.calibrate:
            calibrate(db, people, max(args.seconds, 15.0), size)
        elif not args.no_camera:
            watch_camera(db, people, args.seconds, size)
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    sys.exit(main())
