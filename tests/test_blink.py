from __future__ import annotations

import pytest

from affectlab.blink import BlinkDetector


def ear_sequence(
    seconds: float = 30.0,
    fps: float = 30.0,
    blink_times: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0),
    blink_frames: int = 4,
    open_ear: float = 0.30,
    closed_ear: float = 0.08,
) -> list[tuple[float, float]]:
    n = int(seconds * fps)
    closed = set()
    for start in blink_times:
        first = int(start * fps)
        closed.update(range(first, first + blink_frames))
    return [(i / fps, closed_ear if i in closed else open_ear) for i in range(n)]


def test_counts_blinks_and_rate() -> None:
    detector = BlinkDetector()
    events = 0
    state = None
    for t, ear in ear_sequence():
        state = detector.update(ear, t)
        events += int(state.blink)
    assert state is not None
    assert events == 5
    assert state.blink_count == 5
    assert state.blink_rate_per_min == pytest.approx(10.0, rel=0.05)
    assert state.mean_blink_duration_ms is not None
    assert 60.0 <= state.mean_blink_duration_ms <= 200.0
    assert state.perclos is not None and state.perclos < 0.05


def test_long_closure_is_not_a_blink_but_raises_perclos() -> None:
    detector = BlinkDetector()
    fps = 30.0
    state = None
    for i in range(int(30 * fps)):
        t = i / fps
        ear = 0.08 if 10.0 <= t < 20.0 else 0.30
        state = detector.update(ear, t)
    assert state is not None
    assert state.blink_count == 0
    assert state.perclos == pytest.approx(1 / 3, abs=0.03)


def test_threshold_adapts_to_open_eye_baseline() -> None:
    detector = BlinkDetector()
    assert detector.threshold == pytest.approx(0.21)
    for i in range(60):
        detector.update(0.36, i / 30.0)
    assert detector.threshold == pytest.approx(0.27, abs=0.01)
    assert BlinkDetector(ear_threshold=0.2).threshold == 0.2


def test_direct_closure_drives_perclos() -> None:
    detector = BlinkDetector()
    fps = 30.0
    state = None
    for i in range(int(20 * fps)):
        t = i / fps
        state = detector.update(0.3, t, closure=1.0 if t < 10.0 else 0.0)
    assert state is not None
    assert state.perclos == pytest.approx(0.5, abs=0.02)


def test_single_frame_dip_is_ignored() -> None:
    detector = BlinkDetector(min_closed_frames=2)
    seq = ear_sequence(blink_frames=1)
    for t, ear in seq:
        detector.update(ear, t)
    assert detector.count == 0


def test_low_ear_subject_still_gets_a_baseline_and_blinks() -> None:
    """Open-eye EAR below the 0.21 default must not lock the detector shut."""
    detector = BlinkDetector()
    events = 0
    state = None
    for t, ear in ear_sequence(open_ear=0.19, closed_ear=0.06):
        state = detector.update(ear, t)
        events += int(state.blink)
    assert state is not None
    assert detector.baseline is not None and abs(detector.baseline - 0.19) < 0.01
    assert state.threshold < 0.19
    assert events == 5 and state.blink_count == 5
    assert not state.eyes_closed
    assert state.perclos is not None and state.perclos < 0.05


def test_face_loss_gap_is_not_counted_as_closure_or_time() -> None:
    detector = BlinkDetector()
    fps = 30.0
    state = None
    for i in range(int(20 * fps)):
        state = detector.update(0.30, i / fps)
    assert state is not None and state.perclos == 0.0
    # The face is lost for 30 s and re-detected mid-blink.
    state = detector.update(0.05, 50.0, closure=1.0)
    assert state.perclos is not None and state.perclos < 0.01
    assert detector.observed_seconds < 21.0
    # Blink rate uses observed time, not wall-clock time.
    for i in range(1, int(20 * fps)):
        state = detector.update(0.30, 50.0 + i / fps)
    assert state.blink_rate_per_min is not None
    assert state.blink_rate_per_min < 2.0
