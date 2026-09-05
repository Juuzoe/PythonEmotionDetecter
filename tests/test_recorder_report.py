from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from affectlab.recorder import COLUMNS, SessionRecorder, read_session
from affectlab.types import (
    EMOTIONS,
    Affect,
    BBox,
    Dynamics,
    EmotionEstimate,
    FaceObservation,
    FrameResult,
    HeadPose,
    Vitals,
)


def make_result(i: int, fps: float = 30.0) -> FrameResult:
    t = i / fps
    if i % 50 == 49:  # an occasional lost face
        return FrameResult(i, t, None, {}, None, None, None, Vitals(), Dynamics(), fps)
    label = "happiness" if (t // 5) % 2 == 0 else "sadness"
    probs = {e: 0.02 for e in EMOTIONS}
    probs[label] = 1.0 - 0.02 * (len(EMOTIONS) - 1)
    valence = 0.7 if label == "happiness" else -0.6
    hr = 70.0 + 3.0 * math.sin(t / 5.0)
    return FrameResult(
        frame_index=i,
        timestamp=t,
        face=FaceObservation(bbox=BBox(100, 80, 120, 140)),
        action_units={"AU12": 0.8 if label == "happiness" else 0.1, "AU15": 0.05},
        emotion=EmotionEstimate(probs, "test"),
        affect=Affect(valence, 0.3, valence * 0.9, 0.25),
        head_pose=HeadPose(2.0, -3.0, 0.5),
        vitals=Vitals(
            heart_rate_bpm=hr if t > 8 else None,
            heart_rate_snr_db=7.0,
            heart_rate_quality="good" if t > 8 else "collecting",
            blink_count=int(t // 4),
            blink_rate_per_min=15.0 if t > 10 else None,
            perclos=0.02 if t > 10 else None,
        ),
        dynamics=Dynamics(
            inertia=0.4 if t > 10 else None, valence_sd=0.6, switch_rate_per_min=12.0
        ),
        fps=fps,
    )


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_roundtrip(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"session{suffix}"
    with SessionRecorder(path, flush_every=7) as recorder:
        for i in range(400):
            recorder.write(make_result(i))
    assert recorder.rows == 400
    session = read_session(path)
    assert set(COLUMNS) <= set(session)
    assert len(session["t"]) == 400
    assert session["face"][49] == 0 and math.isnan(session["valence"][49])
    assert session["emotion"][0] == "happiness"
    assert session["AU12"][0] == pytest.approx(0.8)
    assert np.nanmax(session["hr_bpm"]) > 70.0
    assert session["hr_quality"][-2] == "good"  # the very last row is a lost-face frame


def test_summary_and_report(tmp_path: Path) -> None:
    from affectlab.report import build_report, render_markdown, summarize

    path = tmp_path / "session.csv"
    with SessionRecorder(path) as recorder:
        for i in range(30 * 40):
            recorder.write(make_result(i))
    session = read_session(path)
    summary = summarize(session)
    assert summary["frames"] == 1200
    assert summary["duration_s"] == pytest.approx(1199 / 30, abs=0.01)
    assert 0.95 < summary["face_coverage"] < 1.0
    assert summary["emotion_distribution"]["happiness"] == pytest.approx(0.5, abs=0.05)
    assert summary["heart_rate"]["mean"] == pytest.approx(70.0, abs=3.0)
    assert summary["blink_rate"]["mean"] == pytest.approx(15.0)
    assert summary["top_action_units"][0][0] == "AU12"
    markdown = render_markdown(summary, [])
    assert "Heart rate" in markdown and "happiness" in markdown

    pytest.importorskip("matplotlib")
    report_path = build_report(path, tmp_path / "out")
    assert report_path.exists()
    figures = sorted(p.name for p in (tmp_path / "out").glob("*.png"))
    assert {"emotion_timeline.png", "circumplex.png", "vitals.png", "action_units.png"} <= set(
        figures
    )


def test_summary_ignores_frames_without_a_face(tmp_path: Path) -> None:
    from affectlab.report import summarize

    path = tmp_path / "session.csv"
    with SessionRecorder(path) as recorder:
        for i in range(300):
            result = make_result(i)
            if i >= 150:  # a stale heart rate recorded while nobody is in front of the camera
                result = FrameResult(
                    i,
                    i / 30.0,
                    None,
                    {},
                    None,
                    None,
                    None,
                    Vitals(heart_rate_bpm=200.0, heart_rate_quality="good"),
                    Dynamics(),
                    30.0,
                )
            recorder.write(result)
    summary = summarize(read_session(path))
    assert summary["face_coverage"] == pytest.approx(0.49, abs=0.02)
    assert summary["heart_rate"] is None or summary["heart_rate"]["max"] < 100.0


def test_report_leaves_matplotlib_state_alone(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from affectlab.report import build_report

    before = dict(matplotlib.rcParams)
    path = tmp_path / "session.csv"
    with SessionRecorder(path) as recorder:
        for i in range(600):
            recorder.write(make_result(i))
    build_report(path, tmp_path / "out")
    changed = {
        k: (before[k], matplotlib.rcParams[k])
        for k in before
        if before[k] != matplotlib.rcParams[k]
    }
    assert changed == {}
