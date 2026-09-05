from __future__ import annotations

import numpy as np
import pytest

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


def test_bbox_geometry() -> None:
    box = BBox(10, 20, 30, 40)
    assert (box.x2, box.y2) == (40, 60)
    assert box.center == (25.0, 40.0)
    assert box.size == 40.0 and box.area == 1200
    mid = BBox(60, 70, 30, 40)
    grown = mid.expanded(0.5, frame_w=200, frame_h=200, square=True)
    assert grown.w == grown.h == 60
    assert grown.center == pytest.approx(mid.center)
    clipped = box.expanded(0.5, frame_w=50, frame_h=50, square=True)
    assert clipped.x >= 0 and clipped.y >= 0 and clipped.x2 <= 50 and clipped.y2 <= 50
    pts = np.array([[5.2, 6.7], [15.1, 30.2]])
    assert BBox.from_points(pts).as_tuple() == (5, 6, 11, 25)
    assert BBox.from_points(pts, frame_w=12, frame_h=12).as_tuple() == (5, 6, 7, 6)


def test_frame_result_row_is_flat_and_complete() -> None:
    face = FaceObservation(bbox=BBox(1, 2, 3, 4))
    emotion = EmotionEstimate({e: (1.0 if e == "happiness" else 0.0) for e in EMOTIONS}, "test")
    result = FrameResult(
        frame_index=7,
        timestamp=1.23456,
        face=face,
        action_units={"AU12": 0.81234},
        emotion=emotion,
        affect=Affect(0.8, 0.45, 0.4, 0.2),
        head_pose=HeadPose(1.0, 2.0, 3.0),
        vitals=Vitals(heart_rate_bpm=72.4, heart_rate_quality="good", blink_count=3),
        dynamics=Dynamics(inertia=0.5),
        fps=29.97,
    )
    row = result.to_row()
    assert row["frame"] == 7 and row["t"] == 1.2346 and row["face"] == 1
    assert row["emotion"] == "happiness" and row["confidence"] == 1.0
    assert row["p_happiness"] == 1.0 and row["p_sadness"] == 0.0
    assert row["valence"] == 0.8 and row["hr_bpm"] == 72.4 and row["blink_count"] == 3
    assert row["AU12"] == 0.8123 and row["inertia"] == 0.5
    assert all(not isinstance(v, (dict, list)) for v in row.values())


def test_frame_result_without_face() -> None:
    result = FrameResult(0, 0.0, None, {}, None, None, None, Vitals(), Dynamics(), 0.0)
    row = result.to_row()
    assert row["face"] == 0 and row["emotion"] is None and row["p_neutral"] is None
    assert result.emotion is None
