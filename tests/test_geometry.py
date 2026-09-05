from __future__ import annotations

import math

import numpy as np
import pytest

from affectlab.geometry import (
    EAR_LEFT_EYE,
    EAR_RIGHT_EYE,
    PNP_INDICES,
    PNP_MODEL_POINTS,
    bbox_forehead_rect,
    ear_from_landmarks,
    euler_degrees,
    eye_aspect_ratio,
    forehead_polygon,
    head_pose_from_transform,
    head_pose_pnp,
    motion_between,
    polygon_mean_rgb,
    rect_mean_rgb,
    resample_polyline,
    skin_polygons,
)
from affectlab.types import BBox


def eye_points(openness: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, -openness],
            [2.0, -openness],
            [3.0, 0.0],
            [2.0, openness],
            [1.0, openness],
        ]
    )


def test_eye_aspect_ratio() -> None:
    assert eye_aspect_ratio(eye_points(0.45)) == pytest.approx(0.3)
    assert eye_aspect_ratio(eye_points(0.0)) == 0.0
    degenerate = np.zeros((6, 2))
    assert eye_aspect_ratio(degenerate) == 0.0


def test_ear_from_landmarks_averages_both_eyes() -> None:
    lm = np.zeros((478, 3))
    lm[list(EAR_RIGHT_EYE), :2] = eye_points(0.45)
    lm[list(EAR_LEFT_EYE), :2] = eye_points(0.15) + 10.0
    assert ear_from_landmarks(lm) == pytest.approx(0.2)


def rotation_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_euler_decomposition() -> None:
    pose = euler_degrees(np.eye(3))
    assert (pose.yaw, pose.pitch, pose.roll) == pytest.approx((0.0, 0.0, 0.0))
    assert euler_degrees(rotation_y(20.0)).yaw == pytest.approx(20.0)
    assert euler_degrees(rotation_x(15.0)).pitch == pytest.approx(15.0)
    assert euler_degrees(rotation_z(-10.0)).roll == pytest.approx(-10.0)
    transform = np.eye(4)
    transform[:3, :3] = rotation_y(-30.0)
    assert head_pose_from_transform(transform).yaw == pytest.approx(-30.0)


def project(points_model: np.ndarray, rotation_head: np.ndarray, w: int, h: int) -> np.ndarray:
    """Project model points through a head rotation into an OpenCV image."""
    flip = np.diag([1.0, -1.0, -1.0])  # y-up/z-towards-camera to y-down/z-forward
    cam = (flip @ rotation_head @ points_model.T).T + np.array([0.0, 0.0, 1000.0])
    f = float(w)
    return np.column_stack([f * cam[:, 0] / cam[:, 2] + w / 2, f * cam[:, 1] / cam[:, 2] + h / 2])


@pytest.mark.parametrize("yaw", [0.0, 20.0, -25.0])
def test_head_pose_pnp_matches_synthetic_rotation(yaw: float) -> None:
    w, h = 640, 480
    lm = np.zeros((478, 3))
    lm[list(PNP_INDICES), :2] = project(PNP_MODEL_POINTS, rotation_y(yaw), w, h)
    pose = head_pose_pnp(lm, w, h)
    assert pose is not None
    assert pose.yaw == pytest.approx(yaw, abs=1.5)
    assert pose.pitch == pytest.approx(0.0, abs=1.5)
    assert pose.roll == pytest.approx(0.0, abs=1.5)


def test_resample_polyline_is_evenly_spaced() -> None:
    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    pts = resample_polyline(line, 5)
    assert np.allclose(pts[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0])
    assert np.allclose(pts[:, 1], 0.0)


def test_motion_between_boxes() -> None:
    a = BBox(100, 100, 100, 100)
    assert motion_between(a, a) == 0.0
    assert motion_between(a, BBox(150, 100, 100, 100)) == pytest.approx(0.5)
    assert motion_between(a, BBox(100, 100, 120, 120)) == pytest.approx(
        (math.hypot(10, 10) + 20) / 110
    )


def test_polygon_and_rect_mean_rgb() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 10  # blue
    frame[:, :, 1] = 20  # green
    frame[:, :, 2] = 30  # red
    frame[40:60, 40:60] = (0, 0, 255)
    square = np.array([[40, 40], [59, 40], [59, 59], [40, 59]], dtype=float)
    rgb = polygon_mean_rgb(frame, [square])
    assert rgb is not None
    assert rgb[0] > 240 and rgb[1] < 5 and rgb[2] < 5  # returned as R, G, B
    assert polygon_mean_rgb(frame, [np.zeros((0, 2))]) is None
    rect_rgb = rect_mean_rgb(frame, BBox(0, 0, 10, 10))
    assert rect_rgb is not None and np.allclose(rect_rgb, [30, 20, 10])
    forehead = bbox_forehead_rect(BBox(100, 100, 200, 200))
    assert forehead.x == 150 and forehead.y == 116 and forehead.w == 100 and forehead.h == 40


def test_forehead_polygon_stays_below_hairline() -> None:
    lm = np.zeros((478, 3))
    from affectlab.geometry import FOREHEAD_BROW_TOP, FOREHEAD_HAIRLINE

    for i, idx in enumerate(FOREHEAD_BROW_TOP):
        lm[idx, :2] = (100 + 10 * i, 200.0)
    for i, idx in enumerate(FOREHEAD_HAIRLINE):
        lm[idx, :2] = (100 + 16 * i, 100.0)
    poly = forehead_polygon(lm, hair_fraction=0.5, brow_margin=0.1)
    assert poly.shape == (2 * len(FOREHEAD_BROW_TOP), 2)
    assert poly[:, 1].min() == pytest.approx(150.0)  # halfway to the hairline
    assert poly[:, 1].max() == pytest.approx(190.0)  # slightly above the brows
    assert len(skin_polygons(lm)) == 3


def test_polygon_mean_rgb_handles_partly_outside_and_degenerate_polygons() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    outside = np.array([[-20.0, -20.0], [30.0, -20.0], [30.0, 30.0], [-20.0, 30.0]])
    rgb = polygon_mean_rgb(frame, [outside])
    assert rgb is not None and np.allclose(rgb, [0, 200, 0])
    assert polygon_mean_rgb(frame, [np.array([[1.0, 1.0], [2.0, 2.0]])]) is None
    assert polygon_mean_rgb(frame, [np.array([[60.0, 60.0], [70.0, 60.0], [70.0, 70.0]])]) is None
