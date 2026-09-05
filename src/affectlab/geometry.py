"""Facial geometry: landmark indices, eye and mouth ratios, skin regions, head pose.

Index constants refer to MediaPipe's 478-point face mesh. "Right" and "left"
mean the subject's own side; on an un-mirrored frame the subject's right eye
appears on the image's left.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from affectlab.types import BBox, HeadPose

#: Six-point eye contours in the order used by the eye aspect ratio:
#: outer corner, upper lid x2, inner corner, lower lid x2 (paired 2-6, 3-5).
EAR_RIGHT_EYE: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
EAR_LEFT_EYE: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
MOUTH_CORNERS: tuple[int, int] = (61, 291)
MOUTH_INNER_LIPS: tuple[int, int] = (13, 14)
NOSE_TIP = 1
CHIN = 152

#: Upper edge of both eyebrows, image-left to image-right, via the glabella (9).
FOREHEAD_BROW_TOP: tuple[int, ...] = (70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300)
#: Hairline arc of the face oval, image-left to image-right.
FOREHEAD_HAIRLINE: tuple[int, ...] = (103, 67, 109, 10, 338, 297, 332)
CHEEK_RIGHT: tuple[int, ...] = (116, 117, 118, 119, 100, 126, 209, 49, 129, 203, 205, 187, 123)
CHEEK_LEFT: tuple[int, ...] = (345, 346, 347, 348, 329, 355, 429, 279, 358, 423, 425, 411, 352)
FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)  # fmt: skip

#: Landmarks used for PnP head pose: nose tip, chin, eye outer corners, mouth corners.
PNP_INDICES: tuple[int, ...] = (1, 152, 33, 263, 61, 291)
#: Matching generic 3-D face model points (millimetres, nose tip at the origin).
PNP_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR of six eye landmarks (Soukupová & Čech, 2016)."""
    p = np.asarray(points, dtype=float)[:, :2]
    horizontal = float(np.linalg.norm(p[0] - p[3]))
    if horizontal < 1e-6:
        return 0.0
    vertical = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    return vertical / (2.0 * horizontal)


def ear_from_landmarks(landmarks: np.ndarray) -> float:
    """Mean EAR of both eyes."""
    lm = np.asarray(landmarks)
    return 0.5 * (
        eye_aspect_ratio(lm[list(EAR_RIGHT_EYE)]) + eye_aspect_ratio(lm[list(EAR_LEFT_EYE)])
    )


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """Inner-lip opening relative to mouth width."""
    lm = np.asarray(landmarks, dtype=float)[:, :2]
    width = float(np.linalg.norm(lm[MOUTH_CORNERS[0]] - lm[MOUTH_CORNERS[1]]))
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(lm[MOUTH_INNER_LIPS[0]] - lm[MOUTH_INNER_LIPS[1]])) / width


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points spaced evenly along its length."""
    p = np.asarray(points, dtype=float)[:, :2]
    if len(p) < 2:
        return np.repeat(p[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])])


def forehead_polygon(
    landmarks: np.ndarray, hair_fraction: float = 0.65, brow_margin: float = 0.15
) -> np.ndarray:
    """Forehead skin patch between the brows and (well below) the hairline.

    The upper edge stops at ``hair_fraction`` of the way to the hairline so a
    fringe does not contaminate the colour signal.
    """
    lm = np.asarray(landmarks, dtype=float)
    brow = lm[list(FOREHEAD_BROW_TOP), :2]
    hair = resample_polyline(lm[list(FOREHEAD_HAIRLINE), :2], len(brow))
    lower = brow + brow_margin * (hair - brow)
    upper = brow + hair_fraction * (hair - brow)
    return np.vstack([lower, upper[::-1]])


def cheek_polygons(landmarks: np.ndarray) -> list[np.ndarray]:
    lm = np.asarray(landmarks, dtype=float)
    return [lm[list(CHEEK_RIGHT), :2], lm[list(CHEEK_LEFT), :2]]


def skin_polygons(landmarks: np.ndarray) -> list[np.ndarray]:
    """Forehead and both cheeks: the regions used for rPPG."""
    return [forehead_polygon(landmarks), *cheek_polygons(landmarks)]


def polygon_mean_rgb(frame_bgr: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray | None:
    """Mean colour inside the union of ``polygons``, as [R, G, B].

    The mask is rasterised only over the polygons' bounding box, not the
    whole frame, so the cost scales with the face rather than the image.
    """
    h, w = frame_bgr.shape[:2]
    point_sets: list[np.ndarray] = []
    for poly in polygons:
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            continue
        point_sets.append(np.round(arr[:, :2]).astype(np.int32))
    if not point_sets:
        return None
    every = np.vstack(point_sets)
    x0 = int(max(0, every[:, 0].min()))
    y0 = int(max(0, every[:, 1].min()))
    x1 = int(min(w, every[:, 0].max() + 1))
    y1 = int(min(h, every[:, 1].max() + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    offset = np.array([x0, y0], dtype=np.int32)
    for pts in point_sets:
        cv2.fillPoly(mask, [(pts - offset).reshape(-1, 1, 2)], 255)
    if not mask.any():
        return None
    b, g, r, _ = cv2.mean(frame_bgr[y0:y1, x0:x1], mask=mask)
    return np.array([r, g, b], dtype=float)


def bbox_forehead_rect(bbox: BBox) -> BBox:
    """Approximate forehead rectangle when only a face box is available."""
    return BBox(
        int(bbox.x + 0.25 * bbox.w),
        int(bbox.y + 0.08 * bbox.h),
        max(1, int(0.5 * bbox.w)),
        max(1, int(0.2 * bbox.h)),
    )


def rect_mean_rgb(frame_bgr: np.ndarray, rect: BBox) -> np.ndarray | None:
    crop = rect.crop(frame_bgr)
    if crop.size == 0:
        return None
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    return mean_bgr[::-1].astype(float)


def euler_degrees(rotation: np.ndarray) -> HeadPose:
    """Decompose a rotation matrix into yaw, pitch and roll in degrees."""
    r = np.asarray(rotation, dtype=float)
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return HeadPose(yaw=math.degrees(yaw), pitch=math.degrees(pitch), roll=math.degrees(roll))


def head_pose_from_transform(transform: np.ndarray) -> HeadPose:
    """Head pose from MediaPipe's 4x4 facial transformation matrix."""
    return euler_degrees(np.asarray(transform, dtype=float)[:3, :3])


_CV_TO_MP = np.diag([1.0, -1.0, -1.0])


def head_pose_pnp(landmarks: np.ndarray, frame_w: int, frame_h: int) -> HeadPose | None:
    """Head pose by Perspective-n-Point on six landmarks (fallback path).

    solvePnP returns the model-to-camera rotation in OpenCV's camera frame
    (y down, z forward). Left-multiplying by the y/z flip expresses it in
    MediaPipe's frame (y up, z towards the camera), so a frontal face gives
    zero angles on both paths and positive yaw means the face turns towards
    the image's right.
    """
    lm = np.asarray(landmarks, dtype=float)
    if len(lm) <= max(PNP_INDICES):
        return None
    image_points = np.ascontiguousarray(lm[list(PNP_INDICES), :2], dtype=np.float64)
    focal = float(frame_w)
    camera = np.array(
        [[focal, 0.0, frame_w / 2.0], [0.0, focal, frame_h / 2.0], [0.0, 0.0, 1.0]], dtype=float
    )
    ok, rvec, _ = cv2.solvePnP(
        PNP_MODEL_POINTS, image_points, camera, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return euler_degrees(_CV_TO_MP @ rotation)


def motion_between(previous: BBox, current: BBox) -> float:
    """Face motion between frames as a fraction of face size (shift plus scale change)."""
    size = max(1.0, (previous.size + current.size) / 2.0)
    (px, py), (cx, cy) = previous.center, current.center
    shift = math.hypot(cx - px, cy - py) / size
    scale = abs(current.size - previous.size) / size
    return shift + scale
