"""Shared protocol and helpers for emotion backends."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from affectlab.types import EMOTIONS, BBox, EmotionEstimate, FaceObservation


class BackendUnavailable(RuntimeError):
    """The backend's optional dependency or model is missing."""


@runtime_checkable
class EmotionBackend(Protocol):
    name: str

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        """Estimate emotion probabilities for ``face``; None if not possible."""

    def close(self) -> None: ...


#: Labels used by other libraries mapped onto the canonical set.
LABEL_ALIASES: dict[str, str] = {
    "neutral": "neutral",
    "happy": "happiness",
    "happiness": "happiness",
    "surprise": "surprise",
    "surprised": "surprise",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "scared": "fear",
    "contempt": "contempt",
}


def canonical_label(label: str) -> str | None:
    return LABEL_ALIASES.get(label.strip().lower())


def normalize_probabilities(raw: Mapping[str, float]) -> dict[str, float]:
    """Map arbitrary labels onto :data:`EMOTIONS` and rescale to sum to one."""
    acc = dict.fromkeys(EMOTIONS, 0.0)
    for label, value in raw.items():
        key = canonical_label(label)
        if key is not None and value is not None:
            acc[key] += max(0.0, float(value))
    total = sum(acc.values())
    if total <= 0.0:
        acc["neutral"] = 1.0
        total = 1.0
    return {k: v / total for k, v in acc.items()}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def face_crop(frame_bgr: np.ndarray, bbox: BBox, expand: float = 0.2) -> np.ndarray:
    """Square crop around ``bbox`` grown by ``expand`` and clipped to the frame."""
    h, w = frame_bgr.shape[:2]
    return bbox.expanded(expand, w, h, square=True).crop(frame_bgr)


def face_crop_gray(frame_bgr: np.ndarray, bbox: BBox, size: int, expand: float = 0.2) -> np.ndarray:
    crop = face_crop(frame_bgr, bbox, expand)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
