"""Fallback face detection with OpenCV's YuNet when MediaPipe is unavailable.

YuNet yields boxes only, so downstream stages that need landmarks (action
units, blinks, head pose) are skipped and rPPG falls back to a forehead
rectangle inside the box.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from affectlab.models import ensure_model
from affectlab.types import BBox, FaceObservation


class YuNetDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 500,
    ) -> None:
        path = Path(model_path) if model_path else ensure_model("yunet")
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, nms_threshold, top_k
        )
        self._input_size: tuple[int, int] | None = None

    def process(
        self, frame_bgr: np.ndarray, timestamp_ms: int | None = None
    ) -> list[FaceObservation]:
        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self._detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []
        out: list[FaceObservation] = []
        for row in np.asarray(faces):
            x, y, fw, fh = (round(float(v)) for v in row[:4])
            x, y = max(0, x), max(0, y)
            fw, fh = max(1, min(fw, w - x)), max(1, min(fh, h - y))
            score = float(row[14]) if len(row) > 14 else 1.0
            out.append(FaceObservation(bbox=BBox(x, y, fw, fh), score=score, source="yunet"))
        out.sort(key=lambda f: f.bbox.area, reverse=True)
        return out

    def close(self) -> None:
        return None
