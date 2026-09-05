"""DeepFace backend (optional): ``pip install "affectlab[deepface]"``.

Kept for continuity with the original project and for comparison against the
lighter backends. DeepFace's own face detection is skipped because AffectLab
already knows where the face is.
"""

from __future__ import annotations

import logging

import numpy as np

from affectlab.emotion.base import BackendUnavailable, face_crop, normalize_probabilities
from affectlab.types import EmotionEstimate, FaceObservation

log = logging.getLogger(__name__)


class DeepFaceBackend:
    name = "deepface"

    def __init__(self) -> None:
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise BackendUnavailable(
                'deepface is not installed; run: pip install "affectlab[deepface]"'
            ) from exc
        self._deepface = DeepFace

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        crop = face_crop(frame_bgr, face.bbox, expand=0.3)
        if crop.size == 0:
            return None
        try:
            results = self._deepface.analyze(
                crop,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
        except Exception as exc:  # DeepFace raises plain Exceptions on bad input
            log.debug("DeepFace analyze failed: %s", exc)
            return None
        result = results[0] if isinstance(results, list) else results
        raw = result.get("emotion") or {}
        if not raw:
            return None
        return EmotionEstimate(probabilities=normalize_probabilities(raw), backend=self.name)

    def close(self) -> None:
        return None
