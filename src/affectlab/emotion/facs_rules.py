"""Rule-based emotion inference from Action Units (no learned model)."""

from __future__ import annotations

import numpy as np

from affectlab.facs import infer_emotion
from affectlab.types import EmotionEstimate, FaceObservation


class FacsRuleBackend:
    name = "facs"

    def __init__(self, saturation: float = 0.7, sharpness: float = 2.0) -> None:
        self.saturation = saturation
        self.sharpness = sharpness

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        if not face.blendshapes:
            return None
        return infer_emotion(face.blendshapes, saturation=self.saturation, sharpness=self.sharpness)

    def close(self) -> None:
        return None
