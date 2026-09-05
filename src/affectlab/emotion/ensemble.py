"""Weighted ensemble of emotion backends."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from affectlab.emotion.base import EmotionBackend
from affectlab.types import EMOTIONS, EmotionEstimate, FaceObservation


class EnsembleBackend:
    name = "ensemble"

    def __init__(self, backends: Sequence[tuple[EmotionBackend, float]]) -> None:
        if not backends:
            raise ValueError("an ensemble needs at least one backend")
        self.backends = list(backends)

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        acc = dict.fromkeys(EMOTIONS, 0.0)
        total = 0.0
        evidence: dict[str, float] = {}
        for backend, weight in self.backends:
            estimate = backend.predict(frame_bgr, face)
            if estimate is None or weight <= 0.0:
                continue
            for label in EMOTIONS:
                acc[label] += weight * float(estimate.probabilities.get(label, 0.0))
            total += weight
            evidence.update(estimate.evidence)
        if total <= 0.0:
            return None
        probabilities = {label: value / total for label, value in acc.items()}
        return EmotionEstimate(probabilities=probabilities, backend=self.name, evidence=evidence)

    def close(self) -> None:
        for backend, _ in self.backends:
            backend.close()
