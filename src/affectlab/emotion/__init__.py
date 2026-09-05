"""Interchangeable emotion backends.

* ``facs``: explainable EMFACS-style rules over Action Units (no learned model).
* ``ferplus``: the FER+ convolutional network (Barsoum et al., 2016) via ONNX Runtime.
* ``deepface``: the original DeepFace backend (optional extra, TensorFlow).
* ``ensemble``: weighted average of ``ferplus`` and ``facs`` (default).
"""

from __future__ import annotations

from affectlab.emotion.base import BackendUnavailable, EmotionBackend, normalize_probabilities
from affectlab.emotion.ensemble import EnsembleBackend
from affectlab.emotion.facs_rules import FacsRuleBackend
from affectlab.emotion.ferplus import FerPlusBackend

BACKENDS: tuple[str, ...] = ("ensemble", "ferplus", "facs", "deepface")


def create_backend(name: str = "ensemble") -> EmotionBackend:
    """Instantiate a backend by name. Raises :class:`BackendUnavailable` if deps are missing."""
    key = name.strip().lower()
    if key == "facs":
        return FacsRuleBackend()
    if key == "ferplus":
        return FerPlusBackend()
    if key == "deepface":
        from affectlab.emotion.deepface_backend import DeepFaceBackend

        return DeepFaceBackend()
    if key == "ensemble":
        return EnsembleBackend([(FerPlusBackend(), 0.6), (FacsRuleBackend(), 0.4)])
    raise ValueError(f"unknown emotion backend {name!r}; choose from {BACKENDS}")


__all__ = [
    "BACKENDS",
    "BackendUnavailable",
    "EmotionBackend",
    "EnsembleBackend",
    "FacsRuleBackend",
    "FerPlusBackend",
    "create_backend",
    "normalize_probabilities",
]
