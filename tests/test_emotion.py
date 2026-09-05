from __future__ import annotations

import numpy as np
import pytest

from affectlab.emotion.base import normalize_probabilities, softmax
from affectlab.emotion.ensemble import EnsembleBackend
from affectlab.emotion.facs_rules import FacsRuleBackend
from affectlab.emotion.ferplus import INPUT_SIZE, preprocess
from affectlab.types import EMOTIONS, BBox, EmotionEstimate, FaceObservation


class FakeBackend:
    def __init__(self, name: str, probs: dict[str, float] | None) -> None:
        self.name = name
        self._probs = probs
        self.closed = False

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        if self._probs is None:
            return None
        return EmotionEstimate(dict(self._probs), backend=self.name, evidence={self.name: 1.0})

    def close(self) -> None:
        self.closed = True


def test_normalize_maps_deepface_labels() -> None:
    probs = normalize_probabilities({"happy": 80.0, "sad": 10.0, "angry": 10.0, "unknown": 5.0})
    assert set(probs) == set(EMOTIONS)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["happiness"] == pytest.approx(0.8)
    assert probs["sadness"] == pytest.approx(0.1)
    assert probs["anger"] == pytest.approx(0.1)


def test_normalize_empty_falls_back_to_neutral() -> None:
    assert normalize_probabilities({})["neutral"] == 1.0


def test_softmax() -> None:
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2


def test_ensemble_weights_and_skips_none() -> None:
    a = FakeBackend("a", {"happiness": 1.0})
    b = FakeBackend("b", {"sadness": 1.0})
    c = FakeBackend("c", None)
    ensemble = EnsembleBackend([(a, 3.0), (b, 1.0), (c, 5.0)])
    face = FaceObservation(bbox=BBox(0, 0, 10, 10))
    estimate = ensemble.predict(np.zeros((20, 20, 3), np.uint8), face)
    assert estimate is not None
    assert estimate.probabilities["happiness"] == pytest.approx(0.75)
    assert estimate.probabilities["sadness"] == pytest.approx(0.25)
    assert estimate.dominant == "happiness"
    assert estimate.evidence == {"a": 1.0, "b": 1.0}
    ensemble.close()
    assert a.closed and b.closed and c.closed


def test_ensemble_returns_none_when_all_backends_fail() -> None:
    ensemble = EnsembleBackend([(FakeBackend("x", None), 1.0)])
    face = FaceObservation(bbox=BBox(0, 0, 10, 10))
    assert ensemble.predict(np.zeros((20, 20, 3), np.uint8), face) is None
    with pytest.raises(ValueError):
        EnsembleBackend([])


def test_facs_backend_needs_blendshapes() -> None:
    backend = FacsRuleBackend()
    frame = np.zeros((20, 20, 3), np.uint8)
    assert backend.predict(frame, FaceObservation(bbox=BBox(0, 0, 10, 10))) is None
    face = FaceObservation(
        bbox=BBox(0, 0, 10, 10), blendshapes={"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9}
    )
    estimate = backend.predict(frame, face)
    assert estimate is not None and estimate.dominant == "happiness"


def test_ferplus_preprocess_shape_and_range() -> None:
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    tensor = preprocess(frame, BBox(40, 30, 50, 60))
    assert tensor.shape == (1, 1, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 255.0
    assert tensor.mean() == pytest.approx(200.0, abs=1.0)
