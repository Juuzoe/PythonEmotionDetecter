from __future__ import annotations

import pytest

from affectlab.facs import (
    ACTION_UNITS,
    AU_CODES,
    EMOTION_PROTOTYPES,
    action_units_from_blendshapes,
    describe_action_units,
    infer_emotion,
)
from affectlab.types import EMOTIONS


def test_bilateral_blendshapes_are_averaged() -> None:
    aus = action_units_from_blendshapes({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.4})
    assert aus["AU12"] == pytest.approx(0.6)
    assert aus["AU1"] == 0.0


def test_every_prototype_au_is_defined() -> None:
    codes = set(AU_CODES)
    for weights in EMOTION_PROTOTYPES.values():
        assert set(weights) <= codes
    assert len({au.code for au in ACTION_UNITS}) == len(ACTION_UNITS)


def test_intensities_are_clamped() -> None:
    aus = action_units_from_blendshapes({"jawOpen": 1.7, "browInnerUp": -0.2})
    assert aus["AU26"] == 1.0
    assert aus["AU1"] == 0.0


@pytest.mark.parametrize(
    ("blendshapes", "expected"),
    [
        (
            {
                "mouthSmileLeft": 0.9,
                "mouthSmileRight": 0.9,
                "cheekSquintLeft": 0.5,
                "cheekSquintRight": 0.5,
            },
            "happiness",
        ),
        (
            {
                "browInnerUp": 0.8,
                "browOuterUpLeft": 0.7,
                "browOuterUpRight": 0.7,
                "eyeWideLeft": 0.8,
                "eyeWideRight": 0.8,
                "jawOpen": 0.7,
            },
            "surprise",
        ),
        (
            {
                "browInnerUp": 0.7,
                "browDownLeft": 0.4,
                "browDownRight": 0.4,
                "mouthFrownLeft": 0.7,
                "mouthFrownRight": 0.7,
            },
            "sadness",
        ),
        (
            {
                "browDownLeft": 0.9,
                "browDownRight": 0.9,
                "eyeSquintLeft": 0.6,
                "eyeSquintRight": 0.6,
                "mouthPressLeft": 0.7,
                "mouthPressRight": 0.7,
                "eyeWideLeft": 0.3,
                "eyeWideRight": 0.3,
            },
            "anger",
        ),
        (
            {
                "noseSneerLeft": 0.8,
                "noseSneerRight": 0.8,
                "mouthUpperUpLeft": 0.5,
                "mouthUpperUpRight": 0.5,
                "mouthFrownLeft": 0.3,
                "mouthFrownRight": 0.3,
            },
            "disgust",
        ),
        (
            {
                "browInnerUp": 0.6,
                "browOuterUpLeft": 0.5,
                "browOuterUpRight": 0.5,
                "browDownLeft": 0.4,
                "browDownRight": 0.4,
                "eyeWideLeft": 0.8,
                "eyeWideRight": 0.8,
                "mouthStretchLeft": 0.8,
                "mouthStretchRight": 0.8,
                "jawOpen": 0.4,
            },
            "fear",
        ),
        ({"mouthSmileLeft": 0.6, "mouthSmileRight": 0.0}, "contempt"),
        ({}, "neutral"),
    ],
)
def test_prototypes_yield_expected_label(blendshapes: dict[str, float], expected: str) -> None:
    estimate = infer_emotion(blendshapes)
    assert estimate.dominant == expected
    assert estimate.backend == "facs"
    assert sum(estimate.probabilities.values()) == pytest.approx(1.0)
    assert set(estimate.probabilities) == set(EMOTIONS)


def test_symmetric_smile_is_not_contempt() -> None:
    estimate = infer_emotion({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9})
    assert estimate.dominant == "happiness"
    assert estimate.probabilities["contempt"] < 0.05


def test_evidence_lists_prototype_action_units() -> None:
    estimate = infer_emotion({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9})
    assert set(estimate.evidence) == {"AU6", "AU12"}
    assert estimate.evidence["AU12"] == pytest.approx(0.9, abs=1e-3)


def test_describe_action_units_orders_by_intensity() -> None:
    lines = describe_action_units({"AU12": 0.8, "AU6": 0.3, "AU1": 0.1})
    assert lines[0].startswith("AU12 Lip corner puller 0.80")
    assert len(lines) == 2
