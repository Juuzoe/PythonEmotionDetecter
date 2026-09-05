"""Facial Action Coding System (FACS) estimation from MediaPipe blendshapes.

FACS (Ekman & Friesen, 1978) describes facial behaviour as combinations of
Action Units (AUs), each tied to a muscle group. It is descriptive rather than
interpretive: AU12 means "lip corners pulled up", not "happy". MediaPipe's
Face Landmarker emits 52 ARKit-style blendshape coefficients that are
themselves FACS-inspired, so a transparent table maps them to AU intensities.

The emotion prototypes below follow the EMFACS conventions (Friesen & Ekman,
1983; Ekman, Friesen & Hager, 2002) that are widely used in the affective
computing literature. They are an explainable heuristic baseline, not a
validated diagnostic instrument: see docs/SCIENCE.md and docs/ETHICS.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from affectlab.types import EMOTIONS, EmotionEstimate


@dataclass(frozen=True, slots=True)
class ActionUnit:
    code: str
    name: str
    blendshapes: tuple[str, ...]
    #: How bilateral blendshapes are combined: "mean" or "max".
    reduce: str = "mean"


#: Mapping from FACS Action Units to MediaPipe/ARKit blendshapes.
ACTION_UNITS: tuple[ActionUnit, ...] = (
    ActionUnit("AU1", "Inner brow raiser", ("browInnerUp",)),
    ActionUnit("AU2", "Outer brow raiser", ("browOuterUpLeft", "browOuterUpRight")),
    ActionUnit("AU4", "Brow lowerer", ("browDownLeft", "browDownRight")),
    ActionUnit("AU5", "Upper lid raiser", ("eyeWideLeft", "eyeWideRight")),
    ActionUnit("AU6", "Cheek raiser", ("cheekSquintLeft", "cheekSquintRight")),
    ActionUnit("AU7", "Lid tightener", ("eyeSquintLeft", "eyeSquintRight")),
    ActionUnit("AU9", "Nose wrinkler", ("noseSneerLeft", "noseSneerRight")),
    ActionUnit("AU10", "Upper lip raiser", ("mouthUpperUpLeft", "mouthUpperUpRight")),
    ActionUnit("AU12", "Lip corner puller", ("mouthSmileLeft", "mouthSmileRight")),
    ActionUnit("AU14", "Dimpler", ("mouthDimpleLeft", "mouthDimpleRight")),
    ActionUnit("AU15", "Lip corner depressor", ("mouthFrownLeft", "mouthFrownRight")),
    ActionUnit("AU16", "Lower lip depressor", ("mouthLowerDownLeft", "mouthLowerDownRight")),
    ActionUnit("AU17", "Chin raiser", ("mouthShrugLower",)),
    ActionUnit("AU18", "Lip pucker", ("mouthPucker",)),
    ActionUnit("AU20", "Lip stretcher", ("mouthStretchLeft", "mouthStretchRight")),
    ActionUnit("AU22", "Lip funneler", ("mouthFunnel",)),
    # ARKit has no separate tightener; the pressor coefficient is the closest proxy for both.
    ActionUnit("AU23", "Lip tightener", ("mouthPressLeft", "mouthPressRight")),
    ActionUnit("AU24", "Lip pressor", ("mouthPressLeft", "mouthPressRight")),
    ActionUnit("AU26", "Jaw drop", ("jawOpen",)),
    ActionUnit("AU28", "Lip suck", ("mouthRollLower", "mouthRollUpper")),
    ActionUnit("AU29", "Jaw thrust", ("jawForward",)),
    ActionUnit("AU34", "Cheek puff", ("cheekPuff",)),
    ActionUnit("AU43", "Eyes closed", ("eyeBlinkLeft", "eyeBlinkRight")),
)

AU_BY_CODE: dict[str, ActionUnit] = {au.code: au for au in ACTION_UNITS}
AU_CODES: tuple[str, ...] = tuple(au.code for au in ACTION_UNITS)

#: EMFACS-style prototypes: which AUs, with what relative weight, characterise
#: each categorical expression. Negative weights are inhibitory: an AU that
#: contradicts the expression (a lowered brow or stretched lips rule out
#: surprise, a smile argues against sadness) scales the score down. Contempt
#: is handled separately because its signature is asymmetry (a unilateral
#: AU12/AU14), not intensity.
EMOTION_PROTOTYPES: dict[str, dict[str, float]] = {
    "happiness": {"AU6": 0.5, "AU12": 1.0, "AU15": -0.6},
    "sadness": {"AU1": 1.0, "AU4": 0.6, "AU15": 1.0, "AU17": 0.4, "AU12": -0.8},
    "surprise": {"AU1": 0.7, "AU2": 0.7, "AU5": 1.0, "AU26": 1.0, "AU4": -0.6, "AU20": -0.6},
    "fear": {
        "AU1": 0.6,
        "AU2": 0.5,
        "AU4": 0.6,
        "AU5": 1.0,
        "AU7": 0.4,
        "AU20": 1.0,
        "AU26": 0.5,
        "AU12": -0.5,
    },
    "anger": {"AU4": 1.0, "AU5": 0.6, "AU7": 0.8, "AU23": 1.0, "AU1": -0.5, "AU12": -0.5},
    "disgust": {"AU9": 1.0, "AU10": 0.6, "AU15": 0.5, "AU16": 0.4, "AU17": 0.3, "AU12": -0.3},
}


def action_units_from_blendshapes(blendshapes: Mapping[str, float]) -> dict[str, float]:
    """Convert blendshape coefficients into AU intensities in [0, 1]."""
    result: dict[str, float] = {}
    for au in ACTION_UNITS:
        values = [float(blendshapes.get(name, 0.0)) for name in au.blendshapes]
        if not values:
            continue
        value = max(values) if au.reduce == "max" else sum(values) / len(values)
        result[au.code] = min(1.0, max(0.0, value))
    return result


def asymmetry(blendshapes: Mapping[str, float], left: str, right: str) -> float:
    """Absolute left/right difference of a bilateral blendshape pair."""
    return abs(float(blendshapes.get(left, 0.0)) - float(blendshapes.get(right, 0.0)))


def infer_emotion(
    blendshapes: Mapping[str, float],
    *,
    saturation: float = 0.7,
    sharpness: float = 2.0,
    action_units: Mapping[str, float] | None = None,
) -> EmotionEstimate:
    """Rule-based, explainable emotion estimate from blendshapes.

    ``saturation`` is the AU intensity treated as fully active (blendshape
    scores rarely reach 1.0 for real faces). ``sharpness`` is an exponent that
    makes the resulting distribution more decisive. The returned
    ``evidence`` lists the AU intensities behind the dominant label.
    """
    aus = (
        dict(action_units)
        if action_units is not None
        else action_units_from_blendshapes(blendshapes)
    )

    def active(code: str) -> float:
        return min(1.0, aus.get(code, 0.0) / saturation)

    scores: dict[str, float] = {}
    for emotion, weights in EMOTION_PROTOTYPES.items():
        positive = {code: w for code, w in weights.items() if w > 0}
        support = sum(w * active(code) for code, w in positive.items()) / sum(positive.values())
        inhibition = sum(-w * active(code) for code, w in weights.items() if w < 0)
        scores[emotion] = support * max(0.0, 1.0 - inhibition)

    # Contempt: unilateral lip-corner pull or dimple, suppressed by a symmetric smile.
    unilateral = 0.7 * min(1.0, asymmetry(blendshapes, "mouthSmileLeft", "mouthSmileRight") / 0.35)
    unilateral += 0.3 * min(
        1.0, asymmetry(blendshapes, "mouthDimpleLeft", "mouthDimpleRight") / 0.35
    )
    symmetric_smile = min(
        1.0,
        min(blendshapes.get("mouthSmileLeft", 0.0), blendshapes.get("mouthSmileRight", 0.0)) / 0.5,
    )
    scores["contempt"] = unilateral * (1.0 - symmetric_smile)

    peak = max(scores.values())
    scores["neutral"] = max(0.0, 1.0 - 1.25 * peak)

    powered = {label: scores.get(label, 0.0) ** sharpness for label in EMOTIONS}
    norm = sum(powered.values()) or 1.0
    probabilities = {label: powered[label] / norm for label in EMOTIONS}

    dominant = max(probabilities, key=lambda k: probabilities[k])
    if dominant == "contempt":
        evidence = {
            "AU12 asymmetry": round(asymmetry(blendshapes, "mouthSmileLeft", "mouthSmileRight"), 3),
            "AU14 asymmetry": round(
                asymmetry(blendshapes, "mouthDimpleLeft", "mouthDimpleRight"), 3
            ),
        }
    else:
        weights = EMOTION_PROTOTYPES.get(dominant, {})
        evidence = {code: round(aus.get(code, 0.0), 3) for code, w in weights.items() if w > 0}
    return EmotionEstimate(probabilities=probabilities, backend="facs", evidence=evidence)


def describe_action_units(aus: Mapping[str, float], threshold: float = 0.2) -> list[str]:
    """Human-readable list of active AUs, strongest first, e.g. ``AU12 Lip corner puller 0.81``."""
    active = [(code, value) for code, value in aus.items() if value >= threshold]
    active.sort(key=lambda kv: kv[1], reverse=True)
    return [f"{code} {AU_BY_CODE[code].name} {value:.2f}" for code, value in active]
