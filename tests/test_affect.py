from __future__ import annotations

import math

import numpy as np
import pytest

from affectlab.affect import (
    CIRCUMPLEX,
    AffectTracker,
    ExponentialSmoother,
    expected_affect,
    lag_autocorrelation,
    quadrant,
)
from affectlab.types import EMOTIONS


def test_pure_label_maps_to_its_coordinates() -> None:
    for label in EMOTIONS:
        v, a = expected_affect({label: 1.0})
        assert (v, a) == pytest.approx(CIRCUMPLEX[label])


def test_mixture_is_probability_weighted() -> None:
    v, a = expected_affect({"happiness": 0.5, "sadness": 0.5})
    assert v == pytest.approx(0.5 * (CIRCUMPLEX["happiness"][0] + CIRCUMPLEX["sadness"][0]))
    assert a == pytest.approx(0.5 * (CIRCUMPLEX["happiness"][1] + CIRCUMPLEX["sadness"][1]))


def test_empty_distribution_is_origin() -> None:
    assert expected_affect({}) == (0.0, 0.0)


def test_smoother_follows_time_constant() -> None:
    s = ExponentialSmoother(tau_seconds=1.0)
    assert s.update(1.0, 0.0) == 1.0
    assert s.update(0.0, 1.0) == pytest.approx(math.exp(-1.0))
    assert s.update(0.0, 1.0) == pytest.approx(math.exp(-1.0))  # same time: unchanged
    s.reset()
    assert s.update(5.0, 10.0) == 5.0


def test_lag_autocorrelation_cases() -> None:
    assert lag_autocorrelation(np.ones(20)) is None
    assert lag_autocorrelation(np.array([1.0, 2.0])) is None
    alternating = np.array([1.0, -1.0] * 20)
    assert lag_autocorrelation(alternating) == pytest.approx(-1.0)
    smooth = np.sin(np.linspace(0, 2 * np.pi, 200))
    assert lag_autocorrelation(smooth) > 0.95


def test_quadrants() -> None:
    assert quadrant(0.5, 0.5) == "high_arousal_positive"
    assert quadrant(0.5, -0.5) == "low_arousal_positive"
    assert quadrant(-0.5, -0.5) == "low_arousal_negative"
    assert quadrant(-0.5, 0.5) == "high_arousal_negative"


def test_tracker_dynamics_on_alternating_labels() -> None:
    tracker = AffectTracker(tau_seconds=0.5, window_seconds=60.0)
    fs = 10.0
    last_dynamics = None
    for i in range(int(40 * fs)):
        t = i / fs
        label = "happiness" if int(t) % 2 == 0 else "sadness"
        _, last_dynamics = tracker.update({label: 1.0}, t)
    assert last_dynamics is not None
    # The label flips once per second: roughly 60 switches per minute.
    assert last_dynamics.switch_rate_per_min == pytest.approx(60.0, rel=0.1)
    assert last_dynamics.inertia is not None
    assert last_dynamics.valence_sd is not None and last_dynamics.valence_sd > 0.5
    total = sum(last_dynamics.time_in_state.values())
    assert total == pytest.approx(40.0 - 1 / fs, abs=1e-6)
    assert last_dynamics.time_in_state["happiness"] == pytest.approx(20.0, abs=0.2)


def test_tracker_smoothing_lags_raw() -> None:
    tracker = AffectTracker(tau_seconds=2.0)
    tracker.update({"neutral": 1.0}, 0.0)
    affect, _ = tracker.update({"happiness": 1.0}, 0.1)
    assert affect.valence == pytest.approx(CIRCUMPLEX["happiness"][0])
    assert 0.0 < affect.valence_smoothed < affect.valence


def test_gaps_are_not_credited_and_long_gaps_restart() -> None:
    tracker = AffectTracker(tau_seconds=1.0, gap_seconds=1.0, reset_seconds=5.0)
    tracker.update({"happiness": 1.0}, 0.0)
    tracker.update({"happiness": 1.0}, 1.0)
    # A 3 s gap: not credited to any state, not counted as a switch.
    _, dynamics = tracker.update({"sadness": 1.0}, 4.0)
    assert dynamics.time_in_state["happiness"] == pytest.approx(1.0)
    assert dynamics.time_in_state["sadness"] == 0.0
    assert len(tracker._switches) == 0
    # A 5-minute absence restarts smoothing and the window but keeps time in state.
    affect, dynamics = tracker.update({"happiness": 1.0}, 304.0)
    assert affect.valence_smoothed == pytest.approx(affect.valence)
    assert dynamics.time_in_state["happiness"] == pytest.approx(1.0)
    assert dynamics.valence_sd is None
