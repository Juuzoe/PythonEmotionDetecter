"""Dimensional affect and emotion dynamics.

Categorical labels are a coarse description of affect. Russell's (1980)
circumplex model places affective states on two continuous axes, valence
(unpleasant to pleasant) and arousal (deactivated to activated). Projecting
a probability distribution over emotion labels onto that plane gives a
smooth, continuous signal that can be tracked over time.

From the tracked signal we compute emotion-dynamics statistics used in
affective science: emotional inertia (the lag-1 autocorrelation of affect;
Kuppens, Allen & Sheeber, 2010), variability, and how often the dominant
category switches.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

import numpy as np

from affectlab.types import EMOTIONS, Affect, Dynamics

#: Approximate (valence, arousal) coordinates of each label on the circumplex,
#: in [-1, 1], after Russell (1980) and Posner, Russell & Peterson (2005).
#: Neutral sits at the origin by construction.
CIRCUMPLEX: dict[str, tuple[float, float]] = {
    "neutral": (0.00, 0.00),
    "happiness": (0.80, 0.45),
    "surprise": (0.15, 0.85),
    "fear": (-0.65, 0.75),
    "anger": (-0.70, 0.65),
    "disgust": (-0.70, 0.30),
    "sadness": (-0.75, -0.45),
    "contempt": (-0.55, 0.15),
}

#: Quadrant descriptors, clockwise from the top-right, for display.
QUADRANTS: dict[str, str] = {
    "high_arousal_positive": "excited / elated",
    "low_arousal_positive": "calm / content",
    "low_arousal_negative": "sad / bored",
    "high_arousal_negative": "tense / distressed",
}


def expected_affect(probabilities: Mapping[str, float]) -> tuple[float, float]:
    """Probability-weighted valence and arousal."""
    total = sum(float(probabilities.get(label, 0.0)) for label in EMOTIONS)
    if total <= 0.0:
        return 0.0, 0.0
    valence = sum(float(probabilities.get(k, 0.0)) * CIRCUMPLEX[k][0] for k in EMOTIONS) / total
    arousal = sum(float(probabilities.get(k, 0.0)) * CIRCUMPLEX[k][1] for k in EMOTIONS) / total
    return float(valence), float(arousal)


def quadrant(valence: float, arousal: float) -> str:
    if valence >= 0.0:
        return "high_arousal_positive" if arousal >= 0.0 else "low_arousal_positive"
    return "high_arousal_negative" if arousal >= 0.0 else "low_arousal_negative"


class ExponentialSmoother:
    """Time-aware exponential smoothing with time constant ``tau`` seconds."""

    def __init__(self, tau_seconds: float) -> None:
        if tau_seconds <= 0.0:
            raise ValueError("tau_seconds must be positive")
        self.tau = float(tau_seconds)
        self.value: float | None = None
        self._t: float | None = None

    def update(self, x: float, t: float) -> float:
        if self.value is None or self._t is None or t < self._t:
            self.value = float(x)
        elif t > self._t:
            alpha = 1.0 - math.exp(-(t - self._t) / self.tau)
            self.value += alpha * (float(x) - self.value)
        self._t = t
        return self.value

    def reset(self) -> None:
        self.value = None
        self._t = None


def lag_autocorrelation(x: np.ndarray, lag: int = 1) -> float | None:
    """Pearson correlation between ``x[t]`` and ``x[t + lag]``; None if undefined."""
    x = np.asarray(x, dtype=float)
    if lag < 1 or len(x) <= lag + 2:
        return None
    a = x[:-lag] - x[:-lag].mean()
    b = x[lag:] - x[lag:].mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom < 1e-12:
        return None
    return float((a * b).sum() / denom)


class AffectTracker:
    """Track affect over time and compute dynamics on a rolling window.

    Raw per-frame valence/arousal is smoothed for display with an exponential
    filter. Dynamics (inertia, variability, switch rate) are computed on the
    raw series resampled to a fixed step, so smoothing does not inflate them.

    Time only counts while updates arrive: an interval longer than
    ``gap_seconds`` (the face was lost) is not credited to any state and is
    not counted as a switch, and an interval longer than ``reset_seconds``
    restarts the smoothers and the dynamics window. Time in state
    accumulates over the whole session.
    """

    def __init__(
        self,
        tau_seconds: float = 1.5,
        window_seconds: float = 60.0,
        inertia_step_seconds: float = 1.0,
        min_inertia_samples: int = 10,
        min_rate_seconds: float = 10.0,
        gap_seconds: float = 1.0,
        reset_seconds: float = 5.0,
    ) -> None:
        self.window = float(window_seconds)
        self.step = float(inertia_step_seconds)
        self.min_inertia_samples = int(min_inertia_samples)
        self.min_rate_seconds = float(min_rate_seconds)
        self.gap_seconds = float(gap_seconds)
        self.reset_seconds = float(reset_seconds)
        self._valence = ExponentialSmoother(tau_seconds)
        self._arousal = ExponentialSmoother(tau_seconds)
        self._history: deque[tuple[float, float, float]] = deque()
        self._switches: deque[float] = deque()
        self._last_t: float | None = None
        self._last_dominant: str | None = None
        self.time_in_state: dict[str, float] = dict.fromkeys(EMOTIONS, 0.0)

    def update(self, probabilities: Mapping[str, float], t: float) -> tuple[Affect, Dynamics]:
        dt = None if self._last_t is None else t - self._last_t
        if dt is not None and (dt < 0.0 or dt > self.reset_seconds):
            self._restart()
            dt = None
        gap = dt is not None and dt > self.gap_seconds

        valence, arousal = expected_affect(probabilities)
        vs = self._valence.update(valence, t)
        ars = self._arousal.update(arousal, t)
        dominant = max(EMOTIONS, key=lambda k: float(probabilities.get(k, 0.0)))

        if dt is not None and dt > 0.0 and not gap and self._last_dominant is not None:
            self.time_in_state[self._last_dominant] += dt
        if self._last_dominant is not None and dominant != self._last_dominant and not gap:
            self._switches.append(t)

        self._history.append((t, valence, arousal))
        cutoff = t - self.window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        while self._switches and self._switches[0] < cutoff:
            self._switches.popleft()

        self._last_t = t
        self._last_dominant = dominant
        return Affect(valence, arousal, vs, ars), self.dynamics()

    def dynamics(self) -> Dynamics:
        if len(self._history) < 2:
            return Dynamics(time_in_state=dict(self.time_in_state))
        hist = np.asarray(self._history, dtype=float)
        t, v, a = hist[:, 0], hist[:, 1], hist[:, 2]
        span = float(t[-1] - t[0])

        inertia = None
        if span >= self.step * self.min_inertia_samples:
            grid = np.arange(t[0], t[-1], self.step)
            inertia = lag_autocorrelation(np.interp(grid, t, v), 1)

        switch_rate = None
        if span >= self.min_rate_seconds:
            switch_rate = len(self._switches) / span * 60.0

        return Dynamics(
            inertia=inertia,
            valence_sd=float(v.std()),
            arousal_sd=float(a.std()),
            switch_rate_per_min=switch_rate,
            time_in_state=dict(self.time_in_state),
        )

    def _restart(self) -> None:
        """Forget smoothing and window state after a long gap; keep time in state."""
        self._valence.reset()
        self._arousal.reset()
        self._history.clear()
        self._switches.clear()
        self._last_t = None
        self._last_dominant = None

    def reset(self) -> None:
        self._restart()
        self.time_in_state = dict.fromkeys(EMOTIONS, 0.0)
