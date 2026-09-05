"""Blink detection and eyelid-closure metrics.

The eye aspect ratio (EAR; Soukupová & Čech, 2016) is the ratio of the eye's
vertical to horizontal landmark distances. It is nearly constant while the
eye is open and drops towards zero during a blink, independent of face size
and distance. Blinks are short closures, typically 100 to 400 ms.

Two summary statistics are exposed:

* **blink rate** in blinks per minute. Spontaneous blink rate at rest
  averages about 17 per minute, rises during conversation and falls during
  reading (Bentivoglio et al., 1997), which makes it a rough proxy for
  visual attention and cognitive load.
* **PERCLOS**, the proportion of time the eyes are at least 80 percent
  closed over a window (Wierwille et al., 1994; Dinges & Grace, 1998), a
  validated drowsiness index from driver-monitoring research.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BlinkState:
    ear: float
    threshold: float
    eyes_closed: bool
    #: True on the update at which a blink was completed.
    blink: bool
    blink_count: int
    blink_rate_per_min: float | None
    perclos: float | None
    mean_blink_duration_ms: float | None


class BlinkDetector:
    """EAR-threshold blink detector with an adaptive per-person baseline.

    The open-eye baseline is a high percentile of *all* recent EAR samples
    (eyes are open most of the time, so blinks barely move it), and the
    closed threshold is ``closed_ratio`` times that baseline, clamped to a
    plausible range. Learning from every sample rather than only from frames
    already judged "open" means people with naturally low EAR values still
    get a baseline instead of being stuck permanently "closed".

    Time is only counted while the face is observed: an interval longer than
    ``gap_reset_seconds`` between updates (the face was lost) is excluded
    from the blink-rate and PERCLOS denominators and resets the closure state
    machine.
    """

    def __init__(
        self,
        *,
        ear_threshold: float | None = None,
        closed_ratio: float = 0.75,
        min_closed_frames: int = 2,
        max_blink_seconds: float = 0.5,
        window_seconds: float = 60.0,
        min_window_seconds: float = 10.0,
        baseline_samples: int = 600,
        min_baseline_samples: int = 30,
        baseline_percentile: float = 90.0,
        perclos_closure: float = 0.8,
        gap_reset_seconds: float = 1.0,
    ) -> None:
        self._fixed = ear_threshold
        self.closed_ratio = float(closed_ratio)
        self.min_closed_frames = int(min_closed_frames)
        self.max_blink_seconds = float(max_blink_seconds)
        self.window = float(window_seconds)
        self.min_window = float(min_window_seconds)
        self.min_baseline_samples = int(min_baseline_samples)
        self.baseline_percentile = float(baseline_percentile)
        self.perclos_closure = float(perclos_closure)
        self.gap_reset_seconds = float(gap_reset_seconds)
        self._ears: deque[float] = deque(maxlen=baseline_samples)
        self._blinks: deque[tuple[float, float]] = deque()
        self._closure_hist: deque[tuple[float, float, bool]] = deque()
        self._closed_since: float | None = None
        self._closed_frames = 0
        self._last_t: float | None = None
        self.count = 0

    @property
    def baseline(self) -> float | None:
        if len(self._ears) >= self.min_baseline_samples:
            return float(np.percentile(self._ears, self.baseline_percentile))
        return None

    @property
    def threshold(self) -> float:
        if self._fixed is not None:
            return float(self._fixed)
        base = self.baseline
        if base is None:
            return 0.21
        return float(min(0.30, max(0.12, base * self.closed_ratio)))

    @property
    def observed_seconds(self) -> float:
        """Seconds of face-visible time inside the current window."""
        return float(sum(dt for _, dt, _ in self._closure_hist))

    def update(self, ear: float, t: float, closure: float | None = None) -> BlinkState:
        """Feed one EAR sample at time ``t``.

        ``closure`` is an optional direct eyelid-closure fraction in [0, 1]
        (for example MediaPipe's eyeBlink blendshapes); when omitted it is
        derived from EAR relative to the baseline.
        """
        gap = self._last_t is not None and (
            t - self._last_t > self.gap_reset_seconds or t < self._last_t
        )
        if gap:
            self._closed_since = None
            self._closed_frames = 0

        self._ears.append(float(ear))
        thr = self.threshold
        closed = ear < thr
        blink = False
        if closed:
            if self._closed_since is None:
                self._closed_since = t
                self._closed_frames = 0
            self._closed_frames += 1
        elif self._closed_since is not None:
            duration = t - self._closed_since
            if self._closed_frames >= self.min_closed_frames and duration <= self.max_blink_seconds:
                blink = True
                self.count += 1
                self._blinks.append((t, duration))
            self._closed_since = None
            self._closed_frames = 0

        if closure is None:
            closure = self.closure_from_ear(ear)
        if self._last_t is not None and t > self._last_t and not gap:
            self._closure_hist.append((t, t - self._last_t, closure >= self.perclos_closure))

        cutoff = t - self.window
        while self._blinks and self._blinks[0][0] < cutoff:
            self._blinks.popleft()
        while self._closure_hist and self._closure_hist[0][0] < cutoff:
            self._closure_hist.popleft()

        rate: float | None = None
        perclos: float | None = None
        observed = self.observed_seconds
        if observed >= self.min_window:
            rate = len(self._blinks) / observed * 60.0
            perclos = sum(dt for _, dt, c in self._closure_hist if c) / observed
        mean_duration = None
        if self._blinks:
            mean_duration = 1000.0 * float(np.mean([d for _, d in self._blinks]))

        self._last_t = t
        return BlinkState(
            ear=float(ear),
            threshold=thr,
            eyes_closed=closed,
            blink=blink,
            blink_count=self.count,
            blink_rate_per_min=rate,
            perclos=perclos,
            mean_blink_duration_ms=mean_duration,
        )

    def closure_from_ear(self, ear: float) -> float:
        """Approximate eyelid closure in [0, 1] from EAR and the open-eye baseline."""
        base = self.baseline or 0.28
        closed_ear = 0.33 * base
        return float(np.clip((base - ear) / max(base - closed_ear, 1e-6), 0.0, 1.0))
