"""Core data types shared by every AffectLab module.

These are plain dataclasses with no I/O and no heavy imports, so they can be
used from tests, notebooks, the CLI and the pipeline alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Canonical emotion labels, in FER+ order. Every backend maps onto this set.
EMOTIONS: tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)


@dataclass(slots=True)
class BBox:
    """Axis-aligned face box in pixel coordinates (top-left origin)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def size(self) -> float:
        return float(max(self.w, self.h))

    @property
    def area(self) -> int:
        return self.w * self.h

    def expanded(self, factor: float, frame_w: int, frame_h: int, square: bool = False) -> BBox:
        """Return a copy grown by ``factor`` (0.2 = 20 percent) and clipped to the frame."""
        cx, cy = self.center
        w, h = self.w * (1.0 + factor), self.h * (1.0 + factor)
        if square:
            w = h = max(w, h)
        x1 = max(0, round(cx - w / 2.0))
        y1 = max(0, round(cy - h / 2.0))
        x2 = min(frame_w, round(cx + w / 2.0))
        y2 = min(frame_h, round(cy + h / 2.0))
        return BBox(x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y2, self.x : self.x2]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @classmethod
    def from_points(
        cls, points: np.ndarray, frame_w: int | None = None, frame_h: int | None = None
    ) -> BBox:
        """Tight box around ``points`` (N, >=2), optionally clipped to the frame."""
        x1, y1 = np.floor(points[:, :2].min(axis=0))
        x2, y2 = np.ceil(points[:, :2].max(axis=0))
        if frame_w is not None:
            x1, x2 = max(0.0, x1), min(float(frame_w), x2)
        if frame_h is not None:
            y1, y2 = max(0.0, y1), min(float(frame_h), y2)
        return cls(int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))


@dataclass(slots=True)
class FaceObservation:
    """Everything the face stage knows about one face in one frame.

    ``landmarks`` is an (N, 3) array of pixel x, pixel y and relative depth z
    (478 points from MediaPipe). ``blendshapes`` are the 52 ARKit-style
    coefficients in [0, 1]. ``transform`` is MediaPipe's 4x4 facial
    transformation matrix. Fallback detectors fill only ``bbox``.
    """

    bbox: BBox
    landmarks: np.ndarray | None = None
    blendshapes: dict[str, float] | None = None
    transform: np.ndarray | None = None
    score: float = 1.0
    source: str = "mediapipe"

    @property
    def has_landmarks(self) -> bool:
        return self.landmarks is not None and len(self.landmarks) >= 468


@dataclass(slots=True)
class HeadPose:
    """Head orientation in degrees (MediaPipe camera convention).

    ``yaw`` turns left/right, ``pitch`` nods up/down, ``roll`` tilts sideways.
    """

    yaw: float
    pitch: float
    roll: float


@dataclass(slots=True)
class EmotionEstimate:
    """A probability distribution over :data:`EMOTIONS` from one backend."""

    probabilities: dict[str, float]
    backend: str
    #: Optional explanation, e.g. the Action Units that drove a rule-based decision.
    evidence: dict[str, float] = field(default_factory=dict)

    @property
    def dominant(self) -> str:
        return max(self.probabilities, key=lambda k: self.probabilities[k])

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.dominant])

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:n]


@dataclass(slots=True)
class Affect:
    """Dimensional affect on the circumplex, raw and temporally smoothed, in [-1, 1]."""

    valence: float
    arousal: float
    valence_smoothed: float
    arousal_smoothed: float


@dataclass(slots=True)
class Vitals:
    """Contactless physiological and ocular measurements."""

    heart_rate_bpm: float | None = None
    heart_rate_snr_db: float | None = None
    #: One of "no signal", "collecting", "poor", "fair", "good".
    heart_rate_quality: str = "no signal"
    signal_seconds: float = 0.0
    rmssd_ms: float | None = None
    breathing_rate_bpm: float | None = None
    ear: float | None = None
    blink_count: int = 0
    blink_rate_per_min: float | None = None
    perclos: float | None = None


@dataclass(slots=True)
class Dynamics:
    """Emotion-dynamics statistics over a rolling window."""

    #: Lag-1 autocorrelation of valence at a one-second step (emotional inertia).
    inertia: float | None = None
    valence_sd: float | None = None
    arousal_sd: float | None = None
    #: Dominant-emotion switches per minute.
    switch_rate_per_min: float | None = None
    #: Seconds spent in each dominant emotion over the whole session.
    time_in_state: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FrameResult:
    """The pipeline's complete output for a single frame."""

    frame_index: int
    timestamp: float
    face: FaceObservation | None
    action_units: dict[str, float]
    emotion: EmotionEstimate | None
    affect: Affect | None
    head_pose: HeadPose | None
    vitals: Vitals
    dynamics: Dynamics
    fps: float

    def to_row(self) -> dict[str, Any]:
        """Flatten into a single dict suitable for CSV or JSON lines."""

        def r(x: float | None, nd: int = 4) -> float | None:
            return None if x is None else round(float(x), nd)

        row: dict[str, Any] = {
            "frame": self.frame_index,
            "t": r(self.timestamp),
            "fps": r(self.fps, 2),
            "face": int(self.face is not None),
            "bbox_x": self.face.bbox.x if self.face else None,
            "bbox_y": self.face.bbox.y if self.face else None,
            "bbox_w": self.face.bbox.w if self.face else None,
            "bbox_h": self.face.bbox.h if self.face else None,
            "yaw": r(self.head_pose.yaw, 2) if self.head_pose else None,
            "pitch": r(self.head_pose.pitch, 2) if self.head_pose else None,
            "roll": r(self.head_pose.roll, 2) if self.head_pose else None,
            "emotion": self.emotion.dominant if self.emotion else None,
            "confidence": r(self.emotion.confidence) if self.emotion else None,
            "backend": self.emotion.backend if self.emotion else None,
        }
        for label in EMOTIONS:
            row[f"p_{label}"] = r(self.emotion.probabilities.get(label)) if self.emotion else None
        a = self.affect
        row.update(
            valence=r(a.valence) if a else None,
            arousal=r(a.arousal) if a else None,
            valence_smoothed=r(a.valence_smoothed) if a else None,
            arousal_smoothed=r(a.arousal_smoothed) if a else None,
        )
        d = self.dynamics
        row.update(
            inertia=r(d.inertia),
            valence_sd=r(d.valence_sd),
            arousal_sd=r(d.arousal_sd),
            switch_rate=r(d.switch_rate_per_min, 3),
        )
        v = self.vitals
        row.update(
            hr_bpm=r(v.heart_rate_bpm, 1),
            hr_snr_db=r(v.heart_rate_snr_db, 2),
            hr_quality=v.heart_rate_quality,
            signal_seconds=r(v.signal_seconds, 2),
            rmssd_ms=r(v.rmssd_ms, 1),
            breathing_bpm=r(v.breathing_rate_bpm, 1),
            ear=r(v.ear),
            blink_count=v.blink_count,
            blink_rate=r(v.blink_rate_per_min, 2),
            perclos=r(v.perclos),
        )
        for code, value in self.action_units.items():
            row[code] = r(value)
        return row
