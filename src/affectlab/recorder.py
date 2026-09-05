"""Session recording to CSV or JSON Lines, and reading it back."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from affectlab.facs import AU_CODES
from affectlab.types import EMOTIONS, FrameResult

#: Fixed column order so every session file has the same schema.
COLUMNS: tuple[str, ...] = (
    "frame",
    "t",
    "fps",
    "face",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "yaw",
    "pitch",
    "roll",
    "emotion",
    "confidence",
    "backend",
    *[f"p_{label}" for label in EMOTIONS],
    "valence",
    "arousal",
    "valence_smoothed",
    "arousal_smoothed",
    "inertia",
    "valence_sd",
    "arousal_sd",
    "switch_rate",
    "hr_bpm",
    "hr_snr_db",
    "hr_quality",
    "signal_seconds",
    "rmssd_ms",
    "breathing_bpm",
    "ear",
    "blink_count",
    "blink_rate",
    "perclos",
    *AU_CODES,
)
STRING_COLUMNS: frozenset[str] = frozenset({"emotion", "hr_quality", "backend"})


class SessionRecorder:
    """Append one row per frame. ``.jsonl`` paths write JSON Lines, others CSV."""

    def __init__(self, path: str | Path, flush_every: int = 30) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.path.suffix.lower() in {".jsonl", ".json"}
        self.flush_every = max(1, int(flush_every))
        self.rows = 0
        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter[str] | None = None
        if not self.jsonl:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(COLUMNS), extrasaction="ignore")
            self._writer.writeheader()

    def write(self, result: FrameResult) -> None:
        row = result.to_row()
        if self._writer is None:
            self._fh.write(json.dumps(row) + "\n")
        else:
            self._writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
        self.rows += 1
        if self.rows % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_session(path: str | Path) -> dict[str, np.ndarray]:
    """Load a session file into column arrays (float with NaN, or str)."""
    path = Path(path)
    rows: list[dict[str, Any]]
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    columns = list(COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    session: dict[str, np.ndarray] = {}
    for column in columns:
        if column in STRING_COLUMNS:
            session[column] = np.array(
                [("" if r.get(column) is None else str(r.get(column))) for r in rows], dtype=object
            )
        else:
            session[column] = np.array([_to_float(r.get(column)) for r in rows], dtype=float)
    return session
