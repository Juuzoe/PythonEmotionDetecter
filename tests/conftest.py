from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def astronaut_path() -> Path:
    """Public-domain NASA portrait of Eileen Collins (via scikit-image)."""
    return DATA_DIR / "astronaut.png"


def synthetic_rgb(
    bpm: float = 72.0,
    seconds: float = 20.0,
    fs: float = 30.0,
    noise: float = 0.3,
    drift: bool = True,
    seed: int = 0,
    jitter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-RGB trace of a skin patch with a pulsatile component at ``bpm``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    if jitter:
        t = t + rng.uniform(-jitter, jitter, size=n) / fs
        t = np.sort(t)
    pulse = np.sin(2.0 * np.pi * bpm / 60.0 * t)
    base = np.array([150.0, 110.0, 95.0])
    amplitude = np.array([0.4, 1.0, 0.6])  # the pulse is strongest in green
    rgb = base + pulse[:, None] * amplitude + noise * rng.normal(size=(n, 3))
    if drift:
        rgb += 3.0 * np.sin(2.0 * np.pi * 0.05 * t)[:, None] + 0.1 * t[:, None]
    return t, rgb
