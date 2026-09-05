"""End-to-end tests that need the downloaded models and MediaPipe's native library.

Run with ``pytest -m integration`` after ``affectlab models download``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from affectlab.models import is_cached

pytestmark = pytest.mark.integration


def _require_models() -> None:
    if not (is_cached("face_landmarker") and is_cached("ferplus")):
        pytest.skip("models not downloaded; run: affectlab models download")
    pytest.importorskip("mediapipe")


@pytest.fixture
def astronaut(astronaut_path: Path) -> np.ndarray:
    import cv2

    _require_models()
    return cv2.imread(str(astronaut_path))


def test_image_pipeline_reads_a_smile(astronaut: np.ndarray) -> None:
    from affectlab.pipeline import AffectPipeline, PipelineConfig

    with AffectPipeline(PipelineConfig(backend="ensemble"), video=False) as pipeline:
        result = pipeline.process(astronaut, 0.0)
    assert result.face is not None and result.face.has_landmarks
    assert result.emotion is not None
    assert result.emotion.dominant == "happiness"
    assert result.action_units["AU12"] > 0.5
    assert result.affect is not None and result.affect.valence > 0.3
    assert result.head_pose is not None and abs(result.head_pose.yaw) < 20
    assert result.vitals.ear is not None and 0.2 < result.vitals.ear < 0.45


@pytest.mark.parametrize("backend", ["facs", "ferplus"])
def test_each_backend_agrees_on_the_smile(astronaut: np.ndarray, backend: str) -> None:
    from affectlab.pipeline import AffectPipeline, PipelineConfig

    with AffectPipeline(PipelineConfig(backend=backend), video=False) as pipeline:
        result = pipeline.process(astronaut, 0.0)
    assert result.emotion is not None and result.emotion.backend == backend
    assert result.emotion.dominant == "happiness"


def test_synthetic_video_recovers_pulse(astronaut: np.ndarray) -> None:
    """Modulate the frame's green channel at 72 bpm and expect rPPG to find it."""
    from affectlab.pipeline import AffectPipeline, PipelineConfig

    fps, seconds, bpm = 30.0, 16.0, 72.0
    rng = np.random.default_rng(0)
    base = astronaut.astype(np.float32)
    config = PipelineConfig(backend="facs", rppg_window_seconds=12.0, rppg_min_seconds=8.0)
    with AffectPipeline(config) as pipeline:
        result = None
        for i in range(int(fps * seconds)):
            t = i / fps
            frame = base.copy()
            frame[:, :, 1] += 1.5 * np.sin(2 * np.pi * bpm / 60.0 * t)  # green channel (BGR)
            frame += rng.normal(0.0, 0.5, size=frame.shape).astype(np.float32)
            result = pipeline.process(np.clip(frame, 0, 255).astype(np.uint8), t)
    assert result is not None and result.face is not None
    assert result.vitals.heart_rate_bpm is not None
    assert abs(result.vitals.heart_rate_bpm - bpm) < 5.0
    assert result.vitals.heart_rate_quality in {"good", "fair"}


def test_cli_image_command(astronaut_path: Path, tmp_path: Path) -> None:
    from affectlab.cli import main

    _require_models()
    out = tmp_path / "annotated.png"
    assert main(["image", str(astronaut_path), "--save", str(out), "--backend", "ferplus"]) == 0
    assert out.exists() and out.stat().st_size > 10_000


def test_cli_image_without_landmarks_and_without_emotion(
    astronaut_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The facs backend cannot run on YuNet boxes; the command must still report the face."""
    from affectlab.cli import main

    _require_models()
    if not is_cached("yunet"):
        pytest.skip("yunet model not downloaded")
    assert main(["image", str(astronaut_path), "--no-landmarks", "--backend", "facs"]) == 0
    out = capsys.readouterr().out
    assert "source: yunet" in out and "not available" in out
