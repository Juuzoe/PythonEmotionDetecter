"""Pipeline behaviour with injected fakes: no models, no MediaPipe, no camera."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from affectlab.pipeline import AffectPipeline, PipelineConfig
from affectlab.types import EMOTIONS, BBox, EmotionEstimate, FaceObservation


class ScriptedDetector:
    """Returns a face for frames whose index is in ``present``."""

    def __init__(self, present) -> None:
        self.present = present
        self.calls = 0
        self.closed = False

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None):
        index = self.calls
        self.calls += 1
        if not self.present(index):
            return []
        return [FaceObservation(bbox=BBox(40, 40, 60, 60), source="fake")]

    def close(self) -> None:
        self.closed = True


class ConstantBackend:
    name = "constant"

    def __init__(self, label: str = "happiness") -> None:
        self.label = label

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        probs = dict.fromkeys(EMOTIONS, 0.0)
        probs[self.label] = 1.0
        return EmotionEstimate(probs, backend=self.name)

    def close(self) -> None:
        return None


def pulsing_frame(t: float, bpm: float = 72.0) -> np.ndarray:
    frame = np.full((160, 160, 3), 120, dtype=np.float32)
    frame[:, :, 1] += 3.0 * np.sin(2 * np.pi * bpm / 60.0 * t)
    return frame.astype(np.uint8)


def run(pipeline: AffectPipeline, present, seconds: float, fps: float = 30.0):
    results = []
    for i in range(int(seconds * fps)):
        results.append(pipeline.process(pulsing_frame(i / fps), i / fps))
    return results


def test_stale_state_expires_after_a_gap() -> None:
    fps = 30.0
    # Face present 0-12 s, absent 12-18 s, present again 18-28 s.
    present = lambda i: not (12 * fps <= i < 18 * fps)  # noqa: E731
    with AffectPipeline(
        PipelineConfig(rppg_min_seconds=8.0, hold_seconds=0.5, gap_reset_seconds=1.0),
        detector=ScriptedDetector(present),
        backend=ConstantBackend(),
    ) as pipeline:
        results = run(pipeline, present, 28.0, fps)

    at = lambda s: results[int(s * fps)]  # noqa: E731
    assert at(11.9).face is not None
    assert at(11.9).vitals.heart_rate_bpm is not None
    assert abs(at(11.9).vitals.heart_rate_bpm - 72.0) < 5.0
    assert at(11.9).emotion is not None and at(11.9).emotion.dominant == "happiness"

    # Within the hold the last emotion is still shown; the face is gone.
    just_after = at(12.2)
    assert just_after.face is None and just_after.emotion is not None

    # After the gap everything stale is gone, including the pulse estimate.
    later = at(15.0)
    assert later.face is None
    assert later.emotion is None and later.affect is None
    assert later.vitals.heart_rate_bpm is None
    assert later.vitals.heart_rate_quality == "no signal"

    # When the face returns, inference restarts immediately and the pulse rebuilds.
    back = at(18.0)
    assert back.face is not None and back.emotion is not None
    # The rPPG buffer has to refill from scratch, so the estimate returns only after
    # another full minimum window (8 s), not immediately.
    assert at(24.0).vitals.heart_rate_bpm is None
    assert at(27.9).vitals.heart_rate_bpm is not None
    assert pipeline.frames_with_face == int(22 * fps)


def test_time_in_state_excludes_absence() -> None:
    fps = 10.0
    present = lambda i: i < 5 * fps or i >= 15 * fps  # noqa: E731
    with AffectPipeline(
        PipelineConfig(emotion_every=1),
        detector=ScriptedDetector(present),
        backend=ConstantBackend("sadness"),
    ) as pipeline:
        results = run(pipeline, present, 20.0, fps)
    credited = results[-1].dynamics.time_in_state["sadness"]
    # About 5 s before and 5 s after the gap; the 10 s absence is not credited.
    assert 9.0 <= credited <= 10.5


def test_short_rppg_window_is_clamped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="affectlab.pipeline"):
        pipeline = AffectPipeline(
            PipelineConfig(rppg_window_seconds=5.0, rppg_min_seconds=8.0),
            detector=ScriptedDetector(lambda i: True),
            backend=ConstantBackend(),
        )
    assert pipeline.rppg_min_seconds == pytest.approx(4.5)
    assert "shorter than the minimum" in caplog.text
    results = run(pipeline, lambda i: True, 7.0)
    assert results[-1].vitals.heart_rate_bpm is not None  # possible with a 5 s window now
    pipeline.close()


def test_injected_detector_and_backend_are_closed() -> None:
    detector = ScriptedDetector(lambda i: True)
    pipeline = AffectPipeline(PipelineConfig(), detector=detector, backend=ConstantBackend())
    assert not pipeline.uses_landmarks
    pipeline.close()
    assert detector.closed
