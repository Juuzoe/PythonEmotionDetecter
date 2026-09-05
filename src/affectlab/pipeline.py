"""The AffectLab pipeline: one frame in, one :class:`FrameResult` out.

The pipeline owns no I/O. Feed it BGR frames with timestamps and it returns
a complete, self-describing result per frame. Sources, display and recording
live in separate modules so the same pipeline runs on a webcam, a video
file, a single image or a unit test.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

import numpy as np

from affectlab import geometry
from affectlab.affect import AffectTracker
from affectlab.blink import BlinkDetector, BlinkState
from affectlab.detect import YuNetDetector
from affectlab.emotion import create_backend
from affectlab.emotion.base import EmotionBackend
from affectlab.facs import action_units_from_blendshapes
from affectlab.landmarks import FaceLandmarkerEngine, LandmarkerUnavailable
from affectlab.rppg import PulseEstimate, RppgBuffer, estimate_breathing_rate, estimate_pulse
from affectlab.types import (
    Affect,
    BBox,
    Dynamics,
    EmotionEstimate,
    FaceObservation,
    FrameResult,
    HeadPose,
    Vitals,
)

log = logging.getLogger(__name__)

#: Default frames between emotion inferences, per backend.
EMOTION_CADENCE: dict[str, int] = {"facs": 1, "ferplus": 2, "ensemble": 2, "deepface": 6}


class FaceDetector(Protocol):
    """Anything that turns a frame into faces, largest first."""

    def process(
        self, frame_bgr: np.ndarray, timestamp_ms: int | None = None
    ) -> list[FaceObservation]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PipelineConfig:
    backend: str = "ensemble"
    #: Frames between emotion inferences; None picks a per-backend default.
    emotion_every: int | None = None
    num_faces: int = 1
    use_landmarks: bool = True
    min_detection_confidence: float = 0.5
    rppg_method: str = "pos"
    rppg_window_seconds: float = 20.0
    rppg_min_seconds: float = 8.0
    rppg_update_every: int = 5
    #: Face motion (fraction of face size per frame) above which the rPPG buffer resets.
    motion_reset_threshold: float = 0.12
    affect_tau_seconds: float = 1.5
    dynamics_window_seconds: float = 60.0
    breathing: bool = True
    breathing_window_seconds: float = 30.0
    #: Keep reporting the last emotion and affect for this long after the face is lost.
    hold_seconds: float = 0.5
    #: A face absence longer than this discards the pulse estimate, its signal buffer,
    #: and the last emotion, and is excluded from every time-based statistic.
    gap_reset_seconds: float = 1.0


class AffectPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        video: bool = True,
        detector: FaceDetector | None = None,
        backend: EmotionBackend | None = None,
    ) -> None:
        """Build the pipeline.

        ``detector`` and ``backend`` can be injected (for tests or custom
        models); otherwise MediaPipe (with a YuNet fallback) and the
        configured emotion backend are created.
        """
        self.config = config or PipelineConfig()
        cfg = self.config
        self.landmarker: FaceLandmarkerEngine | None = None
        self.detector: FaceDetector | None = detector
        if self.detector is None and cfg.use_landmarks:
            try:
                self.landmarker = FaceLandmarkerEngine(
                    num_faces=cfg.num_faces,
                    video=video,
                    min_detection_confidence=cfg.min_detection_confidence,
                )
            except LandmarkerUnavailable as exc:
                log.warning(
                    "%s. Falling back to YuNet face boxes: no action units, blinks or head pose.",
                    exc,
                )
        if self.landmarker is None and self.detector is None:
            self.detector = YuNetDetector(score_threshold=cfg.min_detection_confidence)

        self.backend: EmotionBackend = (
            backend if backend is not None else create_backend(cfg.backend)
        )
        if self.landmarker is None and self.backend.name == "facs":
            log.warning(
                "The facs backend needs MediaPipe blendshapes; without landmarks it produces no "
                "emotion estimates. Use --backend ferplus or ensemble."
            )
        self.emotion_every = max(1, cfg.emotion_every or EMOTION_CADENCE.get(self.backend.name, 2))

        self.rppg_min_seconds = float(cfg.rppg_min_seconds)
        if cfg.rppg_window_seconds < cfg.rppg_min_seconds:
            # The buffer drops samples older than the window, so it never quite holds a
            # full window; requiring exactly the window would make an estimate impossible.
            self.rppg_min_seconds = 0.9 * float(cfg.rppg_window_seconds)
            log.warning(
                "rPPG window (%.1f s) is shorter than the minimum signal length (%.1f s); "
                "lowering the minimum to %.1f s. Heart-rate estimates from such short windows "
                "are coarse.",
                cfg.rppg_window_seconds,
                cfg.rppg_min_seconds,
                self.rppg_min_seconds,
            )

        self.affect = AffectTracker(
            tau_seconds=cfg.affect_tau_seconds,
            window_seconds=cfg.dynamics_window_seconds,
            gap_seconds=cfg.gap_reset_seconds,
        )
        self.rppg = RppgBuffer(
            window_seconds=cfg.rppg_window_seconds, gap_reset_seconds=cfg.gap_reset_seconds
        )
        self.blinks = BlinkDetector(gap_reset_seconds=cfg.gap_reset_seconds)
        self.last_pulse: PulseEstimate | None = None
        self.frames_with_face = 0

        self._breath: deque[tuple[float, float]] = deque()
        self._frame_index = 0
        self._last_t: float | None = None
        self._last_face_t: float | None = None
        self._fps = 0.0
        self._last_emotion: EmotionEstimate | None = None
        self._last_affect: Affect | None = None
        self._last_dynamics = Dynamics()
        self._since_pulse = 0
        self._since_breath = 0
        self._last_bbox: BBox | None = None
        self._breathing: float | None = None

    @property
    def uses_landmarks(self) -> bool:
        return self.landmarker is not None

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> FrameResult:
        """Analyse one BGR frame. ``timestamp`` is in seconds; defaults to a monotonic clock."""
        t = time.perf_counter() if timestamp is None else float(timestamp)
        self._update_fps(t)
        index = self._frame_index
        self._frame_index += 1

        faces = self._detect(frame_bgr, t)
        if not faces:
            return self._no_face_result(index, t)

        face = faces[0]
        self.frames_with_face += 1
        self._last_face_t = t
        action_units = action_units_from_blendshapes(face.blendshapes) if face.blendshapes else {}
        head_pose = self._head_pose(face, frame_bgr.shape[1], frame_bgr.shape[0])

        if self._last_emotion is None or index % self.emotion_every == 0:
            estimate = self.backend.predict(frame_bgr, face)
            if estimate is not None:
                self._last_emotion = estimate
                self._last_affect, self._last_dynamics = self.affect.update(
                    estimate.probabilities, t
                )

        self._update_rppg(frame_bgr, face, t)
        blink_state = self._update_blinks(face, t)
        if self.config.breathing and face.has_landmarks:
            self._update_breathing(face, t)

        return FrameResult(
            frame_index=index,
            timestamp=t,
            face=face,
            action_units=action_units,
            emotion=self._last_emotion,
            affect=self._last_affect,
            head_pose=head_pose,
            vitals=self._vitals(blink_state),
            dynamics=self._last_dynamics,
            fps=self._fps,
        )

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        if self.detector is not None:
            self.detector.close()
        self.backend.close()

    def __enter__(self) -> AffectPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ----------------------------------------------------------------- stages

    def _detect(self, frame_bgr: np.ndarray, t: float) -> list[FaceObservation]:
        if self.landmarker is not None:
            return self.landmarker.process(frame_bgr, round(t * 1000.0))
        assert self.detector is not None
        return self.detector.process(frame_bgr, round(t * 1000.0))

    def _no_face_result(self, index: int, t: float) -> FrameResult:
        """Result for a frame without a face: nothing stale survives a real gap."""
        self._last_bbox = None
        absent = float("inf") if self._last_face_t is None else t - self._last_face_t
        if absent > self.config.gap_reset_seconds:
            self.rppg.clear()
            self.last_pulse = None
            self._last_emotion = None
            self._last_affect = None
        hold = absent <= self.config.hold_seconds
        return FrameResult(
            frame_index=index,
            timestamp=t,
            face=None,
            action_units={},
            emotion=self._last_emotion if hold else None,
            affect=self._last_affect if hold else None,
            head_pose=None,
            vitals=self._vitals(None),
            dynamics=self._last_dynamics,
            fps=self._fps,
        )

    @staticmethod
    def _head_pose(face: FaceObservation, frame_w: int, frame_h: int) -> HeadPose | None:
        if face.transform is not None:
            return geometry.head_pose_from_transform(face.transform)
        if face.has_landmarks and face.landmarks is not None:
            return geometry.head_pose_pnp(face.landmarks, frame_w, frame_h)
        return None

    def _update_fps(self, t: float) -> None:
        if self._last_t is not None and t > self._last_t:
            instantaneous = 1.0 / (t - self._last_t)
            self._fps = instantaneous if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instantaneous
        self._last_t = t

    def _update_rppg(self, frame_bgr: np.ndarray, face: FaceObservation, t: float) -> None:
        cfg = self.config
        if (
            self._last_bbox is not None
            and geometry.motion_between(self._last_bbox, face.bbox) > cfg.motion_reset_threshold
        ):
            self.rppg.clear()
            self.last_pulse = None
        self._last_bbox = face.bbox

        if face.has_landmarks and face.landmarks is not None:
            rgb = geometry.polygon_mean_rgb(frame_bgr, geometry.skin_polygons(face.landmarks))
        else:
            rgb = geometry.rect_mean_rgb(frame_bgr, geometry.bbox_forehead_rect(face.bbox))
        if rgb is not None:
            self.rppg.append(t, rgb)

        self._since_pulse += 1
        if self._since_pulse >= cfg.rppg_update_every:
            self._since_pulse = 0
            if len(self.rppg) >= 2:
                times, values = self.rppg.arrays()
                self.last_pulse = estimate_pulse(
                    times, values, method=cfg.rppg_method, min_seconds=self.rppg_min_seconds
                )
            else:
                self.last_pulse = None

    def _update_blinks(self, face: FaceObservation, t: float) -> BlinkState | None:
        if not face.has_landmarks or face.landmarks is None:
            return None
        ear = geometry.ear_from_landmarks(face.landmarks)
        closure = None
        if face.blendshapes:
            closure = 0.5 * (
                face.blendshapes.get("eyeBlinkLeft", 0.0)
                + face.blendshapes.get("eyeBlinkRight", 0.0)
            )
        return self.blinks.update(ear, t, closure)

    def _update_breathing(self, face: FaceObservation, t: float) -> None:
        assert face.landmarks is not None
        if self._breath and t - self._breath[-1][0] > self.config.gap_reset_seconds:
            self._breath.clear()
            self._breathing = None
        nose_y = float(face.landmarks[geometry.NOSE_TIP, 1]) / max(1.0, face.bbox.size)
        self._breath.append((t, nose_y))
        cutoff = t - self.config.breathing_window_seconds
        while self._breath and self._breath[0][0] < cutoff:
            self._breath.popleft()
        self._since_breath += 1
        if self._since_breath >= 30:
            self._since_breath = 0
            if len(self._breath) >= 16:
                arr = np.asarray(self._breath, dtype=float)
                self._breathing = estimate_breathing_rate(arr[:, 0], arr[:, 1])

    def _vitals(self, blink_state: BlinkState | None) -> Vitals:
        vitals = Vitals(
            signal_seconds=self.rppg.seconds,
            breathing_rate_bpm=self._breathing,
            blink_count=self.blinks.count,
        )
        pulse = self.last_pulse
        if pulse is not None:
            vitals.heart_rate_bpm = pulse.bpm
            vitals.heart_rate_snr_db = pulse.snr_db
            vitals.heart_rate_quality = pulse.quality
            vitals.rmssd_ms = pulse.rmssd_ms
        else:
            vitals.heart_rate_quality = "collecting" if len(self.rppg) > 0 else "no signal"
        if blink_state is not None:
            vitals.ear = blink_state.ear
            vitals.blink_rate_per_min = blink_state.blink_rate_per_min
            vitals.perclos = blink_state.perclos
        return vitals
