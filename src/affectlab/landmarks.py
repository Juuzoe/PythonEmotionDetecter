"""MediaPipe Face Landmarker wrapper: 478 landmarks, 52 blendshapes, head transform."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from affectlab.models import ensure_model
from affectlab.types import BBox, FaceObservation


class LandmarkerUnavailable(RuntimeError):
    """MediaPipe is missing, its native library cannot load, or the model is absent."""


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except Exception:  # ImportError or a broken native install
        return False
    return True


class FaceLandmarkerEngine:
    """Detect faces and produce :class:`FaceObservation` objects.

    In video mode MediaPipe tracks faces across frames and requires strictly
    increasing timestamps in milliseconds; the wrapper enforces that.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        num_faces: int = 1,
        video: bool = True,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise LandmarkerUnavailable(
                "mediapipe is not installed (pip install mediapipe)"
            ) from exc
        path = Path(model_path) if model_path else ensure_model("face_landmarker")
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except OSError as exc:
            raise LandmarkerUnavailable(
                f"MediaPipe could not load its native library ({exc}). On Debian or Ubuntu "
                "run: sudo apt install libegl1 libgles2 libgl1"
            ) from exc
        self._mp = mp
        self._video = video
        self._last_ts = -1
        self.num_faces = num_faces

    def process(
        self, frame_bgr: np.ndarray, timestamp_ms: int | None = None
    ) -> list[FaceObservation]:
        """Return faces in ``frame_bgr``, largest first."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            ts = int(timestamp_ms) if timestamp_ms is not None else self._last_ts + 33
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        faces: list[FaceObservation] = []
        blendshapes = result.face_blendshapes or []
        transforms = result.facial_transformation_matrixes or []
        for i, lm in enumerate(result.face_landmarks):
            pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float32)
            bs = (
                {c.category_name: float(c.score) for c in blendshapes[i]}
                if i < len(blendshapes)
                else None
            )
            tf = np.asarray(transforms[i], dtype=np.float32) if i < len(transforms) else None
            faces.append(
                FaceObservation(
                    bbox=BBox.from_points(pts, w, h),
                    landmarks=pts,
                    blendshapes=bs,
                    transform=tf,
                    source="mediapipe",
                )
            )
        faces.sort(key=lambda f: f.bbox.area, reverse=True)
        return faces

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkerEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
