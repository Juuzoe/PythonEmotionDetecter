"""FER+ emotion recognition (Barsoum, Zhang, Canton Ferrer & Zhang, 2016).

FER+ re-labelled the FER2013 faces with ten crowd workers each, so the
network is trained on label *distributions* rather than a single majority
vote. It outputs eight classes, the same set AffectLab uses as canonical.
The ONNX export from the ONNX Model Zoo expects a 64x64 greyscale face with
raw 0-255 intensities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from affectlab.emotion.base import BackendUnavailable, face_crop_gray, softmax
from affectlab.models import ensure_model
from affectlab.types import EMOTIONS, BBox, EmotionEstimate, FaceObservation

INPUT_SIZE = 64
#: Output order of the FER+ network; identical to :data:`EMOTIONS` by design.
FERPLUS_LABELS: tuple[str, ...] = EMOTIONS


def preprocess(frame_bgr: np.ndarray, bbox: BBox, expand: float = 0.2) -> np.ndarray:
    """(1, 1, 64, 64) float32 tensor with raw 0-255 grey intensities."""
    gray = face_crop_gray(frame_bgr, bbox, INPUT_SIZE, expand)
    return gray.astype(np.float32)[None, None, :, :]


class FerPlusBackend:
    name = "ferplus"

    def __init__(
        self, model_path: str | Path | None = None, providers: list[str] | None = None
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailable(
                "onnxruntime is not installed (pip install onnxruntime)"
            ) from exc
        path = Path(model_path) if model_path else ensure_model("ferplus")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, frame_bgr: np.ndarray, face: FaceObservation) -> EmotionEstimate | None:
        if face.bbox.w < 8 or face.bbox.h < 8:
            return None
        logits = self._session.run(None, {self._input_name: preprocess(frame_bgr, face.bbox)})[0]
        probs = softmax(np.asarray(logits).reshape(-1))
        return EmotionEstimate(
            probabilities={label: float(p) for label, p in zip(FERPLUS_LABELS, probs)},
            backend=self.name,
        )

    def close(self) -> None:
        return None
