"""Model registry with verified downloads.

Every model AffectLab can use is listed here with its origin, licence and
SHA-256 digest. Files are fetched on first use into a local cache directory
(``$AFFECTLAB_MODELS_DIR``, else ``~/.cache/affectlab/models``) and verified
before use, so a corrupted or tampered download is rejected rather than
silently loaded. Nothing else is ever downloaded, and no data is uploaded.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ModelError(RuntimeError):
    """A model could not be downloaded or failed verification."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    description: str
    homepage: str


MODELS: dict[str, ModelSpec] = {
    "face_landmarker": ModelSpec(
        name="face_landmarker",
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        size_bytes=3758596,
        license="Apache-2.0",
        description="MediaPipe Face Landmarker: 478 3D landmarks, 52 blendshapes, head transform.",
        homepage="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
    ),
    "ferplus": ModelSpec(
        name="ferplus",
        filename="emotion-ferplus-8.onnx",
        url=(
            "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
            "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
        ),
        sha256="a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c",
        size_bytes=35040571,
        license="MIT (ONNX Model Zoo)",
        description="FER+ emotion CNN (Barsoum et al., 2016): 8 classes from a 64x64 grey face.",
        homepage="https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus",
    ),
    "yunet": ModelSpec(
        name="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size_bytes=232589,
        license="MIT (OpenCV Zoo)",
        description="YuNet face detector, used only when MediaPipe is unavailable.",
        homepage="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    ),
}

ProgressCallback = Callable[[int, int], None]

#: Environment variables that relocate the model cache.
MODELS_DIR_ENV = "AFFECTLAB_MODELS_DIR"
HOME_ENV = "AFFECTLAB_HOME"


def models_dir() -> Path:
    """Directory that holds cached models.

    ``$AFFECTLAB_MODELS_DIR`` names it exactly; otherwise it is
    ``$AFFECTLAB_HOME/models`` or ``~/.cache/affectlab/models``.
    """
    exact = os.environ.get(MODELS_DIR_ENV)
    if exact:
        return Path(exact).expanduser()
    home = os.environ.get(HOME_ENV)
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "affectlab"
    return base / "models"


def model_path(name: str) -> Path:
    return models_dir() / spec(name).filename


def spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as exc:
        known = ", ".join(sorted(MODELS))
        raise ModelError(f"unknown model {name!r}; known models: {known}") from exc


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_cached(name: str) -> bool:
    """Cheap check: the file exists with the expected size. Hashes are verified on download."""
    path = model_path(name)
    return path.is_file() and path.stat().st_size == spec(name).size_bytes


def verify(name: str) -> bool:
    """Full SHA-256 verification of a cached model."""
    path = model_path(name)
    return path.is_file() and sha256_of(path) == spec(name).sha256


def _stderr_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    sys.stderr.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
    if done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    progress: ProgressCallback | None = None,
    timeout: float = 60.0,
) -> Path:
    """Stream ``url`` to ``dest`` atomically, verifying the digest if given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "affectlab/2.0"})
    digest = hashlib.sha256()
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            os.fdopen(tmp_fd, "wb") as out,
        ):
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            raise ModelError(
                f"checksum mismatch for {url}: expected {expected_sha256}, got {digest.hexdigest()}"
            )
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest


def ensure_model(name: str, progress: ProgressCallback | None = _stderr_progress) -> Path:
    """Return the local path of ``name``, downloading and verifying it if needed."""
    model = spec(name)
    path = model_path(name)
    if path.is_file():
        if path.stat().st_size == model.size_bytes:
            return path
        path.unlink()  # truncated or stale; fetch again
    if progress is _stderr_progress:
        sys.stderr.write(f"Downloading {model.description}\n  from {model.url}\n")
    try:
        download_file(model.url, path, model.sha256, progress)
    except ModelError:
        raise
    except Exception as exc:  # network errors, permission errors
        raise ModelError(
            f"could not download model {name!r} from {model.url}: {exc}. "
            f"You can download it manually and place it at {path}."
        ) from exc
    return path


def cached_models() -> dict[str, bool]:
    return {name: is_cached(name) for name in MODELS}
