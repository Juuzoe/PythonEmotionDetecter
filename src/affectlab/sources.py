"""Frame sources: cameras, video files and still images."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


class SourceError(RuntimeError):
    """A camera, file or stream could not be opened or read."""


def parse_source(spec: str | int) -> int | str:
    """``"0"`` becomes camera index 0; anything else is a path or URL."""
    if isinstance(spec, int):
        return spec
    text = str(spec).strip()
    return int(text) if text.isdigit() else text


def is_image_path(spec: str | int) -> bool:
    return isinstance(spec, str) and Path(spec).suffix.lower() in IMAGE_SUFFIXES


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SourceError(f"could not read image {path!s}")
    return image


class FrameSource:
    """Iterate ``(frame_bgr, timestamp_seconds)`` from a camera or video file.

    Camera timestamps come from a monotonic clock; file timestamps are
    ``frame_index / fps`` so that analysis is deterministic and independent
    of processing speed.
    """

    def __init__(
        self, spec: str | int, *, width: int | None = None, height: int | None = None
    ) -> None:
        self.spec = parse_source(spec)
        self.is_camera = isinstance(self.spec, int)
        self.capture = cv2.VideoCapture(self.spec)
        if not self.capture.isOpened():
            kind = "camera" if self.is_camera else "video"
            raise SourceError(f"could not open {kind} {spec!r}")
        if self.is_camera:
            if width:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = (
            0 if self.is_camera else int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        )

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        index = 0
        start = time.perf_counter()
        while True:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            t = time.perf_counter() - start if self.is_camera else index / self.fps
            yield frame, t
            index += 1

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
