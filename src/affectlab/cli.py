"""Command-line interface: ``affectlab live | video | image | report | doctor | models``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from affectlab import __version__
from affectlab.emotion import BACKENDS, BackendUnavailable
from affectlab.models import (
    MODELS,
    MODELS_DIR_ENV,
    ModelError,
    cached_models,
    ensure_model,
    model_path,
)
from affectlab.rppg import METHODS

if TYPE_CHECKING:
    import numpy as np

    from affectlab.hud import Hud
    from affectlab.pipeline import PipelineConfig
    from affectlab.recorder import SessionRecorder
    from affectlab.sources import FrameSource

log = logging.getLogger("affectlab")


# ----------------------------------------------------------------------------- parser


def _add_pipeline_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("analysis")
    group.add_argument(
        "--backend",
        choices=BACKENDS,
        default="ensemble",
        help="emotion backend (default: ensemble of ferplus and facs)",
    )
    group.add_argument(
        "--every", type=int, default=None, metavar="N", help="run emotion inference every N frames"
    )
    group.add_argument(
        "--rppg", choices=METHODS, default="pos", help="pulse extraction method (default: pos)"
    )
    group.add_argument(
        "--rppg-window",
        type=float,
        default=20.0,
        metavar="SECONDS",
        help="length of the rPPG analysis window (default: 20)",
    )
    group.add_argument(
        "--no-landmarks",
        action="store_true",
        help="skip MediaPipe and use YuNet boxes only (no action units, blinks or head pose)",
    )
    group.add_argument(
        "--no-breathing", action="store_true", help="disable the experimental breathing estimate"
    )


def _add_hud_flags(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--no-panel", action="store_true", help="hide the side panel")
    group.add_argument("--landmarks", action="store_true", help="draw landmarks on start")
    group.add_argument("--rois", action="store_true", help="draw the rPPG skin regions on start")
    group.add_argument("--blur", action="store_true", help="pixelate the face (privacy mode)")


def _add_output_options(parser: argparse.ArgumentParser, *, live: bool) -> None:
    group = parser.add_argument_group("output")
    group.add_argument(
        "--record",
        metavar="FILE",
        help="write one row per frame to a .csv or .jsonl session file",
    )
    group.add_argument(
        "--output", metavar="FILE.mp4", help="write the annotated video (HUD included)"
    )
    _add_hud_flags(group)
    if live:
        group.add_argument("--no-hud", action="store_true", help="do not open a window")
        group.add_argument(
            "--max-seconds", type=float, default=None, help="stop after this many seconds"
        )
    else:
        group.add_argument("--show", action="store_true", help="display frames while processing")
        group.add_argument(
            "--max-frames", type=int, default=None, help="stop after this many frames"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affectlab",
        description=(
            "Real-time affective computing from a webcam: FACS action units, dimensional "
            "affect, contactless heart rate and blink dynamics. Everything runs locally."
        ),
    )
    parser.add_argument("--version", action="version", version=f"affectlab {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--models-dir", metavar="DIR", help="where to cache models (default: ~/.cache/affectlab)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="analyse a webcam in real time")
    live.add_argument("--camera", default="0", help="camera index or stream URL (default: 0)")
    live.add_argument("--width", type=int, default=None)
    live.add_argument("--height", type=int, default=None)
    live.add_argument("--mirror", action="store_true", help="flip the image horizontally")
    _add_pipeline_options(live)
    _add_output_options(live, live=True)
    live.set_defaults(func=cmd_live)

    video = sub.add_parser("video", help="analyse a video file offline")
    video.add_argument("path", help="video file")
    _add_pipeline_options(video)
    _add_output_options(video, live=False)
    video.set_defaults(func=cmd_video)

    image = sub.add_parser("image", help="analyse a single image")
    image.add_argument("path", help="image file")
    image.add_argument("--save", metavar="FILE.png", help="write the annotated image")
    image.add_argument("--json", action="store_true", help="print the full result as JSON")
    _add_pipeline_options(image)
    _add_hud_flags(image.add_argument_group("output"))
    image.set_defaults(func=cmd_image)

    report = sub.add_parser("report", help="summarise a recorded session")
    report.add_argument("session", help="session .csv or .jsonl written by --record")
    report.add_argument("--out", metavar="DIR", help="output directory (default: next to session)")
    report.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    report.set_defaults(func=cmd_report)

    doctor = sub.add_parser("doctor", help="check the installation")
    doctor.add_argument("--camera", default=None, help="also try to open this camera index")
    doctor.set_defaults(func=cmd_doctor)

    models = sub.add_parser("models", help="list or download models")
    models.add_argument("action", choices=("list", "download", "path"), nargs="?", default="list")
    models.add_argument("names", nargs="*", help="model names (default: all)")
    models.set_defaults(func=cmd_models)
    return parser


# ----------------------------------------------------------------------------- helpers


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    from affectlab.pipeline import PipelineConfig

    return PipelineConfig(
        backend=args.backend,
        emotion_every=args.every,
        use_landmarks=not args.no_landmarks,
        rppg_method=args.rppg,
        rppg_window_seconds=args.rppg_window,
        breathing=not args.no_breathing,
    )


def _hud(args: argparse.Namespace) -> Hud:
    from affectlab.hud import Hud, HudOptions

    return Hud(
        HudOptions(
            show_panel=not args.no_panel,
            show_landmarks=args.landmarks,
            show_rois=args.rois,
            blur_face=args.blur,
        )
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _quiet_native_logs() -> None:
    try:
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


class _VideoOutput:
    """Annotated-video writer that is robust to the things that silently break VideoWriter.

    Every frame is fitted to the size of the first one (a writer drops frames of any other
    size without raising), the writer is checked to have opened, and for live sources the
    frame rate is measured over the first seconds instead of trusting the camera's nominal
    value, so the file plays back in real time.
    """

    def __init__(self, path: str, source_fps: float, measure_fps: bool) -> None:
        self.path = path
        self.source_fps = float(source_fps)
        self.measure_fps = measure_fps
        self.ok = True
        self.frames_written = 0
        self._writer: Any = None
        self._size: tuple[int, int] | None = None
        self._pending: list[tuple[float, np.ndarray]] = []

    def write(self, canvas: np.ndarray, t: float) -> None:
        from affectlab.hud import fit_canvas

        if self._size is None:
            h, w = canvas.shape[:2]
            self._size = (w + w % 2, h + h % 2)
        fitted = fit_canvas(canvas, self._size)
        if self._writer is None and self.measure_fps:
            self._pending.append((t, fitted))
            span = t - self._pending[0][0]
            if len(self._pending) < 90 and span < 3.0:
                return
            self._open(self._measured_fps())
            self._flush()
            return
        if self._writer is None:
            self._open(self.source_fps)
        self._write(fitted)

    def _measured_fps(self) -> float:
        if len(self._pending) < 2:
            return self.source_fps
        span = self._pending[-1][0] - self._pending[0][0]
        if span <= 0:
            return self.source_fps
        return float(min(120.0, max(1.0, (len(self._pending) - 1) / span)))

    def _open(self, fps: float) -> None:
        import cv2

        assert self._size is not None
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter.fourcc(*"mp4v"), fps, self._size)
        if not writer.isOpened():
            self.ok = False
            log.error("could not open %s for writing (unsupported path or codec)", self.path)
        self._writer = writer

    def _flush(self) -> None:
        for _, frame in self._pending:
            self._write(frame)
        self._pending.clear()

    def _write(self, frame: np.ndarray) -> None:
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)
            self.frames_written += 1

    def close(self) -> bool:
        if self._writer is None and self._pending:
            self._open(self._measured_fps())
            self._flush()
        if self._writer is not None:
            self._writer.release()
        return self.ok and self.frames_written > 0


def _handle_key(
    key: int,
    hud: Hud,
    canvas: np.ndarray,
    recorder: SessionRecorder | None,
    output_locked: bool,
) -> SessionRecorder | None:
    """Apply a keyboard toggle; returns the (possibly new or closed) recorder."""
    import cv2

    from affectlab.recorder import SessionRecorder

    if key == ord("l"):
        hud.options.show_landmarks = not hud.options.show_landmarks
    elif key == ord("o"):
        hud.options.show_rois = not hud.options.show_rois
    elif key == ord("p"):
        if output_locked:
            print("the side panel is fixed while --output is being written", file=sys.stderr)
        else:
            hud.options.show_panel = not hud.options.show_panel
    elif key == ord("b"):
        hud.options.blur_face = not hud.options.blur_face
    elif key == ord("h"):
        hud.options.show_help = not hud.options.show_help
    elif key == ord("s"):
        name = f"affectlab_snapshot_{_timestamp()}.png"
        cv2.imwrite(name, canvas)
        print(f"saved {name}", file=sys.stderr)
    elif key == ord("r"):
        if recorder is None:
            path = f"affectlab_session_{_timestamp()}.csv"
            print(f"recording to {path}", file=sys.stderr)
            return SessionRecorder(path)
        recorder.close()
        print(f"stopped recording ({recorder.rows} rows)", file=sys.stderr)
        return None
    return recorder


def _run_stream(
    args: argparse.Namespace,
    source: FrameSource,
    *,
    show: bool,
    mirror: bool = False,
    max_seconds: float | None = None,
    max_frames: int | None = None,
) -> int:
    import cv2

    from affectlab.pipeline import AffectPipeline
    from affectlab.recorder import SessionRecorder

    hud = _hud(args)
    recorder: SessionRecorder | None = SessionRecorder(args.record) if args.record else None
    output = _VideoOutput(args.output, source.fps, source.is_camera) if args.output else None
    window = "AffectLab"
    processed = 0
    started = time.perf_counter()
    try:
        with AffectPipeline(_pipeline_config(args)) as pipeline:
            for frame, t in source.frames():
                if mirror:
                    frame = cv2.flip(frame, 1)
                result = pipeline.process(frame, t)
                processed += 1
                if recorder is not None:
                    recorder.write(result)
                if show or output is not None:
                    canvas = hud.render(
                        frame, result, pipeline.last_pulse, recording=recorder is not None
                    )
                    if output is not None:
                        output.write(canvas, t)
                    if show:
                        cv2.imshow(window, canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        if key != 255:
                            new_recorder = _handle_key(
                                key, hud, canvas, recorder, output is not None
                            )
                            assert new_recorder is None or isinstance(new_recorder, SessionRecorder)
                            recorder = new_recorder
                elif processed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"\r{processed} frames  {processed / max(elapsed, 1e-6):5.1f} fps",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                if max_seconds is not None and t >= max_seconds:
                    break
                if max_frames is not None and processed >= max_frames:
                    break
    finally:
        if not show and processed >= 100:
            print(file=sys.stderr)
        if output is not None:
            if output.close():
                print(f"wrote {args.output} ({output.frames_written} frames)", file=sys.stderr)
            else:
                print(f"error: could not write {args.output}", file=sys.stderr)
        if recorder is not None:
            recorder.close()
            print(f"recorded {recorder.rows} rows to {recorder.path}", file=sys.stderr)
        if show:
            cv2.destroyAllWindows()
    return 0 if output is None or output.ok else 1


# ----------------------------------------------------------------------------- commands


def cmd_live(args: argparse.Namespace) -> int:
    from affectlab.sources import FrameSource

    if args.no_hud and not args.record and not args.output:
        print(
            "note: --no-hud without --record or --output analyses frames but shows and saves "
            "nothing; press Ctrl-C to stop",
            file=sys.stderr,
        )
    with FrameSource(args.camera, width=args.width, height=args.height) as source:
        hint = "Press Ctrl-C to stop." if args.no_hud else "Press q to quit, h for help."
        print(
            f"camera {args.camera}: {source.width}x{source.height} @ {source.fps:.0f} fps. {hint}",
            file=sys.stderr,
        )
        return _run_stream(
            args, source, show=not args.no_hud, mirror=args.mirror, max_seconds=args.max_seconds
        )


def cmd_video(args: argparse.Namespace) -> int:
    from affectlab.sources import FrameSource

    if args.record is None and args.output is None and not args.show:
        args.record = str(Path(args.path).with_suffix(".affectlab.csv"))
        print(f"no --record given; writing {args.record}", file=sys.stderr)
    with FrameSource(args.path) as source:
        print(
            f"{args.path}: {source.width}x{source.height} @ {source.fps:.1f} fps, "
            f"{source.frame_count} frames",
            file=sys.stderr,
        )
        return _run_stream(args, source, show=args.show, max_frames=args.max_frames)


def cmd_image(args: argparse.Namespace) -> int:
    import cv2

    from affectlab.facs import describe_action_units
    from affectlab.pipeline import AffectPipeline
    from affectlab.sources import read_image

    image = read_image(args.path)
    with AffectPipeline(_pipeline_config(args), video=False) as pipeline:
        result = pipeline.process(image, 0.0)
        if args.json:
            print(json.dumps(result.to_row(), indent=2))
        elif result.face is None:
            print("no face found")
        else:
            print(f"face: {result.face.bbox.as_tuple()}  source: {result.face.source}")
            if result.emotion is None:
                print("  emotion: not available (the facs backend needs MediaPipe landmarks)")
            else:
                for label, p in result.emotion.top(3):
                    print(f"  {label:10s} {p:5.1%}")
            if result.affect:
                print(f"valence {result.affect.valence:+.2f}  arousal {result.affect.arousal:+.2f}")
            if result.head_pose:
                hp = result.head_pose
                print(f"head pose: yaw {hp.yaw:+.1f}  pitch {hp.pitch:+.1f}  roll {hp.roll:+.1f}")
            aus = describe_action_units(result.action_units)
            if aus:
                print("action units:")
                for line in aus[:8]:
                    print(f"  {line}")
            if result.emotion is not None and result.emotion.evidence:
                print(f"evidence: {result.emotion.evidence}")
        if args.save:
            canvas = _hud(args).render(image, result, None)
            cv2.imwrite(args.save, canvas)
            print(f"saved {args.save}", file=sys.stderr)
    return 0 if result.face is not None else 1


def cmd_report(args: argparse.Namespace) -> int:
    from affectlab.report import build_report

    report_path = build_report(args.session, args.out, figures=not args.no_figures)
    print(f"report written to {report_path}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    ok = True
    print(f"affectlab {__version__}  python {platform.python_version()}  {platform.platform()}")
    for module in ("numpy", "scipy", "cv2", "mediapipe", "onnxruntime", "matplotlib", "deepface"):
        try:
            mod = __import__(module)
            print(f"  [ok] {module:12s} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # ImportError or a broken native install
            optional = module in ("matplotlib", "deepface")
            tag = "opt" if optional else "MISSING"
            print(f"  [{tag}] {module:12s} {type(exc).__name__}: {exc}")
            ok = ok and optional
    print("models:")
    for name, cached in cached_models().items():
        print(f"  [{'ok' if cached else '--'}] {name:16s} {model_path(name)}")
    try:
        from affectlab.landmarks import FaceLandmarkerEngine

        if cached_models()["face_landmarker"]:
            FaceLandmarkerEngine(video=False).close()
            print("  [ok] MediaPipe Face Landmarker loads")
        else:
            print("  [--] face_landmarker not downloaded yet (run: affectlab models download)")
    except Exception as exc:
        ok = False
        print(f"  [!!] MediaPipe Face Landmarker failed: {exc}")
    if args.camera is not None:
        try:
            from affectlab.sources import FrameSource

            with FrameSource(args.camera) as source:
                frame, _ = next(source.frames())
                print(f"  [ok] camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}")
        except Exception as exc:
            ok = False
            print(f"  [!!] camera {args.camera}: {exc}")
    print("all good" if ok else "problems found")
    return 0 if ok else 1


def cmd_models(args: argparse.Namespace) -> int:
    names = args.names or list(MODELS)
    for name in names:
        if name not in MODELS:
            print(f"unknown model {name!r}; known: {', '.join(MODELS)}", file=sys.stderr)
            return 2
    if args.action == "list":
        for name in names:
            spec = MODELS[name]
            state = "cached" if cached_models()[name] else "not downloaded"
            print(f"{name:16s} {state:15s} {spec.size_bytes / 1e6:5.1f} MB  {spec.license}")
            print(f"{'':16s} {spec.description}")
    elif args.action == "download":
        for name in names:
            path = ensure_model(name)
            print(f"{name}: {path}")
    else:
        for name in names:
            print(model_path(name))
    return 0


# ----------------------------------------------------------------------------- entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _quiet_native_logs()
    # --models-dir is passed to the model registry through the environment, but only for
    # the duration of this command: main() is importable and callable, and leaving the
    # variable set would hide the default cache from everything that runs afterwards.
    previous = os.environ.get(MODELS_DIR_ENV)
    if args.models_dir:
        os.environ[MODELS_DIR_ENV] = str(Path(args.models_dir).expanduser())
    try:
        return int(args.func(args))
    except (BackendUnavailable, ModelError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        print(f"error: {type(exc).__name__}: {exc} (run with -v for a traceback)", file=sys.stderr)
        return 1
    finally:
        if args.models_dir:
            if previous is None:
                os.environ.pop(MODELS_DIR_ENV, None)
            else:
                os.environ[MODELS_DIR_ENV] = previous


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
