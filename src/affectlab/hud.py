"""Heads-up display: draws the pipeline's output onto frames with OpenCV.

The HUD is deliberately dependency-free (no Qt, no matplotlib) so that it
works wherever ``cv2.imshow`` works and can be written straight into an
annotated video file.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from affectlab import geometry
from affectlab.affect import QUADRANTS, quadrant
from affectlab.facs import AU_BY_CODE
from affectlab.palette import CHROME_DARK, EMOTION_HEX_DARK, STATUS, hex_to_bgr
from affectlab.rppg import PulseEstimate
from affectlab.types import EMOTIONS, BBox, FrameResult

PANEL_WIDTH = 340
#: The panel needs this much height for every section; shorter frames are padded.
MIN_PANEL_HEIGHT = 760
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colours, derived from the shared palette (dark surface).
BG = hex_to_bgr(CHROME_DARK["surface"])
FG = hex_to_bgr(CHROME_DARK["ink"])
MUTED = hex_to_bgr(CHROME_DARK["muted"])
GRID = (58, 56, 54)
TRACK = (52, 50, 48)
GOOD = hex_to_bgr(STATUS["good"])
WARN = hex_to_bgr(STATUS["warning"])
BAD = hex_to_bgr(STATUS["critical"])
EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    label: hex_to_bgr(value) for label, value in EMOTION_HEX_DARK.items()
}


def valence_color(valence: float) -> tuple[int, int, int]:
    """Red for negative valence, grey around zero, green for positive."""
    v = float(np.clip(valence, -1.0, 1.0))
    neutral = np.array([170, 170, 170], dtype=float)
    target = np.array(GOOD if v >= 0 else BAD, dtype=float)
    color = neutral + abs(v) * (target - neutral)
    return int(color[0]), int(color[1]), int(color[2])


def text(
    img: np.ndarray,
    s: str,
    org: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = FG,
    thickness: int = 1,
) -> None:
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def bar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fraction: float,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), TRACK, -1)
    fill = round(w * float(np.clip(fraction, 0.0, 1.0)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


def pixelate(img: np.ndarray, box: BBox, blocks: int = 12) -> None:
    """In-place privacy blur of ``box`` by heavy down- and up-sampling."""
    region = box.crop(img)
    if region.size == 0:
        return
    small = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    img[box.y : box.y2, box.x : box.x2] = cv2.resize(
        small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def fit_canvas(canvas: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pad (bottom/right, background colour) or crop ``canvas`` to ``size`` = (width, height).

    Video writers silently drop frames whose size differs from the first one,
    so every frame handed to a writer goes through this.
    """
    width, height = size
    h, w = canvas.shape[:2]
    if (w, h) == (width, height):
        return canvas
    out = np.empty((height, width, 3), dtype=np.uint8)
    out[:] = BG
    ch, cw = min(h, height), min(w, width)
    out[:ch, :cw] = canvas[:ch, :cw]
    return out


@dataclass(slots=True)
class HudOptions:
    show_panel: bool = True
    show_landmarks: bool = False
    show_rois: bool = False
    show_help: bool = True
    blur_face: bool = False


class Hud:
    """Stateful renderer: keeps a short affect trajectory for the circumplex plot."""

    def __init__(self, options: HudOptions | None = None, trajectory_length: int = 120) -> None:
        self.options = options or HudOptions()
        self.trajectory: deque[tuple[float, float]] = deque(maxlen=trajectory_length)

    def canvas_size(self, frame_shape: tuple[int, ...]) -> tuple[int, int]:
        """(width, height) of the rendered canvas for a frame of ``frame_shape``, even-sized."""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if self.options.show_panel:
            w, h = w + PANEL_WIDTH, max(h, MIN_PANEL_HEIGHT)
        return w + (w % 2), h + (h % 2)

    def render(
        self,
        frame_bgr: np.ndarray,
        result: FrameResult,
        pulse: PulseEstimate | None = None,
        *,
        recording: bool = False,
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        if result.face is not None:
            self._draw_face(canvas, result)
        elif result.emotion is None and self.options.show_panel is False:
            text(canvas, "no face", (10, 24), 0.6, MUTED, 1)
        if result.affect is not None:
            self.trajectory.append((result.affect.valence_smoothed, result.affect.arousal_smoothed))
        if recording:
            cv2.circle(canvas, (18, 18), 7, BAD, -1)
            text(canvas, "REC", (30, 24), 0.5, BAD, 1)
        if not self.options.show_panel:
            return canvas
        height = max(canvas.shape[0], MIN_PANEL_HEIGHT)
        if canvas.shape[0] < height:
            pad = np.full((height - canvas.shape[0], canvas.shape[1], 3), BG, dtype=np.uint8)
            canvas = np.vstack([canvas, pad])
        panel = self._render_panel(height, result, pulse)
        return np.hstack([canvas, panel])

    # ------------------------------------------------------------------ frame

    def _draw_face(self, canvas: np.ndarray, result: FrameResult) -> None:
        face = result.face
        assert face is not None
        box = face.bbox
        if self.options.blur_face:
            pixelate(canvas, box)
        color = valence_color(result.affect.valence_smoothed if result.affect else 0.0)
        cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), color, 2)

        if result.emotion is not None:
            label = f"{result.emotion.dominant} {result.emotion.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            ly = max(th + 6, box.y - 8)
            cv2.rectangle(canvas, (box.x, ly - th - 6), (box.x + tw + 8, ly + 4), BG, -1)
            text(canvas, label, (box.x + 4, ly), 0.6, color, 2)

        if self.options.show_landmarks and face.landmarks is not None:
            for x, y in face.landmarks[:, :2].astype(int):
                cv2.circle(canvas, (int(x), int(y)), 1, (255, 255, 255), -1)

        if self.options.show_rois and face.landmarks is not None:
            for poly in geometry.skin_polygons(face.landmarks):
                pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, WARN, 1, cv2.LINE_AA)

        if result.head_pose is not None:
            hp = result.head_pose
            text(
                canvas,
                f"yaw {hp.yaw:+.0f}  pitch {hp.pitch:+.0f}  roll {hp.roll:+.0f}",
                (box.x, min(canvas.shape[0] - 6, box.y2 + 18)),
                0.42,
                FG,
            )

    # ------------------------------------------------------------------ panel

    def _render_panel(
        self, height: int, result: FrameResult, pulse: PulseEstimate | None
    ) -> np.ndarray:
        panel = np.full((height, PANEL_WIDTH, 3), BG, dtype=np.uint8)
        x0, y = 14, 24

        text(panel, "AffectLab", (x0, y), 0.7, FG, 2)
        backend = (
            result.emotion.backend
            if result.emotion
            else ("no face" if result.face is None else "-")
        )
        fps = f"{result.fps:4.1f} fps  " if result.fps > 0 else ""
        text(panel, f"{fps}{backend}", (x0 + 130, y), 0.45, MUTED)
        y += 14
        cv2.line(panel, (x0, y), (PANEL_WIDTH - x0, y), GRID, 1)
        y += 18

        y = self._section_emotion(panel, x0, y, result)
        y = self._section_affect(panel, x0, y, result)
        y = self._section_action_units(panel, x0, y, result)
        y = self._section_vitals(panel, x0, y, result, pulse)
        y = self._section_dynamics(panel, x0, y, result)

        if self.options.show_help and height - y > 40:
            hy = height - 12
            text(panel, "q quit  l landmarks  o skin  p panel  b blur", (x0, hy - 14), 0.36, MUTED)
            text(panel, "r record  s snapshot  h help", (x0, hy), 0.36, MUTED)
        return panel

    @staticmethod
    def _heading(panel: np.ndarray, x: int, y: int, title: str) -> int:
        text(panel, title, (x, y), 0.42, MUTED)
        return y + 12

    def _section_emotion(self, panel: np.ndarray, x: int, y: int, result: FrameResult) -> int:
        y = self._heading(panel, x, y, "EMOTION")
        probs = result.emotion.probabilities if result.emotion else {}
        dominant = result.emotion.dominant if result.emotion else None
        row_h = 16
        for label in EMOTIONS:
            p = float(probs.get(label, 0.0))
            color = EMOTION_COLORS[label] if label == dominant else MUTED
            text(panel, label, (x, y + 9), 0.4, FG if label == dominant else MUTED)
            bar(panel, x + 88, y, 170, row_h - 5, p, color)
            text(panel, f"{p:4.0%}", (x + 266, y + 9), 0.4, FG if label == dominant else MUTED)
            y += row_h
        if result.emotion and result.emotion.evidence:
            items = list(result.emotion.evidence.items())[:4]
            evidence = "  ".join(f"{k} {v:.2f}" for k, v in items)
            text(panel, evidence[:52], (x, y + 8), 0.36, MUTED)
        y += 14
        return y + 10

    def _section_affect(self, panel: np.ndarray, x: int, y: int, result: FrameResult) -> int:
        y = self._heading(panel, x, y, "AFFECT  (Russell circumplex)")
        size = 150
        cx, cy = x + size // 2, y + size // 2
        r = size // 2
        cv2.circle(panel, (cx, cy), r, GRID, 1, cv2.LINE_AA)
        cv2.line(panel, (cx - r, cy), (cx + r, cy), GRID, 1)
        cv2.line(panel, (cx, cy - r), (cx, cy + r), GRID, 1)
        text(panel, "arousal", (cx - 22, cy - r - 4), 0.33, MUTED)
        text(panel, "-", (cx - r - 10, cy + 4), 0.4, MUTED)
        text(panel, "+", (cx + r + 3, cy + 4), 0.4, MUTED)
        n = len(self.trajectory)
        for i, (v, a) in enumerate(self.trajectory):
            px = int(cx + v * (r - 4))
            py = int(cy - a * (r - 4))
            shade = int(60 + 160 * (i + 1) / max(1, n))
            cv2.circle(panel, (px, py), 2, (shade, shade, shade), -1, cv2.LINE_AA)
        affect = result.affect
        if affect is not None:
            px = int(cx + affect.valence_smoothed * (r - 4))
            py = int(cy - affect.arousal_smoothed * (r - 4))
            cv2.circle(panel, (px, py), 6, valence_color(affect.valence_smoothed), -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), 6, FG, 1, cv2.LINE_AA)
            tx = x + size + 14
            text(panel, f"valence {affect.valence_smoothed:+.2f}", (tx, y + 20), 0.45, FG)
            text(panel, f"arousal {affect.arousal_smoothed:+.2f}", (tx, y + 40), 0.45, FG)
            q = QUADRANTS[quadrant(affect.valence_smoothed, affect.arousal_smoothed)]
            text(panel, q, (tx, y + 62), 0.4, MUTED)
        return y + size + 16

    def _section_action_units(self, panel: np.ndarray, x: int, y: int, result: FrameResult) -> int:
        y = self._heading(panel, x, y, "ACTION UNITS  (FACS)")
        if not result.action_units:
            text(panel, "no landmarks", (x, y + 10), 0.4, MUTED)
            return y + 26
        top = sorted(result.action_units.items(), key=lambda kv: kv[1], reverse=True)[:6]
        for code, value in top:
            name = AU_BY_CODE[code].name
            text(panel, f"{code} {name}"[:24], (x, y + 9), 0.38, FG if value >= 0.2 else MUTED)
            bar(panel, x + 176, y, 100, 9, value, WARN if value >= 0.2 else MUTED)
            text(panel, f"{value:.2f}", (x + 284, y + 9), 0.38, MUTED)
            y += 15
        return y + 10

    def _section_vitals(
        self,
        panel: np.ndarray,
        x: int,
        y: int,
        result: FrameResult,
        pulse: PulseEstimate | None,
    ) -> int:
        y = self._heading(panel, x, y, "VITALS  (contactless, indicative)")
        v = result.vitals
        if v.heart_rate_bpm is not None:
            color = {"good": GOOD, "fair": WARN}.get(v.heart_rate_quality, BAD)
            text(panel, f"{v.heart_rate_bpm:3.0f} bpm", (x, y + 22), 0.8, color, 2)
            snr = f"{v.heart_rate_snr_db:+.1f} dB" if v.heart_rate_snr_db is not None else ""
            text(panel, f"{v.heart_rate_quality} {snr}", (x + 120, y + 22), 0.42, MUTED)
        else:
            text(panel, v.heart_rate_quality, (x, y + 22), 0.6, MUTED, 1)
            if v.signal_seconds > 0:
                text(panel, f"{v.signal_seconds:4.1f} s of signal", (x + 130, y + 22), 0.42, MUTED)
        y += 30
        if pulse is not None and len(pulse.pulse) > 4:
            self._waveform(panel, x, y, PANEL_WIDTH - 2 * x, 34, pulse)
            y += 40
        lines = []
        if v.blink_rate_per_min is not None:
            lines.append(f"blinks {v.blink_rate_per_min:4.1f}/min")
        else:
            lines.append(f"blinks {v.blink_count}")
        if v.perclos is not None:
            lines.append(f"PERCLOS {v.perclos:4.0%}")
        if v.rmssd_ms is not None:
            lines.append(f"RMSSD {v.rmssd_ms:3.0f} ms")
        if v.breathing_rate_bpm is not None:
            lines.append(f"breath {v.breathing_rate_bpm:3.0f}/min")
        text(panel, "   ".join(lines[:2]), (x, y + 10), 0.42, FG)
        if len(lines) > 2:
            text(panel, "   ".join(lines[2:4]), (x, y + 28), 0.42, FG)
            y += 18
        return y + 26

    @staticmethod
    def _waveform(panel: np.ndarray, x: int, y: int, w: int, h: int, pulse: PulseEstimate) -> None:
        samples = pulse.pulse[-int(6 * pulse.fs) :]
        if len(samples) < 2:
            return
        cv2.rectangle(panel, (x, y), (x + w, y + h), TRACK, -1)
        xs = np.linspace(x, x + w - 1, len(samples))
        scale = max(1e-6, float(np.abs(samples).max()))
        ys = y + h / 2 - samples / scale * (h / 2 - 3)
        pts = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, GOOD, 1, cv2.LINE_AA)

    def _section_dynamics(self, panel: np.ndarray, x: int, y: int, result: FrameResult) -> int:
        y = self._heading(panel, x, y, "DYNAMICS  (last 60 s)")
        d = result.dynamics
        inertia = f"{d.inertia:+.2f}" if d.inertia is not None else "  -  "
        sd = f"{d.valence_sd:.2f}" if d.valence_sd is not None else " - "
        switches = f"{d.switch_rate_per_min:.1f}/min" if d.switch_rate_per_min is not None else "-"
        text(panel, f"inertia {inertia}   valence sd {sd}", (x, y + 10), 0.42, FG)
        text(panel, f"switches {switches}", (x, y + 28), 0.42, FG)
        return y + 44
