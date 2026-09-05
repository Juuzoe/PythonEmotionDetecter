from __future__ import annotations

import numpy as np

from affectlab.hud import BG, MIN_PANEL_HEIGHT, PANEL_WIDTH, Hud, HudOptions, fit_canvas
from affectlab.types import (
    EMOTIONS,
    BBox,
    Dynamics,
    EmotionEstimate,
    FaceObservation,
    FrameResult,
    Vitals,
)


def result_with_face() -> FrameResult:
    probs = {e: (1.0 if e == "happiness" else 0.0) for e in EMOTIONS}
    return FrameResult(
        frame_index=1,
        timestamp=0.1,
        face=FaceObservation(bbox=BBox(50, 40, 100, 120)),
        action_units={"AU12": 0.9, "AU6": 0.4},
        emotion=EmotionEstimate(probs, "test", evidence={"AU12": 0.9}),
        affect=None,
        head_pose=None,
        vitals=Vitals(heart_rate_bpm=70.0, heart_rate_quality="good", blink_count=2),
        dynamics=Dynamics(),
        fps=29.0,
    )


def test_canvas_size_matches_render_and_is_even() -> None:
    frame = np.zeros((481, 641, 3), dtype=np.uint8)
    hud = Hud(HudOptions(show_panel=True))
    rendered = hud.render(frame, result_with_face())
    assert rendered.shape[:2] == (MIN_PANEL_HEIGHT, 641 + PANEL_WIDTH)
    size = hud.canvas_size(frame.shape)
    assert size == (641 + PANEL_WIDTH + 1, MIN_PANEL_HEIGHT)
    assert size[0] % 2 == 0 and size[1] % 2 == 0
    hud.options.show_panel = False
    assert hud.canvas_size(frame.shape) == (642, 482)
    assert hud.render(frame, result_with_face()).shape == frame.shape


def test_fit_canvas_pads_and_crops() -> None:
    canvas = np.full((100, 200, 3), 7, dtype=np.uint8)
    same = fit_canvas(canvas, (200, 100))
    assert same is canvas
    padded = fit_canvas(canvas, (220, 130))
    assert padded.shape == (130, 220, 3)
    assert (padded[:100, :200] == 7).all()
    assert tuple(int(v) for v in padded[-1, -1]) == BG
    cropped = fit_canvas(canvas, (150, 80))
    assert cropped.shape == (80, 150, 3) and (cropped == 7).all()


def test_render_handles_missing_face_and_emotion() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    empty = FrameResult(0, 0.0, None, {}, None, None, None, Vitals(), Dynamics(), 0.0)
    for panel in (True, False):
        out = Hud(HudOptions(show_panel=panel, blur_face=True)).render(frame, empty, recording=True)
        assert out.dtype == np.uint8 and out.ndim == 3
