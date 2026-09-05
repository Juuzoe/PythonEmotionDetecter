"""Colours shared by the live HUD (dark surface) and report figures (light surface).

Each emotion has one fixed colour everywhere, so identity never depends on
rank or ordering. The stacking order :data:`EMOTION_ORDER` was checked for
adjacent-pair separation under simulated colour-vision deficiency in both
modes. Status colours are reserved for signal quality and never reused for a
series.
"""

from __future__ import annotations

EMOTION_ORDER: tuple[str, ...] = (
    "neutral",
    "happiness",
    "sadness",
    "surprise",
    "fear",
    "anger",
    "disgust",
    "contempt",
)

EMOTION_HEX_LIGHT: dict[str, str] = {
    "neutral": "#898781",
    "happiness": "#eda100",
    "sadness": "#2a78d6",
    "surprise": "#eb6834",
    "anger": "#e34948",
    "fear": "#4a3aa7",
    "disgust": "#008300",
    "contempt": "#e87ba4",
}

EMOTION_HEX_DARK: dict[str, str] = {
    "neutral": "#898781",
    "happiness": "#c98500",
    "sadness": "#3987e5",
    "surprise": "#d95926",
    "anger": "#e66767",
    "fear": "#9085e9",
    "disgust": "#008300",
    "contempt": "#d55181",
}

#: Single-hue ramp for magnitude (heatmaps, time colouring), light to dark.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

STATUS: dict[str, str] = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

CHROME_LIGHT: dict[str, str] = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

CHROME_DARK: dict[str, str] = {
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    r, g, b = hex_to_rgb(value)
    return b, g, r
