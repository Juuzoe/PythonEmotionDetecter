"""Session reports: summary statistics, a Markdown write-up and figures.

Figures need matplotlib (``pip install "affectlab[report]"``); without it the
Markdown summary is still produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from affectlab.affect import CIRCUMPLEX
from affectlab.facs import AU_BY_CODE, AU_CODES
from affectlab.palette import (
    CHROME_LIGHT,
    EMOTION_HEX_LIGHT,
    EMOTION_ORDER,
    SEQUENTIAL_BLUE,
)
from affectlab.recorder import read_session
from affectlab.types import EMOTIONS

#: Spontaneous blink rate at rest (Bentivoglio et al., 1997), used as a reference line.
RESTING_BLINK_RATE = 17.0


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _stats(x: np.ndarray) -> dict[str, float] | None:
    x = _finite(x)
    if len(x) == 0:
        return None
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": len(x),
    }


def summarize(session: dict[str, np.ndarray]) -> dict[str, Any]:
    """Descriptive statistics for a recorded session."""
    t = session["t"]
    n = len(t)
    if n == 0:
        return {"frames": 0, "duration_s": 0.0}
    face = np.nan_to_num(session["face"], nan=0.0)
    labels = session["emotion"]
    with_face = [str(label) for label, f in zip(labels, face) if f == 1 and label]
    distribution = (
        {label: with_face.count(label) / len(with_face) for label in EMOTIONS} if with_face else {}
    )
    hr = session["hr_bpm"]
    quality = session["hr_quality"].astype(str)
    reliable = np.isfinite(hr) & np.isin(quality, ["good", "fair"])
    aus = {
        code: float(np.nanmean(session[code]))
        for code in AU_CODES
        if code in session and np.isfinite(session[code]).any()
    }
    blink_counts = _finite(session["blink_count"])
    backends = [str(b) for b in session["backend"] if b]
    return {
        "frames": n,
        "duration_s": float(np.nanmax(t) - np.nanmin(t)) if n > 1 else 0.0,
        "face_coverage": float(face.mean()),
        "fps": _stats(session["fps"]),
        "backend": backends[0] if backends else None,
        "emotion_distribution": distribution,
        "dominant_emotion": max(distribution, key=lambda k: distribution[k])
        if distribution
        else None,
        "valence": _stats(session["valence"]),
        "arousal": _stats(session["arousal"]),
        "inertia": _stats(session["inertia"]),
        "switch_rate": _stats(session["switch_rate"]),
        "heart_rate": _stats(hr[reliable]),
        "heart_rate_coverage": float(reliable.mean()),
        "rmssd": _stats(session["rmssd_ms"]),
        "breathing": _stats(session["breathing_bpm"]),
        "blink_count": int(blink_counts.max()) if len(blink_counts) else 0,
        "blink_rate": _stats(session["blink_rate"]),
        "perclos": _stats(session["perclos"]),
        "top_action_units": sorted(aus.items(), key=lambda kv: kv[1], reverse=True)[:6],
    }


def _fmt(stats: dict[str, float] | None, unit: str = "", nd: int = 2) -> str:
    if not stats:
        return "not available"
    return (
        f"{stats['mean']:.{nd}f}{unit} mean, {stats['median']:.{nd}f}{unit} median "
        f"(range {stats['min']:.{nd}f} to {stats['max']:.{nd}f}, n = {stats['n']})"
    )


def render_markdown(summary: dict[str, Any], figures: list[Path]) -> str:
    lines: list[str] = ["# AffectLab session report", ""]
    if summary.get("frames", 0) == 0:
        lines.append("The session file contains no frames.")
        return "\n".join(lines) + "\n"
    duration = summary["duration_s"]
    lines += [
        f"- Duration: {duration:.1f} s ({summary['frames']} frames)",
        f"- Frames with a face: {summary['face_coverage']:.0%}",
        f"- Emotion backend: {summary.get('backend') or 'unknown'}",
        "",
        "## Categorical emotion",
        "",
        "Share of face-bearing frames in which each label was dominant.",
        "",
        "| label | share |",
        "|---|---|",
    ]
    for label in EMOTION_ORDER:
        share = summary["emotion_distribution"].get(label)
        if share is not None:
            lines.append(f"| {label} | {share:.1%} |")
    lines += [
        "",
        "## Dimensional affect",
        "",
        f"- Valence: {_fmt(summary['valence'])}",
        f"- Arousal: {_fmt(summary['arousal'])}",
        f"- Emotional inertia (lag-1 autocorrelation of valence): {_fmt(summary['inertia'])}",
        f"- Dominant-label switches per minute: {_fmt(summary['switch_rate'], nd=1)}",
        "",
        "## Vitals (contactless, indicative only)",
        "",
        f"- Heart rate: {_fmt(summary['heart_rate'], ' bpm', 1)}; usable signal in "
        f"{summary['heart_rate_coverage']:.0%} of frames",
        f"- RMSSD (experimental): {_fmt(summary['rmssd'], ' ms', 0)}",
        f"- Breathing rate (experimental): {_fmt(summary['breathing'], ' per min', 1)}",
        f"- Blinks: {summary['blink_count']} in total; rate "
        f"{_fmt(summary['blink_rate'], ' per min', 1)} "
        f"(resting reference about {RESTING_BLINK_RATE:.0f} per min)",
        f"- PERCLOS: {_fmt(summary['perclos'], '', 3)}",
        "",
        "## Action units",
        "",
        "Mean intensity over the session, strongest first.",
        "",
        "| AU | name | mean intensity |",
        "|---|---|---|",
    ]
    for code, value in summary["top_action_units"]:
        lines.append(f"| {code} | {AU_BY_CODE[code].name} | {value:.2f} |")
    if figures:
        lines += ["", "## Figures", ""]
        for path in figures:
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")
    lines += [
        "",
        "## Caveats",
        "",
        "Facial movements are not a readout of felt emotion; the labels above describe "
        "expressions as an algorithm scored them. Heart rate, RMSSD, breathing and blink metrics "
        "come from an ordinary camera and are indicative, not clinical. See docs/ETHICS.md.",
        "",
    ]
    return "\n".join(lines)


def render_figures(session: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    """Write the standard figures; returns their paths (empty without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return []

    t = session["t"]
    if len(t) < 2:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(t) // 4000)
    sl = slice(None, None, step)
    ts = t[sl]
    c = CHROME_LIGHT
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            "axes.edgecolor": c["axis"],
            "axes.labelcolor": c["ink2"],
            "axes.titlecolor": c["ink"],
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "text.color": c["ink"],
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": c["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    blue = SEQUENTIAL_BLUE[3]
    cmap = LinearSegmentedColormap.from_list("affectlab_blue", list(SEQUENTIAL_BLUE))
    paths: list[Path] = []

    # 1. Emotion probabilities over time (stacked area, fixed order and colours).
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    series = [np.nan_to_num(session[f"p_{label}"][sl], nan=0.0) for label in EMOTION_ORDER]
    ax.stackplot(
        ts,
        series,
        labels=EMOTION_ORDER,
        colors=[EMOTION_HEX_LIGHT[label] for label in EMOTION_ORDER],
        edgecolor=c["surface"],
        linewidth=0.5,
    )
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("probability")
    ax.set_title("Emotion probabilities over time")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    path = out_dir / "emotion_timeline.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    # 2. Affect trajectory on the circumplex, coloured by time.
    fig, ax = plt.subplots(figsize=(6.4, 6), constrained_layout=True)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=c["axis"], linewidth=1.0))
    ax.axhline(0, color=c["axis"], linewidth=0.8)
    ax.axvline(0, color=c["axis"], linewidth=0.8)
    for label, (lv, la) in CIRCUMPLEX.items():
        if label != "neutral":
            ax.annotate(label, (lv, la), color=c["muted"], fontsize=8, ha="center", va="center")
    v = session["valence_smoothed"][sl]
    a = session["arousal_smoothed"][sl]
    ok = np.isfinite(v) & np.isfinite(a)
    if ok.any():
        ax.plot(v[ok], a[ok], color=c["grid"], linewidth=1.0, zorder=1)
        scatter = ax.scatter(v[ok], a[ok], c=ts[ok], cmap=cmap, s=12, edgecolors="none", zorder=2)
        fig.colorbar(scatter, ax=ax, shrink=0.75, label="time (s)")
        ax.scatter(v[ok][0], a[ok][0], s=70, facecolors="none", edgecolors=c["ink"], zorder=3)
        ax.annotate("start", (v[ok][0], a[ok][0]), xytext=(6, 6), textcoords="offset points")
        ax.scatter(v[ok][-1], a[ok][-1], s=70, color=c["ink"], zorder=3)
        ax.annotate("end", (v[ok][-1], a[ok][-1]), xytext=(6, -14), textcoords="offset points")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("valence (unpleasant to pleasant)")
    ax.set_ylabel("arousal (deactivated to activated)")
    ax.set_title("Affect trajectory on the circumplex")
    path = out_dir / "circumplex.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    # 3. Vitals: one measure per panel, shared time axis.
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.8), sharex=True, constrained_layout=True)
    hr = session["hr_bpm"][sl]
    quality = session["hr_quality"][sl].astype(str)
    usable = np.isfinite(hr) & np.isin(quality, ["good", "fair"])
    poor = np.isfinite(hr) & ~usable
    axes[0].plot(ts[usable], hr[usable], color=blue, linewidth=2.0)
    axes[0].scatter(ts[poor], hr[poor], s=6, color=c["muted"], zorder=1)
    axes[0].set_ylabel("heart rate (bpm)")
    axes[0].set_title("Heart rate from rPPG (line: usable quality, dots: poor quality)")
    blink_rate = session["blink_rate"][sl]
    axes[1].plot(ts, blink_rate, color=blue, linewidth=2.0)
    axes[1].axhline(RESTING_BLINK_RATE, color=c["axis"], linewidth=1.0, linestyle="--")
    axes[1].annotate(
        "resting average (Bentivoglio et al., 1997)",
        (ts[0], RESTING_BLINK_RATE),
        xytext=(4, 4),
        textcoords="offset points",
        color=c["muted"],
        fontsize=8,
    )
    axes[1].set_ylabel("blinks per minute")
    axes[1].set_title("Blink rate (60 s window)")
    axes[1].set_ylim(
        0,
        max(
            RESTING_BLINK_RATE * 1.3,
            float(np.nanmax(blink_rate)) * 1.1 if np.isfinite(blink_rate).any() else 0.0,
        ),
    )
    perclos = session["perclos"][sl] * 100.0
    axes[2].plot(ts, perclos, color=blue, linewidth=2.0)
    axes[2].set_ylim(
        0, max(5.0, float(np.nanmax(perclos)) * 1.2 if np.isfinite(perclos).any() else 0.0)
    )
    axes[2].set_ylabel("PERCLOS (%)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Eyelid closure (PERCLOS, 60 s window)")
    path = out_dir / "vitals.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    # 4. Action-unit intensities as a heatmap, strongest first.
    codes = [code for code in AU_CODES if code in session and np.isfinite(session[code]).any()]
    if codes:
        matrix = np.vstack([np.nan_to_num(session[code][sl], nan=0.0) for code in codes])
        order = np.argsort(-matrix.mean(axis=1))
        matrix = matrix[order]
        codes = [codes[i] for i in order]
        fig, ax = plt.subplots(figsize=(10, 0.28 * len(codes) + 1.6), constrained_layout=True)
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=(float(ts[0]), float(ts[-1]), len(codes) - 0.5, -0.5),
        )
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels([f"{code} {AU_BY_CODE[code].name}" for code in codes])
        ax.grid(False)
        ax.set_xlabel("time (s)")
        ax.set_title("Action unit intensity over time")
        fig.colorbar(image, ax=ax, shrink=0.8, label="intensity")
        path = out_dir / "action_units.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def build_report(
    session_path: str | Path, out_dir: str | Path | None = None, figures: bool = True
) -> Path:
    """Summarise ``session_path`` into ``out_dir/report.md`` (plus figures and summary.json)."""
    session_path = Path(session_path)
    session = read_session(session_path)
    target = (
        Path(out_dir)
        if out_dir
        else session_path.with_suffix("").parent / (session_path.stem + "_report")
    )
    target.mkdir(parents=True, exist_ok=True)
    summary = summarize(session)
    figure_paths = render_figures(session, target) if figures else []
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = target / "report.md"
    report_path.write_text(render_markdown(summary, figure_paths), encoding="utf-8")
    return report_path
