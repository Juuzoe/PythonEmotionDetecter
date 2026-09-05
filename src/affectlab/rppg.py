"""Remote photoplethysmography (rPPG): heart rate from skin colour.

Each heartbeat pushes blood into the skin's capillary bed and changes how
much light it absorbs. The change is invisible to the eye but measurable as
a tiny periodic fluctuation in the mean colour of a skin patch filmed by an
ordinary camera (Verkruysse, Svaasand & Nelson, 2008; Poh, McDuff & Picard,
2010).

Three classic extraction methods are implemented:

* ``green``: the normalised green channel (Verkruysse et al., 2008);
* ``chrom``: chrominance-based pulse extraction (de Haan & Jeanne, 2013);
* ``pos``: plane-orthogonal-to-skin (Wang, den Brinker, Stuijk & de Haan,
  2017), the default.

All operate on a window of mean RGB values, followed by detrending, a
Butterworth band-pass over the physiological range and a spectral peak
search. Signal quality is reported as the spectral signal-to-noise ratio
of de Haan & Jeanne (2013). Heart-rate variability (RMSSD) from a 30 fps
camera is indicative only and is labelled as such throughout.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal

#: Pulse band in Hz: 42 to 180 beats per minute.
DEFAULT_BAND: tuple[float, float] = (0.7, 3.0)
#: Breathing band in Hz: 6 to 30 breaths per minute.
BREATHING_BAND: tuple[float, float] = (0.1, 0.5)
METHODS: tuple[str, ...] = ("pos", "chrom", "green")


@dataclass(slots=True)
class PulseEstimate:
    """A heart-rate estimate plus the intermediate signals used to obtain it."""

    bpm: float
    snr_db: float
    quality: str
    method: str
    seconds: float
    fs: float
    pulse: np.ndarray
    freqs: np.ndarray
    power: np.ndarray
    rmssd_ms: float | None = None


def quality_label(snr_db: float) -> str:
    if snr_db >= 6.0:
        return "good"
    if snr_db >= 2.0:
        return "fair"
    return "poor"


def resample_uniform(
    times: np.ndarray, values: np.ndarray, fs: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Linearly resample irregular samples onto a uniform grid.

    Returns ``(grid_times, values_on_grid, fs)``. ``values`` may be 1-D or
    (N, channels).
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("need at least two increasing timestamps")
    if fs is None:
        fs = (len(t) - 1) / (t[-1] - t[0])
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    out = np.column_stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])])
    return grid, out, float(fs)


def bandpass(
    x: np.ndarray, fs: float, band: tuple[float, float] = DEFAULT_BAND, order: int = 3
) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0, robust to short inputs."""
    x = np.asarray(x, dtype=float)
    low, high = band
    high = min(high, 0.45 * fs)
    if low >= high or len(x) < 8:
        return x - x.mean(axis=0)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    padlen = min(len(x) - 1, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)


def _normalise(c: np.ndarray) -> np.ndarray:
    mean = c.mean(axis=0)
    mean[mean == 0] = 1.0
    return c / mean


def method_green(rgb: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Verkruysse et al. (2008): the normalised green channel."""
    return _normalise(rgb)[:, 1] - 1.0


def method_chrom(rgb: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """de Haan & Jeanne (2013): chrominance signals X and Y, alpha-tuned."""
    cn = _normalise(rgb)
    xs = 3.0 * cn[:, 0] - 2.0 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf = bandpass(xs, fs, band)
    yf = bandpass(ys, fs, band)
    alpha = xf.std() / (yf.std() + 1e-12)
    return xf - alpha * yf


_POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def method_pos(
    rgb: np.ndarray, fs: float, band: tuple[float, float], window_seconds: float = 1.6
) -> np.ndarray:
    """Wang et al. (2017): plane-orthogonal-to-skin with overlap-adding."""
    n = len(rgb)
    length = max(3, min(n, round(window_seconds * fs)))
    h = np.zeros(n)
    for end in range(length, n + 1):
        seg = _normalise(rgb[end - length : end])
        s = seg @ _POS_PROJECTION.T
        s1, s2 = s[:, 0], s[:, 1]
        pulse = s1 + (s1.std() / (s2.std() + 1e-12)) * s2
        h[end - length : end] += pulse - pulse.mean()
    return h


def extract_pulse(
    rgb: np.ndarray, fs: float, method: str = "pos", band: tuple[float, float] = DEFAULT_BAND
) -> np.ndarray:
    """Run one extraction method, then detrend, band-pass and standardise."""
    rgb = np.asarray(rgb, dtype=float)
    if method == "pos":
        raw = method_pos(rgb, fs, band)
    elif method == "chrom":
        raw = method_chrom(rgb, fs, band)
    elif method == "green":
        raw = method_green(rgb, fs, band)
    else:
        raise ValueError(f"unknown rPPG method {method!r}; choose from {METHODS}")
    pulse = bandpass(signal.detrend(raw), fs, band)
    std = pulse.std()
    centred = pulse - pulse.mean()
    return centred / std if std > 0 else centred


def power_spectrum(
    x: np.ndarray, fs: float, band: tuple[float, float], oversample: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed, zero-padded periodogram restricted to ``band``."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return np.array([]), np.array([])
    nfft = int(2 ** math.ceil(math.log2(n * oversample)))
    spectrum = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[mask], spectrum[mask]


def spectral_snr_db(
    freqs: np.ndarray,
    power: np.ndarray,
    f0: float,
    fundamental_halfwidth: float = 0.1,
    harmonic_halfwidth: float = 0.2,
) -> float:
    """Power near the peak and its first harmonic versus everything else in band."""
    if len(freqs) == 0:
        return float("nan")
    signal_mask = (np.abs(freqs - f0) <= fundamental_halfwidth) | (
        np.abs(freqs - 2.0 * f0) <= harmonic_halfwidth
    )
    signal_power = float(power[signal_mask].sum())
    noise_power = float(power[~signal_mask].sum())
    if noise_power <= 1e-20:
        return 40.0
    if signal_power <= 1e-20:
        return -40.0
    return 10.0 * math.log10(signal_power / noise_power)


def interbeat_intervals_ms(
    pulse: np.ndarray, fs: float, min_ibi_ms: float = 300.0, max_ibi_ms: float = 1500.0
) -> np.ndarray:
    """Peak-to-peak intervals of a standardised pulse waveform, in milliseconds."""
    pulse = np.asarray(pulse, dtype=float)
    distance = max(1, round(fs * min_ibi_ms / 1000.0))
    std = float(pulse.std())
    peaks, _ = signal.find_peaks(
        pulse, distance=distance, prominence=0.5 * std if std > 0 else None
    )
    if len(peaks) < 2:
        return np.array([])
    ibi = np.diff(peaks) / fs * 1000.0
    return ibi[(ibi >= min_ibi_ms) & (ibi <= max_ibi_ms)]


def rmssd_ms(ibi: np.ndarray, min_intervals: int = 5) -> float | None:
    """Root mean square of successive inter-beat differences."""
    ibi = np.asarray(ibi, dtype=float)
    if len(ibi) < min_intervals:
        return None
    d = np.diff(ibi)
    return float(math.sqrt(float((d * d).mean())))


def estimate_pulse(
    times: np.ndarray,
    rgb: np.ndarray,
    method: str = "pos",
    band: tuple[float, float] = DEFAULT_BAND,
    min_seconds: float = 8.0,
    min_samples: int = 16,
) -> PulseEstimate | None:
    """Estimate heart rate from timestamped mean RGB values of a skin region.

    Returns None while there is not yet enough signal.
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(rgb, dtype=float)
    if len(t) < min_samples or c.ndim != 2 or c.shape[1] != 3 or len(c) != len(t):
        return None
    seconds = float(t[-1] - t[0])
    if seconds < min_seconds:
        return None
    _, uniform, fs = resample_uniform(t, c)
    effective_band = (band[0], min(band[1], 0.45 * fs))
    if effective_band[1] <= effective_band[0]:
        return None
    pulse = extract_pulse(uniform, fs, method, effective_band)
    freqs, power = power_spectrum(pulse, fs, effective_band)
    if len(freqs) == 0 or not np.isfinite(power).all() or power.max() <= 0:
        return None
    peak = int(np.argmax(power))
    f0 = float(freqs[peak])
    snr = spectral_snr_db(freqs, power, f0)
    ibi = interbeat_intervals_ms(pulse, fs)
    return PulseEstimate(
        bpm=60.0 * f0,
        snr_db=snr,
        quality=quality_label(snr),
        method=method,
        seconds=seconds,
        fs=fs,
        pulse=pulse,
        freqs=freqs,
        power=power,
        rmssd_ms=rmssd_ms(ibi),
    )


def estimate_breathing_rate(
    times: np.ndarray,
    values: np.ndarray,
    band: tuple[float, float] = BREATHING_BAND,
    min_seconds: float = 15.0,
) -> float | None:
    """Breaths per minute from a slowly oscillating signal such as head height.

    Experimental: the spectral resolution of a short window is coarse.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    if len(t) < 16 or len(x) != len(t) or t[-1] - t[0] < min_seconds:
        return None
    _, uniform, fs = resample_uniform(t, x)
    if fs < 2.0 * band[1]:
        return None
    y = bandpass(signal.detrend(uniform[:, 0]), fs, band, order=2)
    freqs, power = power_spectrum(y, fs, band)
    if len(freqs) == 0 or power.max() <= 0:
        return None
    return float(60.0 * freqs[int(np.argmax(power))])


class RppgBuffer:
    """Rolling buffer of timestamped mean RGB samples.

    Samples older than ``window_seconds`` are dropped. A gap longer than
    ``gap_reset_seconds`` (the face was lost) clears the buffer, because the
    signal on either side of a gap is not continuous.
    """

    def __init__(self, window_seconds: float = 20.0, gap_reset_seconds: float = 1.0) -> None:
        self.window = float(window_seconds)
        self.gap = float(gap_reset_seconds)
        self._t: deque[float] = deque()
        self._rgb: deque[tuple[float, float, float]] = deque()

    def append(self, t: float, rgb: np.ndarray) -> None:
        if self._t:
            last = self._t[-1]
            if t <= last:
                return
            if t - last > self.gap:
                self.clear()
        self._t.append(float(t))
        self._rgb.append((float(rgb[0]), float(rgb[1]), float(rgb[2])))
        cutoff = t - self.window
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

    def clear(self) -> None:
        self._t.clear()
        self._rgb.clear()

    def __len__(self) -> int:
        return len(self._t)

    @property
    def seconds(self) -> float:
        return float(self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self._t, dtype=float), np.asarray(self._rgb, dtype=float).reshape(-1, 3)
