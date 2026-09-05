from __future__ import annotations

import numpy as np
import pytest

from affectlab.rppg import (
    RppgBuffer,
    bandpass,
    estimate_breathing_rate,
    estimate_pulse,
    interbeat_intervals_ms,
    power_spectrum,
    quality_label,
    rmssd_ms,
)
from tests.conftest import synthetic_rgb


@pytest.mark.parametrize("method", ["pos", "chrom", "green"])
def test_recovers_heart_rate(method: str) -> None:
    t, rgb = synthetic_rgb(bpm=72.0)
    estimate = estimate_pulse(t, rgb, method=method)
    assert estimate is not None
    assert estimate.method == method
    assert abs(estimate.bpm - 72.0) < 3.0
    assert estimate.snr_db > 0.0
    assert estimate.quality in {"good", "fair", "poor"}
    assert len(estimate.pulse) > 0 and np.isfinite(estimate.pulse).all()


def test_recovers_other_rates_with_irregular_timestamps() -> None:
    for bpm in (55.0, 96.0, 130.0):
        t, rgb = synthetic_rgb(bpm=bpm, jitter=0.3, seed=3)
        estimate = estimate_pulse(t, rgb)
        assert estimate is not None
        assert abs(estimate.bpm - bpm) < 4.0, bpm


def test_returns_none_without_enough_signal() -> None:
    t, rgb = synthetic_rgb(seconds=5.0)
    assert estimate_pulse(t, rgb, min_seconds=8.0) is None
    assert estimate_pulse(t[:5], rgb[:5]) is None


def test_clean_signal_has_higher_snr_than_noisy() -> None:
    t, clean = synthetic_rgb(noise=0.05)
    _, noisy = synthetic_rgb(noise=2.0)
    a = estimate_pulse(t, clean)
    b = estimate_pulse(t, noisy)
    assert a is not None and b is not None
    assert a.snr_db > b.snr_db


def test_bandpass_removes_dc_and_keeps_pulse_band() -> None:
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    x = 100.0 + np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 8.0 * t)
    y = bandpass(x, fs)
    assert abs(y.mean()) < 1e-2
    freqs, power = power_spectrum(y, fs, (0.5, 14.0))
    assert freqs[np.argmax(power)] == pytest.approx(1.2, abs=0.05)


def test_short_input_does_not_crash_bandpass() -> None:
    y = bandpass(np.arange(5.0), 30.0)
    assert len(y) == 5


def test_interbeat_intervals_and_rmssd() -> None:
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    pulse = np.sin(2 * np.pi * 1.0 * t)
    ibi = interbeat_intervals_ms(pulse, fs)
    assert len(ibi) >= 15
    assert np.allclose(ibi, 1000.0, atol=40.0)
    assert rmssd_ms(ibi) is not None and rmssd_ms(ibi) < 40.0
    assert rmssd_ms(np.array([800.0, 810.0])) is None


def test_quality_labels() -> None:
    assert quality_label(10.0) == "good"
    assert quality_label(3.0) == "fair"
    assert quality_label(-1.0) == "poor"


def test_buffer_window_and_gap_reset() -> None:
    buffer = RppgBuffer(window_seconds=5.0, gap_reset_seconds=1.0)
    for i in range(300):
        buffer.append(i / 30.0, np.array([1.0, 2.0, 3.0]))
    assert buffer.seconds <= 5.0 + 1e-9
    assert len(buffer) <= 151
    buffer.append(20.0, np.array([1.0, 2.0, 3.0]))  # a 10 s gap clears the buffer
    assert len(buffer) == 1
    buffer.append(19.0, np.array([1.0, 2.0, 3.0]))  # out-of-order samples are ignored
    assert len(buffer) == 1
    times, values = buffer.arrays()
    assert times.shape == (1,) and values.shape == (1, 3)


def test_breathing_rate() -> None:
    fs = 30.0
    t = np.arange(0, 30, 1 / fs)
    y = (
        0.5
        + 0.02 * np.sin(2 * np.pi * 0.25 * t)
        + 0.001 * np.random.default_rng(0).normal(size=len(t))
    )
    rate = estimate_breathing_rate(t, y)
    assert rate is not None
    assert abs(rate - 15.0) < 2.5
    assert estimate_breathing_rate(t[:100], y[:100]) is None
