"""Core DSP helpers built on numpy only.

Everything in this module works on plain numpy arrays, so it can be reused
without touching the plotting or file IO layers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

EPS = 1e-12
DB_FLOOR = -160.0
WindowName = Literal["hann", "hamming", "blackman", "rect"]

#: Centre frequencies (Hz) of standard octave bands.
OCTAVE_BANDS: tuple[float, ...] = (
    31.5,
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
)


def rms(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Root mean square of ``x``. Returns a float for an empty input."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return np.sqrt(np.mean(np.square(x), axis=axis))


def peak(x: np.ndarray) -> float:
    """Largest absolute sample value."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x))) if x.size else 0.0


def amplitude_to_db(
    a: np.ndarray | float,
    *,
    ref: float = 1.0,
    floor: float = DB_FLOOR,
) -> np.ndarray | float:
    """Convert (linear) amplitude to dB, clamped at ``floor``."""
    a = np.abs(np.asarray(a, dtype=np.float64)) / ref
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 20.0 * np.log10(np.maximum(a, EPS))
    out = np.maximum(out, floor)
    return float(out) if out.ndim == 0 else out


def power_to_db(
    p: np.ndarray | float,
    *,
    ref: float = 1.0,
    floor: float = DB_FLOOR,
) -> np.ndarray | float:
    """Convert (linear) power to dB, clamped at ``floor``."""
    p = np.asarray(p, dtype=np.float64) / ref
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(p, EPS))
    out = np.maximum(out, floor)
    return float(out) if out.ndim == 0 else out


def window_function(name: WindowName = "hann", size: int = 2048) -> np.ndarray:
    """Periodic analysis window of ``size`` samples."""
    if size <= 0:
        raise ValueError("size must be positive")
    if size == 1:
        return np.ones(1, dtype=np.float64)
    n = np.arange(size, dtype=np.float64)
    if name == "hann":
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / size)
    elif name == "blackman":
        w = (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * n / size)
            + 0.08 * np.cos(4.0 * np.pi * n / size)
        )
    elif name == "rect":
        w = np.ones(size, dtype=np.float64)
    else:
        raise ValueError(f"unknown window {name!r}")
    return w


def frame_signal(
    x: np.ndarray,
    frame_size: int,
    hop_size: int | None = None,
    *,
    pad: bool = True,
) -> np.ndarray:
    """Split ``x`` into overlapping frames.

    Returns a read-only view of shape ``(n_frames, frame_size)``. If ``x`` is
    shorter than one frame it is zero-padded when ``pad`` is true, otherwise a
    ``ValueError`` is raised.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    hop_size = frame_size if hop_size is None else int(hop_size)
    if hop_size <= 0:
        raise ValueError("hop_size must be positive")
    if x.size < frame_size:
        if not pad:
            raise ValueError("signal shorter than one frame")
        x = np.pad(x, (0, frame_size - x.size))
    return np.lib.stride_tricks.sliding_window_view(x, frame_size)[::hop_size]


def stft(
    x: np.ndarray,
    sample_rate: float,
    *,
    frame_size: int = 2048,
    hop_size: int | None = None,
    window: WindowName = "hann",
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier transform.

    Returns ``(magnitude, freqs, times)`` where ``magnitude`` has shape
    ``(n_frames, frame_size // 2 + 1)`` and is normalised so that a
    full-scale sine reads ``1.0`` (0 dBFS) at its peak bin and a sine of
    amplitude 0.5 reads ``0.5`` (-6 dBFS).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if hop_size is None:
        hop_size = max(1, frame_size // 4)

    pad = frame_size // 2 if center else 0
    xp = np.pad(x, (pad, pad)) if pad else x
    frames = frame_signal(xp, frame_size, hop_size, pad=center)

    win = window_function(window, frame_size)
    spectrum = np.fft.rfft(frames * win, axis=1)
    magnitude = np.abs(spectrum) * (2.0 / win.sum())
    # Do not double single-sided DC (and Nyquist) bins.
    magnitude[:, 0] *= 0.5
    if frame_size % 2 == 0:
        magnitude[:, -1] *= 0.5

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate))
    times = (np.arange(frames.shape[0]) * hop_size - pad + frame_size / 2) / float(
        sample_rate
    )
    return magnitude, freqs, times


def peak_envelope(x: np.ndarray, n_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope of ``x`` split into ``n_buckets`` equal slices.

    Useful for drawing a waveform with more samples than pixels without
    losing transients (plain decimation would).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    n_buckets = int(max(1, min(n_buckets, x.size)))
    starts = np.linspace(0, x.size, n_buckets + 1).astype(np.int64)[:-1]
    lo = np.minimum.reduceat(x, starts)
    hi = np.maximum.reduceat(x, starts)
    return lo, hi


def short_term_rms(
    x: np.ndarray,
    sample_rate: float,
    *,
    window_s: float = 0.05,
    hop_s: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Short-term RMS level, e.g. 50 ms windows every 10 ms.

    Returns ``(times, rms_values)`` with times at the frame centres.
    """
    frame_size = max(1, round(window_s * sample_rate))
    hop_size = max(1, round(hop_s * sample_rate))
    frames = frame_signal(x, frame_size, hop_size)
    levels = np.sqrt(np.mean(np.square(frames), axis=1))
    times = (np.arange(frames.shape[0]) * hop_size + frame_size / 2) / float(
        sample_rate
    )
    return times, levels


def spectral_centroid(freqs: np.ndarray, power: np.ndarray) -> float:
    """Power-weighted mean frequency (Hz)."""
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * power) / total)


def spectral_rolloff(
    freqs: np.ndarray, power: np.ndarray, roll_percent: float = 0.85
) -> float:
    """Frequency below which ``roll_percent`` of the total power lies."""
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, roll_percent * total))
    return float(freqs[min(idx, freqs.size - 1)])


def refine_peak(freqs: np.ndarray, db: np.ndarray, index: int) -> tuple[float, float]:
    """Parabolic interpolation of a spectral peak given dB magnitudes.

    Returns ``(frequency_hz, level_db)`` at the interpolated vertex. Falls back
    to the raw bin when the peak sits on the edge of the spectrum.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    db = np.asarray(db, dtype=np.float64)
    if 0 < index < db.size - 1:
        a, b, c = float(db[index - 1]), float(db[index]), float(db[index + 1])
        denom = a - 2.0 * b + c
        if denom != 0.0:
            delta = 0.5 * (a - c) / denom
            if -0.5 <= delta <= 0.5:
                step = float(freqs[1] - freqs[0])
                level = b - 0.25 * (a - c) * delta
                return float(freqs[index] + delta * step), level
    return float(freqs[index]), float(db[index])


def spectral_flatness(power: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum.

    Near 0 for tonal signals, near 1 for white noise.
    """
    power = np.asarray(power, dtype=np.float64)
    if power.size == 0:
        return 0.0
    arithmetic = float(np.mean(power))
    if arithmetic <= 0:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(np.maximum(power, EPS)))))
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))


def band_energy(
    freqs: np.ndarray,
    power: np.ndarray,
    edges: tuple[float, ...] = OCTAVE_BANDS,
) -> np.ndarray:
    """Summed power per band.

    Each band spans one octave centred on its nominal frequency, i.e.
    ``[f / sqrt(2), f * sqrt(2))``. Only energy between the first band's lower
    edge and the last band's upper edge is counted, so the returned array does
    not necessarily sum to the total signal power.
    """
    values = []
    for centre in edges:
        lo = centre / np.sqrt(2.0)
        hi = centre * np.sqrt(2.0)
        mask = (freqs >= lo) & (freqs < hi)
        values.append(float(np.sum(power[mask])))
    return np.asarray(values, dtype=np.float64)
