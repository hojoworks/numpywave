"""Synthetic test signals so you can try the analyser without a real file."""

from __future__ import annotations

import numpy as np


def _fade(n: int, sample_rate: int, fade_s: float = 0.02) -> np.ndarray:
    """Linear fade in/out envelope of ``n`` samples."""
    ramp = max(1, round(fade_s * sample_rate))
    env = np.ones(n)
    ramp = min(ramp, n)
    env[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
    env[-ramp:] *= np.linspace(1.0, 0.0, ramp, endpoint=False)
    return env


def make_demo_signal(
    duration: float = 8.0,
    sample_rate: int = 44100,
    *,
    seed: int = 7,
) -> np.ndarray:
    """Build a short, varied mono test signal.

    Sections: harmonic chord, linear chirp, noise burst, silence, louder chord
    and a click train. Peak is normalised to 0.95, so no clipping.
    """
    rng = np.random.default_rng(seed)
    n = round(duration * sample_rate)
    x = np.zeros(n)

    def put(start: float, end: float, signal: np.ndarray, gain: float = 1.0) -> None:
        i = round(start * sample_rate)
        length = round(end * sample_rate) - i
        length = max(0, min(n - i, length, signal.size))
        if length:
            x[i : i + length] += gain * signal[:length]

    total = max(float(duration), 0.5)

    # 1) soft harmonic chord with tremolo
    i0, i1 = 0.0, total * 0.22
    n1 = int((i1 - i0) * sample_rate)
    tt = np.arange(n1) / sample_rate
    chord = (
        0.30 * np.sin(2 * np.pi * 220.0 * tt)
        + 0.18 * np.sin(2 * np.pi * 330.0 * tt + 0.3)
        + 0.10 * np.sin(2 * np.pi * 440.0 * tt + 1.1)
    )
    put(i0, i1, chord * _fade(n1, sample_rate, 0.15) * (0.85 + 0.15 * np.sin(2 * np.pi * 5 * tt)))

    # 2) linear chirp 200 Hz -> 4 kHz
    i0, i1 = total * 0.22, total * 0.42
    n2 = int((i1 - i0) * sample_rate)
    tt = np.arange(n2) / sample_rate
    f0, f1 = 200.0, 4000.0
    phase = 2 * np.pi * (f0 * tt + (f1 - f0) / (2 * (i1 - i0)) * tt**2)
    put(i0, i1, 0.35 * np.sin(phase) * _fade(n2, sample_rate, 0.05))

    # 3) noise burst
    i0, i1 = total * 0.42, total * 0.56
    n3 = int((i1 - i0) * sample_rate)
    noise = rng.standard_normal(n3)
    put(i0, i1, 0.12 * noise * _fade(n3, sample_rate, 0.1))

    # 4) silence (0.56 - 0.68)

    # 5) louder chord with a richer spectrum
    i0, i1 = total * 0.68, total * 0.88
    n5 = int((i1 - i0) * sample_rate)
    tt = np.arange(n5) / sample_rate
    chord = sum(
        (0.20 / k) * np.sin(2 * np.pi * 110.0 * k * tt + 0.2 * k) for k in range(1, 9)
    )
    put(i0, i1, chord * _fade(n5, sample_rate, 0.08))

    # 6) click train, every 50 ms
    i0 = int(total * 0.88 * sample_rate)
    click_len = int(0.02 * sample_rate)
    tt = np.arange(click_len) / sample_rate
    click = np.sin(2 * np.pi * 1500 * tt) * np.exp(-tt * 250.0)
    step = int(0.05 * sample_rate)
    for start in range(i0, n, step):
        end = min(n, start + click_len)
        x[start:end] += 0.5 * click[: end - start]

    peak = float(np.max(np.abs(x))) or 1.0
    return np.asarray(x * (0.95 / peak), dtype=np.float64)
