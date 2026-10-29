"""Matplotlib drawing helpers: waveform, level, spectrogram, spectrum."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .analysis import Analysis

WAVE_COLOR = "#2563eb"
LEVEL_COLOR = "#0891b2"
SPEC_COLOR = "#7c3aed"
PEAK_COLOR = "#dc2626"
GRID_COLOR = "#cbd5e1"
FILL_COLOR = "#f1f5f9"
SILENCE_COLOR = "#94a3b8"


def _style(ax: Axes) -> None:
    ax.set_facecolor(FILL_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=10, fontweight="bold", pad=6)


def _fmax(analysis: Analysis, fmax: float | None) -> float:
    nyquist = analysis.sample_rate / 2
    return float(nyquist if fmax is None else min(fmax, nyquist))


def draw_waveform(
    ax: Axes,
    analysis: Analysis,
    *,
    n_buckets: int = 2000,
    label: str = "peak envelope",
) -> Axes:
    """Draw the min/max waveform envelope."""
    times, lo, hi = analysis.envelope(n_buckets)
    ax.fill_between(times, lo, hi, color=WAVE_COLOR, linewidth=0, label=label)
    ax.axhline(0.0, color="#64748b", linewidth=0.6, alpha=0.6)
    limit = max(0.02, float(analysis.peak) * 1.1)
    ax.set_ylim(-limit, limit)
    ax.set_xlim(0.0, analysis.duration)
    ax.set_ylabel("amplitude")
    _style(ax)
    _label(ax, "waveform")
    return ax


def draw_level(
    ax: Axes,
    analysis: Analysis,
    *,
    label: str = "short-term RMS",
) -> Axes:
    """Draw the short-term level in dBFS with the silence threshold marked."""
    db = analysis.short_term_db
    ax.plot(
        analysis.short_term_times,
        db,
        color=LEVEL_COLOR,
        linewidth=0.9,
        label=f"{label} ({analysis.level_window_s * 1000:.0f} ms)",
    )
    ax.axhline(
        analysis.silence_threshold_db,
        color=SILENCE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"silence {analysis.silence_threshold_db:g} dBFS",
    )
    low = max(-120.0, np.floor((float(np.percentile(db, 1)) - 3.0) / 5.0) * 5.0)
    high = min(6.0, float(np.max(db)) + 3.0)
    ax.set_ylim(low, max(high, low + 10.0))
    ax.set_xlim(0.0, analysis.duration)
    ax.set_ylabel("dBFS")
    _style(ax)
    _label(ax, "level")
    return ax


def draw_spectrogram(
    ax: Axes,
    analysis: Analysis,
    *,
    fmax: float | None = None,
    dynamic_range_db: float = 90.0,
    cmap: str = "magma",
) -> QuadMesh:
    """Draw a dB spectrogram (time on x, frequency on y)."""
    top = _fmax(analysis, fmax)
    mask = analysis.spectrogram_freqs <= top
    freqs = analysis.spectrogram_freqs[mask]
    data = analysis.spectrogram_db[:, mask]
    vmax = float(np.max(data))
    mesh = ax.pcolormesh(
        analysis.spectrogram_times,
        freqs,
        data.T,
        shading="auto",
        cmap=cmap,
        vmin=vmax - dynamic_range_db,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_ylim(0.0, top)
    ax.set_xlim(0.0, analysis.duration)
    ax.set_ylabel("Hz")
    _style(ax)
    ax.grid(False)
    _label(ax, "spectrogram")
    return mesh


def draw_spectrum(
    ax: Axes,
    analysis: Analysis,
    *,
    fmin: float = 20.0,
    fmax: float | None = None,
    show_bands: bool = True,
    show_peak: bool = True,
) -> Line2D:
    """Draw the long-term average spectrum on a log frequency axis."""
    top = _fmax(analysis, fmax)
    freqs = analysis.spectrum_freqs
    db = analysis.spectrum_db
    mask = (freqs >= max(fmin, freqs[1])) & (freqs <= top)
    if not mask.any():
        mask = np.ones_like(freqs, dtype=bool)

    if show_bands:
        for centre in analysis.band_centres:
            lo, hi = centre / np.sqrt(2.0), centre * np.sqrt(2.0)
            if lo > top or hi < fmin:
                continue
            ax.axvspan(lo, hi, color=WAVE_COLOR, alpha=0.05, linewidth=0)
            if hi < top:
                ax.axvline(hi, color=GRID_COLOR, linewidth=0.5, alpha=0.8)

    (line,) = ax.semilogx(freqs[mask], db[mask], color=SPEC_COLOR, linewidth=0.9)

    if show_peak and analysis.dominant_frequency > 0:
        ax.axvline(
            analysis.dominant_frequency,
            color=PEAK_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.annotate(
            f"peak {analysis.dominant_frequency:.0f} Hz",
            xy=(analysis.dominant_frequency, analysis.dominant_db),
            xytext=(6, 8),
            textcoords="offset points",
            color=PEAK_COLOR,
            fontsize=9,
        )

    left = max(fmin, float(freqs[1]))
    ax.set_xlim(left, max(left, top))
    visible = db[mask]
    low = max(-100.0, float(np.percentile(visible, 1)) - 4.0)
    ax.set_ylim(min(low, -20.0), 3.0)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dBFS")
    _style(ax)
    _label(ax, "average spectrum")
    return line


def _title(analysis: Analysis) -> str:
    name = Path(analysis.source).name if analysis.source else "in-memory array"
    return (
        f"{name}   ·   {analysis.duration:.2f} s   ·   "
        f"{analysis.sample_rate / 1000:.1f} kHz   ·   "
        f"{analysis.channels} ch   ·   "
        f"peak {analysis.peak_db:.1f} dBFS   ·   "
        f"RMS {analysis.rms_db:.1f} dBFS   ·   "
        f"crest {analysis.crest_db:.1f} dB"
    )


def plot_overview(
    analysis: Analysis,
    *,
    title: str | None = None,
    fmax: float | None = None,
    figsize: tuple[float, float] = (12.0, 9.0),
    dynamic_range_db: float = 90.0,
    cmap: str = "magma",
) -> Figure:
    """Four-panel overview: waveform, level, spectrogram, spectrum."""
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.set_facecolor("white")
    grid = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.2, 2.6, 1.9])

    ax_wave = fig.add_subplot(grid[0])
    ax_level = fig.add_subplot(grid[1], sharex=ax_wave)
    ax_spec = fig.add_subplot(grid[2], sharex=ax_wave)
    ax_fft = fig.add_subplot(grid[3])

    draw_waveform(ax_wave, analysis)
    draw_level(ax_level, analysis)
    mesh = draw_spectrogram(
        ax_spec,
        analysis,
        fmax=fmax,
        dynamic_range_db=dynamic_range_db,
        cmap=cmap,
    )
    fig.colorbar(mesh, ax=ax_spec, pad=0.005, label="dBFS")
    draw_spectrum(ax_fft, analysis, fmax=fmax)

    ax_wave.set_xlabel("time (s)")
    ax_level.tick_params(labelbottom=False)
    ax_spec.set_xlabel("time (s)")
    ax_wave.legend(loc="upper right", fontsize=8, frameon=False)
    ax_level.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(title or _title(analysis), fontsize=12, x=0.01, ha="left")
    return fig


def save_overview(
    analysis: Analysis,
    path: str | Path,
    *,
    dpi: int = 150,
    **kwargs,
) -> Path:
    """Render :func:`plot_overview` to ``path`` and close the figure."""
    fig = plot_overview(analysis, **kwargs)
    try:
        fig.savefig(str(path), dpi=dpi, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return Path(path)


def show_overview(analysis: Analysis, **kwargs) -> Figure:
    """Open an interactive window with :func:`plot_overview`."""
    fig = plot_overview(analysis, **kwargs)
    plt.show()
    return fig
