"""One-call waveform analysis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dsp import (
    DB_FLOOR,
    OCTAVE_BANDS,
    WindowName,
    amplitude_to_db,
    band_energy,
    peak,
    peak_envelope,
    power_to_db,
    refine_peak,
    rms,
    short_term_rms,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
    stft,
)
from .io import audio_info, load_audio, to_mono


def _band_label(centre: float) -> str:
    return f"{centre:g} Hz"


@dataclass(eq=False)
class Analysis:
    """Result of :func:`analyze`. All arrays are plain numpy."""

    source: str | None
    sample_rate: int
    channels: int
    samples: np.ndarray  #: mono mix used for the analysis
    duration: float

    peak: float
    peak_db: float
    rms: float
    rms_db: float
    crest_db: float
    dc_offset: float
    clipped_samples: int
    clip_threshold: float
    zcr: float

    silence_threshold_db: float
    silence_ratio: float
    dynamic_range_db: float  #: p95 - p10 of level over non-silent frames
    short_term_times: np.ndarray
    short_term_db: np.ndarray
    level_window_s: float
    level_hop_s: float

    frame_size: int
    hop_size: int
    window: str
    spectrogram_db: np.ndarray
    spectrogram_freqs: np.ndarray
    spectrogram_times: np.ndarray

    spectrum_freqs: np.ndarray
    spectrum_db: np.ndarray
    spectral_centroid: float
    spectral_rolloff: float
    spectral_flatness: float
    dominant_frequency: float
    dominant_db: float

    band_centres: tuple[float, ...]
    band_db: np.ndarray
    band_fractions: np.ndarray

    format: str | None = None
    subtype: str | None = None

    @property
    def n_samples(self) -> int:
        return int(self.samples.size)

    @property
    def is_clipped(self) -> bool:
        return self.clipped_samples > 0

    def bands(self) -> dict[str, float]:
        """Band levels in dB as ``{"1000 Hz": -18.3, ...}``."""
        return {
            _band_label(c): round(float(v), 2)
            for c, v in zip(self.band_centres, self.band_db)
        }

    def summary(self) -> dict[str, Any]:
        """JSON-friendly scalar summary (no arrays)."""
        return {
            "source": self.source,
            "format": self.format,
            "subtype": self.subtype,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_s": round(self.duration, 4),
            "n_samples": self.n_samples,
            "peak_dbfs": round(self.peak_db, 2),
            "rms_dbfs": round(self.rms_db, 2),
            "crest_factor_db": round(self.crest_db, 2),
            "dc_offset": round(self.dc_offset, 6),
            "clipped_samples": self.clipped_samples,
            "zcr": round(self.zcr, 5),
            "silence_ratio": round(self.silence_ratio, 4),
            "silence_threshold_db": self.silence_threshold_db,
            "dynamic_range_db": round(self.dynamic_range_db, 2),
            "spectral_centroid_hz": round(self.spectral_centroid, 1),
            "spectral_rolloff_hz": round(self.spectral_rolloff, 1),
            "spectral_flatness": round(self.spectral_flatness, 4),
            "dominant_frequency_hz": round(self.dominant_frequency, 1),
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "band_db": self.bands(),
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialise the summary; also writes it when ``path`` is given."""
        text = json.dumps(self.summary(), indent=indent)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def envelope(self, n_buckets: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(times, lo, hi)`` waveform envelope for plotting."""
        lo, hi = peak_envelope(self.samples, n_buckets)
        if lo.size == 0:
            return np.zeros(0), lo, hi
        edges = np.linspace(0, self.n_samples, lo.size + 1)
        times = (edges[:-1] + edges[1:]) / 2 / self.sample_rate
        return times, lo, hi


def analyze(
    audio: str | Path | np.ndarray,
    sample_rate: int | None = None,
    *,
    frame_size: int = 2048,
    hop_size: int | None = None,
    window: WindowName = "hann",
    bands: Sequence[float] = OCTAVE_BANDS,
    silence_db: float = -50.0,
    clip_threshold: float = 0.999,
    roll_percent: float = 0.85,
    spectrogram_floor_db: float = -120.0,
    level_window_s: float = 0.05,
    level_hop_s: float = 0.01,
) -> Analysis:
    """Analyse an audio file or a numpy array of samples.

    Parameters
    ----------
    audio:
        Path to an audio file, or a float array in ``[-1, 1]`` (1-D mono or
        ``(n, channels)``).
    sample_rate:
        Required when ``audio`` is an array; ignored for files.
    frame_size, hop_size, window:
        STFT settings. ``hop_size`` defaults to ``frame_size // 4``.
    bands:
        Band centre frequencies for the band-energy table.
    silence_db, clip_threshold:
        Thresholds (dBFS) for the silence ratio and (linear) clipping count.
        Frames below ``silence_db`` are also excluded from the dynamic range
        statistic.
    roll_percent:
        Cumulative energy used for the spectral rolloff point.
    spectrogram_floor_db:
        Lower clamp for the stored spectrogram (keeps plots sane).
    """
    fmt = subtype = None
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        info = audio_info(path)
        data, sr = load_audio(path, mono=False)
        source = str(path)
        channels = int(info["channels"])
        fmt, subtype = info["format"], info["subtype"]
    else:
        data = np.asarray(audio, dtype=np.float64)
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing an array")
        sr = int(sample_rate)
        source = None
        if data.ndim > 2:
            raise ValueError(f"expected 1-D or 2-D audio, got shape {data.shape}")
        channels = 1 if data.ndim == 1 else int(data.shape[1])

    mono = np.ascontiguousarray(to_mono(data))
    if mono.size == 0:
        raise ValueError("audio is empty")

    if frame_size < 2:
        raise ValueError("frame_size must be at least 2")
    if frame_size > mono.size:
        frame_size = int(2 ** np.floor(np.log2(mono.size)))
        frame_size = max(2, frame_size)
    hop_size = hop_size or max(1, frame_size // 4)

    peak_amp = peak(mono)
    rms_amp = float(rms(mono))
    peak_db = float(amplitude_to_db(peak_amp))
    rms_db = float(amplitude_to_db(rms_amp))
    crest_db = peak_db - rms_db

    st_times, st_rms = short_term_rms(
        mono, sr, window_s=level_window_s, hop_s=level_hop_s
    )
    st_db = amplitude_to_db(st_rms)

    mag, freqs, times = stft(
        mono, sr, frame_size=frame_size, hop_size=hop_size, window=window
    )
    power = np.mean(np.square(mag), axis=0)
    mean_mag = np.mean(mag, axis=0)
    mean_mag_db = amplitude_to_db(mean_mag)

    dominant_frequency = 0.0
    dominant_db = DB_FLOOR
    if freqs.size > 1:
        idx = 1 + int(np.argmax(power[1:]))
        dominant_frequency, dominant_db = refine_peak(freqs, mean_mag_db, idx)

    band_centres = tuple(float(f) for f in bands)
    band_powers = band_energy(freqs, power, band_centres)
    total_band_power = float(np.sum(band_powers))
    band_fractions = (
        band_powers / total_band_power
        if total_band_power > 0
        else np.zeros_like(band_powers)
    )

    sign_changes = np.count_nonzero(np.diff(np.signbit(mono)))
    active_db = st_db[st_db >= silence_db]
    if active_db.size < 10:
        active_db = st_db
    p10, p95 = (float(v) for v in np.percentile(active_db, (10, 95)))

    return Analysis(
        source=source,
        sample_rate=sr,
        channels=channels,
        samples=mono,
        duration=mono.size / sr,
        peak=peak_amp,
        peak_db=peak_db,
        rms=rms_amp,
        rms_db=rms_db,
        crest_db=crest_db,
        dc_offset=float(np.mean(mono)),
        clipped_samples=int(np.count_nonzero(np.abs(mono) >= clip_threshold)),
        clip_threshold=clip_threshold,
        zcr=float(sign_changes / max(1, mono.size - 1)),
        silence_threshold_db=silence_db,
        silence_ratio=float(np.mean(st_db < silence_db)),
        dynamic_range_db=max(0.0, p95 - p10),
        short_term_times=st_times,
        short_term_db=st_db,
        level_window_s=level_window_s,
        level_hop_s=level_hop_s,
        frame_size=frame_size,
        hop_size=hop_size,
        window=window,
        spectrogram_db=np.maximum(amplitude_to_db(mag), spectrogram_floor_db),
        spectrogram_freqs=freqs,
        spectrogram_times=times,
        spectrum_freqs=freqs,
        spectrum_db=amplitude_to_db(mean_mag),
        spectral_centroid=spectral_centroid(freqs, power),
        spectral_rolloff=spectral_rolloff(freqs, power, roll_percent),
        spectral_flatness=spectral_flatness(power),
        dominant_frequency=dominant_frequency,
        dominant_db=dominant_db,
        band_centres=band_centres,
        band_db=np.asarray(power_to_db(band_powers), dtype=np.float64),
        band_fractions=band_fractions,
        format=fmt,
        subtype=subtype,
    )
