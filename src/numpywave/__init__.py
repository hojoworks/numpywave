"""numpywave - numpy audio waveform analysis with a matplotlib overview."""

from __future__ import annotations

from .analysis import Analysis, analyze
from .demo import make_demo_signal
from .dsp import (
    OCTAVE_BANDS,
    amplitude_to_db,
    band_energy,
    frame_signal,
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
    window_function,
)
from .io import audio_info, load_audio, to_mono, write_audio
from .plots import (
    draw_level,
    draw_spectrogram,
    draw_spectrum,
    draw_waveform,
    plot_overview,
    save_overview,
    show_overview,
)

__version__ = "0.1.0"

__all__ = [
    "OCTAVE_BANDS",
    "Analysis",
    "amplitude_to_db",
    "analyze",
    "audio_info",
    "band_energy",
    "draw_level",
    "draw_spectrogram",
    "draw_spectrum",
    "draw_waveform",
    "frame_signal",
    "load_audio",
    "make_demo_signal",
    "peak",
    "peak_envelope",
    "plot_overview",
    "power_to_db",
    "refine_peak",
    "rms",
    "save_overview",
    "short_term_rms",
    "show_overview",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_rolloff",
    "stft",
    "to_mono",
    "window_function",
    "write_audio",
]
