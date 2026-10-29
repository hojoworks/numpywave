"""Audio file IO (soundfile handles WAV, FLAC, OGG, AIFF, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix ``(n, channels)`` to ``(n,)`` by averaging channels."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        return samples
    if samples.ndim == 2:
        return samples.mean(axis=1)
    raise ValueError(f"expected 1-D or 2-D audio, got shape {samples.shape}")


def load_audio(
    path: str | Path,
    *,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Read an audio file as float64 in ``[-1, 1]``.

    Returns ``(samples, sample_rate)``. Mono files come back as ``(n,)``,
    multichannel files as ``(n, channels)`` unless ``mono=True``.
    """
    data, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    if data.ndim == 2 and mono:
        data = to_mono(data)
    return np.ascontiguousarray(data), int(sample_rate)


def audio_info(path: str | Path) -> dict[str, Any]:
    """Container/channel metadata without decoding the samples."""
    info = sf.info(str(path))
    return {
        "format": info.format,
        "subtype": info.subtype,
        "channels": info.channels,
        "sample_rate": info.samplerate,
        "frames": info.frames,
        "duration": info.duration,
    }


def write_audio(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str | None = None,
) -> None:
    """Write ``samples`` to ``path``, picking a sensible subtype by extension."""
    path = Path(path)
    subtype = subtype or {
        ".wav": "PCM_16",
        ".flac": "PCM_16",
        ".ogg": "VORBIS",
    }.get(path.suffix.lower())
    sf.write(str(path), np.asarray(samples, dtype=np.float64), int(sample_rate),
             subtype=subtype)
