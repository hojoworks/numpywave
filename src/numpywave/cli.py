"""Command line front-end: ``numpywave track.flac``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import Analysis, analyze
from .dsp import OCTAVE_BANDS
from .plots import save_overview, show_overview


def _auto_path(audio: Path, suffix: str) -> Path:
    return audio.with_name(f"{audio.stem}{suffix}")


def format_report(analysis: Analysis, *, show_bands: bool = True) -> str:
    """Human-readable text report."""
    a = analysis
    source = a.source or "<array>"
    fmt = " / ".join(part for part in (a.format, a.subtype) if part) or "unknown"
    lines = [
        "numpywave - waveform analysis",
        f"file          : {source}",
        f"container     : {fmt}",
        (
            f"duration      : {a.duration:8.3f} s   "
            f"samples: {a.n_samples}   "
            f"rate: {a.sample_rate} Hz   "
            f"channels: {a.channels}"
        ),
        "",
        "level",
        f"  peak          : {a.peak_db:8.2f} dBFS   ({a.peak:.4f})",
        f"  rms           : {a.rms_db:8.2f} dBFS   ({a.rms:.4f})",
        f"  crest factor  : {a.crest_db:8.2f} dB",
        f"  dc offset     : {a.dc_offset:+9.5f}",
        f"  clipped       : {a.clipped_samples:8d} samples above {a.clip_threshold:g}",
        f"  zero crossings: {a.zcr:9.4f} per sample",
        f"  dynamic range : {a.dynamic_range_db:8.2f} dB   (p95 - p10, non-silent)",
        (
            f"  silence       : {a.silence_ratio * 100:7.2f} %"
            f"   below {a.silence_threshold_db:g} dBFS"
        ),
        "",
        "spectrum",
        f"  dominant      : {a.dominant_frequency:8.1f} Hz   ({a.dominant_db:.1f} dBFS)",
        f"  centroid      : {a.spectral_centroid:8.1f} Hz",
        f"  rolloff 85%   : {a.spectral_rolloff:8.1f} Hz",
        f"  flatness      : {a.spectral_flatness:8.4f}   (0 = tonal, 1 = noise)",
    ]
    if show_bands:
        lines += ["", "octave bands (dBFS)"]
        for i in range(0, len(a.band_db), 3):
            row = "   ".join(
                f"{label:>9s} {value:7.1f}"
                for label, value in list(a.bands().items())[i : i + 3]
            )
            lines.append("  " + row)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="numpywave",
        description="Analyse an audio file and optionally draw a waveform overview.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="audio file (wav, flac, ogg, aiff, ...)")
    parser.add_argument(
        "-o",
        "--plot",
        nargs="?",
        const="auto",
        metavar="PNG",
        help="write the 4-panel overview to PNG (default name: <audio>_analysis.png)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="auto",
        metavar="JSON",
        help="write the scalar summary as JSON; use '-' for stdout only",
    )
    parser.add_argument("--frame", type=int, default=2048, help="STFT frame size")
    parser.add_argument("--hop", type=int, default=None, help="STFT hop size (default: frame/4)")
    parser.add_argument(
        "--window",
        choices=("hann", "hamming", "blackman", "rect"),
        default="hann",
    )
    parser.add_argument("--fmax", type=float, default=None, help="upper frequency for plots")
    parser.add_argument("--silence-db", type=float, default=-50.0)
    parser.add_argument("--dynamic-range-db", type=float, default=90.0,
                        help="spectrogram colour range in dB")
    parser.add_argument("--cmap", default="magma", help="spectrogram colormap")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-bands", action="store_true", help="hide the octave band table")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    audio = args.audio
    if not audio.is_file():
        parser.error(f"no such file: {audio}")

    try:
        analysis = analyze(
            audio,
            frame_size=args.frame,
            hop_size=args.hop,
            window=args.window,
            bands=OCTAVE_BANDS,
            silence_db=args.silence_db,
        )
    except Exception as exc:  # noqa: BLE001 - soundfile raises unrelated types
        parser.error(f"could not analyse {audio}: {exc}")

    json_stdout = args.json == "-"

    def log(message: str) -> None:
        print(message, file=sys.stderr if json_stdout else sys.stdout)

    if not json_stdout:
        print(format_report(analysis, show_bands=not args.no_bands))

    plot_kwargs = {
        "fmax": args.fmax,
        "dynamic_range_db": args.dynamic_range_db,
        "cmap": args.cmap,
    }
    if args.plot:
        target = _auto_path(audio, "_analysis.png") if args.plot == "auto" else Path(args.plot)
        save_overview(analysis, target, dpi=args.dpi, **plot_kwargs)
        log(f"wrote {target}")

    if json_stdout:
        print(analysis.to_json())
    elif args.json:
        target = _auto_path(audio, ".json") if args.json == "auto" else Path(args.json)
        analysis.to_json(target)
        log(f"wrote {target}")

    if args.show:
        show_overview(analysis, **plot_kwargs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
