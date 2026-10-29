import json

import numpy as np
import pytest

from numpywave import (
    Analysis,
    amplitude_to_db,
    analyze,
    audio_info,
    band_energy,
    frame_signal,
    load_audio,
    make_demo_signal,
    peak,
    peak_envelope,
    plot_overview,
    power_to_db,
    refine_peak,
    rms,
    save_overview,
    short_term_rms,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
    stft,
    window_function,
    write_audio,
)
from numpywave.cli import format_report, main

SR = 8000


def sine(freq=440.0, seconds=1.0, sample_rate=SR, amplitude=0.5):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------- basic maths


def test_amplitude_to_db():
    assert amplitude_to_db(1.0) == pytest.approx(0.0)
    assert amplitude_to_db(0.5) == pytest.approx(-6.0206, abs=1e-3)
    assert amplitude_to_db(0.0) == -160.0
    assert power_to_db(0.25) == pytest.approx(-6.0206, abs=1e-3)


def test_rms_and_peak_of_sine():
    x = sine(amplitude=0.5)
    assert peak(x) == pytest.approx(0.5, rel=1e-3)
    assert rms(x) == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)


def test_window_function_periodic_hann():
    w = window_function("hann", 8)
    assert w.shape == (8,)
    assert w[0] == 0.0
    assert w.sum() == pytest.approx(4.0)


# ------------------------------------------------------------------- framing


def test_frame_signal_shape_and_stride():
    x = np.arange(1000, dtype=float)
    frames = frame_signal(x, 256, 128)
    assert frames.shape == ((1000 - 256) // 128 + 1, 256)
    assert frames[0, 0] == 0
    assert frames[1, 0] == 128


def test_frame_signal_pads_short_input():
    frames = frame_signal(np.ones(10), 32, 16)
    assert frames.shape == (1, 32)
    assert frames[0, :10].tolist() == [1.0] * 10


def test_frame_signal_rejects_short_input_when_not_padding():
    with pytest.raises(ValueError):
        frame_signal(np.ones(10), 32, pad=False)


# ---------------------------------------------------------------------- stft


def test_stft_finds_tone_and_shapes():
    x = sine(440.0, 1.0, SR)
    mag, freqs, times = stft(x, SR, frame_size=1024, hop_size=256)

    assert mag.shape == (1 + x.size // 256, 513)
    assert freqs[-1] == pytest.approx(SR / 2)
    assert times[0] == pytest.approx(0.0, abs=1e-9)
    peak_freq = freqs[np.argmax(mag.mean(axis=0))]
    assert peak_freq == pytest.approx(440.0, abs=5.0)
    # a 0.5-amplitude sine sits near -6 dB at its peak bin
    assert amplitude_to_db(mag.max()) == pytest.approx(-6.0, abs=1.5)


def test_stft_power_of_two_signal_without_centering():
    x = np.ones(1024)
    mag, _freqs, times = stft(x, SR, frame_size=256, hop_size=256, center=False)
    assert mag.shape == (4, 129)
    assert times.shape == (4,)


def test_short_term_rms_of_sine():
    x = sine(amplitude=1.0, sample_rate=1000, seconds=1.0)
    _times, levels = short_term_rms(x, 1000, window_s=0.05, hop_s=0.01)
    assert levels.size == 96  # (1000 - 50) // 10 + 1
    assert np.allclose(levels[5:], 1 / np.sqrt(2), atol=0.01)


def test_peak_envelope_keeps_transient():
    x = np.zeros(1000)
    x[517] = 1.0
    lo, hi = peak_envelope(x, 10)
    assert lo.shape == hi.shape == (10,)
    assert hi.max() == 1.0
    assert np.count_nonzero(hi > 0.5) == 1


# ---------------------------------------------------------------- spectrum


def test_spectral_features_of_pure_tone():
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    power = np.zeros_like(freqs)
    power[np.argmin(np.abs(freqs - 1000.0))] = 1.0

    assert spectral_centroid(freqs, power) == pytest.approx(1000.0, abs=5.0)
    assert spectral_rolloff(freqs, power) == pytest.approx(1000.0, abs=5.0)
    assert spectral_flatness(power) < 1e-3


def test_refine_peak_interpolates_between_bins():
    freqs = np.array([100.0, 200.0, 300.0])
    db = np.array([-1.69, -0.09, -0.49])  # parabola with vertex at +0.3 bins
    freq, level = refine_peak(freqs, db, 1)
    assert freq == pytest.approx(230.0)
    assert level == pytest.approx(0.0, abs=1e-9)


def test_spectral_flatness_is_high_for_white_noise():
    rng = np.random.default_rng(0)
    power = np.abs(np.fft.rfft(rng.standard_normal(4096))) ** 2
    assert spectral_flatness(power) > 0.2


def test_band_energy_of_tone():
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    power = np.zeros_like(freqs)
    idx = np.argmin(np.abs(freqs - 1000.0))
    power[idx] = 1.0

    from numpywave import OCTAVE_BANDS

    values = band_energy(freqs, power, OCTAVE_BANDS)
    assert values.shape == (len(OCTAVE_BANDS),)
    assert np.argmax(values) == OCTAVE_BANDS.index(1000.0)
    assert values.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ analyze


def test_analyze_array_basic_stats():
    x = sine(440.0, 1.0, SR, amplitude=0.5)
    a = analyze(x, SR)

    assert isinstance(a, Analysis)
    assert a.channels == 1
    assert a.duration == pytest.approx(1.0)
    assert a.peak_db == pytest.approx(-6.02, abs=0.05)
    assert a.rms_db == pytest.approx(-9.03, abs=0.05)
    assert a.crest_db == pytest.approx(3.01, abs=0.05)
    assert a.dominant_frequency == pytest.approx(440.0, abs=2.0)
    assert a.band_centres[0] == 31.5
    assert a.band_fractions.sum() <= 1.0 + 1e-9
    assert a.spectrogram_db.min() >= -120.0


def test_analyze_detects_silence_and_clipping():
    x = np.zeros(SR)
    x[: SR // 2] = 0.5 * np.sin(2 * np.pi * 300 * np.arange(SR // 2) / SR)
    x[100] = 1.0
    x[200] = -1.0
    a = analyze(x, SR, silence_db=-50.0)

    assert a.silence_ratio == pytest.approx(0.5, abs=0.1)
    assert a.clipped_samples == 2
    assert a.is_clipped
    assert 0.0 <= a.dynamic_range_db < 100.0


def test_analyze_stereo_downmix():
    left = sine(400.0, 0.5, SR, amplitude=0.4)
    right = np.zeros_like(left)
    a = analyze(np.column_stack([left, right]), SR, frame_size=1024)

    assert a.channels == 2
    assert a.rms == pytest.approx(0.4 / np.sqrt(2) / 2, rel=1e-3)
    assert a.samples.shape == left.shape


def test_analyze_handles_short_and_odd_arrays():
    a = analyze(np.array([0.1, -0.1, 0.2]), 1000)
    assert a.frame_size < 4
    assert a.n_samples == 3
    assert a.duration == pytest.approx(0.003)


def test_analyze_empty_array_raises():
    with pytest.raises(ValueError):
        analyze(np.array([]), SR)


def test_analyze_requires_sample_rate_for_arrays():
    with pytest.raises(ValueError):
        analyze(np.ones(100))


def test_summary_is_json_serialisable(tmp_path):
    a = analyze(sine(440.0, 0.5, SR), SR)
    summary = a.summary()
    assert summary["duration_s"] == pytest.approx(0.5)
    assert "band_db" in summary
    assert isinstance(summary["band_db"], dict)

    text = a.to_json()
    assert json.loads(text)["peak_dbfs"] == summary["peak_dbfs"]

    target = tmp_path / "summary.json"
    a.to_json(target)
    assert json.loads(target.read_text()) == summary


def test_demo_signal_is_well_formed():
    x = make_demo_signal(6.0, SR)
    assert x.shape == (6 * SR,)
    assert np.max(np.abs(x)) <= 0.95 + 1e-9
    a = analyze(x, SR)
    assert a.dominant_frequency > 0


# ----------------------------------------------------------------------- io


def test_load_write_roundtrip(tmp_path):
    x = sine(440.0, 0.25, SR, amplitude=0.5)
    path = tmp_path / "tone.wav"
    write_audio(path, x, SR)

    loaded, sr = load_audio(path)
    assert sr == SR
    assert loaded.shape == x.shape
    assert np.allclose(loaded, x, atol=1e-4)

    info = audio_info(path)
    assert info["channels"] == 1
    assert info["sample_rate"] == SR
    assert info["frames"] == x.size


def test_analyze_file_sets_metadata(tmp_path):
    path = tmp_path / "tone.wav"
    write_audio(path, sine(1000.0, 0.3, SR), SR)
    a = analyze(path, frame_size=512)

    assert a.source == str(path)
    assert a.format == "WAV"
    assert a.subtype == "PCM_16"
    assert a.channels == 1
    assert a.dominant_frequency == pytest.approx(1000.0, abs=10.0)


# -------------------------------------------------------------------- plots


def test_save_overview_writes_png(tmp_path):
    a = analyze(make_demo_signal(3.0, SR), SR)
    target = tmp_path / "overview.png"
    save_overview(a, target, dpi=60)
    assert target.exists() and target.stat().st_size > 10_000


def test_plot_overview_has_four_panels():
    a = analyze(sine(440.0, 0.5, SR), SR)
    fig = plot_overview(a)
    assert len(fig.axes) == 5  # 4 panels + colorbar
    fig.clf()


# ---------------------------------------------------------------------- cli


def test_cli_report_and_json(tmp_path, capsys):
    audio = tmp_path / "tone.wav"
    write_audio(audio, sine(1000.0, 0.5, SR), SR)

    code = main([str(audio), "--json", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert code == 0
    assert "waveform analysis" in out
    assert "dominant" in out
    assert "1000" in out
    assert (tmp_path / "out.json").exists()


def test_cli_json_stdout_only(tmp_path, capsys):
    audio = tmp_path / "tone.wav"
    write_audio(audio, sine(220.0, 0.2, SR), SR)

    assert main([str(audio), "--json", "-"]) == 0
    out = capsys.readouterr().out
    assert "waveform analysis" not in out
    assert json.loads(out)["dominant_frequency_hz"] == pytest.approx(220.0, abs=5.0)


def test_cli_json_stdout_stays_clean_with_plot(tmp_path, capsys):
    audio = tmp_path / "tone.wav"
    write_audio(audio, sine(220.0, 0.2, SR), SR)
    plot = tmp_path / "plot.png"

    assert main([str(audio), "--json", "-", "--plot", str(plot), "--dpi", "50"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["format"] == "WAV"
    assert "wrote" in captured.err
    assert plot.exists()


def test_format_report_contains_sections():
    report = format_report(analyze(sine(440.0, 0.3, SR), SR))
    for section in ("level", "spectrum", "octave bands"):
        assert section in report
