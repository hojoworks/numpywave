"""Generate a demo WAV and its overview plot in this folder.

Run with:  uv run python examples/demo.py
"""

from pathlib import Path

import numpywave as nw

here = Path(__file__).parent
audio_path = here / "demo.wav"
plot_path = here / "demo-overview.png"

samples = nw.make_demo_signal(duration=8.0, sample_rate=44100)
nw.write_audio(audio_path, samples, 44100)

analysis = nw.analyze(audio_path)
nw.save_overview(analysis, plot_path, dpi=150)

print(analysis.to_json())
print(f"\nwrote {audio_path}")
print(f"wrote {plot_path}")
