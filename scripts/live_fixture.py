"""Write frontend/tests/fixtures/live-parity.json from the Python live pipeline.

The browser live engine (frontend/lib/live/) ports LiveStream, LiveDecider and
the MFCC-v1 features. Its tests replay this fixture, so rerun after changing
any of them:

    uv run --project backend python scripts/live_fixture.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.live import LiveDecider, LivePolicy, LiveStream  # noqa: E402
from ml.features import LABELS, FeaturePipeline, Normalization  # noqa: E402
from ml.features.pipeline import SAMPLE_RATE  # noqa: E402

OUTPUT = ROOT / "frontend" / "tests" / "fixtures" / "live-parity.json"
RATES = (8000, 16000, 44100, 48000)
FEATURE_STRIDE = 7
SAMPLE_STRIDE = 101


def signal(length: int) -> np.ndarray:
    """Integer-only test signal both languages generate identically: triangle plus LCG noise."""
    state = 12345
    pcm = np.empty(length, dtype=np.int16)
    for n in range(length):
        state = state * 48271 % 2147483647
        t = n % 37
        pcm[n] = (t if t < 19 else 37 - t) * 800 - 7000 + state % 2001 - 1000
    return pcm


def main() -> None:
    policy = LivePolicy()
    normalization = Normalization(
        tuple(np.linspace(-20, 5, 40)),
        tuple(np.linspace(1, 9, 40)),
        1,
        "fixture",
        "fixture",
        "x" * 64,
    )
    pipeline = FeaturePipeline(normalization)
    streams = []
    for rate in RATES:
        pcm = signal(round(rate * 1.25))
        stream = LiveStream(rate, policy)
        # Browser-sized frames, so window cutting crosses frame boundaries.
        for first in range(0, len(pcm), 2048):
            stream.push(pcm[first : first + 2048].astype("<i2").tobytes())
        batch = stream.take_windows()
        windows = []
        for end, window in zip(batch.ends_s, batch.windows, strict=True):
            features = pipeline.from_audio(window, SAMPLE_RATE).reshape(-1)
            windows.append(
                {
                    "end_s": end,
                    "samples": window[::SAMPLE_STRIDE].astype(float).tolist(),
                    "features": features[::FEATURE_STRIDE].astype(float).tolist(),
                }
            )
        streams.append({"rate": rate, "length": len(pcm), "windows": windows})

    rng = np.random.default_rng(7)
    rows = []
    for _ in range(60):
        label = int(rng.integers(len(LABELS)))
        peak = float(rng.choice([0.55, 0.75, 0.95]))
        for _ in range(int(rng.integers(1, 6))):
            noise = rng.dirichlet(np.ones(len(LABELS))) * (1 - peak)
            noise[label] += peak
            rows.append(noise.astype(np.float32))
    decider = LiveDecider(policy, LABELS)
    decisions = []
    for i, row in enumerate(rows):
        for decision in decider.observe(i * policy.hop_s, row):
            decisions.append({**decision.__dict__, "index": i})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "policy": policy.to_dict(),
        "labels": LABELS,
        "mean": normalization.mean,
        "std": normalization.std,
        "feature_stride": FEATURE_STRIDE,
        "sample_stride": SAMPLE_STRIDE,
        "streams": streams,
        "scores": [row.astype(float).tolist() for row in rows],
        "decisions": decisions,
    }
    OUTPUT.write_text(json.dumps(document) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
