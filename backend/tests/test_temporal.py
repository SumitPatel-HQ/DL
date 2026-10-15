from pathlib import Path

import numpy as np

from ml.data.ami_temporal import BACKGROUND, FRAME_CLASSES, KEYWORDS
from ml.features import FeaturePipeline, Normalization
from ml.temporal import (
    TemporalConfig,
    TemporalNet,
    decode,
    load_checkpoint,
    posteriors,
    save_checkpoint,
    spot_audio,
)

CONFIG = TemporalConfig(smooth_frames=1, threshold=0.5, min_frames=3, suppression_s=1.0)


def scores(frames: int, **spans: tuple[int, int, float]) -> np.ndarray:
    out = np.zeros((frames, len(FRAME_CLASSES)), dtype=np.float32)
    out[:, BACKGROUND] = 1.0
    for keyword, (start, end, value) in spans.items():
        out[start:end, KEYWORDS.index(keyword)] = value
        out[start:end, BACKGROUND] = 1 - value
    return out


def add(out: np.ndarray, keyword: str, start: int, end: int, value: float) -> None:
    out[start:end, KEYWORDS.index(keyword)] = value
    out[start:end, BACKGROUND] = 1 - value


def test_decode_emits_detection_with_span_confidence_and_threshold() -> None:
    (d,) = decode(scores(100, go=(20, 30, 0.9)), CONFIG)
    assert d.keyword == "go" and d.windows == 10
    assert abs(d.start_s - 0.20) < 1e-9 and abs(d.end_s - (0.29 + 0.025)) < 1e-9
    assert abs(d.confidence - 0.9) < 1e-6 and d.threshold == 0.5


def test_decode_drops_short_runs_and_low_scores() -> None:
    assert decode(scores(100, go=(20, 22, 0.9)), CONFIG) == []
    assert decode(scores(100, go=(20, 30, 0.4)), CONFIG) == []


def test_decode_suppresses_same_keyword_but_not_different_keywords() -> None:
    near = scores(200, go=(20, 30, 0.9), stop=(24, 34, 0.8))
    assert sorted(d.keyword for d in decode(near, CONFIG)) == ["go", "stop"]
    close = scores(200, go=(20, 30, 0.9))
    add(close, "go", 50, 60, 0.7)  # centers 0.3 s apart
    (kept,) = decode(close, TemporalConfig(smooth_frames=1, min_frames=3, suppression_s=1.0))
    assert abs(kept.confidence - 0.9) < 1e-6
    far = scores(400, go=(20, 30, 0.9))
    add(far, "go", 200, 210, 0.7)
    assert len(decode(far, CONFIG)) == 2


def test_chunked_posteriors_cover_every_frame_and_sum_to_one() -> None:
    model = TemporalNet(8, 1).eval()
    features = np.random.default_rng(0).normal(size=(2500, 40)).astype(np.float32)
    out = posteriors(model, features)
    assert out.shape == (2500, len(FRAME_CLASSES))
    assert np.allclose(out.sum(axis=1), 1, atol=1e-5)


def test_checkpoint_round_trip_and_spot_audio_shape(tmp_path: Path) -> None:
    norm = Normalization((0.0,) * 40, (1.0,) * 40, 10, "d", "s", "m")
    network = {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}
    path = tmp_path / "t.pt"
    digest = save_checkpoint(path, TemporalNet(8, 1), norm, {"network": network})
    loaded = load_checkpoint(path)
    assert loaded.sha256 == digest and isinstance(loaded.pipeline(), FeaturePipeline)
    audio = np.random.default_rng(0).normal(0, 0.1, 16000 * 3).astype(np.float32)
    for d in spot_audio(loaded, audio, TemporalConfig(threshold=0.01, min_frames=1)):
        assert d.keyword in KEYWORDS and 0 <= d.start_s < d.end_s <= 3.1
