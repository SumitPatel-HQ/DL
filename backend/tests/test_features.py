import json
import wave
from pathlib import Path

import numpy as np
import pytest

from ml.features import FeaturePipeline, Normalization, canonical_audio


@pytest.fixture
def wav_fixture(tmp_path: Path) -> Path:
    """Generated signal for plumbing only, never evaluation evidence."""
    path = tmp_path / "stereo.wav"
    t = np.arange(8000) / 8000
    samples = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(np.column_stack((samples, samples)).tobytes())
    return path


def test_wav_and_live_pcm_share_canonical_features(wav_fixture: Path) -> None:
    pipeline = FeaturePipeline()
    with wave.open(str(wav_fixture), "rb") as wav:
        pcm = np.frombuffer(wav.readframes(8000), dtype="<i2").reshape(-1, 2)
    from_wav = pipeline.from_wav(wav_fixture)
    from_pcm = pipeline.from_audio(pcm.astype(np.float32) / 32768, 8000)
    assert from_wav.shape == (98, 40)
    assert from_wav.dtype == np.float32
    np.testing.assert_array_equal(from_wav, from_pcm)


def test_canonical_audio_mixes_channels_and_preserves_duration() -> None:
    audio = canonical_audio(np.tile([0.5, -0.5], (8000, 1)), 8000)
    assert audio.shape == (16000,)
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, np.zeros(16000))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_audio_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_audio(np.array([value]), 16000)


def test_silence_produces_finite_features_and_empty_audio_is_rejected() -> None:
    features = FeaturePipeline().from_audio(np.zeros(16000, dtype=np.float32), 16000)
    assert features.shape == (98, 40)
    assert np.isfinite(features).all()
    with pytest.raises(ValueError, match="empty"):
        FeaturePipeline().from_audio(np.array([], dtype=np.float32), 16000)


def test_normalization_is_train_only_serializable_and_checks_compatibility(tmp_path: Path) -> None:
    frames = np.array([[1] * 40, [3] * 40], dtype=np.float32)
    metadata = dict(dataset_version="fixture-v1", split_version="split-v1", manifest_sha256="abc")
    stats = FeaturePipeline.fit_normalization([frames], split="train", **metadata)
    assert stats.mean == (2.0,) * 40
    assert stats.std == (1.0,) * 40
    path = tmp_path / "normalization.json"
    stats.save(path)
    assert Normalization.load(path) == stats
    with pytest.raises(ValueError, match="only.*train"):
        FeaturePipeline.fit_normalization([frames], split="test", **metadata)
    path.write_text(path.read_text().replace("mfcc-v1", "mfcc-v0"))
    with pytest.raises(ValueError, match="Incompatible"):
        Normalization.load(path)


def test_baseline_config_accepts_saved_feature_contract(tmp_path: Path) -> None:
    baseline = json.loads((Path(__file__).parents[1] / "ml/configs/baseline.json").read_text())
    stats = FeaturePipeline.fit_normalization(
        [np.ones((2, 40), dtype=np.float32)],
        split="train",
        dataset_version="fixture-v1",
        split_version="split-v1",
        manifest_sha256="abc",
    )
    artifact = tmp_path / "normalization.json"
    stats.save(artifact)
    metadata = json.loads(artifact.read_text())
    assert baseline["features"] == metadata["features"]
    assert baseline["labels"] == metadata["labels"]


def test_stream_features_match_window_features_on_one_second() -> None:
    import numpy as np

    from ml.features import FeaturePipeline

    audio = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    pipeline = FeaturePipeline()
    assert np.array_equal(pipeline.from_stream(audio, 16000), pipeline.from_audio(audio, 16000))
    longer = np.concatenate([audio, audio])
    assert pipeline.from_stream(longer, 16000).shape == (198, 40)
