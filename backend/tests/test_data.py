import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ml.data import build_manifest, load_manifest, save_manifest, validate_manifest
from ml.data.manifest import Source, speaker_split
from ml.data.prepare import prepare_cache
from ml.features import LABELS
from ml.infer import features_for_inference


def write_wav(path: Path, seconds: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        # Distinct bytes per source, strictly a test fixture.
        pcm = np.random.default_rng(sum(path.name.encode())).integers(
            -1000, 1000, seconds * 16000, dtype=np.int16
        )
        wav.writeframes(pcm.astype("<i2").tobytes())


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for split in ("train", "validation", "test"):
        speaker = next(f"s{i}" for i in range(1000) if speaker_split(f"s{i}") == split)
        for word in (*LABELS[:6], "cat", "dog", "tree"):
            write_wav(tmp_path / word / f"{speaker}_nohash_0.wav")
    for i in range(3):
        write_wav(tmp_path / "_background_noise_" / f"noise{i}.wav", 2)
    return tmp_path


def sources() -> dict[str, Source]:
    return {
        key: Source(
            "fixture://test",
            "test-v1",
            "test-only",
            "fixture",
            "generated test signal; no human speech",
            "tests only",
            "temporary",
        )
        for key in ("speech", "noise")
    }


def test_manifest_roundtrip_has_all_classes_without_speaker_or_source_leakage(
    corpus: Path, tmp_path: Path
) -> None:
    manifest = build_manifest(corpus, sources(), max_per_class=2)
    for split in ("train", "validation", "test"):
        assert {e.label for e in manifest.examples if e.split == split} == set(LABELS)
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    restored = load_manifest(path)
    assert restored == manifest
    validate_manifest(restored, corpus, verify_audio=True)
    example = next(e for e in restored.examples if e.label == "left")
    leaked = replace(example, split="test" if example.split == "train" else "train")
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(replace(restored, examples=(*restored.examples, leaked)), corpus)


def test_mini_eight_is_rejected_as_insufficient_unknown(corpus: Path) -> None:
    for word in ("cat", "dog", "tree"):
        for path in (corpus / word).glob("*.wav"):
            path.unlink()
    with pytest.raises(ValueError, match="Unknown"):
        build_manifest(corpus, sources())


def test_training_cache_matches_inference_on_same_wav(corpus: Path, tmp_path: Path) -> None:
    manifest = build_manifest(corpus, sources(), max_per_class=1)
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    output = tmp_path / "cache"
    stats = prepare_cache(path, corpus, output)
    assert stats.frame_count == 980
    example = next(e for e in manifest.examples if e.split == "train")
    with np.load(output / "train.npz", allow_pickle=False) as cached:
        live = features_for_inference(corpus / example.path, output / "normalization.json")
        np.testing.assert_array_equal(cached["features"][0], live)
        assert cached["features"].shape == (10, 98, 40)
