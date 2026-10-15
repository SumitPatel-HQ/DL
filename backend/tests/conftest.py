"""Shared generated corpus: plumbing fixtures only, never evaluation evidence."""

import wave
from pathlib import Path

import numpy as np
import pytest

from ml.data import build_manifest, save_manifest
from ml.data.manifest import Source, speaker_split
from ml.data.prepare import prepare_cache
from ml.features import LABELS


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


@pytest.fixture
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


@pytest.fixture
def cache(corpus: Path, sources: dict[str, Source], tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    save_manifest(build_manifest(corpus, sources, max_per_class=1), manifest)
    output = tmp_path / "features"
    prepare_cache(manifest, corpus, output)
    return output
