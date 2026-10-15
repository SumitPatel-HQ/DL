import hashlib
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ml.data import build_manifest, validate_manifest
from ml.data.manifest import (
    NOISE_ASSIGNMENT_V2,
    SPLIT_VERSION_V2,
    Source,
    noise_sources_by_split,
    speaker_split,
)
from ml.features import LABELS


@pytest.fixture
def sources() -> dict[str, Source]:
    return {
        key: Source(
            "fixture://test",
            "fixture",
            "test-only",
            "fixture",
            "no human speech",
            "tests",
            "temporary",
        )
        for key in ("speech", "noise")
    }


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    paths = []
    for split in ("train", "validation", "test"):
        speaker = next(f"s{i}" for i in range(1000) if speaker_split(f"s{i}") == split)
        paths.extend(
            (f"{word}/{speaker}_nohash_0.wav", 1600) for word in (*LABELS[:6], "cat", "dog", "tree")
        )
    paths.extend((f"_background_noise_/noise{i}.wav", 32000) for i in range(3))
    for relative, length in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        seed = int(hashlib.sha256(relative.encode()).hexdigest()[:8], 16)
        # Plumbing-only signals, never natural evaluation data.
        pcm = np.random.default_rng(seed).integers(-1000, 1000, length, dtype=np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(pcm.astype("<i2").tobytes())
    return tmp_path


def test_speech_groups_follow_speakers_across_word_files(
    corpus: Path,
    sources: dict[str, Source],
) -> None:
    manifest = build_manifest(corpus, sources, max_per_class=2)
    speech = [e for e in manifest.examples if e.source_id == "speech"]
    assert all(e.source_group == f"speaker:{e.speaker_id}" for e in speech)
    assert len({e.source_group for e in speech}) < len({e.path for e in speech})


def test_legacy_speech_file_groups_still_use_the_speaker_leakage_gate(
    corpus: Path,
    sources: dict[str, Source],
) -> None:
    manifest = build_manifest(corpus, sources, max_per_class=2)
    legacy = replace(
        manifest,
        examples=tuple(
            replace(e, source_group=e.path) if e.source_id == "speech" else e
            for e in manifest.examples
        ),
    )
    validate_manifest(legacy, corpus)
    left = next(e for e in legacy.examples if e.label == "left" and e.split == "train")
    moved = tuple(replace(e, split="test") if e == left else e for e in legacy.examples)
    with pytest.raises(ValueError, match="leakage.*speaker"):
        validate_manifest(replace(legacy, examples=moved), corpus)


def test_noise_group_identifies_parent_recording_and_cannot_be_renamed(
    corpus: Path,
    sources: dict[str, Source],
) -> None:
    manifest = build_manifest(corpus, sources, max_per_class=2)
    noise = next(e for e in manifest.examples if e.source_id == "noise")
    assert noise.source_group == noise.path
    renamed = replace(noise, source_group="different-recording")
    rows = tuple(renamed if e == noise else e for e in manifest.examples)
    with pytest.raises(ValueError, match="Noise.*parent"):
        validate_manifest(replace(manifest, examples=rows), corpus)


def test_noise_segments_cannot_cross_splits_even_with_different_speaker_and_hash(
    corpus: Path,
    sources: dict[str, Source],
) -> None:
    manifest = build_manifest(corpus, sources, max_per_class=2)
    first = next(e for e in manifest.examples if e.source_id == "noise" and e.start_s == 0)
    second = next(e for e in manifest.examples if e.path == first.path and e.start_s == 1)
    moved = replace(
        second,
        split="test" if first.split == "train" else "train",
        speaker_id="different-noise-id",
        sha256="f" * 64,
    )
    rows = tuple(moved if e == second else e for e in manifest.examples)
    with pytest.raises(ValueError, match="leakage.*source"):
        validate_manifest(replace(manifest, examples=rows), corpus)


def write_noise(path: Path) -> None:
    seed = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16)
    pcm = np.random.default_rng(seed).integers(-1000, 1000, 32000, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.astype("<i2").tobytes())


def test_v2_noise_split_keeps_p0_sources_and_gives_three_per_split(
    corpus: Path,
    sources: dict[str, Source],
) -> None:
    noise_dir = corpus / "_background_noise_"
    for i in range(3):
        (noise_dir / f"noise{i}.wav").unlink()
    for stem in NOISE_ASSIGNMENT_V2:
        write_noise(noise_dir / f"{stem}.wav")
    manifest = build_manifest(
        corpus,
        sources,
        max_per_class=2,
        dataset_version="v2-test",
        split_version=SPLIT_VERSION_V2,
    )
    by_split: dict[str, set[str]] = {}
    for e in manifest.examples:
        if e.label == "silence":
            by_split.setdefault(e.split, set()).add(e.speaker_id.removeprefix("noise-source:"))
    assert {k: len(v) for k, v in by_split.items()} == {"train": 3, "validation": 3, "test": 3}
    # Every P0 source stays in its P0 split, so the P0 checkpoint never trained on a V2 test source.
    p0 = {
        "doing_the_dishes": "train",
        "pink_noise": "train",
        "dude_miaowing": "validation",
        "running_tap": "validation",
        "exercise_bike": "test",
        "white_noise": "test",
    }
    for stem, split in p0.items():
        assert stem in by_split[split]
    assert noise_sources_by_split(manifest)["test"] == [
        "exercise_bike",
        "violet_noise",
        "white_noise",
    ]


def test_v1_noise_split_ignores_generated_sources(
    corpus: Path,
    sources: dict[str, Source],
) -> None:
    before = build_manifest(corpus, sources, max_per_class=2)
    write_noise(corpus / "_background_noise_" / "blue_noise.wav")
    after = build_manifest(corpus, sources, max_per_class=2)
    assert before.examples == after.examples
