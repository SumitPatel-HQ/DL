from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ml.data import build_manifest, load_manifest, save_manifest, validate_manifest
from ml.data.manifest import Source
from ml.data.prepare import prepare_cache
from ml.features import LABELS
from ml.infer import features_for_inference


def test_expanded_manifest_requires_new_word_provenance_and_rebuilds_unknown(
    corpus: Path, sources: dict[str, Source], tmp_path: Path
) -> None:
    from conftest import write_wav

    from ml.data.manifest import Source, speaker_split

    expanded = (*LABELS[:6], "next", "pause", "urgent", "unknown", "silence")
    for split in ("train", "validation", "test"):
        speaker = next(f"s{i}" for i in range(1000) if speaker_split(f"s{i}") == split)
        for word in ("next", "pause", "urgent"):
            write_wav(corpus / word / f"{speaker}_nohash_0.wav")
    with pytest.raises(ValueError, match="provenance"):
        build_manifest(corpus, sources, labels=expanded, max_per_class=1)
    licensed = Source("fixture://new", "v1", "test-only", "fixture", "generated", "tests", "temp")
    expanded_sources = {**sources, **dict.fromkeys(("next", "pause", "urgent"), licensed)}
    manifest = build_manifest(corpus, expanded_sources, labels=expanded, max_per_class=1)
    assert manifest.dataset_version == "speech-commands-v0.02-p2-vocab-v1"
    assert manifest.labels == expanded
    for split in ("train", "validation", "test"):
        rows = [e for e in manifest.examples if e.split == split]
        assert {e.label for e in rows} == set(expanded)
        assert {e.original_word for e in rows if e.label == "unknown"} == {"cat", "dog", "tree"}
        assert all(e.source_id == e.label for e in rows if e.label in ("next", "pause", "urgent"))
    path = tmp_path / "expanded.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest


def test_manifest_roundtrip_has_all_classes_without_speaker_or_source_leakage(
    corpus: Path, sources: dict[str, Source], tmp_path: Path
) -> None:
    manifest = build_manifest(corpus, sources, max_per_class=2)
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


def test_mini_eight_is_rejected_as_insufficient_unknown(
    corpus: Path, sources: dict[str, Source]
) -> None:
    for word in ("cat", "dog", "tree"):
        for path in (corpus / word).glob("*.wav"):
            path.unlink()
    with pytest.raises(ValueError, match="Unknown"):
        build_manifest(corpus, sources)


def test_training_cache_matches_inference_on_same_wav(
    corpus: Path, sources: dict[str, Source], tmp_path: Path
) -> None:
    manifest = build_manifest(corpus, sources, max_per_class=1)
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
