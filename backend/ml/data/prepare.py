"""CLI: prepare local manifests and caches, with train-only normalization."""

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml.data.manifest import (
    SPLITS,
    Example,
    Manifest,
    Source,
    build_manifest,
    file_sha256,
    load_manifest,
    local_path,
    save_manifest,
    validate_manifest,
)
from ml.features import FEATURE_CONFIG, LABELS, FeaturePipeline, Normalization, read_wav
from ml.features.pipeline import SAMPLE_RATE, Audio
from ml.vocabulary import EXPANDED_LABELS, validate_labels


def iter_segments(manifest: Manifest, root: Path, split: str) -> Iterator[tuple[Example, Audio]]:
    """Yield each example of a split with its raw audio segment."""
    if split not in SPLITS:
        raise ValueError("Invalid split")
    # The selected subset is small. Decode each source only while visiting its segments.
    previous_path, audio = "", np.empty(0, dtype=np.float32)
    for example in manifest.examples:
        if example.split != split:
            continue
        if example.path != previous_path:
            audio = read_wav(local_path(root, example.path))
            previous_path = example.path
        segment = audio[round(example.start_s * SAMPLE_RATE) : round(example.end_s * SAMPLE_RATE)]
        yield example, segment


def iter_features(
    manifest: Manifest,
    root: Path,
    split: str,
    pipeline: FeaturePipeline,
) -> Iterator[tuple[Example, Audio]]:
    for example, segment in iter_segments(manifest, root, split):
        yield example, pipeline.from_audio(segment, SAMPLE_RATE)


def write_split_cache(
    path: Path,
    rows: list[tuple[Example, Audio]],
    metadata: dict[str, Any],
) -> None:
    """Write one cached split; metadata carries versions, hashes and any condition."""
    labels = tuple(metadata.get("labels", LABELS))
    validate_labels(labels)
    np.savez_compressed(
        path,
        features=np.stack([features for _, features in rows]),
        labels=np.array([labels.index(e.label) for e, _ in rows], dtype=np.int64),
        examples=np.array([json.dumps(as_example(e), sort_keys=True) for e, _ in rows]),
        metadata=np.array(
            json.dumps({"features": FEATURE_CONFIG, "labels": labels, **metadata}, sort_keys=True)
        ),
    )


def prepare_cache(manifest_path: Path, root: Path, output: Path) -> Normalization:
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, root, verify_audio=True)
    digest = file_sha256(manifest_path)
    raw_pipeline = FeaturePipeline()
    stats = FeaturePipeline.fit_normalization(
        (features for _, features in iter_features(manifest, root, "train", raw_pipeline)),
        split="train",
        dataset_version=manifest.dataset_version,
        split_version=manifest.split_version,
        manifest_sha256=digest,
        labels=manifest.labels,
    )
    output.mkdir(parents=True, exist_ok=True)
    stats.save(output / "normalization.json")
    pipeline = FeaturePipeline(stats)
    for split in SPLITS:
        write_split_cache(
            output / f"{split}.npz",
            list(iter_features(manifest, root, split, pipeline)),
            {
                "split": split,
                "labels": manifest.labels,
                "dataset_version": manifest.dataset_version,
                "split_version": manifest.split_version,
                "manifest_sha256": digest,
                "normalization_sha256": file_sha256(output / "normalization.json"),
            },
        )
    return stats


@dataclass(frozen=True)
class CachedSplit:
    features: NDArray[np.float32]
    labels: NDArray[np.int64]
    metadata: dict[str, Any]
    examples: NDArray[Any] | None = None


def load_cache(features_dir: Path, split: str, cache_name: str | None = None) -> CachedSplit:
    """Load one cached split, rejecting stale normalization or incompatible metadata."""
    if split not in SPLITS:
        raise ValueError("Invalid split")
    name = cache_name or split
    with np.load(features_dir / f"{name}.npz", allow_pickle=False) as cached:
        features, labels = cached["features"], cached["labels"]
        metadata: dict[str, Any] = json.loads(str(cached["metadata"]))
        examples = cached["examples"] if "examples" in cached else None
    if metadata["features"] != FEATURE_CONFIG or metadata["split"] != split:
        raise ValueError(f"Incompatible feature metadata, class order or split in {split}")
    validate_labels(tuple(metadata["labels"]))
    if tuple(metadata["labels"]) != Normalization.load(features_dir / "normalization.json").labels:
        raise ValueError("Cache and normalization have different class orders")
    if metadata["normalization_sha256"] != file_sha256(features_dir / "normalization.json"):
        raise ValueError("Cache was built with different normalization; regenerate it")
    if features.ndim != 3 or features.shape[2:] != (40,) or len(features) != len(labels):
        raise ValueError("Expected [examples, frames, 40] features aligned with labels")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(
        (labels < 0) | (labels >= len(metadata["labels"]))
    ):
        raise ValueError("Cache contains invalid class indices")
    return CachedSplit(features.astype(np.float32), labels.astype(np.int64), metadata, examples)


def as_example(example: Example) -> dict[str, str | float]:
    return {
        "path": example.path,
        "start_s": example.start_s,
        "end_s": example.end_s,
        "speaker_id": example.speaker_id,
        "label": example.label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--root", type=Path, required=True)
    manifest_parser.add_argument("--sources", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--max-per-class", type=int, default=1000)
    manifest_parser.add_argument("--vocabulary", choices=("p0", "expanded"), default="p0")
    cache_parser = sub.add_parser("cache")
    cache_parser.add_argument("--manifest", type=Path, required=True)
    cache_parser.add_argument("--root", type=Path, required=True)
    cache_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        sources = {
            key: Source(**value)
            for key, value in json.loads(args.sources.read_text(encoding="utf-8")).items()
        }
        manifest = build_manifest(
            args.root,
            sources,
            max_per_class=args.max_per_class,
            labels=EXPANDED_LABELS if args.vocabulary == "expanded" else LABELS,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_manifest(manifest, args.output)
        print(f"Manifest: {len(manifest.examples)} clips, {file_sha256(args.output)}")
    else:
        stats = prepare_cache(args.manifest, args.root, args.output)
        print(f"Cached all splits; normalization fitted on {stats.frame_count} training frames")


if __name__ == "__main__":
    main()
