"""CLI: prepare local manifests and caches, with train-only normalization."""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

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
from ml.features.pipeline import Audio


def iter_features(
    manifest: Manifest,
    root: Path,
    split: str,
    pipeline: FeaturePipeline,
) -> Iterator[tuple[Example, Audio]]:
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
        segment = audio[round(example.start_s * 16000) : round(example.end_s * 16000)]
        yield example, pipeline.from_audio(segment, 16000)


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
    )
    output.mkdir(parents=True, exist_ok=True)
    stats.save(output / "normalization.json")
    pipeline = FeaturePipeline(stats)
    for split in SPLITS:
        rows = list(iter_features(manifest, root, split, pipeline))
        np.savez_compressed(
            output / f"{split}.npz",
            features=np.stack([features for _, features in rows]),
            labels=np.array([LABELS.index(e.label) for e, _ in rows], dtype=np.int64),
            examples=np.array([json.dumps(as_example(e), sort_keys=True) for e, _ in rows]),
            metadata=np.array(
                json.dumps(
                    {
                        "features": FEATURE_CONFIG,
                        "labels": LABELS,
                        "split": split,
                        "dataset_version": manifest.dataset_version,
                        "split_version": manifest.split_version,
                        "manifest_sha256": digest,
                        "normalization_sha256": file_sha256(output / "normalization.json"),
                    },
                    sort_keys=True,
                )
            ),
        )
    return stats


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
        manifest = build_manifest(args.root, sources, max_per_class=args.max_per_class)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_manifest(manifest, args.output)
        print(f"Manifest: {len(manifest.examples)} clips, {file_sha256(args.output)}")
    else:
        stats = prepare_cache(args.manifest, args.root, args.output)
        print(f"Cached all splits; normalization fitted on {stats.frame_count} training frames")


if __name__ == "__main__":
    main()
