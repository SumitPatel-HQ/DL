"""Prepare speech-commands-v0.02-p1-noise-v1: manifest v2 plus clean and noisy caches."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ml.data.manifest import (
    DATASET_VERSION_V2,
    SPLIT_VERSION_V2,
    SPLITS,
    Manifest,
    Source,
    build_manifest,
    file_sha256,
    noise_sources_by_split,
    save_manifest,
    validate_manifest,
)
from ml.data.noise import DEFAULT_SNR_DB, ensure_additional_noise_sources, mix_noise
from ml.data.prepare import iter_features, iter_segments, write_split_cache
from ml.features import FeaturePipeline, Normalization
from ml.features.pipeline import SAMPLE_RATE

MAX_PER_CLASS = 128
NOISE_SEED = 2026
SYNTHESIZED_NOTE = (
    " Blue, brown and violet noise are project-synthesized (ml.data.noise, fixed seeds)."
)


def prepare_noise_dataset(
    speech_commands_root: Path,
    sources_path: Path,
    output_dir: Path,
    base_normalization_path: Path,
) -> tuple[Manifest, Path]:
    """Build manifest v2 (three noise sources per split) and its clean and noisy caches.

    The checkpoint's own fitted normalization is reused, so a P0 checkpoint can be
    evaluated on this dataset version without refitting anything.
    """
    ensure_additional_noise_sources(speech_commands_root / "_background_noise_")
    sources = {
        key: Source(**value)
        for key, value in json.loads(sources_path.read_text(encoding="utf-8")).items()
    }
    sources["noise"] = replace(
        sources["noise"], attribution=sources["noise"].attribution + SYNTHESIZED_NOTE
    )
    manifest = build_manifest(
        speech_commands_root,
        sources,
        max_per_class=MAX_PER_CLASS,
        dataset_version=DATASET_VERSION_V2,
        split_version=SPLIT_VERSION_V2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    save_manifest(manifest, manifest_path)
    validate_manifest(manifest, speech_commands_root, verify_audio=True)

    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "normalization.json").write_bytes(base_normalization_path.read_bytes())
    pipeline = FeaturePipeline(Normalization.load(features_dir / "normalization.json"))
    shared = {
        "dataset_version": manifest.dataset_version,
        "split_version": manifest.split_version,
        "manifest_sha256": file_sha256(manifest_path),
        "normalization_sha256": file_sha256(features_dir / "normalization.json"),
        "noise_sources_by_split": noise_sources_by_split(manifest),
    }
    for split in SPLITS:
        rows = list(iter_features(manifest, speech_commands_root, split, pipeline))
        write_split_cache(features_dir / f"{split}.npz", rows, {**shared, "split": split})

    # Noisy test condition: every non-Silence clip gets a test-split Silence clip mixed in.
    test_noise = [
        segment
        for example, segment in iter_segments(manifest, speech_commands_root, "test")
        if example.label == "silence"
    ]
    rng = np.random.default_rng(NOISE_SEED)
    noisy_rows = []
    for i, (example, segment) in enumerate(iter_segments(manifest, speech_commands_root, "test")):
        if example.label != "silence":
            segment = mix_noise(segment, test_noise[i % len(test_noise)], DEFAULT_SNR_DB, rng)
        noisy_rows.append((example, pipeline.from_audio(segment, SAMPLE_RATE)))
    write_split_cache(
        features_dir / "test-noisy.npz",
        noisy_rows,
        {**shared, "split": "test", "condition": "noisy", "snr_db": DEFAULT_SNR_DB},
    )
    return manifest, features_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Speech Commands root")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        type=Path,
        required=True,
        help="normalization.json fitted for the checkpoint under evaluation",
    )
    args = parser.parse_args()
    manifest, features_dir = prepare_noise_dataset(
        args.root, args.sources, args.output, args.normalization
    )
    print(
        f"{manifest.dataset_version} / {manifest.split_version}: {len(manifest.examples)} examples"
    )
    print(f"  features: {features_dir}")


if __name__ == "__main__":
    main()
