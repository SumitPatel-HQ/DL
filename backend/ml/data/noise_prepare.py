"""Prepare speech-commands-v0.02-p1-noise-v1 dataset and features cache."""

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
    local_path,
    save_manifest,
    validate_manifest,
)
from ml.data.noise import DEFAULT_SNR_DB, ensure_additional_noise_sources, mix_noise
from ml.data.prepare import as_example
from ml.features import FEATURE_CONFIG, LABELS, FeaturePipeline, Normalization, read_wav


def prepare_noise_dataset(
    speech_commands_root: Path,
    sources_path: Path,
    output_dir: Path,
    base_normalization_path: Path,
) -> tuple[Manifest, Path]:
    """Build manifest v2 with 3 noise sources per split and generate clean + noisy features."""
    # 1. Ensure 9 background noise files exist in speech commands root
    noise_dir = speech_commands_root / "_background_noise_"
    ensure_additional_noise_sources(noise_dir)

    sources = {
        key: Source(**value)
        for key, value in json.loads(sources_path.read_text(encoding="utf-8")).items()
    }
    sources["noise"] = replace(
        sources["noise"],
        attribution=sources["noise"].attribution
        + " Blue, brown and violet noise are project-synthesized (ml.data.noise, fixed seeds).",
    )
    manifest = build_manifest(
        speech_commands_root,
        sources,
        max_per_class=128,
        dataset_version=DATASET_VERSION_V2,
        split_version=SPLIT_VERSION_V2,
    )
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, manifest_path)
    validate_manifest(manifest, speech_commands_root, verify_audio=True)
    digest = file_sha256(manifest_path)

    # 2. Use the checkpoint's fitted normalization
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    norm_bytes = base_normalization_path.read_bytes()
    (features_dir / "normalization.json").write_bytes(norm_bytes)
    norm_digest = file_sha256(features_dir / "normalization.json")
    stats = Normalization.load(features_dir / "normalization.json")
    pipeline = FeaturePipeline(stats)

    # 3. Cache clean features for all splits
    # Preload test noise clips for mixing
    test_noise_clips: list[np.ndarray] = []
    for ex in manifest.examples:
        if ex.split == "test" and ex.label == "silence":
            clip = read_wav(local_path(speech_commands_root, ex.path))[
                round(ex.start_s * 16000) : round(ex.end_s * 16000)
            ]
            test_noise_clips.append(clip)

    for split in SPLITS:
        split_examples = [e for e in manifest.examples if e.split == split]
        features_list = []
        labels_list = []
        examples_meta = []
        for example in split_examples:
            audio = read_wav(local_path(speech_commands_root, example.path))
            segment = audio[round(example.start_s * 16000) : round(example.end_s * 16000)]
            feat = pipeline.from_audio(segment, 16000)
            features_list.append(feat)
            labels_list.append(LABELS.index(example.label))
            examples_meta.append(json.dumps(as_example(example), sort_keys=True))

        np.savez_compressed(
            features_dir / f"{split}.npz",
            features=np.stack(features_list),
            labels=np.array(labels_list, dtype=np.int64),
            examples=np.array(examples_meta),
            metadata=np.array(
                json.dumps(
                    {
                        "features": FEATURE_CONFIG,
                        "labels": LABELS,
                        "split": split,
                        "dataset_version": manifest.dataset_version,
                        "split_version": manifest.split_version,
                        "manifest_sha256": digest,
                        "normalization_sha256": norm_digest,
                    },
                    sort_keys=True,
                )
            ),
        )

    # 4. Cache noisy features for test split
    rng = np.random.default_rng(2026)
    noisy_features = []
    noisy_labels = []
    noisy_examples = []
    test_examples = [e for e in manifest.examples if e.split == "test"]
    for i, example in enumerate(test_examples):
        audio = read_wav(local_path(speech_commands_root, example.path))
        segment = audio[round(example.start_s * 16000) : round(example.end_s * 16000)]
        if example.label != "silence" and len(test_noise_clips) > 0:
            noise_clip = test_noise_clips[i % len(test_noise_clips)]
            mixed = mix_noise(segment, noise_clip, snr_db=DEFAULT_SNR_DB, rng=rng)
        else:
            mixed = segment
        feat = pipeline.from_audio(mixed, 16000)
        noisy_features.append(feat)
        noisy_labels.append(LABELS.index(example.label))
        noisy_examples.append(json.dumps(as_example(example), sort_keys=True))

    np.savez_compressed(
        features_dir / "test-noisy.npz",
        features=np.stack(noisy_features),
        labels=np.array(noisy_labels, dtype=np.int64),
        examples=np.array(noisy_examples),
        metadata=np.array(
            json.dumps(
                {
                    "features": FEATURE_CONFIG,
                    "labels": LABELS,
                    "split": "test",
                    "condition": "noisy",
                    "snr_db": DEFAULT_SNR_DB,
                    "dataset_version": manifest.dataset_version,
                    "split_version": manifest.split_version,
                    "manifest_sha256": digest,
                    "normalization_sha256": norm_digest,
                },
                sort_keys=True,
            )
        ),
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
