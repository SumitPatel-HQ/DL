"""CLI: isolated Evaluation Runs from a saved checkpoint, and same-split comparisons.

Evaluation never fits or tunes anything. Metrics are deterministic for a fixed
checkpoint and cache; timing and memory are measurements of the current machine.
"""

import argparse
import json
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ml.checkpoint import LoadedModel, load_checkpoint
from ml.data.manifest import noise_source_name
from ml.data.prepare import load_cache
from ml.metrics import classification_report, confusion_matrix
from ml.runtime import code_revision, peak_rss_bytes

MACRO_F1_TARGET = 0.85
Condition = Literal["clean", "noisy"]
DATASET_KEYS = ("dataset_version", "split_version", "manifest_sha256", "normalization_sha256")


def single_window_latency(loaded: LoadedModel, window: np.ndarray, runs: int) -> dict[str, Any]:
    """Batch-1 CPU timing for one 1 s window: model only, and MFCC plus model."""
    audio = np.random.default_rng(0).normal(0, 0.05, 16000).astype(np.float32)
    pipeline = loaded.pipeline()
    model_ms, end_to_end_ms = [], []
    for i in range(runs + 10):
        start = time.perf_counter()
        loaded.probabilities(window[None])
        middle = time.perf_counter()
        loaded.probabilities(pipeline.from_audio(audio, 16000)[None])
        end = time.perf_counter()
        if i >= 10:  # Warm-up iterations are excluded.
            model_ms.append((middle - start) * 1000)
            end_to_end_ms.append((end - middle) * 1000)
    return {
        "runs": runs,
        "torch_threads": torch.get_num_threads(),
        "model_forward_ms": {
            "median": float(np.median(model_ms)),
            "p95": float(np.percentile(model_ms, 95)),
        },
        "features_plus_forward_ms": {
            "median": float(np.median(end_to_end_ms)),
            "p95": float(np.percentile(end_to_end_ms, 95)),
        },
        "note": "Timing input is not evaluation data; streaming latency is measured later.",
    }


def silence_recall(
    examples: np.ndarray | None,
    labels: np.ndarray,
    detected: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, float]:
    """Share of each noise source's Silence clips detected as Silence."""
    if examples is None or "silence" not in class_names:
        return {}
    silence = class_names.index("silence")
    total: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for example, label, guess in zip(examples, labels, detected, strict=True):
        source = noise_source_name(json.loads(example).get("speaker_id", ""))
        if label == silence and source:
            total[source] += 1
            hits[source] += int(guess == silence)
    return {source: hits[source] / total[source] for source in sorted(total)}


def evaluate(
    checkpoint: Path,
    features_dir: Path,
    split: str,
    *,
    condition: Condition | None = None,
    latency_runs: int = 200,
    allow_dataset_mismatch: bool = False,
) -> dict[str, Any]:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation Runs use held-out validation or test data only")
    started = time.perf_counter()
    loaded = load_checkpoint(checkpoint)
    cache_name = f"{split}-noisy" if condition == "noisy" else split
    if condition == "noisy" and not (features_dir / f"{cache_name}.npz").exists():
        raise ValueError(f"No noisy feature cache for the {split} split; build it first")
    cached = load_cache(features_dir, split, cache_name=cache_name)
    if tuple(cached.metadata["labels"]) != loaded.labels:
        raise ValueError("Cache and checkpoint class order disagree")
    trained_on = loaded.metadata["dataset"]
    dataset = {key: cached.metadata[key] for key in DATASET_KEYS}
    effective_condition = condition or "clean"
    if dataset["normalization_sha256"] != trained_on["normalization_sha256"]:
        raise ValueError("Cache was built with different normalization than the checkpoint")
    if not allow_dataset_mismatch and dataset != trained_on:
        raise ValueError("Cache does not match the checkpoint's manifest or normalization")
    probabilities = loaded.probabilities(cached.features)
    detected = probabilities.argmax(axis=1).astype(np.int64)
    report = classification_report(
        confusion_matrix(cached.labels, detected, len(loaded.labels)), loaded.labels
    )
    silence_recall_by_source = silence_recall(
        cached.examples, cached.labels, detected, loaded.labels
    )
    noise_sources = sorted(silence_recall_by_source)
    if silence_recall_by_source:
        report["silence_recall_by_source"] = silence_recall_by_source
    eval_seconds = time.perf_counter() - started
    latency = single_window_latency(loaded, cached.features[0], latency_runs)
    experiment = loaded.metadata["experiment_id"]
    # No suffix without a condition keeps the P0 run IDs unchanged.
    suffix = f"-{condition}" if condition else ""
    return {
        "run_id": f"{experiment.lower()}-isolated-{split}{suffix}-{loaded.sha256[:12]}",
        "kind": "isolated",
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "experiment_id": experiment,
        "checkpoint": {
            "path": f"{checkpoint.parent.name}/{checkpoint.name}",
            "sha256": loaded.sha256,
            "bytes": checkpoint.stat().st_size,
            "created_at": loaded.metadata["created_at"],
            "code_revision": loaded.metadata["code_revision"],
        },
        "config": loaded.metadata["config"],
        "config_sha256": loaded.metadata["config_sha256"],
        "network": loaded.metadata["network"],
        "labels": list(loaded.labels),
        "features": loaded.metadata["features"],
        "dataset": {
            **dataset,
            "split": split,
            "condition": effective_condition,
            "noise_sources": noise_sources,
            "snr_db": cached.metadata.get("snr_db") if effective_condition == "noisy" else None,
            "noise_sources_by_split": cached.metadata.get("noise_sources_by_split", {}),
            "checkpoint_dataset_version": (
                trained_on["dataset_version"]
                if trained_on["dataset_version"] != dataset["dataset_version"]
                else None
            ),
        },
        "metrics": report,
        "target": {
            "macro_f1": MACRO_F1_TARGET,
            "met": report["macro_f1"] >= MACRO_F1_TARGET,
            "note": "Hypothesis target; unmet targets are reported, never redefined.",
        },
        "training": {
            key: value for key, value in loaded.metadata["training"].items() if key != "history"
        },
        "curves": loaded.metadata["training"]["history"],
        "runtime": {
            "machine": {"os": platform.platform(), "cpu": platform.processor()},
            "evaluation_seconds": eval_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
            "single_window_latency": latency,
        },
    }


def compare(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Side-by-side runs; refuse comparisons across different data or splits."""
    if len(runs) < 2 or any(run["dataset"] != runs[0]["dataset"] for run in runs):
        raise ValueError("Compare at least two Evaluation Runs on the identical split")
    return {
        "dataset": runs[0]["dataset"],
        "runs": [
            {
                "experiment_id": run["experiment_id"],
                "run_id": run["run_id"],
                "checkpoint_sha256": run["checkpoint"]["sha256"],
                "num_layers": run["network"]["num_layers"],
                "hidden_size": run["network"]["hidden_size"],
                "parameters": run["network"]["parameters"],
                "checkpoint_bytes": run["checkpoint"]["bytes"],
                "macro_f1": run["metrics"]["macro_f1"],
                "accuracy": run["metrics"]["accuracy"],
                "model_forward_median_ms": run["runtime"]["single_window_latency"][
                    "model_forward_ms"
                ]["median"],
                "features_plus_forward_median_ms": run["runtime"]["single_window_latency"][
                    "features_plus_forward_ms"
                ]["median"],
                "mean_train_epoch_seconds": run["training"]["mean_train_epoch_seconds"],
                "best_epoch": run["training"]["best_epoch"],
                "epochs_run": run["training"]["epochs_run"],
                "training_peak_rss_bytes": run["training"]["peak_rss_bytes"],
            }
            for run in runs
        ],
    }


def condition_of(run: dict[str, Any]) -> Condition:
    condition: Condition = run["dataset"].get("condition", "clean")
    return condition


def noise_summary(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    if run["kind"] == "continuous":
        headline = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_alarms_per_hour": metrics["false_alarms"]["per_hour"],
        }
    else:
        headline = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "silence_recall_by_source": metrics.get("silence_recall_by_source", {}),
        }
    dataset = run["dataset"]
    return {
        "run_id": run["run_id"],
        "condition": condition_of(run),
        "noise_sources": dataset.get("noise_sources", []),
        "snr_db": dataset.get("snr_db"),
        **headline,
    }


def compare_noise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair one clean and one noisy run that differ only in dataset condition.

    Checkpoint hash, dataset/split versions and split must match; continuous
    runs must also share the frozen detector configuration and tolerance.
    """
    if len(runs) != 2:
        raise ValueError("Noise comparison needs exactly one clean and one noisy Evaluation Run")
    first, second = runs
    if first.get("kind") != second.get("kind"):
        raise ValueError(f"Kinds differ: {first.get('kind')} vs {second.get('kind')}")
    conditions = [condition_of(run) for run in runs]
    if sorted(conditions) != ["clean", "noisy"]:
        raise ValueError("Noise comparison needs exactly one clean and one noisy Evaluation Run")
    if first["checkpoint"]["sha256"] != second["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs; only the dataset condition may differ")
    for key in ("dataset_version", "split_version", "split"):
        if first["dataset"].get(key) != second["dataset"].get(key):
            raise ValueError(f"Dataset {key} differs; only the dataset condition may differ")
    if first["kind"] == "continuous":
        for key in ("config", "tolerance_s"):
            if first["detector"][key] != second["detector"][key]:
                raise ValueError(f"Detector {key} differs; only the dataset condition may differ")
    return {
        "comparison_type": "noise",
        "kind": first["kind"],
        "checkpoint_sha256": first["checkpoint"]["sha256"],
        "dataset_version": first["dataset"]["dataset_version"],
        "split_version": first["dataset"]["split_version"],
        "split": first["dataset"]["split"],
        "noise_sources": sorted(
            {name for run in runs for name in run["dataset"].get("noise_sources", [])}
        ),
        "noise_sources_by_split": second["dataset"].get("noise_sources_by_split")
        or first["dataset"].get("noise_sources_by_split", {}),
        "runs": [
            noise_summary(run) for run in sorted(runs, key=lambda run: condition_of(run) != "clean")
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--features", type=Path, required=True)
    run_parser.add_argument("--split", choices=("validation", "test"), default="test")
    run_parser.add_argument("--condition", choices=("clean", "noisy"), default=None)
    run_parser.add_argument("--allow-different-dataset", action="store_true")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--latency-runs", type=int, default=200)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    compare_noise_parser = sub.add_parser("compare-noise")
    compare_noise_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    compare_noise_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run":
        result = evaluate(
            args.checkpoint,
            args.features,
            args.split,
            condition=args.condition,
            latency_runs=args.latency_runs,
            allow_dataset_mismatch=args.allow_different_dataset,
        )
        write_json(args.output, result)
        metrics = result["metrics"]
        print(
            f"{result['run_id']}: macro-F1 {metrics['macro_f1']:.4f}, "
            f"accuracy {metrics['accuracy']:.4f} on {metrics['examples']} examples"
        )
        for label, row in metrics["per_class"].items():
            print(
                f"  {label:<8} P {row['precision']:.3f} R {row['recall']:.3f} "
                f"F1 {row['f1']:.3f} n={row['support']}"
            )
        if "silence_recall_by_source" in metrics:
            print("  silence recall by source:")
            for src, rec in metrics["silence_recall_by_source"].items():
                print(f"    {src:<15} {rec * 100:.1f}%")
        print("  confusion (rows true, columns detected):")
        for label, row in zip(result["labels"], metrics["confusion_matrix"]["values"], strict=True):
            print(f"  {label:<8} {' '.join(f'{v:4d}' for v in row)}")
    elif args.command == "compare":
        comparison = compare([json.loads(path.read_text(encoding="utf-8")) for path in args.runs])
        write_json(args.output, comparison)
        for row in comparison["runs"]:
            print(
                f"{row['experiment_id']}: {row['num_layers']}x{row['hidden_size']}, "
                f"{row['parameters']} params, {row['checkpoint_bytes']} bytes, "
                f"macro-F1 {row['macro_f1']:.4f}, "
                f"forward {row['model_forward_median_ms']:.2f} ms median"
            )
    elif args.command == "compare-noise":
        comparison = compare_noise(
            [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
        )
        write_json(args.output, comparison)
        print(f"Saved noise comparison: {args.output}")


if __name__ == "__main__":
    main()
