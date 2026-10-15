"""CLI: isolated Evaluation Runs from a saved checkpoint, and same-split comparisons.

Evaluation never fits or tunes anything. Metrics are deterministic for a fixed
checkpoint and cache; timing and memory are measurements of the current machine.
"""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.checkpoint import LoadedModel, load_checkpoint
from ml.data.prepare import load_cache
from ml.metrics import classification_report, confusion_matrix
from ml.runtime import code_revision, peak_rss_bytes

MACRO_F1_TARGET = 0.85
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


def evaluate(
    checkpoint: Path, features_dir: Path, split: str, *, latency_runs: int = 200
) -> dict[str, Any]:
    if split not in ("validation", "test"):
        raise ValueError("Evaluation Runs use held-out validation or test data only")
    started = time.perf_counter()
    loaded = load_checkpoint(checkpoint)
    cached = load_cache(features_dir, split)
    trained_on = loaded.metadata["dataset"]
    dataset = {key: cached.metadata[key] for key in DATASET_KEYS}
    if dataset != trained_on:
        raise ValueError("Cache does not match the checkpoint's manifest or normalization")
    probabilities = loaded.probabilities(cached.features)
    detected = probabilities.argmax(axis=1).astype(np.int64)
    report = classification_report(
        confusion_matrix(cached.labels, detected, len(loaded.labels)), loaded.labels
    )
    eval_seconds = time.perf_counter() - started
    latency = single_window_latency(loaded, cached.features[0], latency_runs)
    experiment = loaded.metadata["experiment_id"]
    return {
        "run_id": f"{experiment.lower()}-isolated-{split}-{loaded.sha256[:12]}",
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
        "dataset": {**dataset, "split": split},
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
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--latency-runs", type=int, default=200)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = evaluate(
            args.checkpoint, args.features, args.split, latency_runs=args.latency_runs
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
        print("  confusion (rows true, columns detected):")
        for label, row in zip(result["labels"], metrics["confusion_matrix"]["values"], strict=True):
            print(f"  {label:<8} {' '.join(f'{v:4d}' for v in row)}")
    else:
        comparison = compare([json.loads(path.read_text(encoding="utf-8")) for path in args.runs])
        write_json(args.output, comparison)
        for row in comparison["runs"]:
            print(
                f"{row['experiment_id']}: {row['num_layers']}x{row['hidden_size']}, "
                f"{row['parameters']} params, {row['checkpoint_bytes']} bytes, "
                f"macro-F1 {row['macro_f1']:.4f}, "
                f"forward {row['model_forward_median_ms']:.2f} ms median"
            )


if __name__ == "__main__":
    main()
