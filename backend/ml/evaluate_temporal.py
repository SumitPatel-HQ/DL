"""CLI: tune the temporal Detector on validation, then continuous-event Evaluation Runs.

Runs the ADR-0006 prototype through the frozen continuous protocol: the same AMI
recordings, ±0.5 s center tolerance, one-to-one match rule, per-Keyword duplicate
suppression and false-alarm exposure as `ml.evaluate_continuous`. `tune` reads
validation only; `run` on test refuses a detector file tuned on anything else.
The run document keeps the `continuous` shape, so the existing Evaluation Runs
import and comparison seam stores it unchanged.
"""

import argparse
import itertools
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml.data.ami_temporal import FRAME_CLASSES
from ml.data.annotations import validate_dataset
from ml.data.manifest import file_sha256
from ml.evaluate import write_json
from ml.evaluate_continuous import (
    EVENT_F1_TARGET,
    TOLERANCE_S,
    ContinuousSplit,
    composition,
    dataset_info,
    error_review,
    load_split,
    score_recording,
    summarize,
    write_examples,
)
from ml.evaluate_continuous import check_frozen as check_frozen_detector
from ml.runtime import code_revision, peak_rss_bytes
from ml.spotting import Detection
from ml.temporal import LoadedTemporal, TemporalConfig, decode, load_checkpoint, posteriors

DETECTOR_FORMAT = "kws-temporal-detector-v1"
SUPPRESSION_S = 1.0
GRID: dict[str, tuple[Any, ...]] = {
    "smooth_frames": (1, 5, 9),
    "threshold": tuple(round(0.3 + 0.05 * i, 2) for i in range(14)),
    "min_frames": (3, 5, 8, 12),
}
FRAME_S = 0.01


def all_posteriors(
    loaded: LoadedTemporal, data: ContinuousSplit
) -> tuple[dict[str, NDArray[np.float32]], float]:
    started = time.perf_counter()
    pipeline = loaded.pipeline()
    scores = {
        r.recording_id: posteriors(loaded.model, pipeline.from_stream(r.audio, 16000))
        for r in data.recordings
    }
    return scores, time.perf_counter() - started


def evaluate_posteriors(
    data: ContinuousSplit,
    scores: dict[str, NDArray[np.float32]],
    config: TemporalConfig,
    tolerance_s: float,
) -> tuple[dict[str, Any], dict[str, list[Detection]], list[dict[str, Any]]]:
    detections = {r.recording_id: decode(scores[r.recording_id], config) for r in data.recordings}
    results = [
        score_recording(r.references, detections[r.recording_id], tolerance_s)
        for r in data.recordings
    ]
    return summarize(results), detections, results


def nearest_frames(scores: NDArray[np.float32], time_s: float) -> dict[str, Any]:
    """Top class of the frame closest to a time, for error review."""
    i = int(min(max(round(time_s / FRAME_S), 0), len(scores) - 1))
    return {
        "frame_start_s": i * FRAME_S,
        "top_label": FRAME_CLASSES[int(scores[i].argmax())],
        "top_score": float(scores[i].max()),
    }


def checkpoint_info(loaded: LoadedTemporal) -> dict[str, Any]:
    return {
        "path": f"{loaded.path.parent.name}/{loaded.path.name}",
        "sha256": loaded.sha256,
        "experiment_id": loaded.metadata["experiment_id"],
    }


def tune(loaded: LoadedTemporal, data: ContinuousSplit) -> dict[str, Any]:
    """Grid search on validation: best event-F1, then fewest false alarms, then stricter."""
    if data.split != "validation":
        raise ValueError("Detector parameters may only be tuned on validation Recordings")
    scores, _ = all_posteriors(loaded, data)
    candidates = []
    for smooth_frames, threshold, min_frames in itertools.product(*GRID.values()):
        config = TemporalConfig(smooth_frames, threshold, min_frames, SUPPRESSION_S)
        summary, _, _ = evaluate_posteriors(data, scores, config, TOLERANCE_S)
        candidates.append((config, summary))
    candidates.sort(
        key=lambda item: (
            -item[1]["f1"],
            item[1]["false_alarms"]["count"],
            -item[0].threshold,
            -item[0].min_frames,
            -item[0].smooth_frames,
        )
    )

    def row(config: TemporalConfig, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            **config.to_dict(),
            **{key: summary[key] for key in ("precision", "recall", "f1")},
            "false_alarms_per_hour": summary["false_alarms"]["per_hour"],
        }

    best_config, best = candidates[0]
    return {
        "format": DETECTOR_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "checkpoint": checkpoint_info(loaded),
        "config": best_config.to_dict(),
        "tolerance_s": TOLERANCE_S,
        "tuned_on": dataset_info(data),
        "objective": "max validation event-F1; ties: fewer false alarms, higher threshold, "
        f"more minimum frames, more smoothing; suppression fixed at {SUPPRESSION_S} s",
        "grid": {key: list(values) for key, values in GRID.items()},
        "selected": row(best_config, best),
        "top_candidates": [row(c, s) for c, s in candidates[:10]],
        "note": "Frozen before any test Evaluation Run. Validation metrics here are optimistic.",
    }


def run(
    loaded: LoadedTemporal,
    data: ContinuousSplit,
    detector: dict[str, Any],
    examples_dir: Path | None = None,
    examples_per_kind: int = 5,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    check_frozen_detector(detector, loaded, data, DETECTOR_FORMAT)
    config = TemporalConfig(**detector["config"])
    scores, scoring_seconds = all_posteriors(loaded, data)
    summary, detections, results = evaluate_posteriors(
        data, scores, config, detector["tolerance_s"]
    )
    review = error_review(data, scores, results, nearest_frames)
    examples = write_examples(data, review, examples_dir, examples_per_kind) if examples_dir else []
    info = composition(data)
    experiment = loaded.metadata["experiment_id"]
    return {
        "run_id": f"{experiment.lower()}-continuous-{data.split}-{loaded.sha256[:12]}",
        "kind": "continuous",
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "experiment_id": experiment,
        "model_family": "temporal-bilstm-frame",
        "experimental": True,
        "checkpoint": checkpoint_info(loaded),
        "detector": {
            "config": detector["config"],
            "tolerance_s": detector["tolerance_s"],
            "match_rule": "same Keyword, Detection center within tolerance of the reference "
            "center, one-to-one",
            "tuned_on": detector["tuned_on"],
            "detector_created_at": detector["created_at"],
        },
        "dataset": {
            **dataset_info(data),
            "condition": "clean",
            "noise_sources": [],
            "snr_db": None,
            "noise_sources_by_split": {},
            "composition": info,
        },
        "metrics": summary,
        "target": {
            "event_f1": EVENT_F1_TARGET,
            "met": summary["f1"] >= EVENT_F1_TARGET,
            "note": "Preliminary hypothesis target; unmet targets are reported, never redefined.",
        },
        "baseline": baseline,
        "detections": {
            recording_id: [d.to_dict() for d in rows] for recording_id, rows in detections.items()
        },
        "error_review": {
            "method": "Automatic categorization against automatic AMI references. Clips are "
            "listening aids; no human listening verdicts are recorded here.",
            "example_clips": examples,
            **review,
        },
        "limitations": [
            "Reference word timings are AMI forced alignment, not manually verified.",
            "Silence spans are inferred from annotation gaps plus low RMS.",
            "Keyword counts per split are small; per-Keyword metrics are not reliable.",
            "Detection spans come from frame labels and only approximate word boundaries.",
            "Experimental prototype; Confidence is a smoothed frame score, not correctness.",
        ],
        "runtime": {
            "machine": {"os": platform.platform(), "cpu": platform.processor()},
            "scoring_seconds": scoring_seconds,
            "real_time_factor": scoring_seconds / info["duration_s"],
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }


def baseline_reference(path: Path) -> dict[str, Any]:
    """Compact pointer to the sliding-window run this prototype is compared against."""
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": value["run_id"],
        "file_sha256": file_sha256(path),
        "checkpoint_sha256": value["checkpoint"]["sha256"],
        "dataset_version": value["dataset"]["dataset_version"],
        "split_version": value["dataset"]["split_version"],
        "split": value["dataset"]["split"],
        "event_f1": value["metrics"]["f1"],
        "precision": value["metrics"]["precision"],
        "recall": value["metrics"]["recall"],
        "false_alarms_per_hour": value["metrics"]["false_alarms"]["per_hour"],
        "tolerance_s": value["detector"]["tolerance_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("tune", "run"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--isolated-manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    run_parser = sub.choices["run"]
    run_parser.add_argument("--detector", type=Path, required=True)
    run_parser.add_argument("--split", choices=("validation", "test"), default="test")
    run_parser.add_argument("--baseline", type=Path, help="sliding-window run on the same split")
    run_parser.add_argument("--examples", type=Path)

    args = parser.parse_args()
    validate_dataset(
        args.root / "recordings.json", args.root / "events.csv", args.root, args.isolated_manifest
    )
    loaded = load_checkpoint(args.checkpoint)
    if args.command == "tune":
        detector = tune(loaded, load_split(args.root, "validation"))
        write_json(args.output, detector)
        selected = detector["selected"]
        print(
            f"Frozen detector for {detector['checkpoint']['experiment_id']}: {detector['config']}"
        )
        print(
            f"  validation event P {selected['precision']:.3f} R {selected['recall']:.3f} "
            f"F1 {selected['f1']:.3f}, {selected['false_alarms_per_hour']:.1f} false alarms/hour"
        )
        print(f"  detector file sha256 {file_sha256(args.output)}")
        return

    detector = json.loads(args.detector.read_text(encoding="utf-8"))
    baseline = baseline_reference(args.baseline) if args.baseline else None
    result = run(
        loaded,
        load_split(args.root, args.split),
        detector,
        args.examples,
        baseline=baseline,
    )
    result["detector"]["file_sha256"] = file_sha256(args.detector)
    write_json(args.output, result)
    metrics = result["metrics"]
    print(
        f"{result['run_id']}: event P {metrics['precision']:.3f} R {metrics['recall']:.3f} "
        f"F1 {metrics['f1']:.3f} (TP {metrics['true_positive']}, FP "
        f"{metrics['false_positive']}, FN {metrics['false_negative']}), "
        f"{metrics['false_alarms']['per_hour']:.1f} false alarms/hour"
    )
    for keyword, row in metrics["per_keyword"].items():
        print(
            f"  {keyword:<6} TP {row['true_positive']} FP {row['false_positive']} "
            f"FN {row['false_negative']} F1 {row['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
