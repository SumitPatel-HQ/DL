"""CLI: tune the window detector on validation, then continuous-event Evaluation Runs.

`tune` searches detector parameters on validation Recordings only and writes a
frozen detector file. `run` applies that file unchanged; on test it refuses
detectors tuned on anything but disjoint validation Recordings. The ±0.5 s
center tolerance is fixed here and copied into every detector file before test.
"""

import argparse
import itertools
import json
import platform
import time
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml.checkpoint import LoadedModel, load_checkpoint
from ml.data.annotations import read_events, validate_dataset
from ml.data.manifest import file_sha256, local_path
from ml.evaluate import write_json
from ml.features import LABELS, read_wav
from ml.features.pipeline import SAMPLE_RATE, Audio
from ml.metrics import event_scores, false_alarms_per_hour, match_events, span_center
from ml.runtime import code_revision, peak_rss_bytes
from ml.spotting import KEYWORDS, Detection, DetectorConfig, WindowScores, detect, score_windows

DETECTOR_FORMAT = "kws-detector-v1"
TOLERANCE_S = 0.5
EVENT_F1_TARGET = 0.70
NEGATIVE_LABELS = ("unknown", "silence")
GRID: dict[str, tuple[Any, ...]] = {
    "stride_s": (0.2, 0.25),
    "threshold": tuple(round(0.3 + 0.05 * i, 2) for i in range(14)),
    "min_consecutive": (1, 2, 3),
}
# Not tuned: one window width, so one spoken Keyword yields at most one Detection.
SUPPRESSION_S = 1.0
EXAMPLE_PAD_S = 1.5


@dataclass(frozen=True)
class Reference:
    """One annotated span. Ignored Keywords neither count as misses nor as false alarms."""

    label: str
    start_s: float
    end_s: float
    ignored: bool = False

    @property
    def center_s(self) -> float:
        return span_center(self.start_s, self.end_s)


@dataclass(frozen=True)
class SplitRecording:
    recording_id: str
    sha256: str
    duration_s: float
    speaker_ids: tuple[str, ...]
    references: tuple[Reference, ...]
    audio: Audio


@dataclass(frozen=True)
class ContinuousSplit:
    dataset_version: str
    split_version: str
    split: str
    annotation_quality: str
    recordings: tuple[SplitRecording, ...]


def load_split(root: Path, split: str) -> ContinuousSplit:
    """Read one split's audio and annotations; truncated Keywords become ignored references."""
    metadata = json.loads((root / "recordings.json").read_text(encoding="utf-8"))
    provenance_path = root / "keyword-provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else []
    )
    truncated = {
        (row["recording_id"], row["keyword"], round(row["source_start_s"] - row["crop_start_s"], 3))
        for row in provenance
        if row["truncated"]
    }
    rows = defaultdict(list)
    for event in read_events(root / "events.csv"):
        key = (event.recording_id, event.keyword, round(event.start_s, 3))
        rows[event.recording_id].append(
            Reference(event.keyword, event.start_s, event.end_s, key in truncated)
        )
    recordings = tuple(
        SplitRecording(
            row["recording_id"],
            row["sha256"],
            row["duration_s"],
            tuple(row["speaker_ids"]),
            tuple(sorted(rows[row["recording_id"]], key=lambda r: r.start_s)),
            read_wav(local_path(root, row["path"])),
        )
        for row in sorted(metadata["recordings"], key=lambda r: r["recording_id"])
        if row["split"] == split
    )
    if not recordings:
        raise ValueError(f"No continuous Recordings in split {split}")
    return ContinuousSplit(
        metadata["dataset_version"],
        metadata["split_version"],
        split,
        metadata.get("annotation_quality", "unspecified"),
        recordings,
    )


def region_label(references: tuple[Reference, ...], time_s: float) -> str:
    """Annotated label covering a time, or `unlabeled` for gaps outside every span."""
    for reference in references:
        if reference.start_s <= time_s <= reference.end_s:
            return reference.label
    return "unlabeled"


def score_recording(
    references: tuple[Reference, ...], detections: list[Detection], tolerance_s: float
) -> dict[str, Any]:
    """Match Detections to Keyword references; count false alarms inside negative spans."""
    keywords = [r for r in references if r.label in KEYWORDS]
    pairs = match_events(
        [(r.label, r.center_s) for r in keywords],
        [(d.keyword, d.center_s) for d in detections],
        tolerance_s,
    )
    matched = dict(pairs)
    used = {j for _, j in pairs}
    true_positive = [(keywords[i], detections[j]) for i, j in pairs if not keywords[i].ignored]
    missed = [r for i, r in enumerate(keywords) if i not in matched and not r.ignored]
    false = [d for j, d in enumerate(detections) if j not in used]
    negative_seconds = {
        label: sum(r.end_s - r.start_s for r in references if r.label == label)
        for label in NEGATIVE_LABELS
    }
    return {
        "true_positive": true_positive,
        "false_negative": missed,
        "false_positive": [(d, region_label(references, d.center_s)) for d in false],
        "ignored_matches": sum(1 for i, _ in pairs if keywords[i].ignored),
        "negative_seconds": negative_seconds,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-Recording matches into event metrics and false alarms/hour."""
    per_keyword = {}
    for keyword in KEYWORDS:
        per_keyword[keyword] = event_scores(
            sum(1 for r in results for ref, _ in r["true_positive"] if ref.label == keyword),
            sum(1 for r in results for d, _ in r["false_positive"] if d.keyword == keyword),
            sum(1 for r in results for ref in r["false_negative"] if ref.label == keyword),
        )
    overall = event_scores(
        *(
            sum(row[key] for row in per_keyword.values())
            for key in ("true_positive", "false_positive", "false_negative")
        )
    )
    regions = Counter(region for r in results for _, region in r["false_positive"])
    negative = {
        label: sum(r["negative_seconds"][label] for r in results) for label in NEGATIVE_LABELS
    }
    false_alarms = sum(regions[label] for label in NEGATIVE_LABELS)
    return {
        **overall,
        "per_keyword": per_keyword,
        "ignored_matches": sum(r["ignored_matches"] for r in results),
        "false_positive_regions": dict(sorted(regions.items())),
        "false_alarms": {
            "definition": "Unmatched Detections centered inside annotated Unknown or Silence "
            "spans, divided by the total duration of those spans. Unlabeled gaps are not "
            "negative exposure.",
            "count": false_alarms,
            "negative_seconds": sum(negative.values()),
            "per_hour": false_alarms_per_hour(false_alarms, sum(negative.values())),
            "by_region": {
                label: {
                    "count": regions[label],
                    "seconds": negative[label],
                    "per_hour": false_alarms_per_hour(regions[label], negative[label]),
                }
                for label in NEGATIVE_LABELS
                if negative[label] > 0
            },
        },
    }


def evaluate_scores(
    data: ContinuousSplit,
    scores: dict[str, WindowScores],
    config: DetectorConfig,
    tolerance_s: float,
) -> tuple[dict[str, Any], dict[str, list[Detection]], list[dict[str, Any]]]:
    detections = {
        recording.recording_id: detect(scores[recording.recording_id], config)
        for recording in data.recordings
    }
    results = [
        score_recording(recording.references, detections[recording.recording_id], tolerance_s)
        for recording in data.recordings
    ]
    return summarize(results), detections, results


def all_scores(
    loaded: LoadedModel, data: ContinuousSplit, stride_s: float
) -> tuple[dict[str, WindowScores], float]:
    started = time.perf_counter()
    scores = {r.recording_id: score_windows(loaded, r.audio, stride_s) for r in data.recordings}
    return scores, time.perf_counter() - started


def tune(loaded: LoadedModel, data: ContinuousSplit) -> dict[str, Any]:
    """Grid search on validation: best event-F1, then fewest false alarms, then stricter."""
    if data.split != "validation":
        raise ValueError("Detector parameters may only be tuned on validation Recordings")
    candidates = []
    for stride_s in GRID["stride_s"]:
        scores, _ = all_scores(loaded, data, stride_s)
        for threshold, min_consecutive in itertools.product(
            GRID["threshold"], GRID["min_consecutive"]
        ):
            config = DetectorConfig(stride_s, threshold, min_consecutive, SUPPRESSION_S)
            summary, _, _ = evaluate_scores(data, scores, config, TOLERANCE_S)
            candidates.append((config, summary))
    candidates.sort(
        key=lambda item: (
            -item[1]["f1"],
            item[1]["false_alarms"]["count"],
            -item[0].threshold,
            -item[0].min_consecutive,
            -item[0].stride_s,
        )
    )
    best_config, best = candidates[0]

    def row(config: DetectorConfig, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            **config.to_dict(),
            **{key: summary[key] for key in ("precision", "recall", "f1")},
            "false_alarms_per_hour": summary["false_alarms"]["per_hour"],
        }

    return {
        "format": DETECTOR_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "checkpoint": checkpoint_info(loaded),
        "config": best_config.to_dict(),
        "tolerance_s": TOLERANCE_S,
        "tuned_on": dataset_info(data),
        "objective": "max validation event-F1; ties: fewer false alarms, higher threshold, "
        f"more consecutive windows, larger stride; suppression fixed at {SUPPRESSION_S} s",
        "grid": {key: list(values) for key, values in GRID.items()},
        "selected": row(best_config, best),
        "top_candidates": [row(c, s) for c, s in candidates[:10]],
        "note": "Frozen before any test Evaluation Run. Validation metrics here are optimistic.",
    }


def checkpoint_info(loaded: LoadedModel) -> dict[str, Any]:
    return {
        "path": f"{loaded.path.parent.name}/{loaded.path.name}",
        "sha256": loaded.sha256,
        "experiment_id": loaded.metadata["experiment_id"],
    }


def dataset_info(data: ContinuousSplit) -> dict[str, Any]:
    return {
        "dataset_version": data.dataset_version,
        "split_version": data.split_version,
        "split": data.split,
        "recordings": {r.recording_id: r.sha256 for r in data.recordings},
    }


def composition(data: ContinuousSplit) -> dict[str, Any]:
    references = [ref for r in data.recordings for ref in r.references]
    seconds: defaultdict[str, float] = defaultdict(float)
    for ref in references:
        seconds[ref.label] += ref.end_s - ref.start_s
    total = sum(r.duration_s for r in data.recordings)
    return {
        "recordings": len(data.recordings),
        "duration_s": total,
        "speaker_ids": sorted({s for r in data.recordings for s in r.speaker_ids}),
        "keyword_events": dict(
            Counter(ref.label for ref in references if ref.label in KEYWORDS and not ref.ignored)
        ),
        "ignored_keyword_events": sum(1 for ref in references if ref.ignored),
        "annotated_seconds": {label: seconds[label] for label in sorted(seconds)},
        "unlabeled_seconds": total - sum(seconds.values()),
        "annotation_quality": data.annotation_quality,
    }


def nearest_window(scores: WindowScores, time_s: float) -> dict[str, Any]:
    """Top class of the window centered closest to a time, for error review."""
    i = int(np.abs(scores.starts_s + 0.5 - time_s).argmin())
    probabilities = scores.probabilities[i]
    return {
        "window_start_s": float(scores.starts_s[i]),
        "top_label": LABELS[int(probabilities.argmax())],
        "top_score": float(probabilities.max()),
    }


def error_review(
    data: ContinuousSplit,
    scores: dict[str, WindowScores],
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    review: dict[str, list[dict[str, Any]]] = {
        "true_positive": [],
        "false_negative": [],
        "false_positive": [],
    }
    for recording, result in zip(data.recordings, results, strict=True):
        window = scores[recording.recording_id]
        for ref, detection in result["true_positive"]:
            review["true_positive"].append(
                {
                    "recording_id": recording.recording_id,
                    "keyword": ref.label,
                    "reference_start_s": ref.start_s,
                    "reference_end_s": ref.end_s,
                    "detection": detection.to_dict(),
                    "center_error_s": detection.center_s - ref.center_s,
                }
            )
        for ref in result["false_negative"]:
            review["false_negative"].append(
                {
                    "recording_id": recording.recording_id,
                    "keyword": ref.label,
                    "reference_start_s": ref.start_s,
                    "reference_end_s": ref.end_s,
                    "nearest_window": nearest_window(window, ref.center_s),
                }
            )
        for detection, region in result["false_positive"]:
            review["false_positive"].append(
                {
                    "recording_id": recording.recording_id,
                    "detection": detection.to_dict(),
                    "annotated_region": region,
                }
            )
    review["false_positive"].sort(key=lambda row: -row["detection"]["confidence"])
    return review


def write_examples(
    data: ContinuousSplit, review: dict[str, list[dict[str, Any]]], output: Path, per_kind: int
) -> list[str]:
    """Save short PCM clips around errors for listening review; files stay in ignored paths."""
    audio = {r.recording_id: r.audio for r in data.recordings}
    written = []
    output.mkdir(parents=True, exist_ok=True)
    for kind, rows in review.items():
        for n, row in enumerate(rows[:per_kind]):
            if "detection" in row:
                center, keyword = row["detection"]["center_s"], row["detection"]["keyword"]
            else:
                center = (row["reference_start_s"] + row["reference_end_s"]) / 2
                keyword = row["keyword"]
            samples = audio[row["recording_id"]]
            start = max(0, round((center - EXAMPLE_PAD_S) * SAMPLE_RATE))
            end = min(len(samples), round((center + EXAMPLE_PAD_S) * SAMPLE_RATE))
            pcm = np.rint(np.clip(samples[start:end], -1, 32767 / 32768) * 32768).astype("<i2")
            path = output / f"{kind}-{n:02d}-{keyword}-{row['recording_id']}-{center:.2f}s.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                wav.writeframes(pcm.tobytes())
            row["example_clip"] = {"file": path.name, "start_s": start / SAMPLE_RATE}
            written.append(path.name)
    return written


def check_frozen(detector: dict[str, Any], loaded: LoadedModel, data: ContinuousSplit) -> None:
    """Reject detector files that could have seen test data or belong to another model."""
    if detector.get("format") != DETECTOR_FORMAT:
        raise ValueError("Unsupported detector file")
    if detector["checkpoint"]["sha256"] != loaded.sha256:
        raise ValueError("Detector was tuned for a different checkpoint")
    if detector["tolerance_s"] != TOLERANCE_S:
        raise ValueError("Match tolerance differs from the frozen protocol")
    tuned_on = detector["tuned_on"]
    if tuned_on["split"] != "validation":
        raise ValueError("Detector parameters must come from validation tuning")
    if data.split == "test":
        if (tuned_on["dataset_version"], tuned_on["split_version"]) != (
            data.dataset_version,
            data.split_version,
        ):
            raise ValueError("Detector was tuned on a different continuous dataset version")
        if set(tuned_on["recordings"].values()) & {r.sha256 for r in data.recordings}:
            raise ValueError("Detector tuning Recordings overlap the test split")


def run(
    loaded: LoadedModel,
    data: ContinuousSplit,
    detector: dict[str, Any],
    examples_dir: Path | None = None,
    examples_per_kind: int = 5,
) -> dict[str, Any]:
    check_frozen(detector, loaded, data)
    config = DetectorConfig(**detector["config"])
    scores, scoring_seconds = all_scores(loaded, data, config.stride_s)
    summary, detections, results = evaluate_scores(data, scores, config, detector["tolerance_s"])
    review = error_review(data, scores, results)
    examples = write_examples(data, review, examples_dir, examples_per_kind) if examples_dir else []
    info = composition(data)
    experiment = loaded.metadata["experiment_id"]
    return {
        "run_id": f"{experiment.lower()}-continuous-{data.split}-{loaded.sha256[:12]}",
        "kind": "continuous",
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "experiment_id": experiment,
        "checkpoint": checkpoint_info(loaded),
        "detector": {
            "config": detector["config"],
            "tolerance_s": detector["tolerance_s"],
            "match_rule": "same Keyword, Detection center within tolerance of the reference "
            "center, one-to-one",
            "tuned_on": detector["tuned_on"],
            "detector_created_at": detector["created_at"],
        },
        "dataset": {**dataset_info(data), "composition": info},
        "metrics": summary,
        "target": {
            "event_f1": EVENT_F1_TARGET,
            "met": summary["f1"] >= EVENT_F1_TARGET,
            "note": "Preliminary hypothesis target; unmet targets are reported, never redefined.",
        },
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
            "Detection spans come from 1 s windows and only approximate word boundaries.",
        ],
        "runtime": {
            "machine": {"os": platform.platform(), "cpu": platform.processor()},
            "scoring_seconds": scoring_seconds,
            "real_time_factor": scoring_seconds / info["duration_s"],
            "peak_rss_bytes": peak_rss_bytes(),
        },
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
    run_parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    validate_dataset(
        args.root / "recordings.json", args.root / "events.csv", args.root, args.isolated_manifest
    )
    loaded = load_checkpoint(args.checkpoint)
    if args.command == "tune":
        detector = tune(loaded, load_split(args.root, "validation"))
        write_json(args.output, detector)
        print(
            f"Frozen detector for {detector['checkpoint']['experiment_id']}: {detector['config']}"
        )
        selected = detector["selected"]
        print(
            f"  validation event P {selected['precision']:.3f} R {selected['recall']:.3f} "
            f"F1 {selected['f1']:.3f}, {selected['false_alarms_per_hour']:.1f} false alarms/hour"
        )
        print(f"  detector file sha256 {file_sha256(args.output)}")
        return
    detector = json.loads(args.detector.read_text(encoding="utf-8"))
    result = run(loaded, load_split(args.root, args.split), detector, args.examples)
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
