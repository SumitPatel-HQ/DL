"""Rerun saved E02 Evaluation Runs and compare against committed metrics.

Ticket 07 acceptance: one rerun from saved checkpoint and config must
reproduce reported metrics within stated tolerance. This script runs the
documented CLI commands for the isolated test run and the continuous-event
test run, then compares every metric field within tolerance and every
provenance field exactly, including the frozen detector parameters for the
continuous run. Timings, timestamps, and checkpoint file bytes are machine
measurements and are excluded from the comparison.

Requires local ignored data (data/isolated, data/continuous-ami) and the
saved checkpoint (artifacts/models/e02/model.pt). These are not in git;
see docs/data/README.md for preparation. Exit code is 0 when every
compared field matches, 1 otherwise.

Run from the repository root:

    uv run --project backend python scripts/reproduce.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
CHECKPOINT = ROOT / "artifacts" / "models" / "e02" / "model.pt"
DETECTOR = ROOT / "artifacts" / "evaluations" / "detector-e02.json"


@dataclass(frozen=True)
class FieldCheck:
    """One dotted-path comparison: within tolerance for metrics, exact for provenance."""

    path: str
    tolerance: float


@dataclass(frozen=True)
class Rerun:
    """One documented CLI invocation plus the committed file it must reproduce."""

    name: str
    argv: tuple[str, ...]
    expected: Path
    metrics: tuple[FieldCheck, ...]
    provenance: tuple[str, ...]


RERUNS: tuple[Rerun, ...] = (
    Rerun(
        name="isolated_test",
        argv=(
            "ml.evaluate",
            "run",
            "--checkpoint",
            str(CHECKPOINT),
            "--features",
            str(ROOT / "data" / "isolated" / "features"),
            "--split",
            "test",
        ),
        expected=ROOT / "docs" / "evaluations" / "isolated-v1" / "e02-test.json",
        metrics=(
            FieldCheck("metrics.accuracy", 1e-9),
            FieldCheck("metrics.macro_f1", 1e-9),
        ),
        provenance=(
            "run_id",
            "kind",
            "experiment_id",
            "checkpoint.sha256",
            "config_sha256",
            "dataset.dataset_version",
            "dataset.split_version",
        ),
    ),
    Rerun(
        name="continuous_test",
        argv=(
            "ml.evaluate_continuous",
            "run",
            "--checkpoint",
            str(CHECKPOINT),
            "--root",
            str(ROOT / "data" / "continuous-ami"),
            "--isolated-manifest",
            str(ROOT / "data" / "isolated" / "manifest.json"),
            "--detector",
            str(DETECTOR),
            "--split",
            "test",
        ),
        expected=ROOT / "docs" / "evaluations" / "continuous-v1" / "e02-continuous-test.json",
        metrics=(
            FieldCheck("metrics.precision", 1e-9),
            FieldCheck("metrics.recall", 1e-9),
            FieldCheck("metrics.f1", 1e-9),
            FieldCheck("metrics.false_alarms.per_hour", 1e-9),
        ),
        provenance=(
            "run_id",
            "kind",
            "experiment_id",
            "checkpoint.sha256",
            "dataset.dataset_version",
            "dataset.split_version",
            "detector.file_sha256",
            "detector.tolerance_s",
            "detector.config.stride_s",
            "detector.config.threshold",
            "detector.config.min_consecutive",
            "detector.config.suppression_s",
            "detector.config.window_s",
        ),
    ),
)


def lookup(document: dict[str, Any], path: str) -> Any:
    """Follow a dotted path such as metrics.false_alarms.per_hour."""
    value: Any = document
    for part in path.split("."):
        value = value[part]
    return value


def run_module(argv: tuple[str, ...], output: Path) -> dict[str, Any]:
    subprocess.run(
        ["uv", "run", "--locked", "python", "-m", *argv, "--output", str(output)],
        cwd=BACKEND,
        check=True,
    )
    result: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    return result


def compare(expected: dict[str, Any], actual: dict[str, Any], rerun: Rerun) -> list[dict[str, Any]]:
    rows = []
    for check in rerun.metrics:
        expected_value = lookup(expected, check.path)
        actual_value = lookup(actual, check.path)
        difference = abs(float(expected_value) - float(actual_value))
        rows.append(
            {
                "field": check.path,
                "expected": expected_value,
                "actual": actual_value,
                "difference": difference,
                "tolerance": check.tolerance,
                "match": difference <= check.tolerance,
            }
        )
    for path in rerun.provenance:
        expected_value = lookup(expected, path)
        actual_value = lookup(actual, path)
        rows.append(
            {
                "field": path,
                "expected": expected_value,
                "actual": actual_value,
                "difference": 0.0,
                "tolerance": 0.0,
                "match": expected_value == actual_value,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="artifacts/evaluations/rerun-07.json",
        help="Ignored rerun report output, relative to the repository root.",
    )
    arguments = parser.parse_args()
    for required in (
        CHECKPOINT,
        DETECTOR,
        ROOT / "data" / "isolated" / "features",
        ROOT / "data" / "isolated" / "manifest.json",
        ROOT / "data" / "continuous-ami",
    ):
        if not required.exists():
            print(f"Missing local artifact: {required}", flush=True)
            print("Prepare data and checkpoints first; see docs/data/README.md.", flush=True)
            return 2
    runs = {}
    with tempfile.TemporaryDirectory() as temporary:
        for rerun in RERUNS:
            actual = run_module(rerun.argv, Path(temporary) / f"{rerun.name}-rerun.json")
            expected = json.loads(rerun.expected.read_text(encoding="utf-8"))
            runs[rerun.name] = compare(expected, actual, rerun)
    passed = all(row["match"] for rows in runs.values() for row in rows)
    report = {
        "note": (
            "Ticket 07 rerun: metric fields must reproduce within tolerance and "
            "provenance fields (checkpoint, config, dataset, detector) exactly. "
            "Timings, timestamps, and checkpoint file hashes are excluded."
        ),
        "runs": runs,
        "passed": passed,
    }
    report_path = ROOT / arguments.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for rows in runs.values():
        for row in rows:
            status = "match" if row["match"] else "MISMATCH"
            print(f"{status:8} {row['field']:32} expected={row['expected']} actual={row['actual']}")
    print(f"Report: {report_path}")
    print("RERUN PASS" if passed else "RERUN FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
