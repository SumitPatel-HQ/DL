"""Rerun saved E02 Evaluation Runs and compare against committed metrics.

Ticket 07 acceptance: one rerun from saved checkpoint and config must
reproduce reported metrics within stated tolerance. This script runs the
documented CLI commands for the isolated test run and the continuous-event
test run, then compares metric fields only (timings and timestamps are
machine measurements and are excluded from the comparison).

Requires local ignored data (data/isolated, data/continuous-ami) and the
saved checkpoint (artifacts/models/e02/model.pt). These are not in git;
see docs/data/README.md for preparation. Exit code is 0 when every
compared field is within tolerance, 1 otherwise.

Run from the repository root:

    uv run --project backend python scripts/reproduce.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
ISOLATED_TOLERANCE = 1e-9
EVENT_TOLERANCE = 1e-9

ISOLATED_FIELDS = (
    ("metrics.accuracy", ISOLATED_TOLERANCE),
    ("metrics.macro_f1", ISOLATED_TOLERANCE),
)
CONTINUOUS_FIELDS = (
    ("metrics.precision", EVENT_TOLERANCE),
    ("metrics.recall", EVENT_TOLERANCE),
    ("metrics.f1", EVENT_TOLERANCE),
    ("metrics.false_alarms.per_hour", EVENT_TOLERANCE),
)


def nested(document: dict[str, Any], dotted: str) -> Any:
    value = document
    for part in dotted.split("."):
        value = value[part]
    return value


def run_command(arguments: list[str], output: Path) -> None:
    subprocess.run(
        ["uv", "run", "--locked", "python", "-m", *arguments, "--output", str(output)],
        cwd=BACKEND,
        check=True,
    )


def compare(
    expected_path: Path, actual: dict[str, Any], fields: tuple[tuple[str, float], ...]
) -> list[dict[str, Any]]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    rows = []
    for dotted, tolerance in fields:
        expected_value = nested(expected, dotted)
        actual_value = nested(actual, dotted)
        difference = abs(float(expected_value) - float(actual_value))
        rows.append(
            {
                "field": dotted,
                "expected": expected_value,
                "actual": actual_value,
                "difference": difference,
                "tolerance": tolerance,
                "match": difference <= tolerance,
            }
        )
    identity_keys = ["run_id", "kind", "experiment_id"]
    for key in identity_keys:
        rows.append(
            {
                "field": key,
                "expected": expected.get(key),
                "actual": actual.get(key),
                "difference": 0.0,
                "tolerance": 0.0,
                "match": expected.get(key) == actual.get(key),
            }
        )
    for key in ("checkpoint.sha256", "dataset.dataset_version", "dataset.split_version"):
        top, sub = key.split(".")
        match = expected.get(top, {}).get(sub) == actual.get(top, {}).get(sub)
        rows.append(
            {
                "field": key,
                "expected": expected.get(top, {}).get(sub),
                "actual": actual.get(top, {}).get(sub),
                "difference": 0.0,
                "tolerance": 0.0,
                "match": match,
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
    checkpoint = ROOT / "artifacts" / "models" / "e02" / "model.pt"
    for required in (
        checkpoint,
        ROOT / "data" / "isolated" / "features",
        ROOT / "data" / "continuous-ami",
        ROOT / "artifacts" / "evaluations" / "detector-e02.json",
    ):
        if not required.exists():
            print(f"Missing local artifact: {required}", flush=True)
            print("Prepare data and checkpoints first; see docs/data/README.md.", flush=True)
            return 2
    with tempfile.TemporaryDirectory() as temporary:
        isolated_output = Path(temporary) / "e02-test-rerun.json"
        run_command(
            [
                "ml.evaluate",
                "run",
                "--checkpoint",
                str(checkpoint),
                "--features",
                str(ROOT / "data" / "isolated" / "features"),
                "--split",
                "test",
            ],
            isolated_output,
        )
        continuous_output = Path(temporary) / "e02-continuous-test-rerun.json"
        run_command(
            [
                "ml.evaluate_continuous",
                "run",
                "--checkpoint",
                str(checkpoint),
                "--root",
                str(ROOT / "data" / "continuous-ami"),
                "--isolated-manifest",
                str(ROOT / "data" / "isolated" / "manifest.json"),
                "--detector",
                str(ROOT / "artifacts" / "evaluations" / "detector-e02.json"),
                "--split",
                "test",
            ],
            continuous_output,
        )
        isolated = json.loads(isolated_output.read_text(encoding="utf-8"))
        continuous = json.loads(continuous_output.read_text(encoding="utf-8"))
    isolated_rows = compare(
        ROOT / "artifacts" / "evaluations" / "e02-test.json", isolated, ISOLATED_FIELDS
    )
    continuous_rows = compare(
        ROOT / "artifacts" / "evaluations" / "e02-continuous-test.json",
        continuous,
        CONTINUOUS_FIELDS,
    )
    passed = all(row["match"] for row in (*isolated_rows, *continuous_rows))
    report = {
        "note": (
            "Ticket 07 rerun: metric fields must reproduce within tolerance. "
            "Timings, timestamps, and checkpoint file hashes are excluded."
        ),
        "isolated_test": isolated_rows,
        "continuous_test": continuous_rows,
        "passed": passed,
    }
    report_path = ROOT / arguments.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in (*isolated_rows, *continuous_rows):
        status = "match" if row["match"] else "MISMATCH"
        print(f"{status:8} {row['field']:32} expected={row['expected']} actual={row['actual']}")
    print(f"Report: {report_path}")
    print("RERUN PASS" if passed else "RERUN FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
