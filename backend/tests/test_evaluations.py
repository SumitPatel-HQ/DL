"""Evaluation Run API over the committed run files; no model checkpoint is needed."""

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings

COMMITTED = Settings().evaluations

ISOLATED = {
    "e01-isolated-test-519de98bae5d",
    "e01-isolated-validation-519de98bae5d",
    "e02-isolated-test-e47e49f57e81",
    "e02-isolated-validation-e47e49f57e81",
}
CONTINUOUS = {"e02-continuous-test-e47e49f57e81", "e02-continuous-validation-e47e49f57e81"}
NOISE_ISOLATED = {
    "e02-isolated-test-clean-e47e49f57e81",
    "e02-isolated-test-noisy-e47e49f57e81",
}
NOISE_CONTINUOUS = {
    "e02-continuous-test-clean-e47e49f57e81",
    "e02-continuous-test-noisy-e47e49f57e81",
}


def settings(tmp_path: Path, evaluations: Path = COMMITTED) -> Settings:
    return Settings(
        storage=tmp_path / "store",
        checkpoint=tmp_path / "missing.pt",
        evaluations=evaluations,
    )


@contextmanager
def open_client(config: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(config), base_url="http://127.0.0.1") as client:
        yield client


def test_importing_committed_runs_lists_each_with_provenance(tmp_path: Path) -> None:
    with open_client(settings(tmp_path)) as client:
        empty = client.get("/api/evaluations")
        assert empty.status_code == 200
        assert empty.json() == {"code": "not_evaluated", "runs": []}

        response = client.post("/api/evaluations/import")
        assert response.status_code == 200
        assert (
            set(response.json()["imported"])
            == ISOLATED | CONTINUOUS | NOISE_ISOLATED | NOISE_CONTINUOUS
        )
        assert response.json()["rejected"] == []

        listing = client.get("/api/evaluations").json()
        assert listing["code"] == "runs_available"
        runs = {run["id"]: run for run in listing["runs"]}
        assert set(runs) == ISOLATED | CONTINUOUS | NOISE_ISOLATED | NOISE_CONTINUOUS
        assert {runs[run]["kind"] for run in ISOLATED} == {"isolated"}
        assert {runs[run]["kind"] for run in CONTINUOUS} == {"continuous"}

        e02 = runs["e02-isolated-test-e47e49f57e81"]
        assert e02["experiment_id"] == "E02"
        assert e02["created_at"] == "2026-10-15T18:21:27.978713+00:00"
        assert e02["source"] == "isolated-v1/e02-test.json"
        assert e02["provenance"] == {
            "checkpoint_sha256": (
                "e47e49f57e81a4d4352efece7c0af798c37527014117fe0ebc554129b4b7589d"
            ),
            "dataset_version": "speech-commands-v0.02-p0-subset-v1",
            "split_version": "speaker-sha256-80-10-10-noise-source-round-robin-v1",
            "split": "test",
            "config_role": "training",
            "config_sha256": "2e399d0a447b90299b6121bd62a376cfefd99956136768ac069719596060e311",
            "code_revision": {
                "commit": "cd2fd3771040c966300e583107e308dd2730400e",
                "dirty": False,
            },
        }
        assert e02["target"] == {
            "metric": "macro_f1",
            "goal": 0.85,
            "achieved": 0.7402308338629419,
            "met": False,
            "note": "Hypothesis target; unmet targets are reported, never redefined.",
        }

        held_out = runs["e02-continuous-test-e47e49f57e81"]
        assert held_out["provenance"]["config_role"] == "detector"
        assert held_out["provenance"]["config_sha256"] == (
            "c75f728b99e64cebbcdedb6b601832e4a7a352d8cd66537f89467876f0713f0b"
        )
        assert held_out["provenance"]["dataset_version"] == "ami-natural-p0-v1"
        assert held_out["target"]["metric"] == "event_f1"
        assert held_out["target"]["goal"] == 0.7
        assert held_out["target"]["achieved"] == 0.01038961038961039
        assert held_out["target"]["met"] is False

        # P1-03: provenance names noise sources per split for the paired runs.
        noisy = runs["e02-isolated-test-noisy-e47e49f57e81"]
        assert noisy["provenance"]["condition"] == "noisy"
        assert noisy["provenance"]["noise_sources"] == [
            "exercise_bike",
            "violet_noise",
            "white_noise",
        ]
        assert noisy["provenance"]["dataset_version"] == "speech-commands-v0.02-p1-noise-v1"
        assert noisy["provenance"]["split_version"] == "speaker-sha256-80-10-10-noise-sources-v2"
        assert (
            noisy["provenance"]["checkpoint_sha256"]
            == runs["e02-isolated-test-clean-e47e49f57e81"]["provenance"]["checkpoint_sha256"]
        )


def keys(value: object) -> set[str]:
    """Every object key nested anywhere in a JSON value."""
    if isinstance(value, dict):
        return set(value).union(*(keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(item) for item in value))
    return set()


def test_run_detail_comes_from_that_run_alone(tmp_path: Path) -> None:
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        e01 = client.get("/api/evaluations/e01-isolated-test-519de98bae5d").json()
        e02 = client.get("/api/evaluations/e02-isolated-test-e47e49f57e81").json()

        assert e01["provenance"]["checkpoint_sha256"].startswith("519de98bae5d")
        assert e02["provenance"]["checkpoint_sha256"].startswith("e47e49f57e81")
        assert e02["provenance"]["config"]["network"]["hidden_size"] == 128
        assert e01["provenance"]["config"]["network"]["hidden_size"] == 64
        assert e02["metrics"]["examples"] == 1017
        assert e02["metrics"]["accuracy"] == 0.7335299901671584
        assert e02["metrics"]["macro_f1"] == 0.7402308338629419
        unknown = e02["metrics"]["per_class"][6]
        assert unknown == {
            "label": "unknown",
            "precision": 0.42207792207792205,
            "recall": 0.5078125,
            "f1": 0.46099290780141844,
            "support": 128,
        }
        matrix = e02["metrics"]["confusion_matrix"]
        assert matrix["labels"] == [
            "left",
            "right",
            "up",
            "down",
            "go",
            "stop",
            "unknown",
            "silence",
        ]
        assert (matrix["rows"], matrix["columns"]) == ("true", "detected")
        assert sum(map(sum, matrix["values"])) == 1017
        assert e02["training"]["best_epoch"] == 10
        assert len(e02["training"]["curves"]) == 20
        assert len(e01["training"]["curves"]) == 19
        assert e02["training"]["curves"][0] == {
            "epoch": 1,
            "train_loss": 1.7907609871520416,
            "train_accuracy": 0.269267364414843,
            "validation_loss": 1.8255013227462769,
            "validation_accuracy": 0.27897838899803534,
            "validation_macro_f1": 0.23383668163619165,
        }
        assert e02["experimental"] is False

        events = client.get("/api/evaluations/e02-continuous-test-e47e49f57e81").json()
        assert events["kind"] == "continuous"
        assert events["experimental"] is True
        assert events["provenance"]["config"] == {
            "stride_s": 0.2,
            "threshold": 0.5,
            "min_consecutive": 1,
            "suppression_s": 1.0,
            "window_s": 1.0,
        }
        metrics = events["metrics"]
        assert (metrics["detections"], metrics["true_positive"]) == (369, 2)
        assert (metrics["false_positive"], metrics["false_negative"]) == (367, 14)
        assert metrics["f1"] == 0.01038961038961039
        assert metrics["false_alarms"]["count"] == 351
        assert metrics["false_alarms"]["negative_seconds"] == 570.217
        assert metrics["false_alarms"]["per_hour"] == 2215.998470757624
        assert metrics["false_alarms"]["by_region"]["silence"] == {
            "count": 1,
            "seconds": 8.827000000000012,
            "per_hour": 407.83958309731446,
        }
        assert metrics["ignored_matches"] == 0
        assert metrics["per_keyword"][2]["keyword"] == "up"
        assert metrics["per_keyword"][2]["true_positive"] == 1
        assert events["composition"]["keyword_events"] == {
            "go": 3,
            "stop": 1,
            "up": 4,
            "right": 6,
            "left": 1,
            "down": 1,
        }
        assert events["composition"]["duration_s"] == 600.0
        assert events["composition"]["speaker_ids"] == ["MIO022", "MIO050", "MIO076", "MIO078"]
        assert events["composition"]["annotation_quality"] == "automatic, not manually verified"
        assert events["detector"]["tuned_on"]["split"] == "validation"
        assert events["detector"]["tolerance_s"] == 0.5
        assert events["limitations"]
        assert "Confidence" in events["confidence_note"]

        # Isolated classification and continuous-event numbers never share a run view,
        # and labeled event scoring never borrows clip accuracy.
        assert "accuracy" not in keys(events)
        assert not {"false_alarms", "per_keyword", "detector", "composition"} & keys(e02)

        missing = client.get("/api/evaluations/e03-isolated-test-000000000000")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "evaluation_not_found"


def test_deleted_runs_stay_deleted_and_leave_explicit_empty_state(tmp_path: Path) -> None:
    config = settings(tmp_path)
    with open_client(config) as client:
        client.post("/api/evaluations/import")
        run = "e02-continuous-test-e47e49f57e81"
        assert client.delete(f"/api/evaluations/{run}").status_code == 204
        gone = client.get(f"/api/evaluations/{run}")
        assert gone.status_code == 404
        assert gone.json()["error"]["code"] == "evaluation_not_found"
        assert client.delete(f"/api/evaluations/{run}").status_code == 404
        remaining = {item["id"] for item in client.get("/api/evaluations").json()["runs"]}
        assert remaining == (ISOLATED | CONTINUOUS | NOISE_ISOLATED | NOISE_CONTINUOUS) - {run}

    # Deletion persists across restart; nothing re-imports runs implicitly.
    with open_client(config) as client:
        listing = client.get("/api/evaluations").json()
        assert run not in {item["id"] for item in listing["runs"]}
        for item in listing["runs"]:
            assert client.delete(f"/api/evaluations/{item['id']}").status_code == 204
        assert client.get("/api/evaluations").json() == {"code": "not_evaluated", "runs": []}

        # An explicit import restores the committed evidence.
        client.post("/api/evaluations/import")
        assert client.get(f"/api/evaluations/{run}").status_code == 200


def committed(name: str) -> dict[str, Any]:
    folder = "continuous-v1" if "continuous" in name or "detector" in name else "isolated-v1"
    result: dict[str, Any] = json.loads((COMMITTED / folder / name).read_text(encoding="utf-8"))
    return result


def write(root: Path, source: str, document: object) -> None:
    path = root / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str) else json.dumps(document))


def test_import_rejects_runs_without_ground_truth_or_consistent_numbers(tmp_path: Path) -> None:
    root = tmp_path / "evaluations"
    write(root, "a/valid.json", committed("e02-test.json"))
    write(root, "a/detector.json", committed("detector-e02.json"))

    unlabeled = committed("e02-test.json")
    unlabeled["run_id"] = "unlabeled-isolated"
    metrics = unlabeled["metrics"]
    metrics["examples"] = 0
    metrics["confusion_matrix"]["values"] = [[0] * 8 for _ in range(8)]
    for counts in metrics["per_class"].values():
        counts["support"] = 0
    write(root, "b/unlabeled-isolated.json", unlabeled)

    no_events = committed("e02-continuous-test.json")
    no_events["run_id"] = "unlabeled-continuous"
    no_events["dataset"]["composition"]["keyword_events"] = {}
    write(root, "b/unlabeled-continuous.json", no_events)

    bragging = committed("e02-continuous-test.json")
    bragging["run_id"] = "bragging"
    bragging["target"]["met"] = True
    write(root, "c/bragging.json", bragging)

    miscounted = committed("e01-test.json")
    miscounted["run_id"] = "miscounted"
    miscounted["metrics"]["confusion_matrix"]["values"][0][0] += 1
    write(root, "c/miscounted.json", miscounted)

    flattered = committed("e01-test.json")
    flattered["run_id"] = "flattered"
    flattered["metrics"]["accuracy"] = 0.99
    write(root, "c/flattered.json", flattered)

    precise = committed("e02-continuous-validation.json")
    precise["run_id"] = "precise"
    precise["metrics"]["precision"] = 0.9
    write(root, "c/precise.json", precise)

    undated = committed("e01-test.json")
    undated["run_id"] = "undated"
    undated["created_at"] = "2026-10-15T11:17:07"
    write(root, "c/undated.json", undated)

    write(root, "c/truncated.json", '{"run_id": "truncated", "kind": "isola')
    write(root, "d/duplicate.json", committed("e02-test.json"))

    with open_client(settings(tmp_path, root)) as client:
        result = client.post("/api/evaluations/import").json()
        assert result == {
            "imported": ["e02-isolated-test-e47e49f57e81"],
            "rejected": [
                {"source": "b/unlabeled-continuous.json", "code": "missing_ground_truth"},
                {"source": "b/unlabeled-isolated.json", "code": "missing_ground_truth"},
                {"source": "c/bragging.json", "code": "invalid_evaluation_run"},
                {"source": "c/flattered.json", "code": "invalid_evaluation_run"},
                {"source": "c/miscounted.json", "code": "invalid_evaluation_run"},
                {"source": "c/precise.json", "code": "invalid_evaluation_run"},
                {"source": "c/truncated.json", "code": "invalid_evaluation_run"},
                {"source": "c/undated.json", "code": "invalid_evaluation_run"},
                {"source": "d/duplicate.json", "code": "duplicate_run_id"},
            ],
        }
        listing = client.get("/api/evaluations").json()
        assert [run["id"] for run in listing["runs"]] == ["e02-isolated-test-e47e49f57e81"]


def test_missing_run_directory_is_an_explicit_import_failure(tmp_path: Path) -> None:
    with open_client(settings(tmp_path, tmp_path / "absent")) as client:
        response = client.post("/api/evaluations/import")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "evaluations_unavailable"
        assert client.get("/api/evaluations").json() == {"code": "not_evaluated", "runs": []}


def test_store_from_before_evaluation_api_upgrades_reserved_table(tmp_path: Path) -> None:
    config = settings(tmp_path)
    config.storage.mkdir(parents=True)
    legacy = sqlite3.connect(config.storage / "metadata.sqlite3")
    legacy.execute(
        "CREATE TABLE evaluation_runs (id TEXT PRIMARY KEY, model_id TEXT NOT NULL, "
        "dataset_id TEXT NOT NULL, split_id TEXT NOT NULL, config_json TEXT NOT NULL, "
        "metrics_json TEXT NOT NULL, artifacts_path TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    legacy.close()
    with open_client(config) as client:
        assert client.post("/api/evaluations/import").status_code == 200
        assert len(client.get("/api/evaluations").json()["runs"]) == 10
    with open_client(config) as client:
        assert len(client.get("/api/evaluations").json()["runs"]) == 10


def test_compare_matched_isolated_pair_reads_both_runs_side_by_side(
    tmp_path: Path,
) -> None:
    """P1-02: comparison loads only from two stored run IDs (E01 vs E02)."""
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e01-isolated-test-519de98bae5d",
                "run_b": "e02-isolated-test-e47e49f57e81",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "comparison_available"
        assert body["kind"] == "isolated"
        assert body["dataset_version"] == "speech-commands-v0.02-p0-subset-v1"
        assert body["split_version"] == "speaker-sha256-80-10-10-noise-source-round-robin-v1"
        assert body["split"] == "test"
        assert [run["id"] for run in body["runs"]] == [
            "e01-isolated-test-519de98bae5d",
            "e02-isolated-test-e47e49f57e81",
        ]
        for run in body["runs"]:
            assert run["kind"] == "isolated"
            metrics = run["metrics"]
            # Every reported number recomputes from the run's own counts.
            values = metrics["confusion_matrix"]["values"]
            correct = sum(values[i][i] for i in range(len(values)))
            assert math.isclose(metrics["accuracy"], correct / metrics["examples"])
            mean_f1 = sum(c["f1"] for c in metrics["per_class"]) / len(metrics["per_class"])
            assert math.isclose(metrics["macro_f1"], mean_f1)
            assert run["training"]["curves"]
            # Isolated and continuous numbers never share a view.
            assert "false_alarms" not in metrics
            assert "per_keyword" not in metrics
        e01, e02 = body["runs"]
        assert e01["metrics"]["macro_f1"] != e02["metrics"]["macro_f1"]
        assert e01["provenance"]["checkpoint_sha256"] != e02["provenance"]["checkpoint_sha256"]


def test_compare_matched_continuous_pair_keeps_event_numbers_separate(
    tmp_path: Path,
) -> None:
    """P1-02: a passing continuous pair carries event numbers only, never accuracy."""
    root = tmp_path / "evaluations"
    write(root, "a/original.json", committed("e02-continuous-test.json"))
    rerun = committed("e02-continuous-test.json")
    rerun["run_id"] = "e02-continuous-test-rerun"
    write(root, "a/rerun.json", rerun)
    with open_client(settings(tmp_path, root)) as client:
        assert client.post("/api/evaluations/import").json()["imported"] == [
            "e02-continuous-test-e47e49f57e81",
            "e02-continuous-test-rerun",
        ]
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-continuous-test-e47e49f57e81",
                "run_b": "e02-continuous-test-rerun",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "continuous"
        for run in body["runs"]:
            metrics = run["metrics"]
            found = metrics["true_positive"] + metrics["false_positive"]
            expected = metrics["true_positive"] + metrics["false_negative"]
            precision = metrics["true_positive"] / found if found else 0.0
            recall = metrics["true_positive"] / expected if expected else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            assert math.isclose(metrics["precision"], precision)
            assert math.isclose(metrics["recall"], recall)
            assert math.isclose(metrics["f1"], f1)
            assert metrics["false_alarms"]["count"] > 0
            assert metrics["per_keyword"]
            assert "accuracy" not in keys(run)


def test_compare_rejects_mismatched_kind_and_split(tmp_path: Path) -> None:
    """P1-02: mismatched kind or dataset/split version is rejected actionably."""
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        kind = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-isolated-test-e47e49f57e81",
                "run_b": "e02-continuous-test-e47e49f57e81",
            },
        )
        assert kind.status_code == 400
        assert kind.json()["error"]["code"] == "comparison_kind_mismatch"
        assert "never share a view" in kind.json()["error"]["message"]

        split = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e01-isolated-test-519de98bae5d",
                "run_b": "e01-isolated-validation-519de98bae5d",
            },
        )
        assert split.status_code == 400
        assert split.json()["error"]["code"] == "comparison_dataset_mismatch"
        assert "identical dataset version" in split.json()["error"]["message"]

        continuous_split = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-continuous-test-e47e49f57e81",
                "run_b": "e02-continuous-validation-e47e49f57e81",
            },
        )
        assert continuous_split.status_code == 400
        assert continuous_split.json()["error"]["code"] == "comparison_dataset_mismatch"


def test_compare_deleted_run_clears_with_no_fallback(tmp_path: Path) -> None:
    """P1-02: deleting a compared run clears the comparison; nothing falls back."""
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        run = "e01-isolated-test-519de98bae5d"
        assert client.delete(f"/api/evaluations/{run}").status_code == 204
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": run,
                "run_b": "e02-isolated-test-e47e49f57e81",
            },
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "evaluation_not_found"
        # The surviving run is still directly readable; nothing falls back.
        survivor = client.get("/api/evaluations/e02-isolated-test-e47e49f57e81")
        assert survivor.status_code == 200


def test_compare_noise_isolated_pair_shares_checkpoint_and_names_sources(
    tmp_path: Path,
) -> None:
    """P1-03: clean/noisy isolated runs pair under one comparison; only condition differs."""
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-isolated-test-clean-e47e49f57e81",
                "run_b": "e02-isolated-test-noisy-e47e49f57e81",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "comparison_available"
        assert body["comparison_type"] == "noise"
        assert body["kind"] == "isolated"
        assert body["dataset_version"] == "speech-commands-v0.02-p1-noise-v1"
        assert body["split_version"] == "speaker-sha256-80-10-10-noise-sources-v2"
        assert body["split"] == "test"
        assert body["noise_sources"] == ["exercise_bike", "violet_noise", "white_noise"]
        clean, noisy = body["runs"]
        assert [run["id"] for run in body["runs"]] == [
            "e02-isolated-test-clean-e47e49f57e81",
            "e02-isolated-test-noisy-e47e49f57e81",
        ]
        # Same checkpoint hash and protocol; only the dataset condition differs.
        assert clean["provenance"]["checkpoint_sha256"] == noisy["provenance"]["checkpoint_sha256"]
        assert clean["provenance"]["condition"] == "clean"
        assert noisy["provenance"]["condition"] == "noisy"
        assert clean["provenance"]["noise_sources"] == noisy["provenance"]["noise_sources"]
        for run in body["runs"]:
            metrics = run["metrics"]
            # Silence recall is reported per noise source.
            assert set(metrics["silence_recall_by_source"]) == {
                "exercise_bike",
                "violet_noise",
                "white_noise",
            }
            # Every reported number recomputes from the run's own counts.
            values = metrics["confusion_matrix"]["values"]
            correct = sum(values[i][i] for i in range(len(values)))
            assert math.isclose(metrics["accuracy"], correct / metrics["examples"])
            mean_f1 = sum(c["f1"] for c in metrics["per_class"]) / len(metrics["per_class"])
            assert math.isclose(metrics["macro_f1"], mean_f1)
            assert "false_alarms" not in metrics
        assert clean["metrics"]["macro_f1"] != noisy["metrics"]["macro_f1"]


def test_compare_noise_continuous_pair_excludes_unlabeled_gaps(tmp_path: Path) -> None:
    """P1-03: clean/noisy continuous runs pair; unlabeled gaps stay out of exposure."""
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-continuous-test-clean-e47e49f57e81",
                "run_b": "e02-continuous-test-noisy-e47e49f57e81",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["comparison_type"] == "noise"
        assert body["kind"] == "continuous"
        assert body["split"] == "test"
        clean, noisy = body["runs"]
        assert clean["provenance"]["checkpoint_sha256"] == noisy["provenance"]["checkpoint_sha256"]
        # Same frozen detector protocol on both conditions.
        assert clean["provenance"]["config_sha256"] == noisy["provenance"]["config_sha256"]
        assert clean["detector"]["tolerance_s"] == noisy["detector"]["tolerance_s"] == 0.5
        for run in body["runs"]:
            metrics = run["metrics"]
            composition = run["composition"]
            # Unlabeled gaps are excluded from negative exposure.
            assert composition["unlabeled_seconds"] > 0
            expected_negative = (
                composition["annotated_seconds"]["unknown"]
                + composition["annotated_seconds"]["silence"]
            )
            assert math.isclose(metrics["false_alarms"]["negative_seconds"], expected_negative)
            assert "accuracy" not in keys(run)


def test_compare_noise_rejects_mismatched_provenance(tmp_path: Path) -> None:
    """P1-03: pairs that differ by more than condition are rejected actionably."""
    root = tmp_path / "evaluations"
    clean_doc = json.loads(
        (COMMITTED / "noise-v1" / "e02-isolated-test-clean.json").read_text(encoding="utf-8")
    )
    write(root, "a/clean.json", clean_doc)
    other_checkpoint = json.loads(
        (COMMITTED / "noise-v1" / "e02-isolated-test-noisy.json").read_text(encoding="utf-8")
    )
    other_checkpoint["run_id"] = "e02-isolated-test-noisy-other-ckpt"
    other_checkpoint["checkpoint"]["sha256"] = "0" * 64
    write(root, "a/other-checkpoint.json", other_checkpoint)
    other_dataset = json.loads(
        (COMMITTED / "noise-v1" / "e02-isolated-test-noisy.json").read_text(encoding="utf-8")
    )
    other_dataset["run_id"] = "e02-isolated-test-noisy-other-data"
    other_dataset["dataset"]["dataset_version"] = "speech-commands-v0.02-p0-subset-v1"
    write(root, "a/other-dataset.json", other_dataset)
    with open_client(settings(tmp_path, root)) as client:
        assert client.post("/api/evaluations/import").status_code == 200
        checkpoint = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-isolated-test-clean-e47e49f57e81",
                "run_b": "e02-isolated-test-noisy-other-ckpt",
            },
        )
        assert checkpoint.status_code == 400
        assert checkpoint.json()["error"]["code"] == "comparison_provenance_mismatch"
        assert "only the dataset condition may differ" in checkpoint.json()["error"]["message"]

        dataset = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-isolated-test-clean-e47e49f57e81",
                "run_b": "e02-isolated-test-noisy-other-data",
            },
        )
        assert dataset.status_code == 400
        assert dataset.json()["error"]["code"] == "comparison_provenance_mismatch"

        # A clean run from the P0 dataset version is not pairable with a noise-v1 run.
    cross_store = tmp_path / "cross-store"
    with open_client(settings(cross_store)) as committed_client:
        committed_client.post("/api/evaluations/import")
        cross = committed_client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-isolated-test-e47e49f57e81",
                "run_b": "e02-isolated-test-noisy-e47e49f57e81",
            },
        )
        assert cross.status_code == 400
        assert cross.json()["error"]["code"] == "comparison_provenance_mismatch"


def test_compare_noise_continuous_rejects_protocol_drift(tmp_path: Path) -> None:
    """P1-03: continuous noise pairs must share detector config and tolerance."""
    root = tmp_path / "evaluations"
    clean_doc = json.loads(
        (COMMITTED / "noise-v1" / "e02-continuous-test-clean.json").read_text(encoding="utf-8")
    )
    write(root, "a/clean.json", clean_doc)
    drifted = json.loads(
        (COMMITTED / "noise-v1" / "e02-continuous-test-noisy.json").read_text(encoding="utf-8")
    )
    drifted["run_id"] = "e02-continuous-test-noisy-drifted"
    drifted["detector"]["config"]["threshold"] = 0.9
    write(root, "a/drifted.json", drifted)
    with open_client(settings(tmp_path, root)) as client:
        assert client.post("/api/evaluations/import").status_code == 200
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-continuous-test-clean-e47e49f57e81",
                "run_b": "e02-continuous-test-noisy-drifted",
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "comparison_protocol_mismatch"


def test_compare_noise_deleted_run_clears_with_no_fallback(tmp_path: Path) -> None:
    """P1-03: deleting one side of a noise pair clears the comparison; nothing falls back."""
    with open_client(settings(tmp_path)) as client:
        client.post("/api/evaluations/import")
        run = "e02-isolated-test-noisy-e47e49f57e81"
        assert client.delete(f"/api/evaluations/{run}").status_code == 204
        response = client.get(
            "/api/evaluations/compare",
            params={
                "run_a": "e02-isolated-test-clean-e47e49f57e81",
                "run_b": run,
            },
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "evaluation_not_found"
        # The surviving clean run is still directly readable; nothing falls back.
        survivor = client.get("/api/evaluations/e02-isolated-test-clean-e47e49f57e81")
        assert survivor.status_code == 200
        assert client.get("/api/evaluations").json()["code"] == "runs_available"


def test_legacy_predicted_columns_are_accepted_and_returned_as_detected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evaluations"
    document = committed("e02-test.json")
    assert document["metrics"]["confusion_matrix"]["columns"] == "detected"
    document["metrics"]["confusion_matrix"]["columns"] = "predicted"
    write(root, "a/legacy.json", document)
    with open_client(settings(tmp_path, root)) as client:
        assert client.post("/api/evaluations/import").json()["imported"] == [
            "e02-isolated-test-e47e49f57e81"
        ]
        detail = client.get("/api/evaluations/e02-isolated-test-e47e49f57e81").json()
        assert (
            detail["metrics"]["confusion_matrix"]["rows"],
            detail["metrics"]["confusion_matrix"]["columns"],
        ) == (
            "true",
            "detected",
        )


def test_noisy_run_without_noise_sources_or_snr_is_rejected(tmp_path: Path) -> None:
    """P1-03: a noisy document must name its noise; it cannot import as unlabeled."""
    root = tmp_path / "evaluations"
    doc = json.loads(
        (COMMITTED / "noise-v1" / "e02-isolated-test-noisy.json").read_text(encoding="utf-8")
    )
    doc["dataset"]["noise_sources"] = []
    write(root, "a/noisy.json", doc)
    with open_client(settings(tmp_path, root)) as client:
        body = client.post("/api/evaluations/import").json()
        assert body["imported"] == []
        assert len(body["rejected"]) == 1
