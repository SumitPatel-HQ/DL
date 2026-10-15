"""Temporal prototype through the frozen continuous protocol (fixture audio, random weights)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from app.services.evaluations import DOCUMENT, ContinuousRun
from ml.data.ami_temporal import KEYWORDS
from ml.evaluate_continuous import (
    TOLERANCE_S,
    ContinuousSplit,
    Reference,
    SplitRecording,
)
from ml.evaluate_temporal import DETECTOR_FORMAT, GRID, run, tune
from ml.features import Normalization
from ml.temporal import LoadedTemporal, TemporalNet


def split(name: str, recording_id: str) -> ContinuousSplit:
    rng = np.random.default_rng(len(recording_id))
    audio = rng.normal(0, 0.05, 16000 * 6).astype(np.float32)
    references = (
        *(Reference(k, 0.2 + 0.3 * i, 0.4 + 0.3 * i) for i, k in enumerate(KEYWORDS)),
        Reference("unknown", 2.0, 3.0),
        Reference("silence", 4.0, 5.0),
    )
    recording = SplitRecording(recording_id, recording_id[0] * 64, 6.0, ("spk",), references, audio)
    return ContinuousSplit("ami-fixture", "split-fixture", name, "automatic", (recording,))


@pytest.fixture
def loaded() -> LoadedTemporal:
    torch.manual_seed(0)
    norm = Normalization((0.0,) * 40, (1.0,) * 40, 1, "fixture", "fixture", "x" * 64)
    model = TemporalNet(8, 1).eval()
    return LoadedTemporal(model, norm, {"experiment_id": "T01"}, Path("t01/model.pt"), "a" * 64)


def frozen(loaded: LoadedTemporal, **overrides: Any) -> dict[str, Any]:
    validation = split("validation", "val")
    value = {
        "format": DETECTOR_FORMAT,
        "created_at": "2026-10-16T00:00:00+00:00",
        "checkpoint": {"sha256": loaded.sha256},
        "config": {"smooth_frames": 5, "threshold": 0.5, "min_frames": 4, "suppression_s": 1.0},
        "tolerance_s": TOLERANCE_S,
        "tuned_on": {
            "dataset_version": validation.dataset_version,
            "split_version": validation.split_version,
            "split": "validation",
            "recordings": {"val": "v" * 64},
        },
    }
    return {**value, **overrides}


def test_tuning_reads_validation_only_and_freezes_the_protocol(loaded: LoadedTemporal) -> None:
    with pytest.raises(ValueError, match="validation"):
        tune(loaded, split("test", "tst"))
    detector = tune(loaded, split("validation", "val"))
    assert detector["format"] == DETECTOR_FORMAT
    assert detector["tolerance_s"] == TOLERANCE_S
    assert detector["tuned_on"]["split"] == "validation"
    assert detector["selected"]["smooth_frames"] in GRID["smooth_frames"]


def test_test_run_refuses_detectors_not_frozen_on_validation(loaded: LoadedTemporal) -> None:
    test = split("test", "tst")
    run(loaded, test, frozen(loaded))
    bad = [
        frozen(loaded, tolerance_s=1.0),
        frozen(loaded, checkpoint={"sha256": "d" * 64}),
        frozen(loaded, format="kws-detector-v1"),
        frozen(loaded, tuned_on={**frozen(loaded)["tuned_on"], "split": "test"}),
        frozen(
            loaded,
            tuned_on={
                **frozen(loaded)["tuned_on"],
                "recordings": {"tst": test.recordings[0].sha256},
            },
        ),
    ]
    for detector in bad:
        with pytest.raises(ValueError):
            run(loaded, test, detector)


def test_run_document_is_a_valid_experimental_continuous_run(loaded: LoadedTemporal) -> None:
    baseline = {"run_id": "e02-continuous-test-x", "event_f1": 0.010}
    document = run(loaded, split("test", "tst"), frozen(loaded), baseline=baseline)
    document["detector"]["file_sha256"] = "c" * 64
    assert document["kind"] == "continuous" and document["experimental"] is True
    assert document["target"]["event_f1"] == 0.70
    assert document["baseline"] == baseline
    assert set(document["metrics"]["per_keyword"]) == {"left", "right", "up", "down", "go", "stop"}
    for rows in document["detections"].values():
        for detection in rows:
            assert {"keyword", "start_s", "end_s", "confidence", "threshold"} <= set(detection)
    # The existing import validator reconciles every count and rate from the document itself.
    imported = DOCUMENT.validate_python(document)
    assert isinstance(imported, ContinuousRun)
    metrics = document["metrics"]
    assert metrics["true_positive"] + metrics["false_negative"] == 6
