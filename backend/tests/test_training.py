import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.checkpoint import load_checkpoint
from ml.evaluate import compare, evaluate
from ml.features import LABELS
from ml.metrics import classification_report, confusion_matrix
from ml.networks import LstmClassifier
from ml.train import train


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "T01",
                "seed": 7,
                "network": {"type": "lstm", "hidden_size": 8, "num_layers": 2, "dropout": 0.1},
                "training": {
                    "batch_size": 16,
                    "optimizer": "adam",
                    "learning_rate": 0.01,
                    "loss": "cross_entropy",
                    "max_epochs": 3,
                    "early_stopping": {"monitor": "validation_loss", "patience": 2},
                    "grad_clip_norm": 5.0,
                    "num_threads": 1,
                },
            }
        )
    )
    return path


def test_metrics_match_hand_computed_fixture() -> None:
    matrix = confusion_matrix(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0]), 3)
    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    report = classification_report(matrix, ("a", "b", "c"))
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["per_class"]["b"] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2}
    )
    assert report["per_class"]["c"]["f1"] == 0
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + 0) / 3)


def test_lstm_emits_one_logit_per_class_with_dropout_outside_single_layer() -> None:
    model = LstmClassifier(hidden_size=16, num_layers=1, dropout=0.3)
    assert model.lstm.dropout == 0
    assert model(torch.zeros(3, 98, 40)).shape == (3, len(LABELS))
    with pytest.raises(ValueError):
        LstmClassifier(hidden_size=16, num_layers=3)


def test_training_is_seeded_and_checkpoint_reloads_with_contract(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    first = train(config, cache, tmp_path / "a")
    second = train(config, cache, tmp_path / "b")
    loss_keys = ("train_loss", "validation_loss")
    for a, b in zip(first["training"]["history"], second["training"]["history"], strict=True):
        assert [a[k] for k in loss_keys] == [b[k] for k in loss_keys]

    loaded = load_checkpoint(tmp_path / "a" / "model.pt")
    again = load_checkpoint(tmp_path / "b" / "model.pt")
    for name, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor, again.model.state_dict()[name])
    assert loaded.labels == LABELS
    assert loaded.metadata["network"]["num_layers"] == 2
    assert loaded.sha256 == first["checkpoint"]["sha256"]
    window = np.zeros((1, 98, 40), dtype=np.float32)
    np.testing.assert_allclose(loaded.probabilities(window).sum(axis=1), 1, rtol=1e-6)
    assert (tmp_path / "a" / "curves.csv").read_text().startswith("epoch,train_loss")


def test_checkpoint_with_different_class_order_is_rejected(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    train(config, cache, tmp_path / "run")
    path = tmp_path / "run" / "model.pt"
    payload: dict[str, Any] = torch.load(path, weights_only=True)
    payload["labels"] = list(reversed(payload["labels"]))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(path)


def test_evaluation_is_reproducible_and_reports_every_class(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    train(config, cache, tmp_path / "run")
    checkpoint = tmp_path / "run" / "model.pt"
    first = evaluate(checkpoint, cache, "test", latency_runs=2)
    second = evaluate(checkpoint, cache, "test", latency_runs=2)
    assert first["metrics"] == second["metrics"]
    assert first["run_id"] == second["run_id"]
    assert set(first["metrics"]["per_class"]) == set(LABELS)
    matrix = np.array(first["metrics"]["confusion_matrix"]["values"])
    assert matrix.shape == (8, 8) and matrix.sum() == first["metrics"]["examples"]
    assert first["dataset"]["split"] == "test"
    assert first["checkpoint"]["sha256"] == load_checkpoint(checkpoint).sha256
    assert compare([first, second])["runs"][0]["macro_f1"] == first["metrics"]["macro_f1"]
    validation = evaluate(checkpoint, cache, "validation", latency_runs=2)
    with pytest.raises(ValueError, match="identical split"):
        compare([first, validation])
    with pytest.raises(ValueError, match="held-out"):
        evaluate(checkpoint, cache, "train", latency_runs=2)
