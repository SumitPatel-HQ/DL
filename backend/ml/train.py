"""CLI: train a CPU LSTM on cached MFCC features with Adam and validation early stopping."""

import argparse
import copy
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ml.checkpoint import save_checkpoint
from ml.data.manifest import file_sha256
from ml.data.prepare import CachedSplit, load_cache
from ml.features import Normalization
from ml.features.pipeline import LABELS
from ml.metrics import classification_report, confusion_matrix
from ml.networks import LstmClassifier, parameter_count
from ml.runtime import code_revision, peak_rss_bytes, seed_everything

PROVENANCE_KEYS = ("dataset_version", "split_version", "manifest_sha256", "normalization_sha256")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    grad_clip_norm: float
    num_threads: int

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ExperimentConfig":
        network, training = value["network"], value["training"]
        if network["type"] != "lstm" or training["optimizer"] != "adam":
            raise ValueError("P0 trains LSTM networks with Adam only")
        if training["loss"] != "cross_entropy" or training["early_stopping"]["monitor"] != (
            "validation_loss"
        ):
            raise ValueError("P0 uses cross-entropy with validation-loss early stopping")
        if training["batch_size"] not in (16, 32):
            raise ValueError("P0 batch size is 16 or 32")
        return cls(
            value["experiment_id"],
            value["seed"],
            network["hidden_size"],
            network["num_layers"],
            network["dropout"],
            training["batch_size"],
            training["learning_rate"],
            training["max_epochs"],
            training["early_stopping"]["patience"],
            training["grad_clip_norm"],
            training["num_threads"],
        )


def dataset_provenance(*splits: CachedSplit) -> dict[str, str]:
    values = [{key: split.metadata[key] for key in PROVENANCE_KEYS} for split in splits]
    if any(value != values[0] for value in values):
        raise ValueError("Splits come from different manifests or normalization")
    return values[0]


def validation_pass(
    model: LstmClassifier, features: torch.Tensor, labels: torch.Tensor
) -> tuple[float, dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        loss = float(nn.functional.cross_entropy(logits, labels))
    detected = logits.argmax(dim=1).numpy().astype(np.int64)
    matrix = confusion_matrix(labels.numpy().astype(np.int64), detected, len(LABELS))
    return loss, classification_report(matrix, LABELS)


def train(config_path: Path, features_dir: Path, output_dir: Path) -> dict[str, Any]:
    raw_config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_json(raw_config)
    seed_everything(config.seed, config.num_threads)
    train_split, validation_split = (load_cache(features_dir, s) for s in ("train", "validation"))
    dataset = dataset_provenance(train_split, validation_split)
    normalization = Normalization.load(features_dir / "normalization.json")
    if normalization.manifest_sha256 != dataset["manifest_sha256"]:
        raise ValueError("Normalization was fitted on a different manifest")

    x_train, y_train = torch.from_numpy(train_split.features), torch.from_numpy(train_split.labels)
    x_val = torch.from_numpy(validation_split.features)
    y_val = torch.from_numpy(validation_split.labels)
    model = LstmClassifier(config.hidden_size, config.num_layers, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    order_generator = torch.Generator().manual_seed(config.seed)

    history: list[dict[str, float | int]] = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, correct = 0.0, 0
        order = torch.randperm(len(x_train), generator=order_generator)
        for batch in torch.split(order, config.batch_size):
            optimizer.zero_grad()
            logits = model(x_train[batch])
            loss = nn.functional.cross_entropy(logits, y_train[batch])
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            correct += int((logits.argmax(dim=1) == y_train[batch]).sum())
        epoch_seconds = time.perf_counter() - epoch_started
        validation_loss, report = validation_pass(model, x_val, y_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(x_train),
                "train_accuracy": correct / len(x_train),
                "validation_loss": validation_loss,
                "validation_accuracy": report["accuracy"],
                "validation_macro_f1": report["macro_f1"],
                "train_epoch_seconds": epoch_seconds,
            }
        )
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    total_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)

    metadata: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "config": raw_config,
        "config_sha256": hashlib.sha256(
            json.dumps(raw_config, sort_keys=True).encode()
        ).hexdigest(),
        "network": {
            "type": "lstm",
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "pooling": LstmClassifier.pooling,
            "parameters": parameter_count(model),
        },
        "dataset": dataset,
        "training": {
            "train_examples": len(x_train),
            "validation_examples": len(x_val),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "stopped_early": len(history) < config.max_epochs,
            "mean_train_epoch_seconds": float(
                np.mean([row["train_epoch_seconds"] for row in history])
            ),
            "total_seconds": total_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
            "torch_threads": torch.get_num_threads(),
            "history": history,
        },
    }
    checkpoint = output_dir / "model.pt"
    sha256 = save_checkpoint(checkpoint, model, normalization, metadata)
    summary = {
        **metadata,
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": sha256,
            "bytes": checkpoint.stat().st_size,
        },
        "normalization_file_sha256": file_sha256(features_dir / "normalization.json"),
    }
    (output_dir / "training.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    with (output_dir / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = train(args.config, args.features, args.output)
    training = summary["training"]
    print(
        f"{summary['experiment_id']}: best epoch {training['best_epoch']}/"
        f"{training['epochs_run']}, validation loss {training['best_validation_loss']:.4f}, "
        f"{training['mean_train_epoch_seconds']:.2f}s/epoch, "
        f"peak RAM {training['peak_rss_bytes'] / 2**20:.0f} MiB, "
        f"checkpoint {summary['checkpoint']['sha256']}"
    )


if __name__ == "__main__":
    main()
