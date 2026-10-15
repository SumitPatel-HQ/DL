"""CLI: train the temporal frame-labeling BiLSTM on cached AMI meeting frames.

Only train recordings update weights and normalization. Held-out dev meetings (other
speakers) pick the best epoch. The frozen validation and test crops are never read here;
detector parameters are tuned later on validation alone.
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ml.data.ami_temporal import BACKGROUND, FRAME_CLASSES
from ml.data.manifest import file_sha256
from ml.features import FeaturePipeline
from ml.metrics import classification_report, confusion_matrix
from ml.runtime import code_revision, peak_rss_bytes, seed_everything
from ml.temporal import TemporalNet, posteriors, save_checkpoint

SPLIT_VERSION = "ami-meeting-speaker-disjoint-v1"


@dataclass(frozen=True)
class TemporalTraining:
    experiment_id: str
    seed: int
    hidden_size: int
    num_layers: int
    dropout: float
    crop_frames: int
    batch_size: int
    crops_per_epoch: int
    keyword_crop_fraction: float
    keyword_weight: float
    learning_rate: float
    max_epochs: int
    patience: int
    grad_clip_norm: float
    num_threads: int

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "TemporalTraining":
        network, training = value["network"], value["training"]
        if network["type"] != "bilstm-frame" or training["optimizer"] != "adam":
            raise ValueError("The temporal prototype trains a frame BiLSTM with Adam only")
        return cls(
            value["experiment_id"],
            value["seed"],
            network["hidden_size"],
            network["num_layers"],
            network["dropout"],
            training["crop_frames"],
            training["batch_size"],
            training["crops_per_epoch"],
            training["keyword_crop_fraction"],
            training["keyword_weight"],
            training["learning_rate"],
            training["max_epochs"],
            training["patience"],
            training["grad_clip_norm"],
            training["num_threads"],
        )


@dataclass(frozen=True)
class FrameRecording:
    recording_id: str
    features: NDArray[np.float32]
    labels: NDArray[np.int64]


def load_recordings(root: Path, manifest: dict[str, Any], role: str) -> list[FrameRecording]:
    rows = [r for r in manifest["recordings"] if r["role"] == role]
    out = []
    for row in rows:
        path = root / "features" / row["file"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Cached frames changed since preparation: {path.name}")
        with np.load(path) as data:
            out.append(FrameRecording(row["recording_id"], data["features"], data["labels"]))
    return out


def sample_crops(
    recordings: list[FrameRecording],
    normalized: list[NDArray[np.float32]],
    config: TemporalTraining,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crops; a fraction are centered on a Keyword frame of an evenly chosen class."""
    lengths = np.array([len(r.labels) for r in recordings], dtype=np.float64)
    by_class = {
        k: [(i, np.flatnonzero(r.labels == k)) for i, r in enumerate(recordings)]
        for k in range(BACKGROUND)
    }
    by_class = {k: [(i, f) for i, f in v if len(f)] for k, v in by_class.items()}
    xs, ys = [], []
    for _ in range(config.batch_size):
        if rng.random() < config.keyword_crop_fraction:
            k = int(rng.integers(BACKGROUND))
            weights = np.array([len(f) for _, f in by_class[k]], dtype=np.float64)
            i, frames = by_class[k][int(rng.choice(len(weights), p=weights / weights.sum()))]
            center = int(rng.choice(frames))
            start = center - int(rng.integers(config.crop_frames))
        else:
            i = int(rng.choice(len(recordings), p=lengths / lengths.sum()))
            start = int(rng.integers(len(recordings[i].labels)))
        start = int(np.clip(start, 0, len(recordings[i].labels) - config.crop_frames))
        xs.append(normalized[i][start : start + config.crop_frames])
        ys.append(recordings[i].labels[start : start + config.crop_frames])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def dev_pass(
    model: TemporalNet,
    dev: list[FrameRecording],
    normalized: list[NDArray[np.float32]],
    weight: torch.Tensor,
) -> tuple[float, dict[str, Any]]:
    """Weighted frame loss and frame-level report over every dev meeting."""
    model.eval()
    probabilities = np.concatenate([posteriors(model, x) for x in normalized])
    labels = np.concatenate([r.labels for r in dev])
    loss = nn.functional.nll_loss(
        torch.log(torch.from_numpy(probabilities).clamp_min(1e-9)),
        torch.from_numpy(labels),
        weight=weight,
    )
    matrix = confusion_matrix(
        labels, probabilities.argmax(axis=1).astype(np.int64), len(FRAME_CLASSES)
    )
    return float(loss), classification_report(matrix, FRAME_CLASSES)


def train(config_path: Path, root: Path, output_dir: Path) -> dict[str, Any]:
    config = TemporalTraining.from_json(json.loads(config_path.read_text(encoding="utf-8")))
    seed_everything(config.seed, config.num_threads)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_set = load_recordings(root, manifest, "train")
    dev_set = load_recordings(root, manifest, "dev")
    manifest_sha = file_sha256(root / "manifest.json")
    normalization = FeaturePipeline.fit_normalization(
        (r.features for r in train_set),
        split="train",
        dataset_version=manifest["dataset_version"],
        split_version=SPLIT_VERSION,
        manifest_sha256=manifest_sha,
    )
    mean, std = np.array(normalization.mean), np.array(normalization.std)

    def normalize(r: FrameRecording) -> NDArray[np.float32]:
        return np.asarray((r.features - mean) / std, dtype=np.float32)

    train_x = [normalize(r) for r in train_set]
    dev_x = [normalize(r) for r in dev_set]
    weight = torch.ones(len(FRAME_CLASSES))
    weight[:BACKGROUND] = config.keyword_weight

    model = TemporalNet(config.hidden_size, config.num_layers, config.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int]] = []
    best_loss, best_score, best_epoch, stale = float("inf"), -1.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        steps = config.crops_per_epoch // config.batch_size
        for _ in range(steps):
            x, y = sample_crops(train_set, train_x, config, rng)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                model(x).reshape(-1, len(FRAME_CLASSES)), y.reshape(-1), weight=weight
            )
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            loss_sum += loss.item()
        dev_loss, report = dev_pass(model, dev_set, dev_x, weight)
        keyword_f1 = float(
            np.mean([report["per_class"][c]["f1"] for c in FRAME_CLASSES[:BACKGROUND]])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / steps,
                "dev_loss": dev_loss,
                "dev_frame_accuracy": report["accuracy"],
                "dev_keyword_frame_f1": keyword_f1,
                "train_epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(history[-1], flush=True)
        # Dev loss mostly rewards predicting background, so the best epoch maximizes the
        # mean Keyword frame F1 on the held-out dev meetings instead.
        if keyword_f1 > best_score:
            best_loss, best_score, best_epoch, stale = dev_loss, keyword_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    total_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    _, dev_report = dev_pass(model, dev_set, dev_x, weight)
    metadata: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "seed": config.seed,
        "network": {
            "type": "bilstm-frame",
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        },
        "dataset": {
            "dataset_version": manifest["dataset_version"],
            "split_version": SPLIT_VERSION,
            "manifest_sha256": manifest_sha,
            "train_recordings": [r.recording_id for r in train_set],
            "dev_recordings": [r.recording_id for r in dev_set],
        },
        "training": {
            "config": json.loads(config_path.read_text(encoding="utf-8"))["training"],
            "best_epoch": best_epoch,
            "best_dev_loss": best_loss,
            "best_dev_keyword_frame_f1": best_score,
            "selection": "max dev Keyword frame F1",
            "total_seconds": total_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    }
    checkpoint = output_dir / "model.pt"
    digest = save_checkpoint(checkpoint, model, normalization, metadata)
    summary = {
        **metadata,
        "checkpoint_sha256": digest,
        "history": history,
        "dev_frame_report": dev_report,
        "note": "Frame-level dev report only; Detection quality is measured by the frozen "
        "continuous protocol, never by these frame scores.",
    }
    (output_dir / "training.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = train(args.config, args.root, args.output)
    print(f"checkpoint sha256 {summary['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
