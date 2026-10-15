"""Self-describing checkpoints: weights plus class order, feature config and normalization."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ml.data.manifest import file_sha256
from ml.features import FEATURE_CONFIG, LABELS, FeaturePipeline, Normalization
from ml.networks import LstmClassifier

CHECKPOINT_FORMAT = "kws-lstm-checkpoint-v1"


@dataclass(frozen=True)
class LoadedModel:
    model: LstmClassifier
    labels: tuple[str, ...]
    normalization: Normalization
    metadata: dict[str, Any]
    path: Path
    sha256: str

    def pipeline(self) -> FeaturePipeline:
        return FeaturePipeline(self.normalization)

    def probabilities(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Softmax scores for [batch, frames, 40] normalized features. Not accuracy."""
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(np.ascontiguousarray(features)))
            return np.asarray(torch.softmax(logits, dim=1).numpy(), dtype=np.float32)


def save_checkpoint(
    path: Path,
    model: LstmClassifier,
    normalization: Normalization,
    metadata: dict[str, Any],
) -> str:
    """Write the checkpoint and return its SHA-256. Metadata must not contain weights."""
    if "network" not in metadata:
        raise ValueError("Checkpoint metadata requires network dimensions")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **metadata,
        "format": CHECKPOINT_FORMAT,
        "labels": list(LABELS),
        "features": FEATURE_CONFIG,
        "normalization": asdict(normalization),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return file_sha256(path)


def load_checkpoint(path: Path) -> LoadedModel:
    """Rebuild the network without retraining; reject incompatible feature or class contracts."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format")
    if payload["labels"] != list(LABELS) or payload["features"] != FEATURE_CONFIG:
        raise ValueError("Checkpoint class order or feature config is incompatible")
    network = payload["network"]
    if network.get("type") != "lstm" or network.get("pooling") != LstmClassifier.pooling:
        raise ValueError("Only mean-pooled LSTM checkpoints are supported")
    model = LstmClassifier(network["hidden_size"], network["num_layers"], network["dropout"])
    model.load_state_dict(payload.pop("state_dict"))
    model.eval()
    values = dict(payload["normalization"])
    values["mean"], values["std"] = tuple(values["mean"]), tuple(values["std"])
    return LoadedModel(
        model,
        LABELS,
        Normalization(**values),
        payload,
        path,
        file_sha256(path),
    )
