"""Experimental temporal Detector: a bidirectional LSTM labels every MFCC frame.

Per ADR-0006 this is a separately scoped prototype beside the sliding-window
detector. It returns the same `Detection` shape (Keyword, start/end, Confidence,
threshold metadata) and is judged only through the frozen continuous protocol.
Confidence is a smoothed frame posterior, not a measure of correctness.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ml.data.ami_temporal import BACKGROUND, FRAME_CLASSES, KEYWORDS
from ml.data.manifest import file_sha256
from ml.features import FEATURE_CONFIG, FeaturePipeline, Normalization
from ml.features.pipeline import FRAME_SAMPLES, HOP_SAMPLES, N_MFCC, SAMPLE_RATE, Audio
from ml.spotting import Detection, suppress_duplicates

CHECKPOINT_FORMAT = "kws-temporal-checkpoint-v1"
FRAME_S = HOP_SAMPLES / SAMPLE_RATE
CHUNK_FRAMES = 1000
CONTEXT_FRAMES = 200


class TemporalNet(nn.Module):
    """Input [batch, frames, 40]; output unnormalized logits [batch, frames, 7]."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers not in (1, 2) or not 0 <= dropout < 1:
            raise ValueError("Expected positive hidden size, 1-2 layers and dropout in [0, 1)")
        self.lstm = nn.LSTM(
            N_MFCC,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, len(FRAME_CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(features)
        logits: torch.Tensor = self.head(self.dropout(outputs))
        return logits


@dataclass(frozen=True)
class TemporalConfig:
    """Post-processing parameters. Tuned on validation, then frozen before test."""

    smooth_frames: int = 5
    threshold: float = 0.5
    min_frames: int = 4
    suppression_s: float = 1.0

    def __post_init__(self) -> None:
        if self.smooth_frames < 1 or self.min_frames < 1 or self.suppression_s < 0:
            raise ValueError("Expected smooth_frames >= 1, min_frames >= 1, suppression >= 0")
        if not 0 < self.threshold < 1:
            raise ValueError("Expected threshold in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedTemporal:
    model: TemporalNet
    normalization: Normalization
    metadata: dict[str, Any]
    path: Path
    sha256: str

    def pipeline(self) -> FeaturePipeline:
        return FeaturePipeline(self.normalization)


def posteriors(model: TemporalNet, features: NDArray[np.float32]) -> NDArray[np.float32]:
    """Frame softmax scores. Long input runs in chunks with context so state never drifts."""
    n = len(features)
    out = np.empty((n, len(FRAME_CLASSES)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, n)
            lo, hi = max(0, start - CONTEXT_FRAMES), min(n, end + CONTEXT_FRAMES)
            logits = model(torch.from_numpy(np.ascontiguousarray(features[lo:hi]))[None])[0]
            out[start:end] = torch.softmax(logits, dim=1).numpy()[start - lo : end - lo]
    return out


def smooth(scores: NDArray[np.float32], frames: int) -> NDArray[np.float32]:
    """Centered moving average along time, per class."""
    if frames == 1:
        return scores
    kernel = np.ones(frames) / frames
    padded = np.pad(scores, ((frames // 2, frames - 1 - frames // 2), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, c], kernel, mode="valid") for c in range(scores.shape[1])], axis=1
    ).astype(np.float32)


def decode(probabilities: NDArray[np.float32], config: TemporalConfig) -> list[Detection]:
    """Threshold smoothed frame scores into Detections.

    A frame votes for a Keyword only if that Keyword is its top class and its smoothed
    score reaches the threshold. Runs shorter than `min_frames` are dropped. Confidence is
    the run's maximum smoothed score. Same-Keyword Detections whose centers lie closer
    than `suppression_s` keep only the most confident one.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != len(FRAME_CLASSES):
        raise ValueError("Frame scores do not match the class list")
    scores = smooth(probabilities, config.smooth_frames)
    top = scores.argmax(axis=1)
    top_score = scores.max(axis=1)
    votes = np.where((top != BACKGROUND) & (top_score >= config.threshold), top, -1)
    candidates = []
    first = 0
    for i in range(1, len(votes) + 1):
        if i < len(votes) and votes[i] == votes[first]:
            continue
        if votes[first] >= 0 and i - first >= config.min_frames:
            candidates.append(
                Detection(
                    KEYWORDS[votes[first]],
                    first * FRAME_S,
                    (i - 1) * FRAME_S + FRAME_SAMPLES / SAMPLE_RATE,
                    float(top_score[first:i].max()),
                    config.threshold,
                    i - first,
                )
            )
        first = i
    return suppress_duplicates(candidates, config.suppression_s)


def spot_audio(loaded: LoadedTemporal, audio: Audio, config: TemporalConfig) -> list[Detection]:
    """End to end on canonical 16 kHz mono audio, mirroring `ml.spotting.spot_audio`."""
    features = loaded.pipeline().from_stream(audio, SAMPLE_RATE)
    return decode(posteriors(loaded.model, features), config)


def save_checkpoint(
    path: Path, model: TemporalNet, normalization: Normalization, metadata: dict[str, Any]
) -> str:
    if "network" not in metadata:
        raise ValueError("Checkpoint metadata requires network dimensions")
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = asdict(normalization)
    stats["mean"], stats["std"] = (
        [float(v) for v in stats["mean"]],
        [float(v) for v in stats["std"]],
    )
    torch.save(
        {
            **metadata,
            "format": CHECKPOINT_FORMAT,
            "frame_classes": list(FRAME_CLASSES),
            "features": FEATURE_CONFIG,
            "normalization": stats,
            "state_dict": model.state_dict(),
        },
        path,
    )
    return file_sha256(path)


def load_checkpoint(path: Path) -> LoadedTemporal:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported temporal checkpoint format")
    if payload["frame_classes"] != list(FRAME_CLASSES) or payload["features"] != FEATURE_CONFIG:
        raise ValueError("Checkpoint class order or feature config is incompatible")
    network = payload["network"]
    model = TemporalNet(network["hidden_size"], network["num_layers"], network["dropout"])
    model.load_state_dict(payload.pop("state_dict"))
    model.eval()
    values = dict(payload["normalization"])
    values["mean"], values["std"] = tuple(values["mean"]), tuple(values["std"])
    return LoadedTemporal(model, Normalization(**values), payload, path, file_sha256(path))
