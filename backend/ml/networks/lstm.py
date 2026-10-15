"""Unidirectional LSTM over MFCC frames, classified from mean-pooled top-layer outputs."""

import torch
from torch import nn

from ml.features.pipeline import LABELS, N_MFCC


class LstmClassifier(nn.Module):
    """Input [batch, frames, 40] batch-first; output unnormalized class logits.

    Mean pooling over all frames replaced the final state after a validation-only
    probe: Speech Commands words often end well before the clip does, so the
    final state mostly summarizes trailing non-speech. `dropout` is applied to the pooled vector
    outside the LSTM, so it also regularizes single-layer models. Stacked models
    reuse it between layers.
    """

    pooling = "mean-over-frames"

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        n_features: int = N_MFCC,
        n_classes: int = len(LABELS),
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers not in (1, 2) or not 0 <= dropout < 1:
            raise ValueError("Expected positive hidden size, 1-2 layers and dropout in [0, 1)")
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Every window is exactly 98 frames, so pooling needs no mask.
        outputs, _ = self.lstm(features)
        logits: torch.Tensor = self.head(self.dropout(outputs.mean(dim=1)))
        return logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
