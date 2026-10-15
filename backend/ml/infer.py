"""Inference preprocessing. Checkpoint loading and classification arrive in ticket 3."""

from pathlib import Path

from ml.features import FeaturePipeline, Normalization
from ml.features.pipeline import Audio


def features_for_inference(wav_path: Path, normalization_path: Path) -> Audio:
    """Use the training artifact without fitting statistics on inference audio."""
    return FeaturePipeline(Normalization.load(normalization_path)).from_wav(wav_path)
