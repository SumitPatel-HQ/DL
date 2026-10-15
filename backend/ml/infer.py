"""Inference preprocessing and one-window classification from a saved checkpoint."""

from pathlib import Path

from ml.checkpoint import LoadedModel
from ml.features import FeaturePipeline, Normalization
from ml.features.pipeline import Audio


def features_for_inference(wav_path: Path, normalization_path: Path) -> Audio:
    """Use the training artifact without fitting statistics on inference audio."""
    return FeaturePipeline(Normalization.load(normalization_path)).from_wav(wav_path)


def classify_wav(loaded: LoadedModel, wav_path: Path) -> dict[str, float]:
    """Confidence per class for the first second of a WAV. Scores are not accuracy."""
    scores = loaded.probabilities(loaded.pipeline().from_wav(wav_path)[None])[0]
    return {label: float(score) for label, score in zip(loaded.labels, scores, strict=True)}
