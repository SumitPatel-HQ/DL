"""Continuous Keyword Detection by sliding the isolated classifier over a Recording.

Experimental per ADR-0002: Detections are approximate spans from 1 s windows, and
Confidence is the classifier's softmax score, not a measure of correctness.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml.checkpoint import LoadedModel
from ml.features import LABELS
from ml.features.pipeline import SAMPLE_RATE, WINDOW_SAMPLES, Audio
from ml.metrics import span_center
from ml.vocabulary import keywords_for

KEYWORDS = tuple(label for label in LABELS if label not in ("unknown", "silence"))
WINDOW_S = WINDOW_SAMPLES / SAMPLE_RATE


@dataclass(frozen=True)
class DetectorConfig:
    """Window/stride/threshold parameters. Tuned on validation, then frozen before test."""

    stride_s: float = 0.25
    threshold: float = 0.5
    min_consecutive: int = 2
    suppression_s: float = 1.0
    window_s: float = WINDOW_S

    def __post_init__(self) -> None:
        if self.window_s != WINDOW_S:
            raise ValueError("The isolated classifier only scores 1 s windows")
        if not 0 < self.stride_s <= self.window_s:
            raise ValueError("Stride must be positive and at most one window")
        if round(self.stride_s * SAMPLE_RATE) != self.stride_s * SAMPLE_RATE:
            raise ValueError("Stride must be a whole number of samples")
        if not 0 < self.threshold < 1 or self.min_consecutive < 1 or self.suppression_s < 0:
            raise ValueError("Expected threshold in (0, 1), min_consecutive >= 1, suppression >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    keyword: str
    start_s: float
    end_s: float
    confidence: float
    threshold: float
    windows: int

    @property
    def center_s(self) -> float:
        return span_center(self.start_s, self.end_s)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "center_s": self.center_s}


@dataclass(frozen=True)
class WindowScores:
    """Per-window softmax scores; debugging evidence, not ground-truth probabilities."""

    starts_s: NDArray[np.float64]
    probabilities: NDArray[np.float32]
    stride_s: float


def window_starts(n_samples: int, stride_s: float) -> NDArray[np.int64]:
    """Window start samples; audio shorter than one window yields one padded window."""
    stride = round(stride_s * SAMPLE_RATE)
    return np.arange(0, max(n_samples - WINDOW_SAMPLES, 0) + 1, stride, dtype=np.int64)


def score_windows(
    loaded: LoadedModel, audio: Audio, stride_s: float, batch_size: int = 256
) -> WindowScores:
    """Run the shared feature pipeline and model on every window of canonical audio."""
    if audio.ndim != 1 or audio.dtype != np.float32 or audio.size == 0:
        raise ValueError("Expected non-empty canonical mono float32 audio")
    pipeline = loaded.pipeline()
    starts = window_starts(len(audio), stride_s)
    chunks = []
    for first in range(0, len(starts), batch_size):
        features = np.stack(
            [
                pipeline.from_audio(audio[start : start + WINDOW_SAMPLES], SAMPLE_RATE)
                for start in starts[first : first + batch_size]
            ]
        )
        chunks.append(loaded.probabilities(features))
    return WindowScores(starts / SAMPLE_RATE, np.concatenate(chunks), stride_s)


def detect(
    scores: WindowScores,
    config: DetectorConfig,
    keywords: tuple[str, ...] = KEYWORDS,
    labels: tuple[str, ...] = LABELS,
) -> list[Detection]:
    """Threshold, adjacent-window consistency, merging and duplicate suppression.

    A window votes for a Keyword only when that Keyword is its top class and its
    score reaches the threshold, so Unknown and Silence can reject it. Runs of
    consecutive votes for one Keyword merge into one candidate spanning their
    windows; runs shorter than `min_consecutive` are dropped. Candidates for the
    same Keyword whose centers lie closer than `suppression_s` keep only the most
    confident one. Different Keywords never suppress each other.
    """
    if config.stride_s != scores.stride_s:
        raise ValueError("Window scores were computed with a different stride")
    if not set(keywords) <= set(keywords_for(labels)):
        raise ValueError("Detections are only available for trained Keywords")
    if scores.probabilities.shape != (len(scores.starts_s), len(labels)):
        raise ValueError("Window scores do not match the label list")
    top = scores.probabilities.argmax(axis=1)
    top_score = scores.probabilities.max(axis=1)
    wanted = [labels.index(keyword) for keyword in keywords]
    votes = np.where(np.isin(top, wanted) & (top_score >= config.threshold), top, -1)
    candidates = []
    first = 0
    for i in range(1, len(votes) + 1):
        if i < len(votes) and votes[i] == votes[first]:
            continue
        if votes[first] >= 0 and i - first >= config.min_consecutive:
            candidates.append(
                Detection(
                    labels[votes[first]],
                    float(scores.starts_s[first]),
                    float(scores.starts_s[i - 1] + config.window_s),
                    float(top_score[first:i].max()),
                    config.threshold,
                    i - first,
                )
            )
        first = i
    return suppress_duplicates(candidates, config.suppression_s)


def suppress_duplicates(candidates: list[Detection], suppression_s: float) -> list[Detection]:
    """Per-Keyword suppression: of Detections with centers closer than the limit, keep the best."""
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=lambda d: (-d.confidence, d.start_s)):
        if all(
            d.keyword != candidate.keyword or abs(candidate.center_s - d.center_s) >= suppression_s
            for d in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda d: d.start_s)


def spot_audio(
    loaded: LoadedModel,
    audio: Audio,
    config: DetectorConfig,
    keywords: tuple[str, ...] | None = None,
) -> list[Detection]:
    """End to end: score every window of canonical 16 kHz mono audio, then `detect`."""
    return detect(
        score_windows(loaded, audio, config.stride_s),
        config,
        keywords if keywords is not None else keywords_for(loaded.labels),
        loaded.labels,
    )
