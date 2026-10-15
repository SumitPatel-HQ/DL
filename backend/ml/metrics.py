"""Isolated classification and continuous-event metrics.

Zero-division precision/recall/F1 are reported as 0.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(
    targets: NDArray[np.int64], detected: NDArray[np.int64], n_classes: int
) -> NDArray[np.int64]:
    """Rows are true classes, columns are detected classes."""
    if targets.shape != detected.shape:
        raise ValueError("Targets and detections must align")
    if targets.size and (
        min(targets.min(), detected.min()) < 0 or max(targets.max(), detected.max()) >= n_classes
    ):
        raise ValueError("Class index out of range")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (targets, detected), 1)
    return matrix


def classification_report(matrix: NDArray[np.int64], labels: tuple[str, ...]) -> dict[str, Any]:
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix does not match labels")
    true_positive = np.diag(matrix).astype(np.float64)
    detected_total, support = matrix.sum(axis=0), matrix.sum(axis=1)
    precision = np.divide(
        true_positive, detected_total, out=np.zeros(len(labels)), where=detected_total > 0
    )
    recall = np.divide(true_positive, support, out=np.zeros(len(labels)), where=support > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros(len(labels)), where=total > 0)
    return {
        "examples": int(matrix.sum()),
        "accuracy": float(true_positive.sum() / max(int(matrix.sum()), 1)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"rows": "true", "columns": "detected", "values": matrix.tolist()},
    }


def span_center(start_s: float, end_s: float) -> float:
    """Midpoint of a time span; the point that event matching compares."""
    return (start_s + end_s) / 2


def match_events(
    references: Sequence[tuple[str, float]],
    detections: Sequence[tuple[str, float]],
    tolerance_s: float,
) -> list[tuple[int, int]]:
    """One-to-one (reference, detection) index pairs: same Keyword, centers within tolerance.

    Scanning both sides in time order and pairing each reference with the earliest
    unused detection inside its tolerance window gives a maximum matching, because
    every window has the same width.
    """
    if tolerance_s < 0:
        raise ValueError("Tolerance must not be negative")
    pairs = []
    for keyword in {label for label, _ in references}:
        refs = sorted((t, i) for i, (label, t) in enumerate(references) if label == keyword)
        dets = sorted((t, j) for j, (label, t) in enumerate(detections) if label == keyword)
        k = 0
        for center, i in refs:
            while k < len(dets) and dets[k][0] < center - tolerance_s:
                k += 1
            if k < len(dets) and dets[k][0] <= center + tolerance_s:
                pairs.append((i, dets[k][1]))
                k += 1
    return sorted(pairs)


def event_scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    """Event precision/recall/F1; each is 0 when its denominator is 0."""
    if min(true_positive, false_positive, false_negative) < 0:
        raise ValueError("Event counts must not be negative")
    detected_total, actual = true_positive + false_positive, true_positive + false_negative
    precision = true_positive / detected_total if detected_total else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def false_alarms_per_hour(false_alarms: int, negative_seconds: float) -> float:
    """False Detections divided by evaluated negative audio hours.

    With no negative audio the rate is undefined and reported as 0. A false alarm
    needs negative audio to fall in, so a positive count without it is invalid.
    """
    if false_alarms < 0 or negative_seconds < 0:
        raise ValueError("False alarm count and negative duration must not be negative")
    if negative_seconds == 0:
        if false_alarms:
            raise ValueError("False alarms need negative audio exposure")
        return 0.0
    return false_alarms * 3600 / negative_seconds
