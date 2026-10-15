"""Isolated classification metrics. Zero-division precision/recall/F1 are reported as 0."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(
    targets: NDArray[np.int64], predictions: NDArray[np.int64], n_classes: int
) -> NDArray[np.int64]:
    """Rows are true classes, columns are predicted classes."""
    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must align")
    if targets.size and (
        min(targets.min(), predictions.min()) < 0
        or max(targets.max(), predictions.max()) >= n_classes
    ):
        raise ValueError("Class index out of range")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def classification_report(matrix: NDArray[np.int64], labels: tuple[str, ...]) -> dict[str, Any]:
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix does not match labels")
    true_positive = np.diag(matrix).astype(np.float64)
    predicted, support = matrix.sum(axis=0), matrix.sum(axis=1)
    precision = np.divide(true_positive, predicted, out=np.zeros(len(labels)), where=predicted > 0)
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
        "confusion_matrix": {"rows": "true", "columns": "predicted", "values": matrix.tolist()},
    }
