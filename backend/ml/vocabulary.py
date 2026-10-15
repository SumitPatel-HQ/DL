"""Versioned class orders. Expanded vocabulary is for offline experiments until reviewed."""

LABELS = ("left", "right", "up", "down", "go", "stop", "unknown", "silence")

NEW_KEYWORDS = ("next", "pause", "urgent")
EXPANDED_LABELS = (*LABELS[:-2], *NEW_KEYWORDS, "unknown", "silence")
EXPANDED_DATASET_VERSION = "speech-commands-v0.02-p2-vocab-v1"
VOCABULARY_REVIEW_ACTION = (
    "Expanded vocabulary is offline-only pending licensed recordings, retraining, "
    "fresh isolated and continuous Evaluation Runs, and a written review. "
    "Restore the P0 checkpoint and its detector to serve the existing Keywords."
)


def validate_labels(labels: tuple[str, ...]) -> None:
    if labels not in (LABELS, EXPANDED_LABELS):
        raise ValueError("Unsupported class order")


def keywords_for(labels: tuple[str, ...]) -> tuple[str, ...]:
    validate_labels(labels)
    return labels[:-2]
