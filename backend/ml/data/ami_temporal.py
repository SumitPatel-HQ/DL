"""Training data for the temporal prototype: AMI meeting selection and frame labels.

Training meetings may be scripted-scenario meetings (ES/IS/TS) as well as natural
ones, but never share a speaker with the validation or test crops. Selection looks
at Keyword counts only, never at model scores.
"""

from collections import Counter

import numpy as np
from numpy.typing import NDArray

from ml.data.ami import KEYWORDS as KEYWORD_SET
from ml.data.ami import Corpus, Word
from ml.features import LABELS
from ml.features.pipeline import FRAME_SAMPLES, HOP_SAMPLES, SAMPLE_RATE

KEYWORDS = tuple(label for label in LABELS if label in KEYWORD_SET)
BACKGROUND = len(KEYWORDS)
FRAME_CLASSES = (*KEYWORDS, "background")
MIN_MEETING_S = 1200
TRAIN_MEETINGS = 18
DEV_MEETINGS = 4
# Held-out source meetings for early stopping; chosen first so their speakers are excluded
# from training.
EXCLUDED_MEETINGS = ("IN1002", "EN2004a")


def frame_centers_s(n_frames: int) -> NDArray[np.float64]:
    """Center time of each MFCC-v1 frame (25 ms window, 10 ms hop)."""
    return (np.arange(n_frames) * HOP_SAMPLES + FRAME_SAMPLES / 2) / SAMPLE_RATE


def frame_labels(
    words: tuple[Word, ...], n_frames: int, offset_s: float = 0.0
) -> NDArray[np.int64]:
    """Keyword class per frame whose center lies in an aligned Keyword; else background."""
    labels = np.full(n_frames, BACKGROUND, dtype=np.int64)
    centers = frame_centers_s(n_frames)
    for word in words:
        if word.kind != "w" or word.token not in KEYWORDS or word.truncated:
            continue
        inside = (centers >= word.start_s - offset_s) & (centers <= word.end_s - offset_s)
        labels[inside] = KEYWORDS.index(word.token)
    return labels


def select_meetings(corpus: Corpus) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Pick train and early-stopping meetings with no speaker shared with val/test or each other.

    Meetings rank by Keyword count weighted toward rare Keywords (`left`, `stop`).
    Dev meetings are chosen first; training meetings may not share their speakers.
    """
    forbidden = {s for m in EXCLUDED_MEETINGS for s in corpus.meetings[m].speaker_ids}
    counts: dict[str, Counter[str]] = {}
    for meeting_id, meeting in corpus.meetings.items():
        if (
            meeting_id not in corpus.words
            or meeting.duration_s < MIN_MEETING_S
            or forbidden & set(meeting.speaker_ids)
        ):
            continue
        counts[meeting_id] = Counter(
            w.token
            for w in corpus.words[meeting_id]
            if w.kind == "w" and w.token in KEYWORD_SET and not w.truncated
        )
    pool = sum(counts.values(), Counter())
    weight = {k: 1 / np.sqrt(max(pool[k], 1)) for k in KEYWORDS}

    def rank(meeting_id: str) -> tuple[float, str]:
        return -sum(weight[k] * n for k, n in counts[meeting_id].items()), meeting_id

    ordered = sorted(counts, key=rank)
    dev: list[str] = []
    dev_speakers: set[str] = set()
    for meeting_id in ordered[::-1][: len(ordered) // 3]:  # mid/low ranks keep best for training
        speakers = set(corpus.meetings[meeting_id].speaker_ids)
        if len(dev) < DEV_MEETINGS and not speakers & dev_speakers and counts[meeting_id]:
            dev.append(meeting_id)
            dev_speakers |= speakers
    train: list[str] = []
    for meeting_id in ordered:
        speakers = set(corpus.meetings[meeting_id].speaker_ids)
        if not speakers & dev_speakers and len(train) < TRAIN_MEETINGS:
            train.append(meeting_id)
    return tuple(sorted(train)), tuple(sorted(dev))


AUGMENT_STREAMS = 48
AUGMENT_STREAM_S = 120
AUGMENT_CLIPS_PER_STREAM = 36
AUGMENT_SNR_DB = (0.0, 12.0)
ENVELOPE_FRACTION = 0.1


def voiced_span(clip: NDArray[np.float32]) -> tuple[int, int]:
    """Sample range where a 10 ms smoothed envelope stays above 10% of its peak."""
    envelope = np.convolve(np.abs(clip), np.ones(HOP_SAMPLES) / HOP_SAMPLES, mode="same")
    above = np.flatnonzero(envelope >= ENVELOPE_FRACTION * envelope.max())
    if above.size == 0:
        return 0, len(clip)
    return int(above[0]), int(above[-1]) + 1


def paste_keywords(
    stream: NDArray[np.float32],
    clips: list[tuple[str, NDArray[np.float32]]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], list[tuple[str, float, float]]]:
    """Add isolated Keyword clips over meeting audio at random SNR; return their aligned spans.

    Augmentation only: it gives rare Keywords (`left`, `stop`) more training examples
    in real meeting acoustics. Evaluation audio is never augmented.
    """
    mixed = stream.astype(np.float64).copy()
    speech_rms = float(np.sqrt(np.mean(mixed**2))) + 1e-9
    spans = []
    for keyword, clip in clips:
        start, end = voiced_span(clip)
        word = clip[start:end].astype(np.float64)
        if len(word) == 0 or len(word) >= len(mixed):
            continue
        snr = rng.uniform(*AUGMENT_SNR_DB)
        word = word / (np.sqrt(np.mean(word**2)) + 1e-9) * speech_rms * 10 ** (snr / 20)
        at = int(rng.integers(0, len(mixed) - len(word)))
        mixed[at : at + len(word)] += word
        spans.append((keyword, at / SAMPLE_RATE, (at + len(word)) / SAMPLE_RATE))
    return np.clip(mixed, -1, 32767 / 32768).astype(np.float32), spans


def spans_to_labels(
    spans: list[tuple[str, float, float]], labels: NDArray[np.int64]
) -> NDArray[np.int64]:
    out = labels.copy()
    centers = frame_centers_s(len(labels))
    for keyword, start, end in spans:
        out[(centers >= start) & (centers <= end)] = KEYWORDS.index(keyword)
    return out
