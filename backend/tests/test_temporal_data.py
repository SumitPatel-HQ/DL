import numpy as np

from ml.data.ami import Word
from ml.data.ami_temporal import BACKGROUND, KEYWORDS, frame_centers_s, frame_labels


def word(token: str, start: float, end: float, **kw: object) -> Word:
    return Word(token, start, end, "S1", **kw)  # type: ignore[arg-type]


def test_frame_centers_follow_mfcc_framing() -> None:
    assert frame_centers_s(3).tolist() == [0.0125, 0.0225, 0.0325]


def test_keyword_frames_cover_aligned_word_and_rest_is_background() -> None:
    labels = frame_labels((word("go", 0.50, 0.70),), 100)
    go = KEYWORDS.index("go")
    covered = np.flatnonzero(labels == go)
    centers = frame_centers_s(100)[covered]
    assert centers.min() >= 0.50 and centers.max() <= 0.70
    assert len(covered) == 20
    assert (labels[labels != go] == BACKGROUND).all()


def test_non_keywords_truncated_and_nonwords_are_background() -> None:
    words = (
        word("hello", 0.1, 0.4),
        word("go", 0.5, 0.7, truncated=True),
        word("stop", 0.8, 0.9, kind="uncertain"),
    )
    assert (frame_labels(words, 100) == BACKGROUND).all()


def test_offset_shifts_words_into_crop_coordinates() -> None:
    labels = frame_labels((word("up", 10.5, 10.7),), 100, offset_s=10.0)
    assert (labels == KEYWORDS.index("up")).sum() == 20


def test_paste_keywords_adds_clip_energy_and_reports_span() -> None:
    from ml.data.ami_temporal import paste_keywords, spans_to_labels

    rng = np.random.default_rng(1)
    stream = (rng.normal(0, 0.05, 16000 * 10)).astype(np.float32)
    clip = np.zeros(16000, dtype=np.float32)
    clip[4000:8000] = np.sin(np.arange(4000) / 5).astype(np.float32)
    mixed, spans = paste_keywords(stream, [("stop", clip)], rng)
    assert len(spans) == 1 and spans[0][0] == "stop"
    keyword, start, end = spans[0]
    assert abs((end - start) - 0.25) < 0.02
    inside = mixed[int(start * 16000) : int(end * 16000)]
    assert np.sqrt(np.mean(inside**2)) > np.sqrt(np.mean(stream**2))
    labels = spans_to_labels(spans, np.full(998, BACKGROUND, dtype=np.int64))
    assert (labels == KEYWORDS.index("stop")).sum() > 20
