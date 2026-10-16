from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.checkpoint import LoadedModel
from ml.evaluate_continuous import (
    TOLERANCE_S,
    ContinuousSplit,
    Reference,
    SplitRecording,
    check_frozen,
    evaluate_scores,
    score_recording,
    summarize,
    tune,
)
from ml.features import LABELS, Normalization
from ml.metrics import event_scores, false_alarms_per_hour, match_events
from ml.networks import LstmClassifier
from ml.spotting import (
    Detection,
    DetectorConfig,
    WindowScores,
    detect,
    score_windows,
    spot_audio,
    window_levels_db,
    window_starts,
)

STRIDE = 0.25


def test_expanded_evaluation_counts_new_keyword_hits_and_unknown_rejection() -> None:
    labels = (*LABELS[:-2], "next", "pause", "urgent", "unknown", "silence")
    votes = ("next", "pause", "urgent", "unknown", "silence")
    probabilities = np.zeros((5, 11), dtype=np.float32)
    for i, label in enumerate(votes):
        probabilities[i, labels.index(label)] = 1.0
    scores_by_id = {"expanded": WindowScores(np.arange(5, dtype=np.float64), probabilities, 1.0)}
    references = tuple(Reference(label, i + 0.1, i + 0.9) for i, label in enumerate(votes))
    data = ContinuousSplit(
        "fixture",
        "fixture",
        "test",
        "fixture-only",
        (
            SplitRecording(
                "expanded",
                "a" * 64,
                5.0,
                ("speaker",),
                references,
                np.zeros(80000, dtype=np.float32),
            ),
        ),
    )
    metrics, detections, _ = evaluate_scores(
        data, scores_by_id, DetectorConfig(stride_s=1.0, min_consecutive=1), 0.5, labels
    )
    assert [d.keyword for d in detections["expanded"]] == ["next", "pause", "urgent"]
    assert metrics["true_positive"] == 3
    assert metrics["false_positive"] == metrics["false_negative"] == 0
    assert metrics["false_alarms"]["negative_seconds"] == pytest.approx(1.6)


def scores(*rows: tuple[str, float]) -> WindowScores:
    """One window per row: the named class gets the score, the rest share the remainder."""
    probabilities = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    for i, (label, score) in enumerate(rows):
        probabilities[i] = (1 - score) / (len(LABELS) - 1)
        probabilities[i, LABELS.index(label)] = score
    return WindowScores(np.arange(len(rows)) * STRIDE, probabilities, STRIDE)


@pytest.fixture
def loaded() -> LoadedModel:
    torch.manual_seed(0)
    normalization = Normalization((0.0,) * 40, (1.0,) * 40, 1, "fixture", "fixture", "x" * 64)
    model = LstmClassifier(hidden_size=8, num_layers=1).eval()
    metadata = {"experiment_id": "T01"}
    return LoadedModel(model, LABELS, normalization, metadata, Path("t/model.pt"), "a" * 64)


def test_consecutive_votes_merge_into_one_detection_with_threshold_metadata() -> None:
    windows = scores(("silence", 0.9), ("go", 0.7), ("go", 0.9), ("go", 0.6), ("unknown", 0.8))
    config = DetectorConfig(STRIDE, threshold=0.5, min_consecutive=2)
    (detection,) = detect(windows, config)
    assert detection == Detection("go", 0.25, 1.75, pytest.approx(0.9), 0.5, 3)
    assert detection.center_s == pytest.approx(1.0)


def test_isolated_or_low_score_or_rejected_windows_do_not_detect() -> None:
    config = DetectorConfig(STRIDE, threshold=0.8, min_consecutive=2)
    # One confident window, then a run below threshold, then Unknown winning the window.
    windows = scores(("left", 0.95), ("silence", 0.9), ("up", 0.7), ("up", 0.7), ("unknown", 0.9))
    assert detect(windows, config) == []
    assert [d.keyword for d in detect(windows, DetectorConfig(STRIDE, 0.8, 1))] == ["left"]


def test_duplicate_suppression_keeps_most_confident_nearby_detection_per_keyword() -> None:
    windows = scores(("left", 0.7), ("right", 0.95), ("silence", 0.9), ("left", 0.8))
    config = DetectorConfig(STRIDE, threshold=0.5, min_consecutive=1, suppression_s=1.0)
    kept = detect(windows, config)
    # The nearby left pair collapses to its more confident member; right is untouched.
    assert [(d.keyword, d.confidence) for d in kept] == [
        ("right", pytest.approx(0.95)),
        ("left", pytest.approx(0.8)),
    ]
    far = DetectorConfig(STRIDE, threshold=0.5, min_consecutive=1, suppression_s=0.2)
    assert [d.keyword for d in detect(windows, far)] == ["left", "right", "left"]


def test_confident_keyword_does_not_suppress_a_different_nearby_keyword() -> None:
    windows = scores(("go", 0.99), ("up", 0.6))
    config = DetectorConfig(STRIDE, threshold=0.5, min_consecutive=1, suppression_s=1.0)
    assert [d.keyword for d in detect(windows, config)] == ["go", "up"]


def test_selected_keywords_only_and_untrained_words_refused() -> None:
    windows = scores(("left", 0.9), ("silence", 0.9), ("go", 0.9))
    config = DetectorConfig(STRIDE, threshold=0.5, min_consecutive=1, suppression_s=0)
    assert [d.keyword for d in detect(windows, config, ("go",))] == ["go"]
    with pytest.raises(ValueError):
        detect(windows, config, ("unknown",))
    with pytest.raises(ValueError):
        detect(windows, DetectorConfig(0.2, 0.5, 1))


def test_margin_rejects_windows_without_a_distinct_peak() -> None:
    windows = scores(("left", 0.55), ("silence", 0.9), ("left", 0.55))
    # The first window's runner-up (Unknown) is close; the last window has a clear peak.
    windows.probabilities[0, LABELS.index("unknown")] = 0.45
    config = DetectorConfig(STRIDE, 0.5, 1, suppression_s=0)
    assert len(detect(windows, config)) == 2
    strict = DetectorConfig(STRIDE, 0.5, 1, suppression_s=0, margin=0.2)
    assert [d.start_s for d in detect(windows, strict)] == [0.5]


def test_level_gate_drops_quiet_windows_and_needs_levels() -> None:
    windows = scores(("go", 0.9), ("silence", 0.9), ("go", 0.9))
    gated = WindowScores(windows.starts_s, windows.probabilities, STRIDE, np.array([3.0, 0, 20.0]))
    config = DetectorConfig(STRIDE, 0.5, 1, suppression_s=0, min_level_db=12)
    assert [d.start_s for d in detect(gated, config)] == [0.5]
    with pytest.raises(ValueError):
        detect(windows, config)


def test_window_levels_are_relative_to_the_recording_floor() -> None:
    audio = np.full(48000, 1e-4, dtype=np.float32)
    audio[16000:32000] = 0.1
    levels = window_levels_db(audio, np.array([0, 16000, 32000]))
    assert levels[0] == pytest.approx(0, abs=1e-6)
    assert levels[1] == pytest.approx(60, abs=0.5)


def test_detector_config_rejects_invalid_parameters() -> None:
    for kwargs in (
        {"window_s": 0.5},
        {"stride_s": 0},
        {"stride_s": 1.5},
        {"threshold": 1.0},
        {"min_consecutive": 0},
        {"margin": 1.0},
        {"min_level_db": -1.0},
    ):
        with pytest.raises(ValueError):
            DetectorConfig(**kwargs)


def test_window_scoring_uses_the_shared_pipeline_per_window(loaded: LoadedModel) -> None:
    assert window_starts(8000, STRIDE).tolist() == [0]
    audio = np.random.default_rng(1).normal(0, 0.1, 51200).astype(np.float32)
    result = score_windows(loaded, audio, STRIDE, batch_size=4)
    assert result.starts_s.tolist() == [i * STRIDE for i in range(9)]
    third = loaded.pipeline().from_audio(audio[8000:24000], 16000)
    np.testing.assert_allclose(
        result.probabilities[2], loaded.probabilities(third[None])[0], rtol=1e-5, atol=1e-6
    )
    assert all(isinstance(d, Detection) for d in spot_audio(loaded, audio, DetectorConfig(STRIDE)))


def test_event_matching_is_one_to_one_same_keyword_within_tolerance() -> None:
    references = [("go", 0.0), ("go", 0.8), ("up", 3.0)]
    detections = [("go", 0.45), ("go", 1.3), ("go", 1.35), ("left", 3.0), ("up", 3.5)]
    # Nearest-pair-first would give 0.45 to the 0.8 reference and miss the first one.
    assert match_events(references, detections, 0.6) == [(0, 0), (1, 1), (2, 4)]
    assert match_events(references, detections, 0.4) == [(1, 0)]


def test_event_metrics_and_false_alarm_rate_math() -> None:
    assert event_scores(2, 2, 6) == pytest.approx(
        {
            "true_positive": 2,
            "false_positive": 2,
            "false_negative": 6,
            "precision": 0.5,
            "recall": 0.25,
            "f1": 1 / 3,
        }
    )
    assert event_scores(0, 0, 0)["f1"] == 0
    assert false_alarms_per_hour(3, 1800) == pytest.approx(6.0)
    assert false_alarms_per_hour(0, 0) == 0.0
    for count, seconds in ((1, 0), (-1, 10), (0, -1)):
        with pytest.raises(ValueError):
            false_alarms_per_hour(count, seconds)


def test_summary_without_negative_audio_reports_zero_false_alarm_rate() -> None:
    references = (Reference("go", 1.0, 1.3),)
    detections = [Detection("go", 0.7, 1.7, 0.9, 0.5, 2)]
    summary = summarize([score_recording(references, detections, TOLERANCE_S)])
    assert summary["f1"] == 1.0
    assert summary["false_alarms"]["count"] == 0
    assert summary["false_alarms"]["per_hour"] == 0.0
    assert summary["false_alarms"]["by_region"] == {}


def test_recording_scoring_counts_false_alarms_only_inside_negative_spans() -> None:
    references = (
        Reference("unknown", 0.0, 4.0),
        Reference("go", 4.0, 4.3),
        Reference("up", 6.0, 6.2, ignored=True),
        Reference("silence", 7.0, 9.0),
        Reference("stop", 10.0, 10.3),
    )
    detections = [
        Detection("go", 3.7, 4.7, 0.9, 0.5, 2),  # matches go
        Detection("left", 1.0, 2.0, 0.9, 0.5, 1),  # false alarm in Unknown
        Detection("up", 5.6, 6.6, 0.9, 0.5, 1),  # matches an ignored reference
        Detection("down", 7.5, 8.5, 0.9, 0.5, 1),  # false alarm in Silence
        Detection("right", 5.0, 5.4, 0.9, 0.5, 1),  # false, unlabeled gap
    ]
    result = score_recording(references, detections, TOLERANCE_S)
    summary = summarize([result])
    assert (summary["true_positive"], summary["false_positive"], summary["false_negative"]) == (
        1,
        3,
        1,
    )
    assert summary["ignored_matches"] == 1
    assert summary["false_positive_regions"] == {"silence": 1, "unknown": 1, "unlabeled": 1}
    assert summary["false_alarms"]["count"] == 2
    assert summary["false_alarms"]["per_hour"] == pytest.approx(2 * 3600 / 6)
    assert summary["per_keyword"]["stop"]["false_negative"] == 1


def split(name: str, sha: str = "b" * 64) -> ContinuousSplit:
    references = (Reference("unknown", 0.0, 1.0), Reference("go", 1.2, 1.4))
    audio = np.zeros(32000, dtype=np.float32)
    recording = SplitRecording("r1", sha, 2.0, ("s1",), references, audio)
    return ContinuousSplit("v1", "s1", name, "fixture", (recording,))


def detector_file(loaded: LoadedModel, **overrides: Any) -> dict[str, Any]:
    value = {
        "format": "kws-detector-v1",
        "checkpoint": {"sha256": loaded.sha256},
        "tolerance_s": TOLERANCE_S,
        "tuned_on": {
            "dataset_version": "v1",
            "split_version": "s1",
            "split": "validation",
            "recordings": {"r0": "c" * 64},
        },
    }
    return {**value, **overrides}


def test_tuning_refuses_test_and_frozen_detector_guards_test(loaded: LoadedModel) -> None:
    with pytest.raises(ValueError, match="validation"):
        tune(loaded, split("test"))
    check_frozen(detector_file(loaded), loaded, split("test"))
    tuned = tune(loaded, split("validation"))
    assert tuned["tolerance_s"] == TOLERANCE_S
    assert tuned["tuned_on"]["split"] == "validation"
    bad = [
        detector_file(loaded, tolerance_s=1.0),
        detector_file(loaded, checkpoint={"sha256": "d" * 64}),
        detector_file(loaded, tuned_on={**detector_file(loaded)["tuned_on"], "split": "test"}),
        detector_file(
            loaded, tuned_on={**detector_file(loaded)["tuned_on"], "recordings": {"r1": "b" * 64}}
        ),
    ]
    for detector in bad:
        with pytest.raises(ValueError):
            check_frozen(detector, loaded, split("test"))
