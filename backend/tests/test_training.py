import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ml.checkpoint import load_checkpoint
from ml.data.manifest import Source
from ml.data.noise import mix_noise
from ml.evaluate import compare, compare_noise, evaluate, silence_recall
from ml.features import LABELS
from ml.metrics import classification_report, confusion_matrix
from ml.networks import LstmClassifier
from ml.train import train


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "T01",
                "seed": 7,
                "network": {"type": "lstm", "hidden_size": 8, "num_layers": 2, "dropout": 0.1},
                "training": {
                    "batch_size": 16,
                    "optimizer": "adam",
                    "learning_rate": 0.01,
                    "loss": "cross_entropy",
                    "max_epochs": 3,
                    "early_stopping": {"monitor": "validation_loss", "patience": 2},
                    "grad_clip_norm": 5.0,
                    "num_threads": 1,
                },
            }
        )
    )
    return path


def test_metrics_match_hand_computed_fixture() -> None:
    matrix = confusion_matrix(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0]), 3)
    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    report = classification_report(matrix, ("a", "b", "c"))
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["per_class"]["b"] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2}
    )
    assert report["per_class"]["c"]["f1"] == 0
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + 0) / 3)


def test_lstm_emits_one_logit_per_class_with_dropout_outside_single_layer() -> None:
    model = LstmClassifier(hidden_size=16, num_layers=1, dropout=0.3)
    assert model.lstm.dropout == 0
    assert model(torch.zeros(3, 98, 40)).shape == (3, len(LABELS))
    with pytest.raises(ValueError):
        LstmClassifier(hidden_size=16, num_layers=3)


def test_training_is_seeded_and_checkpoint_reloads_with_contract(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    first = train(config, cache, tmp_path / "a")
    second = train(config, cache, tmp_path / "b")
    loss_keys = ("train_loss", "validation_loss")
    for a, b in zip(first["training"]["history"], second["training"]["history"], strict=True):
        assert [a[k] for k in loss_keys] == [b[k] for k in loss_keys]

    loaded = load_checkpoint(tmp_path / "a" / "model.pt")
    again = load_checkpoint(tmp_path / "b" / "model.pt")
    for name, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor, again.model.state_dict()[name])
    assert loaded.labels == LABELS
    assert loaded.metadata["network"]["num_layers"] == 2
    assert loaded.sha256 == first["checkpoint"]["sha256"]
    window = np.zeros((1, 98, 40), dtype=np.float32)
    np.testing.assert_allclose(loaded.probabilities(window).sum(axis=1), 1, rtol=1e-6)
    assert (tmp_path / "a" / "curves.csv").read_text().startswith("epoch,train_loss")


def test_checkpoint_with_different_class_order_is_rejected(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    train(config, cache, tmp_path / "run")
    path = tmp_path / "run" / "model.pt"
    payload: dict[str, Any] = torch.load(path, weights_only=True)
    payload["labels"] = list(reversed(payload["labels"]))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="class order"):
        load_checkpoint(path)


def test_evaluation_is_reproducible_and_reports_every_class(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    train(config, cache, tmp_path / "run")
    checkpoint = tmp_path / "run" / "model.pt"
    first = evaluate(checkpoint, cache, "test", latency_runs=2)
    second = evaluate(checkpoint, cache, "test", latency_runs=2)
    assert first["metrics"] == second["metrics"]
    assert first["run_id"] == second["run_id"]
    assert set(first["metrics"]["per_class"]) == set(LABELS)
    matrix = np.array(first["metrics"]["confusion_matrix"]["values"])
    assert matrix.shape == (8, 8) and matrix.sum() == first["metrics"]["examples"]
    assert first["dataset"]["split"] == "test"
    assert first["checkpoint"]["sha256"] == load_checkpoint(checkpoint).sha256
    assert compare([first, second])["runs"][0]["macro_f1"] == first["metrics"]["macro_f1"]
    validation = evaluate(checkpoint, cache, "validation", latency_runs=2)
    with pytest.raises(ValueError, match="identical split"):
        compare([first, validation])
    with pytest.raises(ValueError, match="held-out"):
        evaluate(checkpoint, cache, "train", latency_runs=2)


def test_expanded_vocabulary_trains_reloads_and_evaluates_with_its_own_class_order(
    corpus: Path, sources: dict[str, Source], config: Path, tmp_path: Path
) -> None:
    from conftest import write_wav

    from ml.data.manifest import build_manifest, save_manifest, speaker_split
    from ml.data.prepare import prepare_cache

    labels = (*LABELS[:-2], "next", "pause", "urgent", "unknown", "silence")
    for split in ("train", "validation", "test"):
        speaker = next(f"s{i}" for i in range(1000) if speaker_split(f"s{i}") == split)
        for word in ("next", "pause", "urgent"):
            write_wav(corpus / word / f"{speaker}_nohash_0.wav")
    manifest_path = tmp_path / "expanded.json"
    save_manifest(
        build_manifest(
            corpus,
            {**sources, **dict.fromkeys(("next", "pause", "urgent"), sources["speech"])},
            labels=labels,
            max_per_class=1,
        ),
        manifest_path,
    )
    features = tmp_path / "expanded-features"
    prepare_cache(manifest_path, corpus, features)
    summary = train(config, features, tmp_path / "expanded-model")
    checkpoint = tmp_path / "expanded-model" / "model.pt"
    loaded = load_checkpoint(checkpoint)
    assert loaded.labels == labels
    assert summary["labels"] == list(labels)
    assert loaded.probabilities(np.zeros((1, 98, 40), dtype=np.float32)).shape == (1, 11)
    run = evaluate(checkpoint, features, "test", latency_runs=1)
    assert run["labels"] == list(labels)
    assert set(run["metrics"]["per_class"]) == set(labels)
    assert np.array(run["metrics"]["confusion_matrix"]["values"]).shape == (11, 11)
    assert run["dataset"]["dataset_version"] == "speech-commands-v0.02-p2-vocab-v1"

    # Continuous evaluation must count all nine Keywords, with the frozen protocol.
    from ml.evaluate_continuous import ContinuousSplit, Reference, SplitRecording
    from ml.evaluate_continuous import run as run_continuous
    from ml.spotting import DetectorConfig

    data = ContinuousSplit(
        "expanded-continuous-fixture-v1",
        "speakers-v1",
        "test",
        "fixture-only",
        (
            SplitRecording(
                "test",
                "b" * 64,
                12.0,
                ("held-out",),
                (
                    *(Reference(word, i + 0.1, i + 0.4) for i, word in enumerate(labels[:-2])),
                    Reference("unknown", 9.0, 10.0),
                    Reference("silence", 10.0, 12.0),
                ),
                np.zeros(12 * 16000, dtype=np.float32),
            ),
        ),
    )
    detector = {
        "format": "kws-detector-v1",
        "checkpoint": {"sha256": loaded.sha256},
        "config": DetectorConfig().to_dict(),
        "tolerance_s": 0.5,
        "created_at": "2026-10-16T00:00:00+00:00",
        "tuned_on": {
            "split": "validation",
            "dataset_version": data.dataset_version,
            "split_version": data.split_version,
            "recordings": {"validation": "c" * 64},
        },
    }
    continuous = run_continuous(loaded, data, detector)
    assert set(continuous["metrics"]["per_keyword"]) == set(labels[:-2])
    assert set(continuous["dataset"]["composition"]["keyword_events"]) == set(labels[:-2])
    metrics = continuous["metrics"]
    assert metrics["true_positive"] + metrics["false_negative"] == 9


def noisy_cache(cache: Path) -> None:
    """Copy the test cache as a noisy condition; plumbing only, not real noise."""
    with np.load(cache / "test.npz", allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    metadata = json.loads(str(arrays["metadata"]))
    arrays["metadata"] = np.array(json.dumps({**metadata, "condition": "noisy", "snr_db": 10.0}))
    arrays["features"] = arrays["features"] + 0.5
    np.savez_compressed(cache / "test-noisy.npz", **arrays)


def paired_runs(cache: Path, config: Path, tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train(config, cache, tmp_path / "run")
    checkpoint = tmp_path / "run" / "model.pt"
    noisy_cache(cache)
    clean = evaluate(checkpoint, cache, "test", condition="clean", latency_runs=2)
    noisy = evaluate(checkpoint, cache, "test", condition="noisy", latency_runs=2)
    return clean, noisy


def test_noise_pair_shares_checkpoint_and_protocol_and_reports_silence_per_source(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    clean, noisy = paired_runs(cache, config, tmp_path)
    assert clean["run_id"] != noisy["run_id"]
    assert clean["dataset"]["condition"] == "clean" and noisy["dataset"]["condition"] == "noisy"
    assert noisy["dataset"]["snr_db"] == 10.0 and clean["dataset"]["snr_db"] is None
    assert noisy["dataset"]["noise_sources"]
    assert set(noisy["metrics"]["silence_recall_by_source"]) == set(
        noisy["dataset"]["noise_sources"]
    )
    comparison = compare_noise([noisy, clean])
    assert comparison["comparison_type"] == "noise"
    assert [run["condition"] for run in comparison["runs"]] == ["clean", "noisy"]
    assert comparison["runs"][1]["macro_f1"] == noisy["metrics"]["macro_f1"]
    assert comparison["runs"][1]["snr_db"] == 10.0
    assert comparison["noise_sources"] == noisy["dataset"]["noise_sources"]


def test_noisy_evaluation_without_a_noisy_cache_is_refused(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    train(config, cache, tmp_path / "run")
    with pytest.raises(ValueError, match="noisy feature cache"):
        evaluate(tmp_path / "run" / "model.pt", cache, "test", condition="noisy", latency_runs=2)


def test_noise_comparison_rejects_mismatched_provenance(
    cache: Path, config: Path, tmp_path: Path
) -> None:
    clean, noisy = paired_runs(cache, config, tmp_path)
    with pytest.raises(ValueError, match="exactly one clean and one noisy"):
        compare_noise([clean, clean])
    other_checkpoint = {**noisy, "checkpoint": {**noisy["checkpoint"], "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="Checkpoint hash differs"):
        compare_noise([clean, other_checkpoint])
    other_dataset = {**noisy, "dataset": {**noisy["dataset"], "dataset_version": "other"}}
    with pytest.raises(ValueError, match="dataset_version differs"):
        compare_noise([clean, other_dataset])
    other_split = {**noisy, "dataset": {**noisy["dataset"], "split": "validation"}}
    with pytest.raises(ValueError, match="split differs"):
        compare_noise([clean, other_split])


def test_continuous_noise_comparison_requires_the_same_frozen_detector() -> None:
    detector = {"config": {"threshold": 0.5}, "tolerance_s": 0.5}

    def run(condition: str, **detector_changes: Any) -> dict[str, Any]:
        return {
            "kind": "continuous",
            "run_id": f"continuous-{condition}",
            "metrics": {
                "precision": 0.1,
                "recall": 0.2,
                "f1": 0.13,
                "false_alarms": {"per_hour": 5.0},
            },
            "checkpoint": {"sha256": "a" * 64},
            "detector": {**detector, **detector_changes},
            "dataset": {
                "dataset_version": "ami",
                "split_version": "s",
                "split": "test",
                "condition": condition,
                "noise_sources": ["white_noise"] if condition == "noisy" else [],
            },
        }

    paired = compare_noise([run("noisy"), run("clean")])
    assert paired["noise_sources"] == ["white_noise"]
    assert [row["condition"] for row in paired["runs"]] == ["clean", "noisy"]
    with pytest.raises(ValueError, match="config differs"):
        compare_noise([run("clean"), run("noisy", config={"threshold": 0.9})])
    with pytest.raises(ValueError, match="tolerance_s differs"):
        compare_noise([run("clean"), run("noisy", tolerance_s=1.0)])
    isolated = {**run("noisy"), "kind": "isolated"}
    with pytest.raises(ValueError, match="Kinds differ"):
        compare_noise([run("clean"), isolated])


def test_silence_recall_is_reported_per_noise_source() -> None:
    names = ("left", "silence")
    examples = np.array(
        [
            json.dumps({"speaker_id": "noise-source:a"}),
            json.dumps({"speaker_id": "noise-source:a"}),
            json.dumps({"speaker_id": "noise-source:b"}),
            json.dumps({"speaker_id": "spk1"}),
        ]
    )
    labels = np.array([1, 1, 1, 0])
    detected = np.array([1, 0, 1, 1])
    assert silence_recall(examples, labels, detected, names) == {"a": 0.5, "b": 1.0}
    assert silence_recall(None, labels, detected, names) == {}


def test_noise_mixing_rejects_empty_noise_and_hits_the_target_snr() -> None:
    signal = (0.5 * np.sin(np.linspace(0, 200, 16000))).astype(np.float32)
    with pytest.raises(ValueError, match="empty noise"):
        mix_noise(signal, np.empty(0, dtype=np.float32))
    noise = np.random.default_rng(1).standard_normal(48000).astype(np.float32) * 0.1
    mixed = mix_noise(signal, noise, 10.0, np.random.default_rng(2))
    added = mixed - signal
    snr = 20 * np.log10(np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(added**2)))
    assert snr == pytest.approx(10.0, abs=0.1)


def test_live_training_balance_uses_the_median_keyword_count() -> None:
    from ml.train_live import KEYWORDS, keyword_count
    from ml.vocabulary import keywords_for

    assert KEYWORDS == keywords_for(LABELS)
    # 'left' (label 0) is rare here; the median keeps Silence/TTS balance from collapsing.
    labels = [0] * 2 + [1] * 10 + [2] * 10 + [3] * 12 + [4] * 10 + [5] * 9 + [6] * 50
    assert keyword_count(labels) == 10
