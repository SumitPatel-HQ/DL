"""Evaluation Runs imported from committed run files. Requests never name files."""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from app.db.store import Store
from app.errors import ServiceError

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Score = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0)]
# Above this size a file is not one of the committed run documents (largest ~165 KB).
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


def missing_ground_truth() -> PydanticCustomError:
    return PydanticCustomError(
        "missing_ground_truth", "Evaluation Runs need labeled references to report metrics."
    )


def agrees(reported: float, measured: float) -> bool:
    """Reported metrics must be recomputable from the run's own counts."""
    return math.isclose(reported, measured, abs_tol=1e-9)


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class Document(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)


class Revision(Document):
    commit: str
    dirty: bool


class Checkpoint(Document):
    sha256: Sha256


class Dataset(Document):
    dataset_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    condition: Literal["clean", "noisy"] = "clean"
    noise_sources: list[str] = Field(default_factory=list)
    snr_db: float | None = None
    # Noise sources of every split in the dataset version, not just the evaluated one.
    noise_sources_by_split: dict[str, list[str]] = Field(default_factory=dict)
    # Dataset version the checkpoint trained on, when it differs from the evaluated one.
    checkpoint_dataset_version: str | None = None

    @model_validator(mode="after")
    def noisy_runs_name_their_noise(self) -> "Dataset":
        if self.condition == "noisy" and (not self.noise_sources or self.snr_db is None):
            raise ValueError("A noisy run must name its noise sources and SNR")
        return self


class Target(Document):
    met: bool
    note: str
    macro_f1: Score | None = None
    event_f1: Score | None = None


class Run(Document):
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,99}$")
    created_at: str
    code_revision: Revision
    experiment_id: str = Field(min_length=1)
    checkpoint: Checkpoint
    target: Target

    @field_validator("created_at")
    @classmethod
    def timestamped(cls, value: str) -> str:
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("Timestamp needs a UTC offset")
        return value


class ClassMetrics(Document):
    precision: Score
    recall: Score
    f1: Score
    support: Count


class ConfusionMatrix(Document):
    rows: Literal["true"]
    # Canonical columns label is "detected" per CONTEXT.md Detection vocabulary.
    # "predicted" is accepted on import for committed runs and normalized.
    columns: Literal["detected", "predicted"]
    values: list[list[Count]]

    @model_validator(mode="after")
    def canonical_columns(self) -> "ConfusionMatrix":
        if self.columns != "detected":
            object.__setattr__(self, "columns", "detected")
        return self


class IsolatedMetrics(Document):
    examples: Count
    accuracy: Score
    macro_f1: Score
    per_class: dict[str, ClassMetrics]
    confusion_matrix: ConfusionMatrix
    silence_recall_by_source: dict[str, Score] = Field(default_factory=dict)


class Epoch(Document):
    epoch: Annotated[int, Field(ge=1)]
    train_loss: float
    train_accuracy: Score
    validation_loss: float
    validation_accuracy: Score
    validation_macro_f1: Score


class Training(Document):
    train_examples: Count
    validation_examples: Count
    epochs_run: Count
    best_epoch: Annotated[int, Field(ge=1)]
    best_validation_loss: float
    stopped_early: bool


class IsolatedRun(Run):
    kind: Literal["isolated"]
    config: dict[str, Any]
    config_sha256: Sha256
    labels: list[str]
    dataset: Dataset
    metrics: IsolatedMetrics
    training: Training
    curves: list[Epoch]

    @model_validator(mode="after")
    def reconciled(self) -> "IsolatedRun":
        metrics, labels = self.metrics, self.labels
        if metrics.examples == 0:
            raise missing_ground_truth()
        if len(set(labels)) != len(labels) or list(metrics.per_class) != labels:
            raise ValueError("Per-class metrics must follow the label order")
        matrix = metrics.confusion_matrix.values
        if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
            raise ValueError("Confusion matrix must be square over the labels")
        supports = [metrics.per_class[label].support for label in labels]
        if [sum(row) for row in matrix] != supports or sum(supports) != metrics.examples:
            raise ValueError("Confusion matrix rows must sum to per-class support")
        correct = sum(matrix[index][index] for index in range(len(labels)))
        mean_f1 = sum(counts.f1 for counts in metrics.per_class.values()) / len(labels)
        if not agrees(metrics.accuracy, correct / metrics.examples) or not agrees(
            metrics.macro_f1, mean_f1
        ):
            raise ValueError("Accuracy and macro-F1 must follow from the confusion matrix")
        if not self.curves or self.training.best_epoch not in {e.epoch for e in self.curves}:
            raise ValueError("Best epoch must appear in the training curves")
        if self.target.macro_f1 is None or self.target.met != (
            metrics.macro_f1 >= self.target.macro_f1
        ):
            raise ValueError("Target must state whether macro-F1 met its goal")
        return self


class TunedOn(Document):
    dataset_version: str
    split_version: str
    split: str


class Detector(Document):
    config: dict[str, Any]
    file_sha256: Sha256
    tolerance_s: Annotated[float, Field(gt=0)]
    match_rule: str
    tuned_on: TunedOn
    detector_created_at: str


class Composition(Document):
    recordings: Count
    duration_s: Annotated[float, Field(ge=0)]
    keyword_events: dict[str, Count]
    ignored_keyword_events: Count
    annotated_seconds: dict[str, Annotated[float, Field(ge=0)]]
    unlabeled_seconds: Annotated[float, Field(ge=0)]
    annotation_quality: str
    speaker_ids: list[str]


class ContinuousDataset(Dataset):
    composition: Composition


class KeywordMetrics(Document):
    true_positive: Count
    false_positive: Count
    false_negative: Count
    precision: Score
    recall: Score
    f1: Score


class RegionFalseAlarms(Document):
    count: Count
    seconds: Annotated[float, Field(ge=0)]
    per_hour: Annotated[float, Field(ge=0)]


class FalseAlarms(Document):
    definition: str
    count: Count
    negative_seconds: Annotated[float, Field(ge=0)]
    per_hour: Annotated[float, Field(ge=0)]
    by_region: dict[str, RegionFalseAlarms]


class ContinuousMetrics(KeywordMetrics):
    per_keyword: dict[str, KeywordMetrics]
    false_alarms: FalseAlarms
    ignored_matches: Count


class ContinuousRun(Run):
    kind: Literal["continuous"]
    detector: Detector
    dataset: ContinuousDataset
    metrics: ContinuousMetrics
    detections: dict[str, list[dict[str, Any]]]
    limitations: list[str]

    @model_validator(mode="after")
    def reconciled(self) -> "ContinuousRun":
        metrics, events = self.metrics, self.dataset.composition.keyword_events
        if sum(events.values()) == 0 or metrics.false_alarms.negative_seconds == 0:
            raise missing_ground_truth()
        if set(metrics.per_keyword) != set(events):
            raise ValueError("Per-Keyword metrics must cover the scored Keyword events")
        if metrics.true_positive + metrics.false_negative != sum(events.values()):
            raise ValueError("Hits and misses must account for every Keyword event")
        detections = sum(map(len, self.detections.values()))
        if metrics.true_positive + metrics.false_positive != detections:
            raise ValueError("Hits and false Detections must account for every Detection")
        for counts in (metrics, *metrics.per_keyword.values()):
            found = counts.true_positive + counts.false_positive
            expected = counts.true_positive + counts.false_negative
            precision = counts.true_positive / found if found else 0.0
            recall = counts.true_positive / expected if expected else 0.0
            if not (
                agrees(counts.precision, precision)
                and agrees(counts.recall, recall)
                and agrees(counts.f1, f1(precision, recall))
            ):
                raise ValueError("Event precision, recall, and F1 must follow from the counts")
        if self.target.event_f1 is None or self.target.met != (metrics.f1 >= self.target.event_f1):
            raise ValueError("Target must state whether event-F1 met its goal")
        return self


EvaluationRun = IsolatedRun | ContinuousRun
# mypy cannot type an Annotated special form as a TypeAdapter argument.
DOCUMENT: TypeAdapter[EvaluationRun] = TypeAdapter(
    Annotated[EvaluationRun, Field(discriminator="kind")]  # type: ignore[arg-type]
)


CONFIDENCE_NOTE = (
    "Detection Confidence is a classifier score, not a measure of correctness. "
    "Event metrics come from labeled references; unlabeled audio shows Confidence only."
)


DatasetKey = tuple[str, str, str]


def dataset_key(run: EvaluationRun) -> DatasetKey:
    """Identity a comparison requires: dataset version, split version, split."""
    dataset = run.dataset
    return (dataset.dataset_version, dataset.split_version, dataset.split)


def check_model_pair(run_a: EvaluationRun, run_b: EvaluationRun) -> None:
    """Model comparison: identical dataset version, split version, and split."""
    if dataset_key(run_a) != dataset_key(run_b):
        raise ServiceError(
            "comparison_dataset_mismatch",
            "Runs use different datasets or splits "
            f"({describe_dataset(run_a)} vs {describe_dataset(run_b)}). Pick two runs "
            "with identical dataset version, split version, and split.",
            400,
        )


def check_noise_pair(run_a: EvaluationRun, run_b: EvaluationRun) -> None:
    """Noise comparison: only the dataset condition may differ."""
    if run_a.checkpoint.sha256 != run_b.checkpoint.sha256:
        raise ServiceError(
            "comparison_provenance_mismatch",
            "Clean and noisy runs must share the checkpoint hash; "
            "only the dataset condition may differ.",
            400,
        )
    if dataset_key(run_a) != dataset_key(run_b):
        raise ServiceError(
            "comparison_provenance_mismatch",
            "Clean and noisy runs must use the identical dataset version, split version, "
            f"and split ({describe_dataset(run_a)} vs {describe_dataset(run_b)}); "
            "only the dataset condition may differ.",
            400,
        )
    if (
        isinstance(run_a, ContinuousRun)
        and isinstance(run_b, ContinuousRun)
        and (
            run_a.detector.config != run_b.detector.config
            or run_a.detector.tolerance_s != run_b.detector.tolerance_s
        )
    ):
        raise ServiceError(
            "comparison_protocol_mismatch",
            "Clean and noisy continuous runs must share detector configuration and tolerance.",
            400,
        )


def describe_dataset(run: EvaluationRun) -> str:
    return "/".join(dataset_key(run))


def provenance(run: EvaluationRun) -> dict[str, Any]:
    if isinstance(run, IsolatedRun):
        role, config_sha256 = "training", run.config_sha256
    else:
        role, config_sha256 = "detector", run.detector.file_sha256
    facts: dict[str, Any] = {
        "checkpoint_sha256": run.checkpoint.sha256,
        "dataset_version": run.dataset.dataset_version,
        "split_version": run.dataset.split_version,
        "split": run.dataset.split,
        "config_role": role,
        "config_sha256": config_sha256,
        "code_revision": run.code_revision.model_dump(),
    }
    if run.dataset.noise_sources or run.dataset.condition != "clean":
        facts["condition"] = run.dataset.condition
        facts["noise_sources"] = run.dataset.noise_sources
        facts["snr_db"] = run.dataset.snr_db
        facts["noise_sources_by_split"] = run.dataset.noise_sources_by_split
        facts["checkpoint_dataset_version"] = run.dataset.checkpoint_dataset_version
    return facts


def target(run: EvaluationRun) -> dict[str, Any]:
    if isinstance(run, IsolatedRun):
        metric, goal, achieved = "macro_f1", run.target.macro_f1, run.metrics.macro_f1
    else:
        metric, goal, achieved = "event_f1", run.target.event_f1, run.metrics.f1
    return {
        "metric": metric,
        "goal": goal,
        "achieved": achieved,
        "met": run.target.met,
        "note": run.target.note,
    }


class EvaluationService:
    def __init__(self, store: Store, root: Path) -> None:
        self.store = store
        self.root = root

    def import_committed(self) -> dict[str, Any]:
        """Upsert every committed run file; other JSON (detectors, comparisons) is skipped."""
        if not self.root.is_dir():
            raise ServiceError("evaluations_unavailable", "Committed runs are not installed.", 503)
        imported: list[str] = []
        rejected: list[dict[str, str]] = []
        rows = []
        for path in sorted(self.root.glob("*/*.json")):
            source = path.relative_to(self.root).as_posix()
            code = None
            try:
                if path.stat().st_size > MAX_DOCUMENT_BYTES:
                    raise ValueError("Oversized document")
                text = path.read_text(encoding="utf-8")
                raw = json.loads(text)
                if not isinstance(raw, dict) or "run_id" not in raw:
                    continue
                run = DOCUMENT.validate_json(text)
            except ValidationError as exc:
                labeled = all(error["type"] != "missing_ground_truth" for error in exc.errors())
                code = "invalid_evaluation_run" if labeled else "missing_ground_truth"
            except (ValueError, OSError):
                code = "invalid_evaluation_run"
            if code is None and run.run_id in imported:
                code = "duplicate_run_id"
            if code is not None:
                rejected.append({"source": source, "code": code})
                continue
            facts = provenance(run)
            rows.append(
                (
                    run.run_id,
                    run.kind,
                    facts["checkpoint_sha256"],
                    facts["dataset_version"],
                    facts["split_version"],
                    facts["split"],
                    facts["config_sha256"],
                    source,
                    text,
                    run.created_at,
                )
            )
            imported.append(run.run_id)
        with self.store.transaction() as db:
            db.executemany(
                "INSERT OR REPLACE INTO evaluation_runs (id,kind,checkpoint_sha256,"
                "dataset_version,split_version,split,config_sha256,source,document_json,"
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return {"imported": imported, "rejected": rejected}

    def list_runs(self) -> dict[str, Any]:
        with self.store.transaction() as db:
            rows = db.execute(
                "SELECT source,document_json FROM evaluation_runs ORDER BY created_at,id"
            ).fetchall()
        runs = [self._summary(DOCUMENT.validate_json(row[1]), row[0]) for row in rows]
        return {"code": "runs_available" if runs else "not_evaluated", "runs": runs}

    def get(self, run_id: str) -> dict[str, Any]:
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT source,document_json FROM evaluation_runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise ServiceError("evaluation_not_found", "Evaluation Run does not exist.", 404)
        run = DOCUMENT.validate_json(row[1])
        detail = self._summary(run, row[0])
        detail["provenance"]["config"] = (
            run.config if isinstance(run, IsolatedRun) else run.detector.config
        )
        detail["experimental"] = isinstance(run, ContinuousRun)
        if isinstance(run, IsolatedRun):
            metrics = run.metrics
            detail["labels"] = run.labels
            detail["metrics"] = {
                "examples": metrics.examples,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "per_class": [
                    {"label": label, **metrics.per_class[label].model_dump()}
                    for label in run.labels
                ],
                "confusion_matrix": {
                    "labels": run.labels,
                    **metrics.confusion_matrix.model_dump(),
                },
                "silence_recall_by_source": metrics.silence_recall_by_source,
            }
            detail["training"] = {
                **run.training.model_dump(),
                "curves": [epoch.model_dump() for epoch in run.curves],
            }
            return detail
        scored = run.metrics
        detail["metrics"] = {
            "detections": sum(map(len, run.detections.values())),
            **scored.model_dump(exclude={"per_keyword"}),
            "per_keyword": [
                {"keyword": keyword, **counts.model_dump()}
                for keyword, counts in scored.per_keyword.items()
            ],
        }
        detail["detector"] = run.detector.model_dump(exclude={"config", "file_sha256"})
        detail["composition"] = run.dataset.composition.model_dump()
        detail["limitations"] = run.limitations
        detail["confidence_note"] = CONFIDENCE_NOTE
        return detail

    def delete(self, run_id: str) -> None:
        with self.store.transaction() as db:
            deleted = db.execute("DELETE FROM evaluation_runs WHERE id=?", (run_id,)).rowcount
        if not deleted:
            raise ServiceError("evaluation_not_found", "Evaluation Run does not exist.", 404)

    def compare(self, run_a_id: str, run_b_id: str) -> dict[str, Any]:
        """Side-by-side view over two stored runs of the same kind.

        Same condition and dataset/split: a model comparison. Different conditions:
        a noise comparison, which also needs the same checkpoint hash and, for
        continuous runs, the same detector protocol. Reads only stored runs;
        mismatches are rejected with an actionable error. Isolated and continuous
        numbers never share a response: each run keeps its own detail shape from get().
        """
        with self.store.transaction() as db:
            row_a = db.execute(
                "SELECT source,document_json FROM evaluation_runs WHERE id=?", (run_a_id,)
            ).fetchone()
            row_b = db.execute(
                "SELECT source,document_json FROM evaluation_runs WHERE id=?", (run_b_id,)
            ).fetchone()
        if row_a is None or row_b is None:
            missing = run_a_id if row_a is None else run_b_id
            raise ServiceError(
                "evaluation_not_found",
                f"Evaluation Run {missing} does not exist; comparison views clear "
                "to Not evaluated with no fallback to another run.",
                404,
            )
        run_a = DOCUMENT.validate_json(row_a[1])
        run_b = DOCUMENT.validate_json(row_b[1])
        if run_a.kind != run_b.kind:
            raise ServiceError(
                "comparison_kind_mismatch",
                f"Runs are different kinds ({run_a.kind} vs {run_b.kind}) and never "
                "share a view. Pick two isolated or two continuous runs.",
                400,
            )

        if run_a.dataset.condition != run_b.dataset.condition:
            check_noise_pair(run_a, run_b)
            comparison_type = "noise"
        else:
            check_model_pair(run_a, run_b)
            comparison_type = "model"

        detail_a = self.get(run_a_id)
        detail_b = self.get(run_b_id)
        noise_sources = sorted({*run_a.dataset.noise_sources, *run_b.dataset.noise_sources})
        by_split = {**run_a.dataset.noise_sources_by_split, **run_b.dataset.noise_sources_by_split}
        return {
            "code": "comparison_available",
            "comparison_id": f"{run_a_id}..{run_b_id}",
            "comparison_type": comparison_type,
            "kind": run_a.kind,
            "dataset_version": run_a.dataset.dataset_version,
            "split_version": run_a.dataset.split_version,
            "split": run_a.dataset.split,
            "noise_sources": noise_sources,
            "noise_sources_by_split": by_split,
            "runs": [detail_a, detail_b],
        }

    @staticmethod
    def _summary(run: EvaluationRun, source: str) -> dict[str, Any]:
        return {
            "id": run.run_id,
            "kind": run.kind,
            "experiment_id": run.experiment_id,
            "created_at": run.created_at,
            "source": source,
            "provenance": provenance(run),
            "target": target(run),
        }
