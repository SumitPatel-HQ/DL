"""One bounded analysis worker. Only generated recording IDs address local audio."""

import json
import sqlite3
import time
import wave
from collections import deque
from dataclasses import asdict
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Any, NamedTuple
from uuid import UUID

import numpy as np

from app.constants import MODEL_SETUP_ACTION
from app.db.store import ACTIVE_STATUSES, Store, is_active_status
from app.errors import ServiceError
from app.settings import Settings
from ml.checkpoint import LoadedModel, load_checkpoint
from ml.features import FEATURE_CONFIG, LABELS
from ml.features.pipeline import SAMPLE_RATE, WINDOW_SAMPLES, read_wav
from ml.spotting import (
    KEYWORDS,
    DetectorConfig,
    WindowScores,
    detect,
    score_windows,
    window_levels_db,
    window_starts,
)
from ml.vocabulary import VOCABULARY_REVIEW_ACTION

EVIDENCE = "docs/evaluations/continuous-v1/e02-continuous-test.json"
LIMITATION = "Experimental sliding-window Detection; natural-speech feasibility target was not met."


class WavInfo(NamedTuple):
    duration_s: float
    sample_rate: int
    channels: int


def inspect_wav(path: Path, max_duration_s: float) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as wav:
            frames, rate, channels = wav.getnframes(), wav.getframerate(), wav.getnchannels()
            if channels not in (1, 2) or not 8000 <= rate <= 96000:
                raise ServiceError("unsupported_wav", "Use mono/stereo PCM WAV at 8-96 kHz.")
            if wav.getsampwidth() not in (1, 2, 3, 4) or frames == 0:
                raise ServiceError("invalid_wav", "Use non-empty integer PCM WAV audio.")
            if frames / rate > max_duration_s:
                raise ServiceError(
                    "recording_too_long", "Recording exceeds the duration limit.", 413
                )
            expected = frames * channels * wav.getsampwidth()
            actual = 0
            while chunk := wav.readframes(16384):
                actual += len(chunk)
            if actual != expected:
                raise ServiceError("invalid_wav", "WAV sample data is truncated.")
            return WavInfo(frames / rate, rate, channels)
    except (wave.Error, EOFError) as exc:
        raise ServiceError("invalid_wav", "Upload a valid uncompressed integer PCM WAV.") from exc


class RecordingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audio_dir = settings.storage / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(settings.storage / "metadata.sqlite3")
        # A stopped process may leave upload temporaries or uncommitted audio bytes.
        with self.store.transaction() as db:
            owned = {row[0] for row in db.execute("SELECT audio_path FROM recordings")}
        for path in self.audio_dir.iterdir():
            if path.name.endswith(".deleting") and path.name.removesuffix(".deleting") in owned:
                path.replace(path.with_suffix(""))
                continue
            if path.is_file() and path.name not in owned:
                path.unlink()
        self.loaded: LoadedModel | None = None
        self.config: DetectorConfig | None = None
        self.model_error = "checkpoint_not_installed"
        if settings.checkpoint.is_file():
            try:
                self.loaded = load_checkpoint(settings.checkpoint)
                if self.loaded.labels != LABELS:
                    raise ServiceError("vocabulary_not_reviewed", VOCABULARY_REVIEW_ACTION, 503)
                detector = json.loads(settings.detector.read_text(encoding="utf-8"))
                if (
                    detector["format"] != "kws-detector-v1"
                    or detector["checkpoint"]["sha256"] != self.loaded.sha256
                ):
                    raise ValueError("Detector checkpoint mismatch")
                self.config = DetectorConfig(**detector["config"])
                with self.store.transaction() as db:
                    db.execute(
                        "INSERT OR IGNORE INTO models "
                        "(id,version,checkpoint_path,sha256,labels_json,feature_config_json) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            self.loaded.sha256,
                            self.loaded.metadata.get("experiment_id", "lstm"),
                            str(settings.checkpoint),
                            self.loaded.sha256,
                            json.dumps(self.loaded.labels),
                            json.dumps(FEATURE_CONFIG),
                        ),
                    )
            except Exception as exc:
                self.loaded = None
                self.config = None
                self.model_error = (
                    exc.code if isinstance(exc, ServiceError) else "model_incompatible"
                )
        self.model_setup_action = (
            VOCABULARY_REVIEW_ACTION
            if self.model_error == "vocabulary_not_reviewed"
            else MODEL_SETUP_ACTION
        )
        self.condition = Condition(self.store.lock)
        self.queue: deque[str] = deque()
        self.stopping = Event()
        self.worker_error = Event()
        self.worker = Thread(target=self._work, name="kws-analysis", daemon=False)
        self.worker.start()

    def model_health(self) -> dict[str, Any]:
        if self.worker_error.is_set():
            return {
                "status": "unavailable",
                "code": "worker_unavailable",
                "setup_action": "Metadata writes failed. Worker is waiting for storage recovery.",
            }
        if self.loaded is None:
            return {
                "status": "unavailable",
                "code": self.model_error,
                "setup_action": self.model_setup_action,
            }
        return {
            "status": "ready",
            "code": "model_ready",
            "id": self.loaded.sha256,
            "experimental": True,
        }

    def active_model(self) -> dict[str, Any]:
        if self.worker_error.is_set():
            raise ServiceError(
                "worker_unavailable", "Worker is waiting for metadata storage recovery.", 503
            )
        if self.loaded is None or self.config is None:
            raise ServiceError(self.model_error, self.model_setup_action, 503)
        return {
            "id": self.loaded.sha256,
            "version": self.loaded.metadata.get("experiment_id"),
            "labels": self.loaded.labels,
            "keywords": KEYWORDS,
            "features": FEATURE_CONFIG,
            "detector": asdict(self.config),
            "experimental": True,
            "limitation": LIMITATION,
            "evaluation_evidence": EVIDENCE,
        }

    def get(self, recording: str) -> dict[str, Any]:
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM recordings WHERE id=?", (recording,)).fetchone()
        if row is None:
            raise ServiceError("recording_not_found", "Recording does not exist.", 404)
        return self._metadata(row)

    def list_recordings(self) -> list[dict[str, Any]]:
        with self.store.transaction() as db:
            rows = db.execute("SELECT * FROM recordings ORDER BY created_at,id").fetchall()
        return [self._metadata(row) for row in rows]

    @staticmethod
    def _metadata(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result.pop("audio_path")
        result["keywords"] = json.loads(result.pop("keywords_json") or "[]")
        result["detector"] = json.loads(result.pop("detector_json") or "null")
        return result

    def add(self, recording: str, filename: str, path: Path) -> dict[str, Any]:
        info = inspect_wav(path, self.settings.max_duration_s)
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO recordings "
                "(id,filename,duration_s,sample_rate,channels,audio_path,status) "
                "VALUES (?,?,?,?,?,?,'uploaded')",
                (recording, filename, info.duration_s, info.sample_rate, info.channels, path.name),
            )
        return self.get(recording)

    def analyze(self, recording: str, keywords: list[str]) -> dict[str, Any]:
        with self.condition:
            current = self.get(recording)
            # Keyword validation precedes the model checkpoint check so a bad
            # Keyword is reported as 422 even when no checkpoint is installed.
            if (
                not keywords
                or len(set(keywords)) != len(keywords)
                or not set(keywords) <= set(KEYWORDS)
            ):
                raise ServiceError("unsupported_keyword", "Select unique trained Keywords.", 422)
            self.active_model()
            if is_active_status(current["status"]):
                raise ServiceError(
                    "analysis_in_progress", "This Recording is already being analyzed.", 409
                )
            if len(self.queue) >= self.settings.queue_capacity:
                raise ServiceError("queue_full", "Analysis queue is full; retry later.", 429)
            assert self.loaded is not None and self.config is not None
            with self.store.transaction() as db:
                db.execute("DELETE FROM detections WHERE recording_id=?", (recording,))
                db.execute(
                    "UPDATE recordings SET status='queued',error_code=NULL,model_id=?,"
                    "keywords_json=?,detector_json=? WHERE id=?",
                    (
                        self.loaded.sha256,
                        json.dumps(keywords),
                        json.dumps(asdict(self.config)),
                        recording,
                    ),
                )
            self.queue.append(recording)
            result = self.get(recording)
            self.condition.notify()
            return result

    def results(self, recording: str) -> dict[str, Any]:
        with self.store.lock:
            current = self.get(recording)
            with self.store.transaction() as db:
                rows = db.execute(
                    "SELECT id,keyword,start_s,end_s,confidence,threshold,windows FROM detections "
                    "WHERE recording_id=? ORDER BY start_s,id",
                    (recording,),
                ).fetchall()
        code = "detections_found" if rows else "no_detections"
        if current["status"] != "completed":
            code = current["error_code"] or "analysis_not_completed"
        return {
            "recording_id": recording,
            "model_id": current["model_id"],
            "status": current["status"],
            "code": code,
            "detections": [dict(row) for row in rows],
            "detector": current["detector"],
            "timestamp_quality": "approximate_sliding_window",
            "experimental": True,
            "limitation": LIMITATION,
            "evaluation_evidence": EVIDENCE,
        }

    def audio_path(self, recording: str) -> Path:
        self.get(recording)
        return self.audio_dir / f"{recording}.wav"

    def delete(self, recording: str) -> None:
        with self.condition:
            # Validate IDs before addressing staged bytes for a previously committed deletion.
            try:
                if UUID(recording).hex != recording:
                    raise ValueError("Noncanonical ID")
            except ValueError as exc:
                raise ServiceError("recording_not_found", "Recording does not exist.", 404) from exc
            path = self.audio_dir / f"{recording}.wav"
            staged = path.with_suffix(".wav.deleting")
            try:
                try:
                    self.get(recording)
                except ServiceError:
                    if staged.is_file():
                        staged.unlink()
                        return
                    raise
                if path.is_file():
                    self._stage(path, staged)
                try:
                    with self.store.transaction() as db:
                        db.execute("DELETE FROM recordings WHERE id=?", (recording,))
                except Exception:
                    if staged.is_file():
                        staged.replace(path)
                    raise
                if recording in self.queue:
                    self.queue.remove(recording)
                staged.unlink(missing_ok=True)
            except OSError as exc:
                raise ServiceError(
                    "delete_failed", "Audio could not be removed; retry deletion.", 409
                ) from exc

    @staticmethod
    def _stage(path: Path, staged: Path) -> None:
        # Inference reads audio without holding Store.lock, so a concurrent
        # worker read can briefly hold the file open. Windows denies renaming
        # open files, so retry the staging rename briefly; other platforms are
        # unaffected. Persistent faults still surface as delete_failed.
        deadline = time.monotonic() + 2
        while True:
            try:
                path.replace(staged)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def _work(self) -> None:
        failed: str | None = None
        while True:
            with self.condition:
                if self.stopping.is_set():
                    return
                if failed is not None:
                    try:
                        self._fail(failed, "persistence_error")
                    except sqlite3.Error:
                        self.condition.wait(timeout=0.25)
                        continue
                    failed = None
                    self.worker_error.clear()
                self.condition.wait_for(lambda: self.queue or self.stopping.is_set())
                if self.stopping.is_set():
                    return
                recording = self.queue.popleft()
            try:
                with self.store.transaction() as db:
                    db.execute("UPDATE recordings SET status='running' WHERE id=?", (recording,))
                self._infer(recording)
            except Exception as exc:
                try:
                    self._fail(
                        recording,
                        "persistence_error"
                        if isinstance(exc, sqlite3.Error)
                        else "analysis_failed",
                    )
                except sqlite3.Error:
                    failed = recording
                    self.worker_error.set()

    def _fail(self, recording: str, code: str) -> None:
        with self.store.transaction() as db:
            db.execute(
                "UPDATE recordings SET status='failed',error_code=? WHERE id=?", (code, recording)
            )

    def _infer(self, recording: str) -> None:
        assert self.loaded is not None and self.config is not None
        # Short metadata read under the lock; audio decoding and scoring run
        # without holding Store.lock so concurrent metadata GETs never block
        # on inference I/O. Only the queued->running transition above, the
        # per-batch liveness checks below, and the final completed write hold
        # the lock.
        current = self.get(recording)
        audio = read_wav(self.audio_dir / f"{recording}.wav")
        starts = window_starts(len(audio), self.config.stride_s)
        chunks = []
        for first in range(0, len(starts), 32):
            with self.store.lock:
                if self.stopping.is_set():
                    return
                self.get(recording)
            subset = starts[first : first + 32]
            scored = score_windows(
                self.loaded,
                audio[subset[0] : subset[-1] + WINDOW_SAMPLES],
                self.config.stride_s,
                batch_size=32,
            )
            chunks.append(scored.probabilities)
        detections = detect(
            WindowScores(
                starts / SAMPLE_RATE,
                np.concatenate(chunks),
                self.config.stride_s,
                window_levels_db(audio, starts),
            ),
            self.config,
            tuple(current["keywords"]),
            self.loaded.labels,
        )
        with self.store.transaction() as db:
            if (
                self.stopping.is_set()
                or not db.execute(
                    "SELECT 1 FROM recordings WHERE id=?",
                    (recording,),
                ).fetchone()
            ):
                return
            db.executemany(
                "INSERT INTO detections (recording_id,model_id,keyword,start_s,end_s,confidence,"
                "threshold,windows) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        recording,
                        self.loaded.sha256,
                        d.keyword,
                        d.start_s,
                        min(d.end_s, current["duration_s"]),
                        d.confidence,
                        d.threshold,
                        d.windows,
                    )
                    for d in detections
                ],
            )
            db.execute("UPDATE recordings SET status='completed' WHERE id=?", (recording,))

    def close(self) -> None:
        with self.condition:
            self.stopping.set()
            self.queue.clear()
            self.condition.notify()
        self.worker.join()
        try:
            with self.store.transaction() as db:
                db.execute(
                    "UPDATE recordings SET status='failed',error_code='service_interrupted' "
                    "WHERE status IN (?,?)",
                    ACTIVE_STATUSES,
                )
        except sqlite3.Error:
            # Startup recovers persisted queued/running states when storage is writable again.
            pass
        finally:
            self.store.close()
