"""One bounded analysis worker. Only generated recording IDs address local audio."""

import json
import wave
from collections import deque
from dataclasses import asdict
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Any

import numpy as np

from app.constants import MODEL_SETUP_ACTION
from app.db.store import Store
from app.errors import ServiceError
from app.settings import Settings
from ml.checkpoint import LoadedModel, load_checkpoint
from ml.features import FEATURE_CONFIG
from ml.features.pipeline import SAMPLE_RATE, WINDOW_SAMPLES, read_wav
from ml.spotting import KEYWORDS, DetectorConfig, WindowScores, detect, score_windows, window_starts

EVIDENCE = "docs/evaluations/continuous-v1/e02-continuous-test.json"
LIMITATION = "Experimental sliding-window Detection; natural-speech feasibility target was not met."


def inspect_wav(path: Path, max_duration_s: float) -> tuple[float, int, int]:
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
            return frames / rate, rate, channels
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
            if path.is_file() and path.name not in owned:
                path.unlink()
        self.loaded: LoadedModel | None = None
        self.config: DetectorConfig | None = None
        self.model_error = "checkpoint_not_installed"
        if settings.checkpoint.is_file():
            try:
                self.loaded = load_checkpoint(settings.checkpoint)
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
            except Exception:
                self.loaded = None
                self.config = None
                self.model_error = "model_incompatible"
        self.condition = Condition(self.store.lock)
        self.queue: deque[str] = deque()
        self.stopping = Event()
        self.worker = Thread(target=self._work, name="kws-analysis", daemon=False)
        self.worker.start()

    def model_health(self) -> dict[str, Any]:
        if self.loaded is None:
            return {
                "status": "unavailable",
                "code": self.model_error,
                "setup_action": MODEL_SETUP_ACTION,
            }
        return {
            "status": "ready",
            "code": "model_ready",
            "id": self.loaded.sha256,
            "experimental": True,
        }

    def active_model(self) -> dict[str, Any]:
        if self.loaded is None or self.config is None:
            raise ServiceError(self.model_error, MODEL_SETUP_ACTION, 503)
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
        result = dict(row)
        result.pop("audio_path")
        result["keywords"] = json.loads(result.pop("keywords_json") or "[]")
        result["detector"] = json.loads(result.pop("detector_json") or "null")
        return result

    def add(self, recording: str, filename: str, path: Path) -> dict[str, Any]:
        duration, rate, channels = inspect_wav(path, self.settings.max_duration_s)
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO recordings "
                "(id,filename,duration_s,sample_rate,channels,audio_path,status) "
                "VALUES (?,?,?,?,?,?,'uploaded')",
                (recording, filename, duration, rate, channels, path.name),
            )
        return self.get(recording)

    def analyze(self, recording: str, keywords: list[str]) -> dict[str, Any]:
        with self.condition:
            current = self.get(recording)
            self.active_model()
            if (
                not keywords
                or len(set(keywords)) != len(keywords)
                or not set(keywords) <= set(KEYWORDS)
            ):
                raise ServiceError("unsupported_keyword", "Select unique trained Keywords.", 422)
            if current["status"] in ("running", "queued"):
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
            self.get(recording)
            # The worker closes the source before inference and checks existence before saving.
            try:
                self.audio_path(recording).unlink(missing_ok=True)
            except OSError as exc:
                raise ServiceError(
                    "delete_failed", "Audio could not be removed; retry deletion.", 409
                ) from exc
            with self.store.transaction() as db:
                db.execute("DELETE FROM recordings WHERE id=?", (recording,))
            if recording in self.queue:
                self.queue.remove(recording)

    def _work(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.queue or self.stopping.is_set())
                if self.stopping.is_set():
                    return
                recording = self.queue.popleft()
                with self.store.transaction() as db:
                    db.execute("UPDATE recordings SET status='running' WHERE id=?", (recording,))
            try:
                self._infer(recording)
            except Exception:
                with self.store.transaction() as db:
                    db.execute(
                        "UPDATE recordings SET status='failed',error_code='analysis_failed' "
                        "WHERE id=?",
                        (recording,),
                    )

    def _infer(self, recording: str) -> None:
        assert self.loaded is not None and self.config is not None
        with self.store.lock:
            current = self.get(recording)
            audio = read_wav(self.audio_path(recording))
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
            WindowScores(starts / SAMPLE_RATE, np.concatenate(chunks), self.config.stride_s),
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
        with self.store.transaction() as db:
            db.execute(
                "UPDATE recordings SET status='failed',error_code='service_interrupted' "
                "WHERE status IN ('running','queued')"
            )
        self.store.close()
