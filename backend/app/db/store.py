"""SQLite metadata, with recording-owned results deleted by foreign-key cascade."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

# evaluation_runs and live_events carry no API yet: their import/read routes
# (/api/evaluations) and the live WebSocket/action implementation belong to
# ticket 06. The tables exist here so the ticket-05 persistence seam (models,
# recordings, detections, evaluation runs, live events per KWS_SPEC) has a
# stable schema for ticket 06 to build on, without silent schema-only drift.
SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY, version TEXT NOT NULL, checkpoint_path TEXT NOT NULL,
    sha256 TEXT NOT NULL, labels_json TEXT NOT NULL, feature_config_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY, filename TEXT NOT NULL, duration_s REAL NOT NULL,
    sample_rate INTEGER NOT NULL, channels INTEGER NOT NULL, audio_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('uploaded','queued','running','completed','failed')),
    error_code TEXT, model_id TEXT REFERENCES models(id), keywords_json TEXT,
    detector_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY, recording_id TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES models(id), keyword TEXT NOT NULL,
    start_s REAL NOT NULL, end_s REAL NOT NULL, confidence REAL NOT NULL,
    threshold REAL NOT NULL, windows INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id), dataset_id TEXT NOT NULL,
    split_id TEXT NOT NULL, config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
    artifacts_path TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS live_events (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, keyword TEXT,
    score REAL, accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
    action TEXT CHECK(action IN
        ('next_slide','previous_slide','play','pause','volume_up','volume_down')),
    action_status TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS detections_recording ON detections(recording_id);
"""

# Recordings being analyzed by the bounded worker. RecordingService holds
# Store.lock only for these short metadata transitions, never across audio
# reads or model inference, so concurrent metadata GETs stay responsive.
ACTIVE_STATUSES = ("queued", "running")


def is_active_status(status: str) -> bool:
    """Whether a Recording status means the worker owns the job."""
    return status in ACTIVE_STATUSES


class Store:
    def __init__(self, path: Path) -> None:
        self.lock = RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(SCHEMA)
        with self.transaction() as db:
            db.execute(
                "UPDATE recordings SET status='failed', error_code='service_interrupted' "
                "WHERE status IN (?,?)",
                ACTIVE_STATUSES,
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock, self.connection:
            yield self.connection

    def close(self) -> None:
        self.connection.close()
