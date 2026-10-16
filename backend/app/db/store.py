"""SQLite metadata, with recording-owned results deleted by foreign-key cascade."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

# live_events holds live decision metadata only (label, score, decision,
# action, action result); raw mic audio is never stored. Evaluation Runs are
# imported from committed run files and keep the whole validated document; the
# provenance columns make every row
# traceable to checkpoint, dataset/split version, config, and timestamp without
# parsing JSON. Runs may describe checkpoints that are not installed (E01), so
# they reference checkpoints by hash rather than by a models row.
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
    id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('isolated','continuous')),
    checkpoint_sha256 TEXT NOT NULL, dataset_version TEXT NOT NULL,
    split_version TEXT NOT NULL, split TEXT NOT NULL, config_sha256 TEXT NOT NULL,
    source TEXT NOT NULL, document_json TEXT NOT NULL, created_at TEXT NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS live_events (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, keyword TEXT,
    score REAL, accepted INTEGER NOT NULL CHECK(accepted IN (0,1)),
    action TEXT CHECK(action IN
        ('next_slide','previous_slide','play','pause','volume_up','volume_down')),
    action_status TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS detections_recording ON detections(recording_id);
CREATE TABLE IF NOT EXISTS external_action_logs (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, event_id INTEGER,
    kind TEXT NOT NULL, target_id TEXT, action TEXT, status TEXT NOT NULL,
    reason TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS negative_session_reports (
    id TEXT PRIMARY KEY, document_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
SCHEMA_VERSION = 2

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
        if self.connection.execute("PRAGMA user_version").fetchone()[0] < 1:
            # Version 0 reserved evaluation_runs with no writer, so the old
            # shape never held data and is replaced rather than migrated.
            self.connection.execute("DROP TABLE IF EXISTS evaluation_runs")
        self.connection.executescript(SCHEMA)
        self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
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
