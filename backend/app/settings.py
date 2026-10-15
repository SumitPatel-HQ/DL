"""Local operator configuration. Requests cannot choose checkpoints or storage paths."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    storage: Path = ROOT / "data" / "service"
    checkpoint: Path = ROOT / "artifacts" / "models" / "e02" / "model.pt"
    detector: Path = ROOT / "docs" / "evaluations" / "continuous-v1" / "detector-e02.json"
    evaluations: Path = ROOT / "docs" / "evaluations"
    max_upload_bytes: int = 40 * 1024 * 1024
    max_duration_s: float = 600
    queue_capacity: int = 2
    requests_per_minute: int = 60
    concurrent_uploads: int = 2
    origins: tuple[str, ...] = ("http://127.0.0.1:3000", "http://localhost:3000")

    def __post_init__(self) -> None:
        if self.max_upload_bytes < 1 or not 0 < self.max_duration_s <= 600:
            raise ValueError("Expected positive upload limit and duration at most 600 seconds")
        if self.queue_capacity < 1:
            raise ValueError("Queue capacity must be positive")
        if self.requests_per_minute < 1 or self.concurrent_uploads < 1:
            raise ValueError("Request and concurrent upload limits must be positive")

    @classmethod
    def from_environment(cls) -> "Settings":
        defaults = cls()
        return cls(
            storage=Path(os.environ.get("KWS_STORAGE", defaults.storage)),
            checkpoint=Path(os.environ.get("KWS_CHECKPOINT", defaults.checkpoint)),
            detector=Path(os.environ.get("KWS_DETECTOR", defaults.detector)),
            evaluations=Path(os.environ.get("KWS_EVALUATIONS", defaults.evaluations)),
        )
