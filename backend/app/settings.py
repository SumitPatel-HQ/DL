"""Operator configuration. Requests cannot choose checkpoints or storage paths.

Locally the API serves the torch checkpoint on loopback. Deployed (Vercel sets
`VERCEL=1`; `KWS_DEPLOYED=1` simulates it) it serves the committed ONNX export
from backend/deploy/, keeps metadata in the writable temp directory, analyzes
inside the request and accepts its own origin instead of loopback only.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
# Vercel rejects request bodies above 4.5 MB before they reach the function.
DEPLOYED_MAX_UPLOAD_BYTES = 4 * 1024 * 1024


def deployed() -> bool:
    return any(os.environ.get(name) == "1" for name in ("VERCEL", "KWS_DEPLOYED"))


@dataclass(frozen=True)
class Settings:
    storage: Path = ROOT / "data" / "service"
    checkpoint: Path = ROOT / "artifacts" / "models" / "live3" / "model.pt"
    detector: Path = ROOT / "artifacts" / "models" / "live3" / "detector.json"
    evaluations: Path = ROOT / "docs" / "evaluations"
    max_upload_bytes: int = 40 * 1024 * 1024
    max_duration_s: float = 600
    queue_capacity: int = 2
    requests_per_minute: int = 60
    concurrent_uploads: int = 2
    origins: tuple[str, ...] = ("http://127.0.0.1:3000", "http://localhost:3000")
    # Deployed: analysis runs inside the analyze request, because a serverless
    # instance may be paused between requests and would strand a worker thread.
    inline_analysis: bool = False
    # Deployed: a browser Origin matching the request's own host is allowed.
    same_origin: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    import_evaluations: bool = False

    def __post_init__(self) -> None:
        if self.max_upload_bytes < 1 or not 0 < self.max_duration_s <= 600:
            raise ValueError("Expected positive upload limit and duration at most 600 seconds")
        if self.queue_capacity < 1:
            raise ValueError("Queue capacity must be positive")
        if self.requests_per_minute < 1 or self.concurrent_uploads < 1:
            raise ValueError("Request and concurrent upload limits must be positive")

    def origin_allowed(self, origin: str, hosts: tuple[str | None, ...]) -> bool:
        """Listed origins always pass; deployed, so does the request's own host."""
        if origin in self.origins:
            return True
        if not self.same_origin:
            return False
        parts = urlsplit(origin)
        return parts.scheme in ("http", "https") and any(
            host is not None and parts.netloc == host for host in hosts
        )

    @classmethod
    def from_environment(cls) -> "Settings":
        if deployed():
            defaults = cls(
                storage=Path(tempfile.gettempdir()) / "kws",
                checkpoint=DEPLOY / "model.onnx",
                detector=DEPLOY / "detector.json",
                evaluations=DEPLOY / "evaluations",
                max_upload_bytes=DEPLOYED_MAX_UPLOAD_BYTES,
                inline_analysis=True,
                same_origin=True,
                allowed_hosts=("*",),
                import_evaluations=True,
            )
        else:
            defaults = cls()
        extra = tuple(
            origin.strip()
            for origin in os.environ.get("KWS_ORIGINS", "").split(",")
            if origin.strip()
        )
        return cls(
            storage=Path(os.environ.get("KWS_STORAGE", defaults.storage)),
            checkpoint=Path(os.environ.get("KWS_CHECKPOINT", defaults.checkpoint)),
            detector=Path(os.environ.get("KWS_DETECTOR", defaults.detector)),
            evaluations=Path(os.environ.get("KWS_EVALUATIONS", defaults.evaluations)),
            max_upload_bytes=defaults.max_upload_bytes,
            origins=defaults.origins + extra,
            inline_analysis=defaults.inline_analysis,
            same_origin=defaults.same_origin,
            allowed_hosts=defaults.allowed_hosts,
            import_evaluations=defaults.import_evaluations,
        )
