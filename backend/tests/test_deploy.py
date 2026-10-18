"""Deployed mode: ONNX serving, inline analysis and same-origin checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import DEPLOY, Settings
from ml.checkpoint import load_checkpoint
from ml.onnx_model import export_onnx, load_onnx
from tests.test_recordings import configured, wav_bytes

ROOT = Path(__file__).resolve().parents[2]


def exported(tmp_path: Path) -> Settings:
    settings = configured(tmp_path)
    onnx = tmp_path / "fixture.onnx"
    export_onnx(settings.checkpoint, onnx)
    return Settings(
        storage=tmp_path / "store",
        checkpoint=onnx,
        detector=settings.detector,
        evaluations=tmp_path / "runs",
        inline_analysis=True,
        same_origin=True,
        allowed_hosts=("*",),
    )


def test_onnx_export_matches_the_checkpoint_at_any_batch_size(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    export_onnx(settings.checkpoint, tmp_path / "m.onnx")
    torch_model, onnx_model = load_checkpoint(settings.checkpoint), load_onnx(tmp_path / "m.onnx")
    assert onnx_model.sha256 == torch_model.sha256
    assert onnx_model.labels == torch_model.labels
    assert onnx_model.normalization == torch_model.normalization
    rng = np.random.default_rng(0)
    for batch in (1, 3, 40):
        features = rng.standard_normal((batch, 98, 40)).astype(np.float32)
        np.testing.assert_allclose(
            onnx_model.probabilities(features), torch_model.probabilities(features), atol=1e-5
        )


def test_deployed_analysis_finishes_inside_the_request(tmp_path: Path) -> None:
    with TestClient(create_app(exported(tmp_path)), base_url="https://kws.example") as client:
        assert client.get("/api/health").json()["model"]["status"] == "ready"
        same = {"Origin": "https://kws.example"}
        files = {"file": ("x.wav", wav_bytes())}
        upload = client.post("/api/recordings", files=files, headers=same)
        recording = upload.json()["id"]
        analyzed = client.post(
            f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]}, headers=same
        )
        assert analyzed.json()["status"] == "completed"
        assert client.get(f"/api/recordings/{recording}/detections").json()["detections"]
        foreign = client.delete(
            f"/api/recordings/{recording}", headers={"Origin": "https://other.example"}
        )
        assert foreign.status_code == 403


def test_local_settings_still_reject_other_hosts_and_origins() -> None:
    settings = Settings()
    assert not settings.same_origin and not settings.inline_analysis
    assert not settings.origin_allowed("https://kws.example", ("kws.example",))
    assert settings.origin_allowed("http://localhost:3000", ("localhost:8000",))


def test_committed_deploy_artifacts_serve_without_torch(tmp_path: Path) -> None:
    """backend/deploy must match its detector and load in a process that never imports torch."""
    if not (DEPLOY / "model.onnx").is_file():
        pytest.skip("Run scripts/export_deploy.py to create deploy artifacts")
    detector = json.loads((DEPLOY / "detector.json").read_text(encoding="utf-8"))
    browser = json.loads(
        (ROOT / "frontend" / "public" / "models" / "kws.json").read_text(encoding="utf-8")
    )
    assert browser["checkpoint_sha256"] == detector["checkpoint"]["sha256"]
    code = "\n".join(
        (
            "import sys",
            "from fastapi.testclient import TestClient",
            "from app.main import create_app",
            "with TestClient(create_app()) as client:",
            "    assert client.get('/api/health').json()['model']['status'] == 'ready'",
            "    assert client.get('/api/evaluations').json()['runs']",
            "assert 'torch' not in sys.modules",
        )
    )
    environment = {**os.environ, "KWS_DEPLOYED": "1", "KWS_STORAGE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / "backend",
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
