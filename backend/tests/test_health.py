from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.constants import MODEL_SETUP_ACTION
from app.main import create_app
from app.settings import Settings


def test_health_distinguishes_running_api_from_unavailable_model(tmp_path: Path) -> None:
    isolated = create_app(Settings(storage=tmp_path, checkpoint=tmp_path / "missing.pt"))
    with TestClient(isolated, base_url="http://127.0.0.1") as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": {
            "status": "unavailable",
            "code": "checkpoint_not_installed",
            "setup_action": MODEL_SETUP_ACTION,
        },
    }


@pytest.mark.parametrize("host", ["[::1]", "attacker.example"])
def test_health_rejects_unsupported_host_headers(host: str, tmp_path: Path) -> None:
    """Exercise HTTP Host filtering, not the server's listening socket."""
    isolated = create_app(Settings(storage=tmp_path, checkpoint=tmp_path / "missing.pt"))
    with TestClient(isolated, base_url=f"http://{host}:8000") as client:
        response = client.get("/api/health")

    assert response.status_code == 400


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_health_accepts_loopback_hosts(host: str, tmp_path: Path) -> None:
    isolated = create_app(Settings(storage=tmp_path, checkpoint=tmp_path / "missing.pt"))
    with TestClient(isolated, base_url=f"http://{host}:8000") as client:
        response = client.get("/api/health")

    assert response.status_code == 200
