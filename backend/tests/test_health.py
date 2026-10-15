import pytest
from fastapi.testclient import TestClient

from app.constants import MODEL_SETUP_ACTION
from app.main import app


def test_health_distinguishes_running_api_from_unavailable_model() -> None:
    with TestClient(app, base_url="http://127.0.0.1") as client:
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


@pytest.mark.parametrize("host", ["localhost", "[::1]", "attacker.example"])
def test_health_rejects_unsupported_host_headers(host: str) -> None:
    """Exercise HTTP Host filtering, not the server's listening socket."""
    with TestClient(app, base_url=f"http://{host}:8000") as client:
        response = client.get("/api/health")

    assert response.status_code == 400
