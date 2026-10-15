from fastapi.testclient import TestClient

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
            "setup_action": (
                "Complete ticket 3 to train and install a compatible checkpoint with "
                "class labels and feature metadata in artifacts/models/. "
                "Model loading is not implemented in this scaffold."
            ),
        },
    }


def test_health_rejects_non_loopback_hosts() -> None:
    with TestClient(app, base_url="http://attacker.example") as client:
        response = client.get("/api/health")

    assert response.status_code == 400
