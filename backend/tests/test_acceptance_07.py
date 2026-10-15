"""Ticket 07 consolidated acceptance run across the seams.

Corrupt audio, Silence, Unknown words, and reconnects in one user journey
over recorded analysis plus a live session. The seams are covered in depth
by test_recordings.py and test_live.py; this file asserts the whole matrix
holds together end to end. Fixtures are deliberately biased plumbing, never
model-quality evidence.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_live import LIVE, start, until, until_decision
from test_recordings import completed, configured, open_client, wav_bytes

from app.settings import Settings
from ml.features import LABELS


def test_acceptance_corrupt_audio_silence_and_keyword_rejection(tmp_path: Path) -> None:
    with open_client(configured(tmp_path, label=LABELS.index("silence"))) as client:
        rejected = client.post("/api/recordings", files={"file": ("bad.wav", b"not WAV")})
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "invalid_wav"
        assert client.get("/api/recordings").json() == []

        recording = client.post(
            "/api/recordings", files={"file": ("silence.wav", wav_bytes())}
        ).json()["id"]
        unsupported = client.post(
            f"/api/recordings/{recording}/analyze", json={"keywords": ["play"]}
        )
        assert unsupported.status_code == 422
        assert unsupported.json()["error"]["code"] == "unsupported_keyword"

        assert (
            client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        ).status_code == 202
        assert completed(client, recording)["status"] == "completed"
        result = client.get(f"/api/recordings/{recording}/detections").json()
        assert result["detections"] == []
        assert result["code"] == "no_detections"

        assert client.delete(f"/api/recordings/{recording}").status_code == 204
        assert client.get("/api/recordings").json() == []


def test_acceptance_missing_checkpoint_stays_explicit(tmp_path: Path) -> None:
    settings = Settings(storage=tmp_path / "store", checkpoint=tmp_path / "missing.pt")
    with open_client(settings) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/models/active").status_code == 503
        recording = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()[
            "id"
        ]
        result = client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "checkpoint_not_installed"


def test_acceptance_live_rejects_then_recovers_after_disconnect(tmp_path: Path) -> None:
    with open_client(configured(tmp_path, label=LABELS.index("go"))) as client:
        actions = client.get("/api/actions").json()
        assert [item["keyword"] for item in actions["actions"]] == [
            "right",
            "left",
            "go",
            "stop",
            "up",
            "down",
        ]
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start(version=2))
            error = socket.receive_json()
            assert (error["type"], error["code"]) == ("error", "unsupported_version")
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()

        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            decision = until_decision(socket)
            assert (decision["label"], decision["accepted"], decision["action"]) == (
                "go",
                True,
                "play",
            )
            socket.send_json(
                {"type": "action_result", "decision_id": decision["id"], "status": "applied"}
            )
            socket.send_json({"type": "stop"})
            until(socket, "stopped")

        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")


def test_acceptance_unknown_word_is_rejected_without_action(tmp_path: Path) -> None:
    client: TestClient
    with open_client(configured(tmp_path, label=LABELS.index("unknown"))) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start(sample_rate=8000))
            until(socket, "ready")
            decision = until_decision(socket, 8000)
            assert (decision["label"], decision["accepted"], decision["reason"]) == (
                "unknown",
                False,
                "unknown",
            )
            assert decision["action"] is None
