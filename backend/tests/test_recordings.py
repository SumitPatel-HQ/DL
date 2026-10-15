"""API fixtures exercise real checkpoint inference, never model-quality evidence."""

import asyncio
import io
import json
import sqlite3
import time
import wave
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.types import Message

from app.main import create_app
from app.settings import Settings
from ml.checkpoint import save_checkpoint
from ml.features import Normalization
from ml.networks import LstmClassifier


def wav_bytes(seconds: float = 2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * int(seconds * 16000))
    return output.getvalue()


def configured(tmp_path: Path, label: int = 4, **overrides: object) -> Settings:
    model = LstmClassifier(8, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[label] = 10
    checkpoint = tmp_path / "fixture.pt"
    digest = save_checkpoint(
        checkpoint,
        model,
        Normalization((0.0,) * 40, (1.0,) * 40, 1, "fixture", "train", "a" * 64),
        {
            "experiment_id": "fixture",
            "network": {
                "type": "lstm",
                "pooling": model.pooling,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
        },
    )
    detector = tmp_path / "detector.json"
    detector.write_text(
        json.dumps(
            {
                "format": "kws-detector-v1",
                "checkpoint": {"sha256": digest},
                "config": {
                    "stride_s": 0.25,
                    "threshold": 0.5,
                    "min_consecutive": 2,
                    "suppression_s": 1.0,
                    "window_s": 1.0,
                },
            }
        )
    )
    return Settings(
        storage=tmp_path / "store", checkpoint=checkpoint, detector=detector, **overrides
    )


def completed(client: TestClient, recording: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get(f"/api/recordings/{recording}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("Analysis did not finish")


def test_upload_analyze_playback_persist_and_delete(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/health").json()["model"]["status"] == "ready"
        active = client.get("/api/models/active").json()
        assert active["keywords"] == ["left", "right", "up", "down", "go", "stop"]
        assert active["experimental"] is True
        audio = wav_bytes()
        response = client.post("/api/recordings", files={"file": ("go.wav", audio)})
        assert response.status_code == 201
        recording = response.json()["id"]
        response = client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        assert response.status_code == 202
        assert completed(client, recording)["status"] == "completed"
        result = client.get(f"/api/recordings/{recording}/detections").json()
        assert result["code"] == "detections_found"
        assert result["experimental"] is True
        assert result["model_id"] == active["id"]
        (detection,) = result["detections"]
        assert detection["keyword"] == "go"
        assert (detection["start_s"], detection["end_s"]) == (0, 2)
        assert detection["confidence"] > 0.99
        assert detection["threshold"] == 0.5
        playback = client.get(f"/api/recordings/{recording}/audio", headers={"Range": "bytes=0-43"})
        assert playback.status_code == 206
        assert playback.content == audio[:44]
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert len(client.get(f"/api/recordings/{recording}/detections").json()["detections"]) == 1
        assert client.delete(f"/api/recordings/{recording}").status_code == 204
        for suffix in ("", "/detections", "/audio"):
            assert client.get(f"/api/recordings/{recording}{suffix}").status_code == 404
        assert list((settings.storage / "audio").iterdir()) == []


@pytest.mark.parametrize(
    ("filename", "body", "code", "status"),
    [
        ("bad.wav", b"not WAV", "invalid_wav", 400),
        ("truncated.wav", wav_bytes()[:-2], "invalid_wav", 400),
        ("long.wav", wav_bytes(3), "recording_too_long", 413),
        ("../escape.wav", wav_bytes(), "invalid_filename", 422),
        ("cmd.exe;calc.wav", wav_bytes(), "invalid_filename", 422),
    ],
    ids=["bad-wav", "truncated", "too-long", "path", "command"],
)
def test_invalid_uploads_leave_no_recordings_or_bytes(
    tmp_path: Path,
    filename: str,
    body: bytes,
    code: str,
    status: int,
) -> None:
    settings = configured(tmp_path, max_duration_s=2)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        result = client.post("/api/recordings", files={"file": (filename, body)})
        assert result.status_code == status
        assert result.json()["error"]["code"] == code
        assert client.get("/api/recordings").json() == []
        assert list((settings.storage / "audio").iterdir()) == []


def test_no_detections_and_keyword_and_command_rejection(tmp_path: Path) -> None:
    with TestClient(
        create_app(configured(tmp_path, label=7)), base_url="http://127.0.0.1"
    ) as client:
        recording = client.post(
            "/api/recordings", files={"file": ("silence.wav", wav_bytes())}
        ).json()["id"]
        for body, code in [
            ({"keywords": ["play"]}, "unsupported_keyword"),
            ({"keywords": ["C:/Windows/cmd.exe"]}, "unsupported_keyword"),
            ({"keywords": ["go"], "command": "calc.exe"}, "invalid_request"),
        ]:
            result = client.post(f"/api/recordings/{recording}/analyze", json=body)
            assert result.status_code == 422
            assert result.json()["error"]["code"] == code
        client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        assert completed(client, recording)["status"] == "completed"
        result = client.get(f"/api/recordings/{recording}/detections").json()
        assert result["detections"] == []
        assert result["code"] == "no_detections"


def test_missing_checkpoint_is_distinct_from_running_service(tmp_path: Path) -> None:
    settings = Settings(storage=tmp_path / "store", checkpoint=tmp_path / "missing.pt")
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/models/active").status_code == 503
        recording = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()[
            "id"
        ]
        result = client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "checkpoint_not_installed"
        assert client.get(f"/api/recordings/{recording}").json()["status"] == "uploaded"


def test_byte_limits_origin_and_extra_upload_fields(tmp_path: Path) -> None:
    settings = configured(tmp_path, max_upload_bytes=128)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        result = client.post("/api/recordings", files={"file": ("big.wav", wav_bytes())})
        assert result.status_code == 413
        assert result.json()["error"]["code"] == "upload_too_large"
        result = client.post("/api/recordings", files={"file": ("big.wav", b"x" * 100000)})
        assert result.status_code == 413
        assert result.json()["error"]["code"] == "upload_too_large"
        result = client.post(
            "/api/recordings",
            files={"file": ("go.wav", wav_bytes(0.001))},
            data={"command": "calc.exe"},
        )
        assert result.status_code == 422
        result = client.get("/api/health", headers={"Origin": "https://attacker.example"})
        assert result.status_code == 403
        assert result.json()["error"]["code"] == "origin_not_allowed"
        assert list((settings.storage / "audio").iterdir()) == []


def test_chunked_oversize_upload_is_rejected_and_cleaned(tmp_path: Path) -> None:
    settings = configured(tmp_path, max_upload_bytes=128)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:

        def chunks():
            yield (
                b"--boundary\r\nContent-Disposition: form-data; "
                b'name="file"; filename="go.wav"\r\n\r\n'
            )
            for _ in range(20):
                yield b"x" * 10000
            yield b"\r\n--boundary--\r\n"

        result = client.post(
            "/api/recordings",
            content=chunks(),
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )
        assert result.status_code == 413
        assert result.json()["error"]["code"] == "upload_too_large"
        assert client.get("/api/recordings").json() == []


def test_worker_failure_does_not_block_next_recording(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        broken = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()[
            "id"
        ]
        # Filesystem is an external boundary: simulate source loss after upload.
        (settings.storage / "audio" / f"{broken}.wav").unlink()
        client.post(f"/api/recordings/{broken}/analyze", json={"keywords": ["go"]})
        failed = completed(client, broken)
        assert failed["status"] == "failed"
        assert failed["error_code"] == "analysis_failed"
        good = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()["id"]
        client.post(f"/api/recordings/{good}/analyze", json={"keywords": ["go"]})
        assert completed(client, good)["status"] == "completed"


def test_one_worker_queue_limit_shutdown_and_restart(tmp_path: Path) -> None:
    settings = configured(tmp_path, queue_capacity=1)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        ids = [
            client.post("/api/recordings", files={"file": ("go.wav", wav_bytes(600))}).json()["id"]
            for _ in range(3)
        ]
        assert (
            client.post(f"/api/recordings/{ids[0]}/analyze", json={"keywords": ["go"]}).status_code
            == 202
        )
        deadline = time.monotonic() + 5
        while client.get(f"/api/recordings/{ids[0]}").json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert (
            client.post(f"/api/recordings/{ids[1]}/analyze", json={"keywords": ["go"]}).status_code
            == 202
        )
        result = client.post(f"/api/recordings/{ids[2]}/analyze", json={"keywords": ["go"]})
        assert result.status_code == 429
        assert result.json()["error"]["code"] == "queue_full"
        states = [client.get(f"/api/recordings/{key}").json()["status"] for key in ids]
        assert states == ["running", "queued", "uploaded"]
        assert (
            client.post(f"/api/recordings/{ids[0]}/analyze", json={"keywords": ["go"]}).status_code
            == 409
        )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        for key in ids[:2]:
            job = client.get(f"/api/recordings/{key}").json()
            assert job["status"] == "failed"
            assert job["error_code"] == "service_interrupted"
        for key in ids:
            assert client.delete(f"/api/recordings/{key}").status_code == 204


def test_delete_running_recording_cannot_restore_results(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        recording = client.post(
            "/api/recordings", files={"file": ("go.wav", wav_bytes(600))}
        ).json()["id"]
        client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        assert client.delete(f"/api/recordings/{recording}").status_code == 204
        good = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()["id"]
        client.post(f"/api/recordings/{good}/analyze", json={"keywords": ["go"]})
        assert completed(client, good)["status"] == "completed"
        assert client.get(f"/api/recordings/{recording}/detections").status_code == 404


def test_upload_disconnect_cleans_partial_data(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    application = create_app(settings)
    messages: list[Message] = []

    async def disconnected_upload() -> None:
        incoming = iter(
            [
                {
                    "type": "http.request",
                    "more_body": True,
                    "body": (
                        b"--boundary\r\nContent-Disposition: form-data; "
                        b'name="file"; filename="go.wav"\r\n\r\n' + b"x" * 1200000
                    ),
                },
                {"type": "http.disconnect"},
            ]
        )

        async def receive() -> Message:
            return next(incoming)

        async def send(message: Message) -> None:
            messages.append(message)

        await application(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/recordings",
                "query_string": b"",
                "root_path": "",
                "headers": [
                    (b"host", b"127.0.0.1"),
                    (b"content-type", b"multipart/form-data; boundary=boundary"),
                ],
                "client": ("127.0.0.1", 123),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        )

    with TestClient(application, base_url="http://127.0.0.1") as client:
        asyncio.run(disconnected_upload())
        assert messages[0]["status"] == 400
        assert b"upload_disconnected" in messages[1]["body"]
        assert client.get("/api/recordings").json() == []
        assert list((settings.storage / "audio").iterdir()) == []


def test_mutation_rate_limit_does_not_block_health(tmp_path: Path) -> None:
    settings = configured(tmp_path, requests_per_minute=2)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        for _ in range(2):
            assert (
                client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).status_code
                == 201
            )
        result = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())})
        assert result.status_code == 429
        assert result.json()["error"]["code"] == "rate_limited"
        assert client.get("/api/health").status_code == 200


def test_incompatible_model_and_detector_do_not_advertise_readiness(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    settings.detector.write_text('{"format":"kws-detector-v1","checkpoint":{"sha256":"wrong"}}')
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get("/api/health").json()["model"]["code"] == "model_incompatible"
        assert client.get("/api/models/active").status_code == 503


def test_database_delete_failure_preserves_playable_audio(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        audio = wav_bytes()
        recording = client.post("/api/recordings", files={"file": ("go.wav", audio)}).json()["id"]
        # External SQLite fault: this deferred constraint fails at COMMIT, after DELETE runs.
        with sqlite3.connect(settings.storage / "metadata.sqlite3") as db:
            db.executescript("""
                CREATE TABLE delete_fault (owner TEXT REFERENCES recordings(id)
                    DEFERRABLE INITIALLY DEFERRED);
                CREATE TRIGGER fail_delete AFTER DELETE ON recordings BEGIN
                    INSERT INTO delete_fault VALUES ('missing');
                END;
            """)
        result = client.delete(f"/api/recordings/{recording}")
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "persistence_error"
        assert client.get(f"/api/recordings/{recording}/audio").content == audio
        with sqlite3.connect(settings.storage / "metadata.sqlite3") as db:
            db.execute("DROP TRIGGER fail_delete")
        assert client.delete(f"/api/recordings/{recording}").status_code == 204


def test_staged_delete_failure_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        recording = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()[
            "id"
        ]
        unlink = Path.unlink

        def unavailable(path: Path, missing_ok: bool = False) -> None:
            if path.suffix == ".deleting":
                raise PermissionError("test filesystem fault")
            unlink(path, missing_ok=missing_ok)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", unavailable)
            result = client.delete(f"/api/recordings/{recording}")
            assert result.status_code == 409
            assert result.json()["error"]["code"] == "delete_failed"
        assert client.delete(f"/api/recordings/{recording}").status_code == 204
        assert list((settings.storage / "audio").iterdir()) == []


def test_restart_restores_uncommitted_staged_delete_and_removes_orphans(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        audio = wav_bytes()
        recording = client.post("/api/recordings", files={"file": ("go.wav", audio)}).json()["id"]
    path = settings.storage / "audio" / f"{recording}.wav"
    path.replace(path.with_suffix(".wav.deleting"))
    orphan = settings.storage / "audio" / "orphan.wav.deleting"
    orphan.write_bytes(b"uncommitted upload or committed delete")
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        assert client.get(f"/api/recordings/{recording}/audio").content == audio
        assert not orphan.exists()
        assert client.delete(f"/api/recordings/{recording}").status_code == 204


def test_status_write_failure_is_visible_and_worker_processes_next_job(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        recording = client.post("/api/recordings", files={"file": ("go.wav", wav_bytes())}).json()[
            "id"
        ]
        with sqlite3.connect(settings.storage / "metadata.sqlite3") as db:
            db.executescript("""
                CREATE TRIGGER fail_status BEFORE UPDATE OF status ON recordings
                WHEN NEW.status IN ('running','failed') BEGIN
                    SELECT RAISE(ABORT, 'temporary test storage fault');
                END;
            """)
        assert (
            client.post(
                f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]}
            ).status_code
            == 202
        )
        deadline = time.monotonic() + 5
        while client.get("/api/health").json()["model"]["status"] != "unavailable":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        rejected = client.post(f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]})
        assert rejected.status_code == 503
        assert rejected.json()["error"]["code"] == "worker_unavailable"
        with sqlite3.connect(settings.storage / "metadata.sqlite3") as db:
            db.execute("DROP TRIGGER fail_status")
        deadline = time.monotonic() + 5
        while client.get("/api/health").json()["model"]["status"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert completed(client, recording)["error_code"] == "persistence_error"
        assert (
            client.post(
                f"/api/recordings/{recording}/analyze", json={"keywords": ["go"]}
            ).status_code
            == 202
        )
        assert completed(client, recording)["status"] == "completed"
