"""Live API action consent. Native OS calls are replaced at the executor boundary."""

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest
from test_live import LIVE, pcm, send, start, until, until_decision
from test_recordings import configured, open_client


class Desktop:
    supported = True

    def __init__(self) -> None:
        self.executed: list[tuple[str, str]] = []

    async def discover(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "player-1",
                "kind": "media",
                "label": "Test player",
                "actions": ["play", "pause"],
            }
        ]

    async def execute(self, target: str, action: str) -> str:
        self.executed.append((target, action))
        return "applied"

    def close(self) -> None:
        pass


def enable(socket: Any, actions: list[str] | None = None) -> dict[str, Any]:
    socket.send_json({"type": "external_targets"})
    while not until(socket, "external_state")["targets"]:
        pass
    socket.send_json(
        {
            "type": "external_grant",
            "target_id": "player-1",
            "actions": actions or ["play"],
            "acknowledge_experimental": True,
        }
    )
    return until(socket, "external_state")


def test_launch_and_microphone_start_do_not_enable_external_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop = Desktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            socket.send_json({"type": "external_targets"})
            state = until(socket, "external_state")
            assert state["enabled"] is False
            decision = until_decision(socket)
            assert decision["action"] == "play"
            assert decision.get("external") is None
            assert desktop.executed == []


def test_permitted_voice_action_executes_once_and_audit_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop = Desktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    settings = configured(tmp_path)
    with open_client(settings) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            assert enable(socket)["enabled"] is True
            decision = until_decision(socket)
            assert decision["external"]["status"] == "applied"
            assert desktop.executed == [("player-1", "play")]
            socket.send_json({"type": "stop"})
            until(socket, "stopped")
        logs = client.get("/api/external-actions/logs").json()["logs"]
        results = [row for row in logs if row["kind"] == "action"]
        assert [(row["status"], row["action"]) for row in results] == [
            ("applied", "play"),
            ("attempted", "play"),
        ]
    with open_client(settings) as client:
        assert client.get("/api/external-actions/logs").json()["logs"] == logs
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            assert until(socket, "external_state")["enabled"] is False


def test_unpermitted_action_is_rejected_without_native_or_demo_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop = Desktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            enable(socket, ["pause"])
            decision = until_decision(socket)
            assert decision["external"]["status"] == "rejected"
            assert decision["external"]["reason"] == "action_not_permitted"
            assert desktop.executed == []


@pytest.mark.parametrize(
    "payload",
    [
        {"target_id": "C:\\Windows\\System32\\cmd.exe"},
        {"actions": ["shell"]},
        {"actions": ["play", {"command": "calc.exe"}]},
        {"acknowledge_experimental": False},
        {"command": "calc.exe"},
        {"target_id": []},
    ],
)
def test_permission_injection_is_rejected_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    desktop = Desktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            until(socket, "external_state")
            socket.send_json({"type": "external_targets"})
            until(socket, "external_state")
            socket.send_json(
                {
                    "type": "external_grant",
                    "target_id": "player-1",
                    "actions": ["play"],
                    "acknowledge_experimental": True,
                    **payload,
                }
            )
            state = until(socket, "external_state")
            assert state["enabled"] is False
            assert state["error"] == "invalid_external_request"
            assert desktop.executed == []
        logs = client.get("/api/external-actions/logs").json()["logs"]
        assert any(row["status"] == "rejected" for row in logs)
        assert "calc.exe" not in str(logs) and "cmd.exe" not in str(logs)


def test_revoke_clears_targets_and_permission_before_more_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop = Desktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            enable(socket)
            socket.send_json({"type": "external_revoke"})
            state = until(socket, "external_state")
            assert not state["enabled"] and state["targets"] == []
            assert until_decision(socket).get("external") is None
            assert desktop.executed == []


def test_short_negative_session_counts_false_actions_and_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desktop = Desktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            enable(socket)
            socket.send_json({"type": "negative_start", "no_keywords_confirmed": True})
            assert until(socket, "external_state")["negative_running"] is True
            until_decision(socket)
            socket.send_json({"type": "negative_finish"})
            report = until(socket, "negative_report")["report"]
            assert report["false_actions"] == 1
            assert report["external_attempts"] == 1
            assert report["negative_exposure_s"] < 1800
            assert report["verdict"] == "not_passed"
        assert client.get("/api/external-actions/negative-reports").json()["reports"] == [report]


def test_revoke_interrupts_pending_native_action_and_preserves_unknown_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DelayedDesktop(Desktop):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.closed = False

        async def discover(self) -> list[dict[str, Any]]:
            self.closed = False
            return await super().discover()

        async def execute(self, target: str, action: str) -> str:
            self.entered.set()
            while not self.closed:
                await asyncio.sleep(0.01)
            raise OSError("Native worker stopped")

        def close(self) -> None:
            self.closed = True

    desktop = DelayedDesktop()
    monkeypatch.setattr("app.api.live.WindowsExecutor", lambda: desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            enable(socket)
            send(socket, pcm(1.0))
            until(socket, "frame")
            send(socket, pcm(0.25))
            assert desktop.entered.wait(5)
            socket.send_json({"type": "external_revoke"})
            assert until(socket, "external_state")["enabled"] is False
            assert until(socket, "decision")["external"]["status"] == "outcome_unknown"
        logs = client.get("/api/external-actions/logs").json()["logs"]
        outcome = next(row for row in logs if row["status"] == "outcome_unknown")
        assert outcome["target_id"] == "player-1"


def test_stop_saves_incomplete_negative_report_without_passing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.live.WindowsExecutor", Desktop)
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            enable(socket)
            socket.send_json({"type": "negative_start", "no_keywords_confirmed": True})
            until(socket, "external_state")
            socket.send_json({"type": "stop"})
            until(socket, "stopped")
        report = client.get("/api/external-actions/negative-reports").json()["reports"][0]
        assert report["end_reason"] == "session_stopped"
        assert report["verdict"] == "not_passed"
        assert report["false_actions"] == 0
        assert len(report["provenance"]["checkpoint_sha256"]) == 64


def test_schema_upgrade_preserves_imported_evaluation_runs(tmp_path: Path) -> None:
    settings = configured(tmp_path)
    with open_client(settings) as client:
        assert client.post("/api/evaluations/import").status_code == 200
        runs = client.get("/api/evaluations").json()
        assert runs["runs"]
    with sqlite3.connect(settings.storage / "metadata.sqlite3") as db:
        db.execute("PRAGMA user_version=1")
        db.execute("DROP TABLE external_action_logs")
        db.execute("DROP TABLE negative_session_reports")
    with open_client(settings) as client:
        assert client.get("/api/evaluations").json() == runs
        assert client.get("/api/external-actions/logs").json() == {"logs": []}
