"""Live voice session: decisions, streaming hygiene and the WebSocket contract.

The checkpoint fixture is deliberately biased to one label. It exercises
plumbing and policy, never speech accuracy.
"""

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_recordings import configured, open_client

from app.services.live import (
    KEYWORD_ACTIONS,
    LiveDecider,
    LivePolicy,
    LiveStream,
)
from app.settings import Settings
from ml.features import LABELS

LIVE = "ws://127.0.0.1/api/live"
POLICY = LivePolicy(
    hop_s=0.25, threshold=0.7, margin=0.0, smooth_windows=1, min_consecutive=2, cooldown_s=1.2
)


def scores(label: str, score: float) -> list[float]:
    rest = (1 - score) / (len(LABELS) - 1)
    return [score if name == label else rest for name in LABELS]


def run(decider: LiveDecider, windows: list[tuple[str, float]], start: float = 1.0) -> list[Any]:
    decisions = []
    for i, (label, score) in enumerate(windows):
        decisions += decider.observe(start + i * POLICY.hop_s, scores(label, score))
    return decisions


def test_actions_are_a_strict_allowlist_of_in_app_actions() -> None:
    assert {mapping.keyword: mapping.action for mapping in KEYWORD_ACTIONS} == {
        "right": "next_slide",
        "left": "previous_slide",
        "go": "play",
        "stop": "pause",
        "up": "volume_up",
        "down": "volume_down",
    }


def test_one_utterance_causes_at_most_one_action() -> None:
    decisions = run(LiveDecider(POLICY), [("go", 0.9)] * 6)
    assert [(d.label, d.accepted, d.reason, d.action) for d in decisions] == [
        ("go", True, "accepted", "play")
    ]


def test_smoothing_requires_consecutive_confident_windows() -> None:
    decisions = run(LiveDecider(POLICY), [("go", 0.9), ("silence", 0.99)])
    assert [(d.label, d.accepted, d.reason) for d in decisions] == [
        ("go", False, "low_confidence"),
        ("silence", False, "silence"),
    ]
    decisions = run(LiveDecider(POLICY), [("up", 0.6), ("up", 0.65), ("up", 0.5), ("unknown", 0.8)])
    assert [(d.label, d.reason) for d in decisions] == [
        ("up", "low_confidence"),
        ("unknown", "unknown"),
    ]
    assert decisions[0].score == pytest.approx(0.65)


def test_cooldown_rejects_a_second_utterance_until_it_expires() -> None:
    decider = LiveDecider(POLICY)
    first = run(decider, [("go", 0.9), ("go", 0.9), ("silence", 0.9)], start=1.0)
    second = run(decider, [("stop", 0.9), ("stop", 0.9), ("silence", 0.9)], start=1.75)
    third = run(decider, [("stop", 0.9), ("stop", 0.9)], start=2.5)
    accepted = [(d.label, d.reason) for d in [*first, *second, *third] if d.label != "silence"]
    assert accepted == [("go", "accepted"), ("stop", "cooldown"), ("stop", "accepted")]
    assert third[-1].action == "pause"


def test_unknown_and_silence_are_rejected_once_per_run() -> None:
    decisions = run(LiveDecider(POLICY), [("unknown", 0.99)] * 4 + [("silence", 0.99)] * 4)
    assert [(d.label, d.accepted, d.reason, d.action) for d in decisions] == [
        ("unknown", False, "unknown", None),
        ("silence", False, "silence", None),
    ]


def test_custom_non_action_labels_emit_their_label_as_reason_for_frontend() -> None:
    labels = ("go", "yes", "unknown", "silence")
    decider = LiveDecider(POLICY, labels)
    by_index = {name: i for i, name in enumerate(labels)}

    def custom(label: str, score: float) -> list[float]:
        rest = (1 - score) / (len(labels) - 1)
        return [score if name == label else rest for name in labels]

    decisions = []
    for i, (label, score) in enumerate([("yes", 0.95)] * 3):
        decisions += decider.observe(1.0 + i * POLICY.hop_s, custom(label, score))
    assert [(d.label, d.accepted, d.reason, d.action) for d in decisions] == [
        ("yes", False, "yes", None)
    ]
    # Frontend contract: rejected decisions carry action None and a non-empty
    # reason that is never "accepted"; accepted decisions carry reason "accepted".
    for d in decisions:
        assert d.action is None and not d.accepted
        assert isinstance(d.reason, str) and d.reason and d.reason != "accepted"
    assert by_index["yes"] == 1


def test_reset_ends_the_utterance_but_keeps_cooldown() -> None:
    decider = LiveDecider(POLICY)
    run(decider, [("go", 0.9), ("go", 0.9)], start=1.0)
    decider.reset()
    decisions = run(decider, [("go", 0.9), ("go", 0.9)], start=1.5)
    assert [(d.reason, d.action) for d in decisions] == [("cooldown", None)]


def pcm(seconds: float, rate: int = 16000) -> bytes:
    """A -20 dBFS tone: loud enough to pass the live RMS gate."""
    t = np.arange(int(seconds * rate)) / rate
    return (3000 * np.sin(2 * np.pi * 440 * t)).astype("<i2").tobytes()


def push(stream: LiveStream, data: bytes) -> None:
    for first in range(0, len(data), 4096):
        stream.push(data[first : first + 4096])


def test_stream_emits_one_window_per_hop_at_16_khz() -> None:
    stream = LiveStream(48000, POLICY)
    push(stream, pcm(0.9, 48000))
    assert stream.take_windows().ends_s == []
    push(stream, pcm(0.6, 48000))
    batch = stream.take_windows()
    assert batch.ends_s == [1.0, 1.25, 1.5]
    assert all(window.shape == (16000,) and window.dtype == np.float32 for window in batch.windows)
    assert batch.dropped_s == 0


def test_stream_drops_stale_audio_instead_of_queueing_it() -> None:
    stream = LiveStream(16000, LivePolicy(hop_s=0.25, max_lag_s=1.0))
    push(stream, pcm(4.0))
    batch = stream.take_windows()
    assert batch.ends_s == [4.0]
    assert batch.dropped_s == pytest.approx(3.0)
    assert stream.retained_seconds() <= 1.0 + 1.0


def test_stream_rejects_odd_and_oversized_frames() -> None:
    stream = LiveStream(16000, POLICY)
    with pytest.raises(ValueError):
        stream.push(b"\0")
    with pytest.raises(ValueError):
        stream.push(b"\0" * 40000)


def start(**overrides: Any) -> dict[str, Any]:
    return {
        "type": "start",
        "version": 1,
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "pcm_s16le",
        **overrides,
    }


def until(socket: Any, kind: str, limit: int = 200) -> dict[str, Any]:
    for _ in range(limit):
        message: dict[str, Any] = socket.receive_json()
        if message["type"] == kind:
            return message
    raise AssertionError(f"No {kind} message")


def send(socket: Any, data: bytes) -> None:
    for first in range(0, len(data), 4096):
        socket.send_bytes(data[first : first + 4096])


def until_decision(socket: Any, rate: int = 16000, hops: int = 8) -> dict[str, Any]:
    """Send audio no faster than it is scored, as a live mic would, until a decision."""
    send(socket, pcm(1.0, rate))
    for _ in range(hops):
        while (message := socket.receive_json())["type"] != "frame":
            if message["type"] == "decision":
                return message
        send(socket, pcm(0.25, rate))
    raise AssertionError("No decision")


def test_accepted_keyword_drives_one_action_and_stores_no_audio(tmp_path: Path) -> None:
    settings = configured(tmp_path, label=LABELS.index("go"))
    with open_client(settings) as client:
        actions = client.get("/api/actions").json()
        assert [c["keyword"] for c in actions["actions"]] == [c.keyword for c in KEYWORD_ACTIONS]
        assert actions["rejected_labels"] == ["unknown", "silence"]
        legacy = client.get("/api/commands").json()
        assert [c["keyword"] for c in legacy["commands"]] == [c.keyword for c in KEYWORD_ACTIONS]
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            ready = until(socket, "ready")
            assert ready["keywords"] == ["left", "right", "up", "down", "go", "stop"]
            assert [c["keyword"] for c in ready["actions"]] == [c.keyword for c in KEYWORD_ACTIONS]
            decision = until_decision(socket)
            assert decision["label"] == "go"
            assert decision["accepted"] is True
            assert decision["action"] == "play"
            assert decision["score"] > 0.99
            socket.send_json(
                {"type": "action_result", "decision_id": decision["id"], "status": "applied"}
            )
            socket.send_json({"type": "stop"})
            messages = []
            while (message := socket.receive_json())["type"] != "stopped":
                messages.append(message)
            assert not [m for m in messages if m["type"] == "decision" and m["accepted"]]
    with sqlite3.connect(settings.storage / "metadata.sqlite3") as db:
        rows = db.execute(
            "SELECT keyword,accepted,action,action_status FROM live_events"
        ).fetchall()
    assert rows == [("go", 1, "play", "applied")]
    assert list((settings.storage / "audio").iterdir()) == []


def test_unknown_input_is_rejected_without_action(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("message", "code"),
    [
        (start(version=2), "unsupported_version"),
        (start(channels=2), "unsupported_audio"),
        (start(sample_rate=4000), "unsupported_audio"),
        (start(encoding="float32"), "unsupported_audio"),
        ({"type": "audio"}, "invalid_message"),
    ],
)
def test_invalid_handshakes_close_with_a_stable_code(
    tmp_path: Path, message: dict[str, Any], code: str
) -> None:
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(message)
            error = socket.receive_json()
            assert (error["type"], error["code"]) == ("error", code)
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()


def test_oversized_frame_closes_the_session(tmp_path: Path) -> None:
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            socket.send_bytes(b"\0" * 40000)
            assert until(socket, "error")["code"] == "invalid_frame"


def test_missing_model_and_busy_session_are_distinct(tmp_path: Path) -> None:
    missing = Settings(storage=tmp_path / "missing", checkpoint=tmp_path / "missing.pt")
    with open_client(missing) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            assert until(socket, "error")["code"] == "checkpoint_not_installed"
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as first:
            first.send_json(start())
            until(first, "ready")
            with client.websocket_connect(LIVE) as second:
                second.send_json(start())
                assert until(second, "error")["code"] == "live_busy"
            first.send_json({"type": "stop"})
            until(first, "stopped")
        with client.websocket_connect(LIVE) as again:
            again.send_json(start())
            until(again, "ready")


def test_disallowed_origin_cannot_open_a_session(tmp_path: Path) -> None:
    with open_client(configured(tmp_path)) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                LIVE, headers={"Origin": "http://evil.example"}
            ) as socket:
                socket.receive_json()


def test_client_disconnect_releases_the_session(tmp_path: Path) -> None:
    client: TestClient
    with open_client(configured(tmp_path)) as client:
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")
            send(socket, pcm(0.5))
        with client.websocket_connect(LIVE) as socket:
            socket.send_json(start())
            until(socket, "ready")



def test_margin_rejects_keywords_that_barely_beat_unknown() -> None:
    policy = LivePolicy(threshold=0.5, margin=0.3, smooth_windows=1, min_consecutive=2)
    near = [0.0] * len(LABELS)
    near[LABELS.index("go")], near[LABELS.index("unknown")] = 0.55, 0.45
    decider = LiveDecider(policy)
    decisions = [d for i in range(4) for d in decider.observe(1.0 + i * 0.25, near)]
    assert not [d for d in decisions if d.accepted]


def test_margin_counts_other_keywords_like_the_recording_detector() -> None:
    policy = LivePolicy(threshold=0.4, margin=0.3, smooth_windows=1, min_consecutive=2)
    close = [0.0] * len(LABELS)
    close[LABELS.index("go")], close[LABELS.index("stop")] = 0.5, 0.45
    close[LABELS.index("unknown")] = 0.05
    decider = LiveDecider(policy)
    decisions = [d for i in range(4) for d in decider.observe(1.0 + i * 0.25, close)]
    assert not [d for d in decisions if d.accepted]


def test_smoothing_ignores_a_single_confident_spike() -> None:
    policy = LivePolicy(threshold=0.7, margin=0.0, smooth_windows=2, min_consecutive=2)
    decider = LiveDecider(policy)
    rows = [scores("unknown", 0.9), scores("go", 0.95), scores("unknown", 0.9), scores("unknown", 0.9)]
    decisions = [d for i, row in enumerate(rows) for d in decider.observe(1.0 + i * 0.25, row)]
    assert not [d for d in decisions if d.accepted]


def test_quiet_windows_are_silence_without_running_the_model(tmp_path: Path) -> None:
    from app.services.live import LiveService

    class Loaded:
        labels = LABELS

        def pipeline(self) -> Any:
            raise AssertionError("model must not run on gated windows")

    service = LiveService.__new__(LiveService)
    service.policy = LivePolicy()
    service.recordings = type("R", (), {"loaded": Loaded()})()  # type: ignore[assignment]
    rows = service.score([np.full(16000, 1e-4, dtype=np.float32)])
    assert LABELS[int(np.argmax(rows[0]))] == "silence"
