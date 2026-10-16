"""Versioned live mic WebSocket and the action map. Loopback, explicit start only.

The browser opens this socket only after the user presses Start. The Stop
button (control message), close, or disconnect ends the session at once;
saying the Keyword "stop" triggers the pause action only and never ends
the session. Nothing keeps listening afterwards.
"""

import asyncio
import contextlib
import json
import sqlite3
import time
from typing import Annotated, Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, Query, Request, WebSocket
from starlette.websockets import WebSocketState

from app.services.external_actions import ExternalSession
from app.services.live import (
    ENCODING,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    PROTOCOL_VERSION,
    Decision,
    LiveDecider,
    LiveService,
    LiveStream,
)
from app.services.windows_actions import WindowsExecutor
from ml.spotting import KEYWORDS

router = APIRouter(prefix="/api")
HANDSHAKE_S = 5.0


def service(request: Request) -> LiveService:
    result: LiveService = request.app.state.live
    return result


@router.get("/actions")
def actions(live: Annotated[LiveService, Depends(service)]) -> dict[str, Any]:
    return live.action_map()


@router.get("/commands", include_in_schema=False)
def commands_legacy(live: Annotated[LiveService, Depends(service)]) -> dict[str, Any]:
    """Deprecated alias for GET /api/actions. Kept for backward compatibility."""
    return live.command_map()


@router.get("/external-actions/logs")
def external_logs(
    live: Annotated[LiveService, Depends(service)], limit: Annotated[int, Query(ge=1, le=500)] = 100
) -> dict[str, Any]:
    with live.recordings.store.transaction() as db:
        rows = db.execute(
            "SELECT * FROM external_action_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"logs": [dict(row) for row in rows]}


@router.get("/external-actions/negative-reports")
def negative_reports(live: Annotated[LiveService, Depends(service)]) -> dict[str, Any]:
    with live.recordings.store.transaction() as db:
        rows = db.execute(
            "SELECT document_json FROM negative_session_reports ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    return {"reports": [json.loads(row[0]) for row in rows]}


class SessionError(Exception):
    def __init__(self, code: str, message: str, close: int) -> None:
        self.code = code
        self.message = message
        self.close = close
        super().__init__(code)


def handshake(message: Any) -> int:
    if not isinstance(message, dict) or message.get("type") != "start":
        raise SessionError("invalid_message", "Send a start message first.", 1008)
    if message.get("version") != PROTOCOL_VERSION:
        raise SessionError("unsupported_version", "Reload the page to match the protocol.", 1008)
    rate = message.get("sample_rate")
    if (
        message.get("encoding") != ENCODING
        or message.get("channels") != 1
        or not isinstance(rate, int)
        or isinstance(rate, bool)
        or not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE
    ):
        raise SessionError("unsupported_audio", "Send mono PCM_s16le at 8-96 kHz.", 1003)
    return rate


def decision_message(decision: Decision, event: int, arrived: float) -> dict[str, Any]:
    return {
        "type": "decision",
        "id": event,
        "t_s": round(decision.t_s, 3),
        "label": decision.label,
        "score": decision.score,
        "accepted": decision.accepted,
        "reason": decision.reason,
        "action": decision.action,
        "processing_ms": round((time.monotonic() - arrived) * 1000, 1),
    }


async def receive_text(socket: WebSocket) -> Any:
    message = await socket.receive()
    if message["type"] == "websocket.disconnect":
        raise SessionError("disconnected", "Client disconnected.", 1000)
    if message.get("text") is None:
        raise SessionError("invalid_message", "Expected a JSON control message.", 1008)
    try:
        return json.loads(message["text"])
    except json.JSONDecodeError as exc:
        raise SessionError("invalid_message", "Expected a JSON control message.", 1008) from exc


@router.websocket("/live")
async def live(socket: WebSocket) -> None:
    live: LiveService = socket.app.state.live
    origin = socket.headers.get("origin")
    if origin is not None and origin not in live.recordings.settings.origins:
        await socket.close(code=1008)
        return
    await socket.accept()
    acquired = False
    external: ExternalSession | None = None
    processor: asyncio.Task[None] | None = None
    try:
        try:
            rate = handshake(await asyncio.wait_for(receive_text(socket), HANDSHAKE_S))
        except TimeoutError as exc:
            raise SessionError("invalid_message", "Start message timed out.", 1008) from exc
        unavailable = live.unavailable()
        if unavailable is not None:
            raise SessionError(unavailable, "The active model is unavailable.", 1011)
        if not live.acquire():
            raise SessionError("live_busy", "Another live session is running.", 1013)
        acquired = True
        session = uuid4().hex
        external = ExternalSession(
            live.recordings.store,
            session,
            WindowsExecutor(),
            {
                "checkpoint_sha256": live.recordings.loaded.sha256
                if live.recordings.loaded
                else None,
                "live_policy": live.policy.to_dict(),
            },
        )
        await socket.send_json(
            {
                "type": "ready",
                "version": PROTOCOL_VERSION,
                "session_id": session,
                "keywords": list(KEYWORDS),
                "actions": live.action_map()["actions"],
                "commands": live.command_map()["commands"],
                "policy": live.policy.to_dict(),
            }
        )
        await socket.send_json(external.state())
        stream = LiveStream(rate, live.policy)
        decider = LiveDecider(live.policy, live.labels())
        wake = asyncio.Event()
        arrived = time.monotonic()

        async def process() -> None:
            labels = live.labels()
            while True:
                await wake.wait()
                wake.clear()
                batch = stream.take_windows()
                generation = external.generation
                if batch.dropped_s:
                    # Stale audio is dropped, never queued; the utterance restarts.
                    decider.reset()
                    await socket.send_json({"type": "backpressure", "dropped_s": batch.dropped_s})
                if not batch.windows:
                    continue
                scores = await asyncio.to_thread(live.score, batch.windows)
                for end, row in zip(batch.ends_s, scores, strict=True):
                    if generation != external.generation:
                        # Audio captured before a permission change cannot trigger the new target.
                        decider.reset()
                        break
                    external.scored(end, live.policy.window_s)
                    top = int(np.argmax(row))
                    await socket.send_json(
                        {
                            "type": "frame",
                            "t_s": round(end, 3),
                            "label": labels[top],
                            "score": float(row[top]),
                        }
                    )
                    for decision in decider.observe(end, row):
                        if generation != external.generation:
                            decider.reset()
                            break
                        event = live.record(session, decision)
                        outcome = await external.dispatch(decision, event)
                        payload = decision_message(decision, event, arrived)
                        if outcome is not None:
                            payload["external"] = outcome
                            with live.recordings.store.transaction() as db:
                                db.execute(
                                    "UPDATE live_events SET action_status=? WHERE id=?",
                                    ("external_" + outcome["status"], event),
                                )
                            await socket.send_json(external.state())
                        await socket.send_json(payload)

        processor = asyncio.create_task(process())
        while True:
            receiving = asyncio.ensure_future(socket.receive())
            done, _ = await asyncio.wait(
                {receiving, processor}, return_when=asyncio.FIRST_COMPLETED
            )
            if processor in done:
                receiving.cancel()
                processor.result()
                return
            message = receiving.result()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                try:
                    stream.push(message["bytes"])
                except ValueError as exc:
                    raise SessionError("invalid_frame", str(exc), 1009) from exc
                arrived = time.monotonic()
                wake.set()
                continue
            try:
                control = json.loads(message.get("text") or "")
            except json.JSONDecodeError:
                control = None
            kind = control.get("type") if isinstance(control, dict) else None
            if isinstance(kind, str) and (
                kind.startswith("external_") or kind.startswith("negative_")
            ):
                generation = external.generation
                response = await external.control(control)
                if generation != external.generation:
                    stream.discard()
                    decider.reset()
                await socket.send_json(response)
                continue
            if kind == "stop":
                external.revoke("session_stopped")
                processor.cancel()
                live.release()
                acquired = False
                await socket.send_json({"type": "stopped"})
                await socket.close(code=1000)
                return
            if kind == "action_result" and isinstance(control, dict):
                if live.acknowledge(session, control.get("decision_id"), control.get("status")):
                    continue
            external.log("control", "rejected", "invalid_message")
            raise SessionError("invalid_message", "Unexpected control message.", 1008)
    except SessionError as exc:
        if exc.code != "disconnected" and socket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await socket.send_json({"type": "error", "code": exc.code, "message": exc.message})
                await socket.close(code=exc.close)
    except Exception:
        if socket.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await socket.send_json(
                    {"type": "error", "code": "live_failed", "message": "Live scoring failed."}
                )
                await socket.close(code=1011)
    finally:
        if external is not None:
            with contextlib.suppress(sqlite3.Error, OSError):
                external.revoke("session_closed")
        if processor is not None:
            processor.cancel()
            with contextlib.suppress(BaseException):
                await processor
        if acquired:
            live.release()
