"""Session-scoped consent and audit policy, separate from inference and OS adapters."""

import asyncio
import json
import platform
import time
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

import psutil

from app.db.store import Store
from app.services.live import Decision

TARGET_ACTIONS = {
    "volume": {"volume_up", "volume_down"},
    "powerpoint": {"next_slide", "previous_slide"},
    "media": {"play", "pause"},
}


class ExternalExecutor(Protocol):
    supported: bool

    async def discover(self) -> list[dict[str, Any]]: ...
    async def execute(self, target: str, action: str) -> str: ...
    def close(self) -> None: ...


def hardware() -> dict[str, Any]:
    cpu = platform.processor()
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    ram = psutil.virtual_memory().total
    return {
        "os": platform.system(),
        "cpu": cpu,
        "ram_bytes": ram,
        "target_hardware": platform.system() == "Windows"
        and "Ryzen 5 5500U" in cpu
        and 14 * 1024**3 <= ram <= 17 * 1024**3,
    }


class ExternalSession:
    def __init__(
        self, store: Store, session: str, executor: ExternalExecutor, provenance: dict[str, Any]
    ) -> None:
        self.store, self.session, self.executor = store, session, executor
        self.targets: dict[str, dict[str, Any]] = {}
        self.target: str | None = None
        self.allowed: set[str] = set()
        self.generation = 0
        self.negative: dict[str, Any] | None = None
        self.last_end: float | None = None
        self.provenance = provenance
        self.executing = False

    def log(
        self,
        kind: str,
        status: str,
        reason: str,
        action: str | None = None,
        event: int | None = None,
        target_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO external_action_logs "
                "(session_id,event_id,kind,target_id,action,status,reason,details_json) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.session,
                    event,
                    kind,
                    target_id or self.target,
                    action,
                    status,
                    reason,
                    json.dumps(details) if details is not None else None,
                ),
            )

    def state(self, error: str | None = None) -> dict[str, Any]:
        return {
            "type": "external_state",
            "supported": self.executor.supported,
            "enabled": self.target is not None,
            "target_id": self.target,
            "allowed_actions": sorted(self.allowed),
            "targets": list(self.targets.values()),
            "experimental": True,
            "error": error,
            "negative_running": self.negative is not None,
        }

    async def control(self, message: dict[str, Any]) -> dict[str, Any]:
        kind = message.get("type")
        try:
            if kind == "external_targets" and set(message) == {"type"}:
                self.revoke("targets_refreshed")
                targets = await self.executor.discover()
                self.targets = {t["id"]: t for t in targets}
                self.log("discovery", "accepted", "targets_listed")
            elif kind == "external_grant" and set(message) == {
                "type",
                "target_id",
                "actions",
                "acknowledge_experimental",
            }:
                # A failed permission replacement revokes the previous grant too.
                self.target = None
                self.allowed.clear()
                self.generation += 1
                self.finish_negative("permission_changed")
                target, actions = message["target_id"], message["actions"]
                if (
                    not self.executor.supported
                    or not isinstance(target, str)
                    or target not in self.targets
                    or message["acknowledge_experimental"] is not True
                    or not isinstance(actions, list)
                    or not actions
                    or not all(isinstance(a, str) for a in actions)
                ):
                    raise ValueError
                info = self.targets[target]
                if not set(actions) <= set(info["actions"]) & TARGET_ACTIONS[info["kind"]]:
                    raise ValueError
                self.target, self.allowed = target, set(actions)
                self.log(
                    "consent",
                    "accepted",
                    "experimental_opt_in",
                    details={
                        "target": info,
                        "allowed_actions": sorted(self.allowed),
                        "provenance": self.provenance,
                    },
                )
            elif kind == "external_revoke" and set(message) == {"type"}:
                self.revoke("user_revoked")
            elif kind == "negative_start" and set(message) == {"type", "no_keywords_confirmed"}:
                if (
                    self.target is None
                    or self.executing
                    or self.negative is not None
                    or message["no_keywords_confirmed"] is not True
                ):
                    raise ValueError
                self.negative = {
                    "id": uuid4().hex,
                    "session_id": self.session,
                    "started_at": datetime.now(UTC).isoformat(),
                    "hardware": hardware(),
                    "target": self.targets[self.target],
                    "allowed_actions": sorted(self.allowed),
                    "accepted_decisions": 0,
                    "external_attempts": 0,
                    "scored_audio_s": 0.0,
                    "clock_start": time.monotonic(),
                    "provenance": self.provenance,
                }
                self.generation += 1
                self.last_end = None
                self.log("negative_session", "accepted", "started")
            elif kind == "negative_finish" and set(message) == {"type"}:
                report = self.finish_negative("finished")
                if report is None:
                    raise ValueError
                return {"type": "negative_report", "report": report}
            else:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            self.log("control", "rejected", "invalid_external_request")
            return self.state("invalid_external_request")
        except (OSError, TimeoutError):
            self.revoke("executor_unavailable")
            return self.state("executor_unavailable")
        return self.state()

    def scored(self, end: float, window_s: float) -> None:
        if self.negative is not None:
            # Exclude the first overlapping window, which can predate test start.
            if self.last_end is not None:
                self.negative["scored_audio_s"] += min(window_s, max(0.0, end - self.last_end))
            self.last_end = end

    async def dispatch(self, decision: Decision, event: int) -> dict[str, str] | None:
        if self.negative is not None and decision.accepted:
            self.negative["accepted_decisions"] += 1
        if not decision.accepted:
            self.log("decision", "rejected", decision.reason, event=event)
            return None
        if self.target is None:
            self.log("decision", "in_app", "external_disabled", decision.action, event)
            return None
        target, generation = self.target, self.generation
        if decision.action not in self.allowed:
            self.log("action", "rejected", "action_not_permitted", decision.action, event)
            return {"status": "rejected", "reason": "action_not_permitted", "target_id": target}
        self.log("action", "attempted", "permitted", decision.action, event)
        if self.negative is not None:
            self.negative["external_attempts"] += 1
        self.executing = True
        try:
            status = await self.executor.execute(target, decision.action)
        except (OSError, TimeoutError):
            status = "outcome_unknown"
        except asyncio.CancelledError:
            self.log(
                "action",
                "outcome_unknown",
                "session_stopped_during_execution",
                decision.action,
                event,
                target,
            )
            raise
        finally:
            self.executing = False
        self.log("action", status, "native_result", decision.action, event, target)
        if generation == self.generation and status in ("failed", "outcome_unknown"):
            self.revoke("execution_failed")
        return {"status": status, "reason": "native_result", "target_id": target}

    def finish_negative(self, reason: str) -> dict[str, Any] | None:
        report, self.negative = self.negative, None
        if report is None:
            return None
        elapsed = time.monotonic() - report.pop("clock_start")
        exposure = min(elapsed, report["scored_audio_s"])
        false_actions = report["accepted_decisions"]
        report.update(
            {
                "ended_at": datetime.now(UTC).isoformat(),
                "end_reason": reason,
                "elapsed_s": elapsed,
                "negative_exposure_s": exposure,
                "false_actions": false_actions,
                "false_actions_per_30_min": false_actions * 1800 / exposure if exposure else None,
                "verdict": "pass"
                if reason == "finished"
                and exposure >= 1800
                and false_actions * 1800 / exposure <= 1
                and report["hardware"]["target_hardware"]
                else "not_passed",
                "annotation": "operator-confirmed no Keywords; not independently verified",
                "model_quality_gate": "unmet; this report cannot authorize model promotion",
            }
        )
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO negative_session_reports (id,document_json) VALUES (?,?)",
                (report["id"], json.dumps(report)),
            )
        self.log("negative_session", "completed", reason)
        return report

    def revoke(self, reason: str) -> None:
        self.generation += 1
        target = self.target
        self.target = None
        self.allowed.clear()
        self.targets.clear()
        self.executor.close()
        # Clearing consent and killing the worker must not depend on a healthy database.
        self.finish_negative(reason)
        self.log("consent", "revoked", reason, target_id=target)
