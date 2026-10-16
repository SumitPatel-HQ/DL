"""Fixed Windows adapters in a disposable process. Never launch apps or run a shell.

Native handles stay in the worker. The browser sees session-local random IDs only.
Revocation terminates the worker, including any blocked COM/WinRT operation.
"""

import asyncio
import multiprocessing as mp
import sys
from multiprocessing.process import BaseProcess
from pathlib import PureWindowsPath
from typing import Any, Protocol
from uuid import uuid4


class ExecutorPipe(Protocol):
    def send(self, obj: Any) -> None: ...
    def recv(self) -> Any: ...
    def poll(self, timeout: float = 0.0) -> bool: ...
    def close(self) -> None: ...


class WindowsExecutor:
    supported = sys.platform == "win32"

    def __init__(self) -> None:
        self.process: BaseProcess | None = None
        self.pipe: ExecutorPipe | None = None
        self.lock = asyncio.Lock()

    async def discover(self) -> list[dict[str, Any]]:
        if not self.supported:
            return []
        result: list[dict[str, Any]] = await self._request({"type": "discover"})
        return result

    async def execute(self, target: str, action: str) -> str:
        result: str = await self._request({"type": "execute", "target": target, "action": action})
        return result

    async def _request(self, message: dict[str, str]) -> Any:
        async with self.lock:
            if self.process is None:
                context = mp.get_context("spawn")
                self.pipe, child = context.Pipe()
                self.process = context.Process(target=worker, args=(child,), daemon=True)
                self.process.start()
                child.close()
            pipe, process = self.pipe, self.process
            assert pipe is not None and process is not None
            try:
                pipe.send(message)
                deadline = asyncio.get_running_loop().time() + 8
                while not pipe.poll():
                    if self.process is not process or not process.is_alive():
                        raise OSError("Executor was revoked or stopped")
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("Native operation timed out; outcome may be unknown")
                    await asyncio.sleep(0.01)
                response = pipe.recv()
                if response["error"]:
                    raise OSError("Native target unavailable or action failed")
                return response["result"]
            except (OSError, EOFError, TimeoutError, asyncio.CancelledError):
                self.close()
                raise

    def close(self) -> None:
        process, self.process = self.process, None
        pipe, self.pipe = self.pipe, None
        if process is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=0.2)
            if not process.is_alive():
                process.close()
        if pipe is not None:
            pipe.close()


class WindowsTargets:
    def __init__(self) -> None:
        self.targets: dict[str, tuple[str, Any]] = {}

    def add(self, kind: str, label: str, native: Any, actions: list[str]) -> dict[str, Any]:
        token = uuid4().hex
        self.targets[token] = (kind, native)
        return {"id": token, "kind": kind, "label": label, "actions": actions}

    async def discover(self) -> list[dict[str, Any]]:
        # Windows-only packages are imported here, never in inference or on launch.
        from comtypes.client import GetActiveObject  # type: ignore[import-untyped]
        from pycaw.pycaw import AudioUtilities  # type: ignore[import-untyped]
        from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager

        self.targets.clear()
        result = []
        try:
            device = AudioUtilities.GetSpeakers()
            if device is not None:
                result.append(
                    self.add(
                        "volume",
                        f"Windows volume: {device.FriendlyName}",
                        device,
                        ["volume_up", "volume_down"],
                    )
                )
        except Exception:
            pass  # A missing audio endpoint must not hide other available targets.
        try:
            app = GetActiveObject("PowerPoint.Application", dynamic=True)
            for index in range(1, app.SlideShowWindows.Count + 1):
                show = app.SlideShowWindows.Item(index)
                result.append(
                    self.add(
                        "powerpoint",
                        f"PowerPoint: {show.Presentation.Name}",
                        show,
                        ["next_slide", "previous_slide"],
                    )
                )
        except Exception:
            pass  # Attach only. Never create an application or start a slideshow.
        try:
            manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
            for session in manager.get_sessions():
                controls = session.get_playback_info().controls
                actions = []
                if controls.is_play_enabled:
                    actions.append("play")
                if controls.is_pause_enabled:
                    actions.append("pause")
                if actions:
                    result.append(
                        self.add(
                            "media",
                            f"Media: {PureWindowsPath(session.source_app_user_model_id).name}",
                            session,
                            actions,
                        )
                    )
        except Exception:
            pass
        return result

    async def execute(self, token: str, action: str) -> str:
        kind, native = self.targets[token]
        if kind == "volume" and action in ("volume_up", "volume_down"):
            # Retain the selected endpoint, even if the Windows default changes.
            volume = native.EndpointVolume
            before = float(volume.GetMasterVolumeLevelScalar())
            after = max(0.0, min(1.0, before + (0.05 if action == "volume_up" else -0.05)))
            volume.SetMasterVolumeLevelScalar(after, None)
            return "no_change" if before == after else "applied"
        if kind == "powerpoint" and action in ("next_slide", "previous_slide"):
            # Retain the selected slideshow object. Never switch to the active window.
            view = native.View
            before = int(view.Slide.SlideIndex)
            destination = before + (1 if action == "next_slide" else -1)
            if not 1 <= destination <= native.Presentation.Slides.Count:
                return "no_change"
            view.GotoSlide(destination)
            return "applied"
        if kind == "media" and action in ("play", "pause"):
            # Retain this exact media session; no global media key or current-session fallback.
            controls = native.get_playback_info().controls
            if action == "play" and controls.is_play_enabled:
                return "applied" if await native.try_play_async() else "failed"
            if action == "pause" and controls.is_pause_enabled:
                return "applied" if await native.try_pause_async() else "failed"
        raise ValueError("Action is not available on the selected target")


def worker(pipe: ExecutorPipe) -> None:
    # A single loop keeps native COM/WinRT objects on their owning thread.
    import comtypes  # type: ignore[import-untyped]

    comtypes.CoInitialize()
    loop = asyncio.new_event_loop()
    targets = WindowsTargets()
    try:
        while True:
            try:
                request = pipe.recv()
            except EOFError:
                return
            try:
                result: Any
                if request["type"] == "discover":
                    result = loop.run_until_complete(targets.discover())
                else:
                    result = loop.run_until_complete(
                        targets.execute(request["target"], request["action"])
                    )
                pipe.send({"error": False, "result": result})
            except Exception:
                pipe.send({"error": True, "result": None})
    finally:
        loop.close()
        comtypes.CoUninitialize()
        pipe.close()
