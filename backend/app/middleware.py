"""Enforce browser origins and byte limits before parsing uploads."""

from collections import deque
from time import monotonic

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.settings import Settings


class BodyTooLarge(Exception):
    pass


class LocalRequestsMiddleware:
    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.mutations: deque[float] = deque()
        self.uploads = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if origin is not None and origin not in self.settings.origins:
            await JSONResponse(
                {
                    "error": {
                        "code": "origin_not_allowed",
                        "message": "Browser origin is not allowed.",
                    }
                },
                403,
            )(scope, receive, send)
            return
        upload = scope["method"] == "POST" and scope["path"] == "/api/recordings"
        if scope["method"] in ("POST", "DELETE"):
            now = monotonic()
            while self.mutations and self.mutations[0] <= now - 60:
                self.mutations.popleft()
            if len(self.mutations) >= self.settings.requests_per_minute:
                await JSONResponse(
                    {
                        "error": {
                            "code": "rate_limited",
                            "message": "Request rate exceeded; retry later.",
                        }
                    },
                    429,
                    headers={"Retry-After": "60"},
                )(scope, receive, send)
                return
            self.mutations.append(now)
        if upload and self.uploads >= self.settings.concurrent_uploads:
            await JSONResponse(
                {
                    "error": {
                        "code": "uploads_busy",
                        "message": "Upload slots are busy; retry later.",
                    }
                },
                429,
            )(scope, receive, send)
            return
        limit = (
            self.settings.max_upload_bytes + 65536 if scope["path"] == "/api/recordings" else 16384
        )
        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            length = limit + 1
        size = 0
        started = False

        async def bounded_receive() -> Message:
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                raise BodyTooLarge
            return message

        async def track_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            if upload:
                self.uploads += 1
            if length < 0 or length > limit:
                raise BodyTooLarge
            await self.app(scope, bounded_receive, track_send)
        except BodyTooLarge:
            if started:
                raise
            await JSONResponse(
                {
                    "error": {
                        "code": "upload_too_large",
                        "message": "Request exceeds the byte limit.",
                    }
                },
                413,
            )(scope, receive, send)
        finally:
            if upload:
                self.uploads -= 1
