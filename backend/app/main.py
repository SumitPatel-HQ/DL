import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from app.api.evaluations import router as evaluations_router
from app.api.health import router as health_router
from app.api.live import router as live_router
from app.api.recordings import router as recordings_router
from app.errors import ServiceError
from app.middleware import LocalRequestsMiddleware
from app.services.evaluations import EvaluationService
from app.services.live import LiveService
from app.services.recordings import RecordingService
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        jobs = RecordingService(config)
        application.state.recordings = jobs
        application.state.evaluations = EvaluationService(jobs.store, config.evaluations)
        application.state.live = LiveService(jobs)
        try:
            yield
        finally:
            jobs.close()

    application = FastAPI(title="Keyword Spotting", version="0.1.0", lifespan=lifespan)
    application.add_middleware(LocalRequestsMiddleware, settings=config)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Range"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, exc.status)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "Request does not match the API contract.",
                }
            },
            422,
        )

    @application.exception_handler(OSError)
    async def storage_error(request: Request, exc: OSError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "storage_error", "message": "Local storage operation failed."}}, 503
        )

    @application.exception_handler(sqlite3.Error)
    async def database_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": "persistence_error", "message": "Metadata operation failed."}}, 503
        )

    @application.get("/health", include_in_schema=False)
    def health_redirect() -> RedirectResponse:
        return RedirectResponse(url="/api/health")

    application.include_router(health_router)
    application.include_router(recordings_router)
    application.include_router(evaluations_router)
    application.include_router(live_router)
    return application


app = create_app()
