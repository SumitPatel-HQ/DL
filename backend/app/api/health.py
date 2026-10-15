from typing import Any

from fastapi import APIRouter

from app.api.recordings import Service
from app.models.health import Health

router = APIRouter(prefix="/api")


@router.get("/health", response_model=Health)
def health(jobs: Service) -> dict[str, Any]:
    return {"status": "ok", "model": jobs.model_health()}
