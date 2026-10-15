from fastapi import APIRouter

from app.models.health import Health, ModelHealth

router = APIRouter(prefix="/api")


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health(model=ModelHealth())
