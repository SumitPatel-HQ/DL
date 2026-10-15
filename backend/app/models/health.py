from typing import Literal

from pydantic import BaseModel

from app.constants import MODEL_SETUP_ACTION


class UnavailableModel(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    code: Literal["checkpoint_not_installed", "model_incompatible", "worker_unavailable"] = (
        "checkpoint_not_installed"
    )
    setup_action: str = MODEL_SETUP_ACTION


class ReadyModel(BaseModel):
    status: Literal["ready"] = "ready"
    code: Literal["model_ready"] = "model_ready"
    id: str
    experimental: Literal[True] = True


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: UnavailableModel | ReadyModel
