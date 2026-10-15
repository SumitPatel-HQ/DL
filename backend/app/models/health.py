from typing import Literal

from pydantic import BaseModel

from app.constants import MODEL_SETUP_ACTION


class ModelHealth(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    code: Literal["checkpoint_not_installed"] = "checkpoint_not_installed"
    setup_action: str = MODEL_SETUP_ACTION


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: ModelHealth
