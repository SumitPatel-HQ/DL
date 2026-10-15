from typing import Literal

from pydantic import BaseModel


class ModelHealth(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    code: Literal["checkpoint_not_installed"] = "checkpoint_not_installed"
    setup_action: str = (
        "Complete ticket 3 to train and install a compatible checkpoint with "
        "class labels and feature metadata in artifacts/models/. "
        "Model loading is not implemented in this scaffold."
    )


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model: ModelHealth
