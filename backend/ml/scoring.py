"""What scoring callers need from a loaded model, without importing a runtime.

`ml.checkpoint.LoadedModel` (torch) and `ml.onnx_model.OnnxModel` (onnxruntime)
both satisfy it, so the API can serve either one.
"""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ml.features import FeaturePipeline, Normalization


class ScoringModel(Protocol):
    @property
    def labels(self) -> tuple[str, ...]: ...

    @property
    def normalization(self) -> Normalization: ...

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def path(self) -> Path: ...

    @property
    def sha256(self) -> str:
        """SHA-256 of the torch checkpoint the weights came from."""
        ...

    def pipeline(self) -> FeaturePipeline: ...

    def probabilities(self, features: NDArray[np.float32]) -> NDArray[np.float32]: ...


def load_model(path: Path) -> ScoringModel:
    """`.onnx` loads with onnxruntime; anything else is a torch checkpoint."""
    if path.suffix == ".onnx":
        from ml.onnx_model import load_onnx

        return load_onnx(path)
    from ml.checkpoint import load_checkpoint

    return load_checkpoint(path)
