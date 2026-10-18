"""ONNX copy of a torch checkpoint, for serving without torch.

The exported graph maps normalized [batch, 98, 40] MFCC windows to softmax
probabilities. Class order, normalization and checkpoint provenance travel in
the model's metadata, so the file is as self-describing as the checkpoint.
`sha256` stays the source checkpoint's hash: detector files and Recordings
keep referring to the same model.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ml.features import FEATURE_CONFIG, FeaturePipeline, Normalization
from ml.features.pipeline import N_MFCC
from ml.vocabulary import validate_labels

ONNX_FORMAT = "kws-lstm-onnx-v1"
WINDOW_FRAMES = 98
METADATA_KEY = "kws"


@dataclass(frozen=True)
class OnnxModel:
    session: ort.InferenceSession
    labels: tuple[str, ...]
    normalization: Normalization
    metadata: dict[str, Any]
    path: Path
    sha256: str

    def pipeline(self) -> FeaturePipeline:
        return FeaturePipeline(self.normalization)

    def probabilities(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Softmax scores for [batch, frames, 40] normalized features. Not accuracy."""
        (scores,) = self.session.run(
            ["probabilities"], {"features": np.ascontiguousarray(features, dtype=np.float32)}
        )
        return np.asarray(scores, dtype=np.float32)


def describe(document: dict[str, Any]) -> tuple[tuple[str, ...], Normalization]:
    if document.get("format") != ONNX_FORMAT or document.get("features") != FEATURE_CONFIG:
        raise ValueError("Unsupported ONNX model metadata")
    labels = tuple(document["labels"])
    validate_labels(labels)
    values = dict(document["normalization"])
    values["mean"], values["std"] = tuple(values["mean"]), tuple(values["std"])
    values["labels"] = tuple(values["labels"])
    if values["labels"] != labels:
        raise ValueError("ONNX model and normalization class order disagree")
    return labels, Normalization(**values)


def load_onnx(path: Path) -> OnnxModel:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    raw = session.get_modelmeta().custom_metadata_map.get(METADATA_KEY)
    if raw is None:
        raise ValueError("ONNX model has no keyword spotting metadata")
    document = json.loads(raw)
    labels, normalization = describe(document)
    return OnnxModel(
        session,
        labels,
        normalization,
        document["checkpoint_metadata"],
        path,
        document["checkpoint_sha256"],
    )


def metadata_document(checkpoint: Path) -> dict[str, Any]:
    from ml.checkpoint import load_checkpoint

    loaded = load_checkpoint(checkpoint)
    return {
        "format": ONNX_FORMAT,
        "labels": list(loaded.labels),
        "features": FEATURE_CONFIG,
        "normalization": asdict(loaded.normalization),
        "checkpoint_sha256": loaded.sha256,
        "checkpoint_metadata": {
            key: value
            for key, value in loaded.metadata.items()
            if key not in ("state_dict", "normalization")
        },
        "window_frames": WINDOW_FRAMES,
    }


def export_onnx(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Write `output` from a torch checkpoint and return its metadata document."""
    import onnx
    import torch

    from ml.checkpoint import load_checkpoint

    loaded = load_checkpoint(checkpoint)
    document = metadata_document(checkpoint)

    class Probabilities(torch.nn.Module):
        def __init__(self, network: torch.nn.Module) -> None:
            super().__init__()
            self.network = network

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.network(features), dim=1)

    module = Probabilities(loaded.model).eval()
    # Unidirectional LSTM with default zero initial states: any batch size is exact.
    example = torch.zeros(2, WINDOW_FRAMES, N_MFCC)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (example,),
        str(output),
        input_names=["features"],
        output_names=["probabilities"],
        dynamic_axes={"features": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    model = onnx.load(str(output))
    entry = model.metadata_props.add()
    entry.key, entry.value = METADATA_KEY, json.dumps(document, sort_keys=True)
    onnx.save(model, str(output))
    return document
