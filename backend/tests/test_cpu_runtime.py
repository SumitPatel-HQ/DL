"""Installation gate, not a trained-model quality measurement."""

import torch


def test_tensor_computation_uses_cpu_only_build() -> None:
    assert torch.version.cuda is None
    assert not torch.cuda.is_available()
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    result = values @ torch.tensor([2.0, 1.0], device="cpu")
    assert result.device.type == "cpu"
    assert result.tolist() == [4.0, 10.0]
