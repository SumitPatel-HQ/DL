"""Capture hardware and installed versions without usernames or speech content."""

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    node = shutil.which("node")
    info = {
        "captured_at": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "python": platform.python_version(),
        "node": subprocess.check_output([node, "--version"], text=True).strip() if node else None,
        "packages": {
            name: version(name)
            for name in (
                "fastapi",
                "uvicorn",
                "torch",
                "numpy",
                "pytest",
                "ruff",
                "mypy",
                "psutil",
            )
        },
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "note": "Environment snapshot only; no training or inference benchmark has been run.",
    }
    target = ROOT / "artifacts" / "system" / "local.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
