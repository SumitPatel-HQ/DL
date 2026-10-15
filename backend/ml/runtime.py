"""Reproducibility and measurement helpers for CPU experiments."""

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int, num_threads: int) -> None:
    """Fixed seeds, deterministic kernels and a fixed thread count for repeatable CPU sums."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def peak_rss_bytes() -> int:
    """Process peak resident memory: peak working set on Windows, ru_maxrss elsewhere."""
    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def code_revision() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}
