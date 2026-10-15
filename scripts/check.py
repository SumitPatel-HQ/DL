"""Run the same checks locally and in CI from any working directory."""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = [
        (
            ["uv", "run", "--locked", "ruff", "check", ".", "../scripts"],
            ROOT / "backend",
        ),
        (
            ["uv", "run", "--locked", "ruff", "format", "--check", ".", "../scripts"],
            ROOT / "backend",
        ),
        (["uv", "run", "--locked", "mypy"], ROOT / "backend"),
        (["npm", "run", "lint"], ROOT / "frontend"),
        (["npm", "run", "typecheck"], ROOT / "frontend"),
        (["uv", "run", "--locked", "pytest"], ROOT / "backend"),
    ]
    for command, directory in commands:
        executable = shutil.which(command[0])
        if executable is None:
            raise SystemExit(f"Install {command[0]} before running checks.")
        print(f"Running {' '.join(command)}", flush=True)
        subprocess.run([executable, *command[1:]], cwd=directory, check=True)


if __name__ == "__main__":
    main()
