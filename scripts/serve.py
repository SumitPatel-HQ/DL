"""Launch the API on loopback. The host cannot be overridden by environment."""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000)
