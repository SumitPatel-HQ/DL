"""Write the committed artifacts a Vercel deployment serves.

Rerun after changing the active checkpoint or committed Evaluation Runs:

    uv run --project backend --extra train python scripts/export_deploy.py

backend/deploy/ holds the ONNX model, its detector and the run files the API
serves without torch. frontend/public/models/ holds the same ONNX model and the
metadata the browser needs to run live sessions on its own.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.live import KEYWORD_ACTIONS, PROTOCOL_VERSION, LivePolicy  # noqa: E402
from ml.onnx_model import export_onnx, load_onnx  # noqa: E402
from ml.spotting import KEYWORDS  # noqa: E402

BACKEND = ROOT / "backend" / "deploy"
PUBLIC = ROOT / "frontend" / "public" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "models" / "live3")
    parser.add_argument("--evaluations", type=Path, default=ROOT / "docs" / "evaluations")
    args = parser.parse_args()

    BACKEND.mkdir(parents=True, exist_ok=True)
    document = export_onnx(args.model / "model.pt", BACKEND / "model.onnx")
    detector = json.loads((args.model / "detector.json").read_text(encoding="utf-8"))
    if detector["checkpoint"]["sha256"] != document["checkpoint_sha256"]:
        raise SystemExit("detector.json does not describe this checkpoint")
    shutil.copyfile(args.model / "detector.json", BACKEND / "detector.json")
    load_onnx(BACKEND / "model.onnx")

    runs = BACKEND / "evaluations"
    shutil.rmtree(runs, ignore_errors=True)
    for path in sorted(args.evaluations.glob("*/*.json")):
        target = runs / path.relative_to(args.evaluations)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    PUBLIC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BACKEND / "model.onnx", PUBLIC / "kws.onnx")
    browser = {
        "version": PROTOCOL_VERSION,
        "checkpoint_sha256": document["checkpoint_sha256"],
        "experiment_id": document["checkpoint_metadata"].get("experiment_id"),
        "labels": document["labels"],
        "keywords": list(KEYWORDS),
        "mean": document["normalization"]["mean"],
        "std": document["normalization"]["std"],
        "features": document["features"],
        "policy": LivePolicy().to_dict(),
        "actions": [
            {"keyword": item.keyword, "action": item.action, "label": item.label}
            for item in KEYWORD_ACTIONS
        ],
    }
    (PUBLIC / "kws.json").write_text(json.dumps(browser, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {document['checkpoint_metadata'].get('experiment_id')} to {BACKEND}")


if __name__ == "__main__":
    main()
