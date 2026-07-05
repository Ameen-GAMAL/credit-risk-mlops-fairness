"""Deploy the monitoring dashboard to Hugging Face Spaces.

Bundles what the Space needs but git deliberately doesn't track:
  * model/model.pkl        — the pinned MLflow model (models live in MLflow,
                             not git; the Space gets a copy at deploy time)
  * src/...                — the FairPipeline module the pickle references

Usage:
  python scripts/deploy_dashboard.py [--run-id <id>] [--token <hf_token>] [--prep-only]

Needs MLFLOW_TRACKING_URI/-USERNAME/-PASSWORD env vars for the model
download (same as every pipeline stage).

NOTE on tokens: a globally-set HF_TOKEN env var silently overrides
`hf auth login` — pass --token explicitly if uploads fail with 403.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DASHBOARD_DIR = ROOT / "monitoring_dashboard"
SPACE_ID = "Am33n-21/credit-risk-fairness-monitor"
SRC_FILES = [
    "src/__init__.py",
    "src/features/__init__.py",
    "src/features/transformers.py",
]


def prep(run_id: str) -> None:
    """Download the model for `run_id` and copy the unpickle dependencies."""
    from src.serving.model_loader import download_model

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = download_model(run_id, tmp)
        target = DASHBOARD_DIR / "model"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_dir / "model.pkl", target / "model.pkl")
        (target / "RUN_ID").write_text(run_id, encoding="utf-8")
    for rel in SRC_FILES:
        dest = DASHBOARD_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    print(f"prepped model {run_id} + src modules into {DASHBOARD_DIR}")


def upload(token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(DASHBOARD_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Deploy dashboard",
    )
    print(f"uploaded -> https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="MLflow run (default: model_info.txt)")
    parser.add_argument("--token", default=None, help="HF write token (default: stored login)")
    parser.add_argument("--prep-only", action="store_true", help="bundle files, skip upload")
    args = parser.parse_args()

    run_id = args.run_id or (ROOT / "model_info.txt").read_text(encoding="utf-8").strip()
    prep(run_id)
    if not args.prep_only:
        upload(args.token)
