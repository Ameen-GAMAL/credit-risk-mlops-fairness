"""Download / load the trained pipeline from MLflow.

Used in two places:
  * Docker build time (CLI): bakes the model for a specific Run ID into the
    image, so containers start without any live MLflow/network dependency —
    which is what makes the Kubernetes readiness probe meaningful.
  * App startup: loads from the baked directory (or downloads on the fly for
    local `uvicorn --reload` development).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow


def download_model(run_id: str, out_dir: str, tracking_uri: str | None = None) -> Path:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model", dst_path=out_dir
    )
    return Path(out_dir) / "model"


def load_model(model_dir: str):
    """Load the sklearn pipeline from a local MLflow artifact directory."""
    return mlflow.sklearn.load_model(model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake an MLflow model into a local dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    path = download_model(args.run_id, args.out, args.tracking_uri)
    print(f"Model for run {args.run_id} -> {path}")
