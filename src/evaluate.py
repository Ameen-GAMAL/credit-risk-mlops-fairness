"""DVC stage `evaluate`: held-out test metrics, logged back to the SAME
MLflow run that trained the model (so `scripts/check_threshold.py` can gate
CI by querying MLflow, mirroring the course Assign-5 pattern)."""
from __future__ import annotations

import json

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import (
    METRICS_DIR,
    MODEL_INFO_PATH,
    TARGET_COL,
    TEST_PATH,
    tracking_uri,
)


def main() -> None:
    run_id = MODEL_INFO_PATH.read_text(encoding="utf-8").strip()
    mlflow.set_tracking_uri(tracking_uri())

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    df = pd.read_csv(TEST_PATH)
    y_true = df[TARGET_COL]
    y_pred = model.predict(df)
    y_proba = model.predict_proba(df)[:, 1]

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, **metrics}, fh, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
