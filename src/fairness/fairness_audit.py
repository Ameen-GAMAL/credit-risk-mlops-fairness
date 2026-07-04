"""DVC stage `fairness_audit` (also run standalone by the CI fairness job).

Computes Fairlearn metrics for the trained model on the held-out test set,
writes `metrics/fairness_report.json`, and logs the numbers to the model's
MLflow run. It always exits 0 — GATING is deliberately a separate concern
(`scripts/check_fairness_gate.py`), so `dvc repro` produces artifacts while
CI decides whether they are acceptable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import mlflow
import pandas as pd

from src.config import (
    METRICS_DIR,
    MODEL_INFO_PATH,
    TARGET_COL,
    TEST_PATH,
    load_params,
    tracking_uri,
)
from src.fairness.metrics import compute_fairness_metrics, evaluate_thresholds


def main() -> None:
    params = load_params()
    fairness_params = params["fairness"]
    protected = fairness_params["protected_attributes"]

    run_id = MODEL_INFO_PATH.read_text(encoding="utf-8").strip()
    mlflow.set_tracking_uri(tracking_uri())
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    df = pd.read_csv(TEST_PATH)
    y_true = df[TARGET_COL]
    y_pred = model.predict(df)

    metrics = compute_fairness_metrics(y_true, y_pred, df[protected])
    violations = evaluate_thresholds(metrics, fairness_params)

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_test_rows": len(df),
        "thresholds": {
            "demographic_parity_threshold": fairness_params["demographic_parity_threshold"],
            "equalized_odds_threshold": fairness_params["equalized_odds_threshold"],
        },
        "metrics": metrics,
        "violations": violations,
        "passed": not violations,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "fairness_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Log to the SAME run as training/evaluation so DP/EO show up next to
    # accuracy in the MLflow UI — fairness as a first-class model metric.
    with mlflow.start_run(run_id=run_id):
        for attr in protected:
            mlflow.log_metrics(
                {
                    f"dp_diff_{attr}": metrics[attr]["demographic_parity_difference"],
                    f"eo_diff_{attr}": metrics[attr]["equalized_odds_difference"],
                }
            )

    print(f"Fairness report -> {out_path}")
    for attr in protected:
        m = metrics[attr]
        print(
            f"  {attr}: DP_diff={m['demographic_parity_difference']:.4f} "
            f"EO_diff={m['equalized_odds_difference']:.4f} "
            f"groups={m['group_counts']}"
        )
    print(f"  passed={report['passed']} violations={len(violations)}")


if __name__ == "__main__":
    main()
