"""Monitoring orchestrator: loop the simulated production batches, generate
Evidently drift reports AND fairness-drift checks per batch, then roll up a
single `summary.json` / `summary.md` (the screenshot artifact: batches 1-3
clean, alarms tripping exactly where drift was injected).

Run locally:      python -m src.monitoring.run_monitoring
Run in CI:        .github/workflows/monitoring.yml (weekly cron + on-demand)
Model selection:  RUN_ID env var, falling back to model_info.txt.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import mlflow
import pandas as pd

from src.config import (
    BATCHES_DIR,
    MODEL_INFO_PATH,
    REFERENCE_PATH,
    REPORTS_DIR,
    TARGET_COL,
    load_params,
    tracking_uri,
)
from src.monitoring.drift_report import PREDICTION_COL, generate_drift_report
from src.monitoring.fairness_drift import compute_fairness_drift, fairness_baseline


def _load_model():
    run_id = os.environ.get("RUN_ID") or MODEL_INFO_PATH.read_text(encoding="utf-8").strip()
    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model"), run_id


def main() -> None:
    params = load_params()
    mon = params["monitoring"]
    protected = params["fairness"]["protected_attributes"]

    model, run_id = _load_model()

    reference = pd.read_csv(REFERENCE_PATH)
    ref_pred = model.predict(reference)
    reference[PREDICTION_COL] = ref_pred
    reference_metrics = fairness_baseline(reference, ref_pred, protected, TARGET_COL)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_path in sorted(BATCHES_DIR.glob("batch_*.csv")):
        name = batch_path.stem
        current = pd.read_csv(batch_path)
        cur_pred = model.predict(current)
        current[PREDICTION_COL] = cur_pred

        drift_summary = generate_drift_report(
            reference,
            current,
            html_path=REPORTS_DIR / f"{name}_drift.html",
            json_path=REPORTS_DIR / f"{name}_drift.json",
        )
        fairness = compute_fairness_drift(
            current,
            cur_pred,
            protected,
            TARGET_COL,
            reference_metrics,
            threshold=mon["fairness_drift_threshold"],
        )
        with open(REPORTS_DIR / f"{name}_fairness_drift.json", "w", encoding="utf-8") as fh:
            json.dump(fairness, fh, indent=2)

        rows.append(
            {
                "batch": name,
                "dataset_drift": drift_summary["dataset_drift"],
                "n_drifted_columns": drift_summary["n_drifted_columns"],
                "prediction_drift": drift_summary["prediction_drift_detected"],
                "fairness_drift": fairness["any_flagged"],
                "any_alarm": bool(drift_summary["dataset_drift"] or fairness["any_flagged"]),
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_run_id": run_id,
        "reference_rows": len(reference),
        "fairness_drift_threshold": mon["fairness_drift_threshold"],
        "reference_fairness": {
            attr: {
                "demographic_parity_difference": round(
                    reference_metrics[attr]["demographic_parity_difference"], 4
                ),
                "equalized_odds_difference": round(
                    reference_metrics[attr]["equalized_odds_difference"], 4
                ),
            }
            for attr in protected
        },
        "batches": rows,
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_markdown(summary)
    _print_table(rows)


def _write_markdown(summary: dict) -> None:
    """Markdown twin of summary.json — piped into $GITHUB_STEP_SUMMARY by CI."""
    lines = [
        "## Drift & Fairness-Drift Monitoring",
        "",
        f"Model run: `{summary['model_run_id']}` | "
        f"fairness-drift threshold: {summary['fairness_drift_threshold']}",
        "",
        "| Batch | Data drift | Drifted cols | Prediction drift | Fairness drift | ALARM |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["batches"]:
        alarm = "🚨" if r["any_alarm"] else "✅"
        lines.append(
            f"| {r['batch']} | {r['dataset_drift']} | {r['n_drifted_columns']} "
            f"| {r['prediction_drift']} | {r['fairness_drift']} | {alarm} |"
        )
    with open(REPORTS_DIR / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _print_table(rows: list[dict]) -> None:
    print(f"{'batch':<10} {'data_drift':<11} {'n_cols':<7} {'pred_drift':<11} "
          f"{'fair_drift':<11} alarm")
    for r in rows:
        print(
            f"{r['batch']:<10} {str(r['dataset_drift']):<11} "
            f"{str(r['n_drifted_columns']):<7} {str(r['prediction_drift']):<11} "
            f"{str(r['fairness_drift']):<11} {'ALARM' if r['any_alarm'] else 'ok'}"
        )


if __name__ == "__main__":
    main()
