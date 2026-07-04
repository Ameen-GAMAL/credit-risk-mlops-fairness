"""Standard data/prediction drift via Evidently (legacy 0.4.x Report API).

One HTML report (human review) + one JSON (machine-readable summary) per
simulated batch, always compared against the versioned reference set.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config import TARGET_COL

PREDICTION_COL = "prediction"


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    json_path: Path,
) -> dict:
    """Run DataDrift + TargetDrift presets; return a compact summary dict.

    Both frames must already carry a `prediction` column (added by the
    orchestrator by running the served model on each batch), so prediction
    drift is measured alongside raw feature drift.
    """
    mapping = ColumnMapping(target=TARGET_COL, prediction=PREDICTION_COL)
    # drift_share=0.1: flag the DATASET as drifted when >10% of columns
    # drift. Evidently's 0.5 default assumes broad corruption; realistic
    # drift (and our injected scenario) hits a handful of columns — 4/24
    # columns drifting is an incident, not statistical noise.
    report = Report(metrics=[DataDriftPreset(drift_share=0.1), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    report.save_json(str(json_path))

    return _summarize(report.as_dict())


def _summarize(report_dict: dict) -> dict:
    """Pull the headline numbers out of Evidently's nested result structure."""
    summary = {
        "dataset_drift": None,
        "n_drifted_columns": None,
        "share_drifted_columns": None,
        "prediction_drift_detected": None,
    }
    for metric in report_dict.get("metrics", []):
        name = metric.get("metric", "")
        result = metric.get("result", {}) or {}
        if name == "DatasetDriftMetric":
            summary["dataset_drift"] = bool(result.get("dataset_drift"))
            summary["n_drifted_columns"] = int(result.get("number_of_drifted_columns", 0))
            summary["share_drifted_columns"] = float(
                result.get("share_of_drifted_columns", 0.0)
            )
        elif name == "ColumnDriftMetric" and result.get("column_name") == PREDICTION_COL:
            summary["prediction_drift_detected"] = bool(result.get("drift_detected"))
    return summary
