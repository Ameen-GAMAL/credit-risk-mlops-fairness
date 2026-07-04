"""The single shared fairness-metric implementation.

Called from three places — the DVC `fairness_audit` stage, the CI gate
(`scripts/check_fairness_gate.py`), and the monitoring fairness-drift check —
so "fairness at training time" and "fairness at monitoring time" are
provably computed identically.
"""
from __future__ import annotations

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import accuracy_score


def compute_fairness_metrics(y_true, y_pred, sensitive: pd.DataFrame) -> dict:
    """Per protected attribute: DP difference, EO difference, and per-group
    diagnostics. All values are plain floats/ints so the result serializes
    straight to JSON."""
    report: dict = {}
    for attr in sensitive.columns:
        sf = sensitive[attr]
        frame = MetricFrame(
            metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )
        report[attr] = {
            "demographic_parity_difference": float(
                demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
            ),
            "equalized_odds_difference": float(
                equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
            ),
            "selection_rate_by_group": {
                str(k): float(v) for k, v in frame.by_group["selection_rate"].items()
            },
            "accuracy_by_group": {
                str(k): float(v) for k, v in frame.by_group["accuracy"].items()
            },
            "group_counts": {str(k): int(v) for k, v in sf.value_counts().items()},
        }
    return report


def evaluate_thresholds(metrics: dict, fairness_params: dict) -> list[dict]:
    """Compare a compute_fairness_metrics() result against params.yaml
    thresholds. Returns a list of violations (empty = gate passes)."""
    thresholds = {
        "demographic_parity_difference": fairness_params["demographic_parity_threshold"],
        "equalized_odds_difference": fairness_params["equalized_odds_threshold"],
    }
    violations = []
    for attr in fairness_params["protected_attributes"]:
        for metric_name, limit in thresholds.items():
            value = metrics[attr][metric_name]
            if value > limit:
                violations.append(
                    {
                        "attribute": attr,
                        "metric": metric_name,
                        "value": round(value, 4),
                        "threshold": limit,
                    }
                )
    return violations
