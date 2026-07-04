"""Fairness-metric drift — the project's differentiator.

Standard monitoring asks "did the data change?". This asks "did the model's
FAIRNESS change?": Fairlearn metrics are recomputed per production batch
(using the exact same `compute_fairness_metrics` the CI gate uses) and
compared against the reference-time baseline. A model can drift into
unfairness long before its accuracy visibly degrades — that delta is what
gets flagged here.

Which metrics FLAG vs merely REPORT is a deliberate measurement decision:
  * demographic parity difference — FLAGGED. Label-free (real production
    labels arrive weeks/months late in credit), and computed on whole-group
    selection rates, so it is estimable at batch size.
  * equalized odds difference — REPORTED, not flagged. It needs true labels
    AND slices each group into TPR/FPR cells of ~15-40 rows at our batch
    size, where observed deltas of 0.3 occur on clean batches (verified
    empirically). Alarming on it would just alarm on noise.
"""
from __future__ import annotations

import pandas as pd

from src.fairness.metrics import compute_fairness_metrics

FLAGGED_METRICS = ["demographic_parity_difference"]
REPORTED_METRICS = ["demographic_parity_difference", "equalized_odds_difference"]


def fairness_baseline(reference: pd.DataFrame, y_pred, protected: list[str], target_col: str) -> dict:
    return compute_fairness_metrics(reference[target_col], y_pred, reference[protected])


def compute_fairness_drift(
    current: pd.DataFrame,
    y_pred,
    protected: list[str],
    target_col: str,
    reference_metrics: dict,
    threshold: float,
) -> dict:
    """Per attribute and per watched metric: reference vs current vs |delta|,
    flagged when |delta| exceeds the monitoring threshold."""
    current_metrics = compute_fairness_metrics(
        current[target_col], y_pred, current[protected]
    )
    drift: dict = {}
    for attr in protected:
        drift[attr] = {}
        for metric_name in REPORTED_METRICS:
            ref_val = reference_metrics[attr][metric_name]
            cur_val = current_metrics[attr][metric_name]
            delta = abs(cur_val - ref_val)
            drift[attr][metric_name] = {
                "reference": round(ref_val, 4),
                "current": round(cur_val, 4),
                "delta": round(delta, 4),
                "drift_flagged": bool(
                    metric_name in FLAGGED_METRICS and delta > threshold
                ),
            }
    drift["any_flagged"] = any(
        m["drift_flagged"] for attr in protected for m in drift[attr].values()
    )
    return drift
