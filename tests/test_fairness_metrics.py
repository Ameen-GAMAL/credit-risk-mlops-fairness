import numpy as np
import pandas as pd
import pytest

from src.fairness.metrics import compute_fairness_metrics, evaluate_thresholds


def _fixture():
    # Group A: selection rate 0.75 (3/4), Group B: 0.25 (1/4) -> DP diff = 0.5
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    sensitive = pd.DataFrame({"grp": ["A", "A", "A", "A", "B", "B", "B", "B"]})
    return y_true, y_pred, sensitive


def test_compute_fairness_metrics_values():
    y_true, y_pred, sensitive = _fixture()
    report = compute_fairness_metrics(y_true, y_pred, sensitive)

    assert set(report.keys()) == {"grp"}
    m = report["grp"]
    assert m["demographic_parity_difference"] == pytest.approx(0.5)
    assert m["selection_rate_by_group"] == {"A": 0.75, "B": 0.25}
    assert m["group_counts"] == {"A": 4, "B": 4}
    assert 0.0 <= m["equalized_odds_difference"] <= 1.0
    # everything must be JSON-serializable plain types
    import json

    json.dumps(report)


def test_evaluate_thresholds_flags_violation():
    y_true, y_pred, sensitive = _fixture()
    metrics = compute_fairness_metrics(y_true, y_pred, sensitive)
    fairness_params = {
        "protected_attributes": ["grp"],
        "demographic_parity_threshold": 0.10,
        "equalized_odds_threshold": 1.00,
    }
    violations = evaluate_thresholds(metrics, fairness_params)
    assert len(violations) == 1
    assert violations[0]["attribute"] == "grp"
    assert violations[0]["metric"] == "demographic_parity_difference"
    assert violations[0]["value"] > 0.10


def test_evaluate_thresholds_passes_when_loose():
    y_true, y_pred, sensitive = _fixture()
    metrics = compute_fairness_metrics(y_true, y_pred, sensitive)
    fairness_params = {
        "protected_attributes": ["grp"],
        "demographic_parity_threshold": 0.60,
        "equalized_odds_threshold": 1.00,
    }
    assert evaluate_thresholds(metrics, fairness_params) == []
