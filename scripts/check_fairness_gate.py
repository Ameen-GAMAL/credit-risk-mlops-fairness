"""CI fairness gate — the step whose exit code blocks deployment.

Reads metrics/fairness_report.json and re-evaluates it against the CURRENT
params.yaml thresholds (never trusting the stored pass/fail, so tightening
a threshold in params.yaml immediately re-gates old reports). Exits 1 on any
violation, which fails the `fairness_audit` job and leaves `deploy` skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fairness.metrics import evaluate_thresholds  # noqa: E402

REPORT_PATH = ROOT / "metrics" / "fairness_report.json"


def main() -> int:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    fairness_params = params["fairness"]

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["metrics"]

    print(f"Fairness gate for MLflow run {report['run_id']} "
          f"(report generated {report['generated_at']})")
    for attr in fairness_params["protected_attributes"]:
        m = metrics[attr]
        print(
            f"  {attr}: DP_diff={m['demographic_parity_difference']:.4f} "
            f"(<= {fairness_params['demographic_parity_threshold']}), "
            f"EO_diff={m['equalized_odds_difference']:.4f} "
            f"(<= {fairness_params['equalized_odds_threshold']})"
        )

    violations = evaluate_thresholds(metrics, fairness_params)
    if violations:
        print("[GATE FAILED] fairness thresholds violated — deployment blocked:")
        for v in violations:
            print(f"  - {v['attribute']}.{v['metric']} = {v['value']} > {v['threshold']}")
        return 1

    print("[GATE PASSED] all fairness metrics within thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
