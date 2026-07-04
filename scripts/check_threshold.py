"""CI gate: verify the trained run exists in MLflow and beats the accuracy/F1
floors from params.yaml. Non-zero exit fails the `validate` job, which blocks
`fairness_audit` and `deploy` downstream. (Course Assign-5 pattern, extended
with an F1 floor.)"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_INFO_PATH, tracking_uri  # noqa: E402


def main() -> int:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    floors = {
        "test_accuracy": params["evaluate"]["min_accuracy"],
        "test_f1": params["evaluate"]["min_f1"],
    }

    run_id = MODEL_INFO_PATH.read_text(encoding="utf-8").strip()
    client = MlflowClient(tracking_uri=tracking_uri())
    run = client.get_run(run_id)  # raises if the run does not exist
    print(f"[OK] MLflow run {run_id} found (status={run.info.status})")

    failures = []
    for metric, floor in floors.items():
        value = run.data.metrics.get(metric)
        if value is None:
            failures.append(f"{metric}: missing from run")
            continue
        verdict = "PASS" if value >= floor else "FAIL"
        print(f"[{verdict}] {metric}={value:.4f} (floor {floor})")
        if value < floor:
            failures.append(f"{metric}={value:.4f} < {floor}")

    if failures:
        print(f"[GATE FAILED] {'; '.join(failures)}")
        return 1
    print("[GATE PASSED] model meets performance floors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
