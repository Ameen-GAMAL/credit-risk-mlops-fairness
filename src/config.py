"""Central configuration: paths, params.yaml access, and column conventions.

Every pipeline stage, CI gate script, and the monitoring layer read from
here so that column semantics (what is a feature vs. a protected attribute
vs. the target) are defined in exactly one place.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

# Windows consoles default to cp1252, which cannot render the emoji MLflow
# prints when terminating a run — crashing the stage AFTER the model was
# logged but BEFORE model_info.txt is written. Force UTF-8 once, centrally
# (every pipeline stage imports this module). No-op on Linux/CI.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "utf-8").lower() not in ("utf-8", "utf8"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"

RAW_DATA_PATH = ROOT / "data" / "raw" / "credit_g.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MONITORING_POOL_PATH = PROCESSED_DIR / "monitoring_pool.csv"
REFERENCE_PATH = ROOT / "data" / "reference" / "reference_batch.csv"
BATCHES_DIR = ROOT / "data" / "simulated_batches"
METRICS_DIR = ROOT / "metrics"
MODEL_INFO_PATH = ROOT / "model_info.txt"
REPORTS_DIR = ROOT / "monitoring" / "reports"

TARGET_COL = "target"          # 1 = good credit (favorable outcome), 0 = bad
RAW_TARGET_COL = "class"       # original credit-g label column ("good"/"bad")

# Columns that must NEVER be model inputs. `personal_status`/`age` are the
# raw sources of the derived protected attributes; the derived attributes
# and the target are excluded for obvious reasons. The model is therefore
# "unaware" of protected attributes — the fairness audit exists precisely
# because unawareness does not prevent disparate outcomes via proxies.
DROP_FROM_FEATURES = ["personal_status", "age", "sex", "age_binary", TARGET_COL]

MLFLOW_EXPERIMENT = "credit-risk-fairness"


def load_params() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def protected_attributes() -> list[str]:
    return load_params()["fairness"]["protected_attributes"]


def feature_columns(df) -> list[str]:
    """Model input columns = everything except protected/target columns."""
    return [c for c in df.columns if c not in DROP_FROM_FEATURES]


def tracking_uri() -> str:
    """DagsHub-hosted MLflow when configured, local ./mlruns otherwise."""
    return os.environ.get("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
