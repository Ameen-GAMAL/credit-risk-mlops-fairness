"""DVC stage `preprocess`: derive protected attributes + target, then split.

Three disjoint splits:
  train.csv           - model training
  test.csv            - evaluation + fairness audit
  monitoring_pool.csv - reserved for simulated "production" batches; never
                        touches training or evaluation, so monitoring results
                        can't leak into model selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    MONITORING_POOL_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    RAW_TARGET_COL,
    TARGET_COL,
    TEST_PATH,
    TRAIN_PATH,
    load_params,
)


def derive_columns(df: pd.DataFrame, age_threshold: int) -> pd.DataFrame:
    """Derive `sex`, `age_binary`, and the binary `target` column.

    - sex: parsed from `personal_status` ("female div/dep/mar" -> female,
      all "male ..." values -> male). The raw column stays in the frame so
      audits can inspect it, but it is excluded from model features.
    - age_binary: "adult" (>= threshold) vs "young" (< threshold) — the
      protected grouping used throughout fairness literature for this dataset.
    - target: 1 = good credit (the favorable outcome a fair model should not
      grant at systematically different rates across groups), 0 = bad.
    """
    out = df.copy()
    out["sex"] = np.where(
        out["personal_status"].astype(str).str.contains("female"), "female", "male"
    )
    out["age_binary"] = np.where(out["age"] >= age_threshold, "adult", "young")
    out[TARGET_COL] = (out[RAW_TARGET_COL].astype(str) == "good").astype(int)
    return out.drop(columns=[RAW_TARGET_COL])


def split_frames(
    df: pd.DataFrame, test_size: float, pool_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / test / monitoring-pool split.

    Fractions are converted to absolute row counts up front (both are
    fractions of the FULL dataset) — re-deriving a relative fraction for the
    second split invites float-rounding surprises at exactly the wrong layer.
    """
    n_pool = round(len(df) * pool_size)
    n_test = round(len(df) * test_size)
    rest, pool = train_test_split(
        df, test_size=n_pool, random_state=random_state, stratify=df[TARGET_COL]
    )
    train, test = train_test_split(
        rest, test_size=n_test, random_state=random_state, stratify=rest[TARGET_COL]
    )
    return train, test, pool


def main() -> None:
    params = load_params()["preprocess"]
    df = pd.read_csv(RAW_DATA_PATH)
    df = derive_columns(df, age_threshold=params["age_threshold"])
    train, test, pool = split_frames(
        df,
        test_size=params["test_size"],
        pool_size=params["monitoring_pool_size"],
        random_state=params["random_state"],
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    pool.to_csv(MONITORING_POOL_PATH, index=False)
    print(
        f"train={len(train)} test={len(test)} monitoring_pool={len(pool)} "
        f"(good-rate: train={train[TARGET_COL].mean():.3f}, "
        f"test={test[TARGET_COL].mean():.3f})"
    )


if __name__ == "__main__":
    main()
