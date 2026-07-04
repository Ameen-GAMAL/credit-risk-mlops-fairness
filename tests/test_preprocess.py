import numpy as np
import pandas as pd

from src.data.preprocess import derive_columns, split_frames
from src.data.make_reference_and_batches import inject_drift


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "personal_status": [
                "male single",
                "female div/dep/mar",
                "male mar/wid",
                "female div/dep/mar",
                "male div/sep",
                "male single",
            ],
            "age": [22, 30, 25, 24, 45, 19],
            "credit_amount": [1000, 2000, 1500, 3000, 2500, 800],
            "class": ["good", "bad", "good", "good", "bad", "good"],
        }
    )


def test_derive_columns_sex_and_age_binary():
    df = derive_columns(_raw_fixture(), age_threshold=25)
    assert list(df["sex"]) == ["male", "female", "male", "female", "male", "male"]
    # age 25 is the inclusive "adult" boundary
    assert list(df["age_binary"]) == ["young", "adult", "adult", "young", "adult", "young"]
    assert list(df["target"]) == [1, 0, 1, 1, 0, 1]
    assert "class" not in df.columns
    assert not df.isna().any().any()


def test_split_frames_disjoint_and_complete():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feature": rng.normal(size=200),
            "target": rng.integers(0, 2, size=200),
        }
    )
    train, test, pool = split_frames(df, test_size=0.2, pool_size=0.3, random_state=42)
    assert len(train) + len(test) + len(pool) == len(df)
    assert len(pool) == 60  # 30% of 200
    assert len(test) == 40  # 20% of the FULL dataset
    all_idx = set(train.index) | set(test.index) | set(pool.index)
    assert len(all_idx) == len(df)  # disjoint splits


def test_inject_drift_shifts_age_down_and_recomputes_groups():
    df = derive_columns(_raw_fixture(), age_threshold=25)
    drifted = inject_drift(df, strength=0.3, age_threshold=25)
    assert (drifted["age"] <= df["age"]).all()
    assert (drifted["age"] >= 18).all()
    # age_binary must be recomputed from the drifted ages, not stale
    expected = np.where(drifted["age"] >= 25, "adult", "young")
    assert list(drifted["age_binary"]) == list(expected)
    # credit_amount inflation targets the (post-drift) young cohort only
    young = drifted["age_binary"] == "young"
    assert (drifted.loc[young, "credit_amount"] > df.loc[young, "credit_amount"]).all()
    assert (drifted.loc[~young, "credit_amount"] == df.loc[~young, "credit_amount"]).all()
