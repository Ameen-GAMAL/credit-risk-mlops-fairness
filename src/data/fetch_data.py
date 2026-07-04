"""DVC stage `fetch_data`: download German Credit (credit-g) from OpenML.

The dataset is the standard benchmark in fairness tooling (Fairlearn's own
examples use it) and needs no manual download/auth, so a fresh clone can
reproduce the whole pipeline.
"""
from __future__ import annotations

from sklearn.datasets import fetch_openml

from src.config import RAW_DATA_PATH, load_params


def main() -> None:
    params = load_params()["fetch_data"]
    bunch = fetch_openml(
        params["openml_name"],
        version=params["openml_version"],
        as_frame=True,
        parser="auto",
    )
    df = bunch.frame
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
