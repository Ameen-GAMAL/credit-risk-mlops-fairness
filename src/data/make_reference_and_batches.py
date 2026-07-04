"""DVC stage `make_reference_and_batches`: build the monitoring simulation.

Generated ONCE, offline, and versioned with DVC — not regenerated at
monitoring time — so drift results are deterministic and reproducible from
a fresh clone.

Design:
  reference_batch.csv  = the full monitoring pool (the "what the world looked
                         like at deployment time" baseline for Evidently).
  batch_01..03         = bootstrap resamples of the pool -> statistically
                         indistinguishable from reference (no drift expected).
  batch_04..05         = same resampling + synthetic drift injection
                         simulating "economic conditions worsened for young
                         applicants":
                           * age shifted DOWNWARD (younger applicant pool) —
                             composition drift on a protected attribute,
                             visible to the plain Evidently column checks.
                           * credit_amount scaled UP for YOUNG rows only —
                             a feature shift targeted at one protected group,
                             so the model starts rejecting that group more
                             and demographic parity genuinely degrades. This
                             is what makes the same batch trip BOTH the data
                             -drift check AND the fairness-drift check — that
                             coupling is the point of this project.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BATCHES_DIR, MONITORING_POOL_PATH, REFERENCE_PATH, load_params


def inject_drift(df: pd.DataFrame, strength: float, age_threshold: int) -> pd.DataFrame:
    """Apply synthetic covariate drift. Kept as one small named function so
    the README can point at it: 'this is the drift we inject, and here is
    the monitoring output catching it'."""
    out = df.copy()
    # Applicant pool skews younger (composition drift on the protected attr)
    out["age"] = (out["age"] * (1.0 - strength)).round().clip(18, 80).astype(int)
    out["age_binary"] = np.where(out["age"] >= age_threshold, "adult", "young")
    # ... and the young cohort asks for markedly larger loans (feature drift
    # TARGETED at one protected group -> the model rejects them more -> the
    # demographic-parity gap widens for real, not just compositionally).
    young = out["age_binary"] == "young"
    out.loc[young, "credit_amount"] = (
        out.loc[young, "credit_amount"] * (1.0 + 2.0 * strength)
    ).round().astype(int)
    return out


def main() -> None:
    params = load_params()
    mon = params["monitoring"]
    age_threshold = params["preprocess"]["age_threshold"]

    pool = pd.read_csv(MONITORING_POOL_PATH)

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(REFERENCE_PATH, index=False)

    for i in range(1, mon["n_batches"] + 1):
        batch = pool.sample(
            n=mon["batch_size"], replace=True, random_state=mon["random_state"] + i
        )
        drifted = i >= mon["inject_drift_from_batch"]
        if drifted:
            batch = inject_drift(
                batch, strength=mon["drift_injection_strength"], age_threshold=age_threshold
            )
        path = BATCHES_DIR / f"batch_{i:02d}.csv"
        batch.to_csv(path, index=False)
        young_share = (batch["age_binary"] == "young").mean()
        print(
            f"{path.name}: n={len(batch)} drift_injected={drifted} "
            f"young_share={young_share:.2f}"
        )


if __name__ == "__main__":
    main()
