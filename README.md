# Credit Risk MLOps — with a Fairness Gate in the Deploy Path

An end-to-end, locally-reproducible MLOps pipeline for a credit-risk
classifier where **AI governance is wired into the pipeline, not written
about after the fact**: deployment is literally unreachable unless a
Fairlearn audit passes, and production monitoring watches *fairness drift*
alongside ordinary data drift.

**Stack:** DVC (data versioning) · MLflow on DagsHub (experiment tracking) ·
GitHub Actions (CI/CD with a fairness gate) · Fairlearn (bias audit +
in-processing mitigation) · Evidently (drift) · FastAPI + Docker (serving,
model baked by MLflow Run ID) · Kubernetes via Minikube/Kind (deployment) ·
Prometheus client (serving metrics)

```
linter ──▶ validate ──▶ fairness_audit ──▶ deploy
           (train +      (Fairlearn gate:    (docker build
            accuracy      DP & EO ≤ 0.10     --build-arg RUN_ID,
            floors)       per group)          model baked from MLflow)
                              │
                              └─ FAIL ⇒ deploy is SKIPPED, by construction
```

**Live proof, not claims:**
[all four jobs green](https://github.com/Ameen-GAMAL/credit-risk-mlops-fairness/actions/runs/28720895645)
· [the gate actually blocking a deploy](https://github.com/Ameen-GAMAL/credit-risk-mlops-fairness/actions/runs/28721066495)
(threshold deliberately tightened below the shipped model's DP — fairness
job red, deploy *skipped*)
· [experiment tracking on DagsHub MLflow](https://dagshub.com/s-amin.mohamed/credit-risk-mlops-fairness.mlflow)
· [**live monitoring dashboard** on HF Spaces](https://huggingface.co/spaces/Am33n-21/credit-risk-fairness-monitor)

## Why this project is interesting

1. **The fairness gate caught a real violation.** The baseline model — which
   never sees `sex` or `age` — still denied young applicants at a 20-point
   higher rate via proxy features. The gate blocked it. That is the entire
   argument for outcome auditing over "we removed the sensitive columns."

2. **The fix is in-processing, not cosmetic.** Fairlearn's
   `ExponentiatedGradient` reduction (equalized-odds constraint, swept
   ε=0.03) brings every audited metric under threshold for **1.7 points of
   accuracy** — and needs no protected attributes at inference, so serving
   stays clean.

3. **Monitoring watches fairness, not just features.** Simulated production
   batches (versioned with DVC, deterministic) include a targeted shock —
   "young applicants start requesting much larger loans" — and the weekly
   monitoring job flags the resulting demographic-parity drift with the
   *same metric implementation* the CI gate uses. A
   [Streamlit dashboard](https://huggingface.co/spaces/Am33n-21/credit-risk-fairness-monitor)
   (in [monitoring_dashboard/](monitoring_dashboard/)) renders the gate
   status and per-batch drift live from the repo's committed JSON artifacts.

## Results (held-out test set, n = 300)

| Model | Accuracy | F1 | DP sex | EO sex | DP age | EO age | Fairness gate |
|---|---|---|---|---|---|---|---|
| Logistic regression (baseline) | **0.793** | 0.861 | 0.069 | 0.177 ✗ | 0.205 ✗ | 0.158 ✗ | ❌ 3 violations |
| + `class_weight=balanced` | 0.753 | 0.806 | 0.077 | 0.068 | 0.156 ✗ | 0.058 | ❌ 1 violation |
| **+ ExponentiatedGradient (shipped)** | 0.777 | 0.850 | **0.028** | **0.083** | **0.096** | **0.097** | ✅ passes |

DP = demographic parity difference, EO = equalized odds difference;
gate threshold 0.10 per protected attribute (`sex`, `age_binary` at 25).
ROC-AUC of the shipped model: 0.834.

## Monitoring output (simulated production, drift injected from batch 4)

| Batch | Data drift | Drifted cols | Fairness drift (ΔDP) | Alarm |
|---|---|---|---|---|
| batch_01 | – | 0 | – | ok |
| batch_02 | – | 0 | – | ok |
| batch_03 | – | 1 | – | ok |
| batch_04 | **drift** | 3 | Δ0.12 (sub-threshold) | 🚨 |
| batch_05 | **drift** | 3 | **Δ0.16 flagged** | 🚨 |

The data-drift alarm fires first (leading indicator), the fairness breach
confirms one batch later — and notably, the *age*-parity gap barely moves
under an age-targeted shock because the equalized-odds constraint absorbs
it; the residual disparity surfaces on the correlated `sex` margin instead.
Full analysis in [docs/model_card.md](docs/model_card.md); per-batch JSON
reports plus a sample Evidently HTML in
[monitoring/reports/](monitoring/reports/) (one command regenerates all).

## Quickstart (fully offline)

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate   # Python 3.12
pip install -r requirements-dev.txt
dvc repro        # fetch → preprocess → batches → train → evaluate → fairness_audit
pytest && flake8 src scripts tests
python scripts/check_fairness_gate.py              # the CI gate, locally
python -m src.monitoring.run_monitoring            # drift + fairness drift
```

Serving, Docker, Kubernetes, and the DagsHub/GitHub hookup:
[docs/runbook.md](docs/runbook.md). Design rationale:
[docs/architecture.md](docs/architecture.md).

## Repo map

```
dvc.yaml / params.yaml        pipeline DAG + single source of tunable truth
src/data/                     fetch, preprocess (protected-attr derivation), batch simulation
src/train.py                  logreg/xgboost + optional EG mitigation → MLflow
src/fairness/metrics.py       ONE fairness implementation, used by gate + audit + monitoring
src/serving/                  FastAPI (/predict /health /model-info /metrics)
src/monitoring/               Evidently drift + fairness-drift orchestration
scripts/check_*.py            the CI gates (exit code = the mechanism)
.github/workflows/            pipeline.yml (gated CI/CD) + monitoring.yml (weekly cron)
docker/ · k8s/                serving image (model baked by Run ID) + Minikube/Kind manifests
docs/                         model card · architecture · runbook
```

## Honest limitations

- 1,000-row benchmark dataset; group cells are small, so equalized odds is
  *reported* in monitoring but only demographic parity (label-free, whole-
  group) is *alarmed*. Thresholds are set from measured noise floors, and
  the reasoning is documented — that discipline is the point of the project.
- Simulated traffic, not real users; drift is injected synthetically (one
  clearly-marked function) so the detectors have known ground truth.
- German Credit's 1970s labels encode their era's biases; this repo is an
  MLOps/governance demonstration, not a deployable credit policy.

## License

MIT — see [LICENSE](LICENSE).
