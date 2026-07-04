# Model Card — Credit Risk Classifier (fairness-gated)

*Follows the structure of [Mitchell et al., "Model Cards for Model
Reporting" (2019)](https://arxiv.org/abs/1810.03993).*

## Model details

| | |
|---|---|
| Model | Logistic regression inside a Fairlearn `ExponentiatedGradient` reduction (equalized-odds constraint, ε = 0.03) — a 14-predictor ensemble; deterministic serving via weight-averaged probabilities thresholded at 0.5 |
| Version | MLflow run recorded in `model_info.txt`; every serving image is tagged with its Run ID |
| Inputs | 18 application features (see `src/serving/schemas.py`) — **excludes** `personal_status`, `age`, and all derived protected attributes |
| Output | Binary: 1 = "good" credit (favorable), 0 = "bad", plus probability |
| Training pipeline | `dvc repro` — fully deterministic (seeded, data pinned by DVC) |

## Intended use

Portfolio/education: demonstrating production MLOps with governance controls
(fairness gate in CI/CD, fairness-drift monitoring). **Not** for real credit
decisions: the dataset is a 1,000-row 1970s German benchmark whose labels
encode their era's lending practices, and modern credit regulation (e.g.
ECOA/GDPR Art. 22) imposes requirements far beyond this demo's scope.

## Training data

OpenML `credit-g` v1 (German Credit / Statlog): 1,000 applications, 20
features, 70/30 good/bad labels. Split 50/30/20 into train (500) / test
(300) / monitoring pool (200), stratified by label.

The test split is deliberately larger than the customary 20%: the fairness
gate computes equalized odds on per-group TPR/FPR cells, and at a 20% split
the female cells were ~20 rows — one individual moved the metric by ±0.05.
**Gate precision was prioritized over monitoring-pool size** (monitoring
batches are bootstrap resamples and don't need unique rows).

## Protected attributes (audited, never model inputs)

- `sex` — parsed from `personal_status` (contains "female" → female)
- `age_binary` — young (< 25) vs adult (≥ 25), the standard grouping for
  this dataset in the fairness literature

`foreign_worker` is arguably a third protected proxy (nationality); it
remains a model feature here and is flagged as an explicit limitation.

## Fairness journey (all numbers: held-out test set, n = 300)

| Iteration | Acc | F1 | DP sex | EO sex | DP age | EO age | Gate (≤ 0.10) |
|---|---|---|---|---|---|---|---|
| 1. Baseline logreg | 0.793 | 0.861 | 0.069 | **0.177** | **0.205** | **0.158** | ❌ |
| 2. `class_weight=balanced` | 0.753 | 0.806 | 0.077 | 0.068 | **0.156** | 0.058 | ❌ |
| 3. **EG, equalized odds, ε=0.03 (shipped)** | 0.777 | 0.850 | 0.028 | 0.083 | 0.096 | 0.097 | ✅ |

Reading of iteration 1: with protected attributes excluded ("fairness
through unawareness"), the model still approved young applicants at a
**20.5-point lower rate** than adults (61.0% vs 81.5% selection rate) —
proxy leakage through correlated features. This is precisely why the gate
audits *outcomes*, not inputs.

Iteration 2 (naive class reweighting) improved equalized odds but left a
15.6-point age parity gap: rebalancing labels does not rebalance groups.

Iteration 3: the in-processing reduction was chosen over
`ThresholdOptimizer` post-processing because the latter requires protected
attributes **at inference time**, which this serving design deliberately
avoids. ε was swept over {0.005–0.05}; counter-intuitively, the *tightest*
constraints over-contorted the small young-applicant group (EO_age stuck at
~0.102) — ε = 0.03 passes all four metrics with the best accuracy
trade-off. Cost of fairness: **1.7 accuracy points** (0.793 → 0.777).

## Gate thresholds and their rationale

- **0.10 absolute** on DP and EO differences per attribute in CI — the
  common "80%-rule analog" rule of thumb, enforced by
  `scripts/check_fairness_gate.py` (non-zero exit ⇒ deploy unreachable).
  Re-evaluated against current `params.yaml` on every run, so tightening
  policy immediately re-gates old models.
- **0.15 on ΔDP** in monitoring (change vs reference, per batch) — set above
  the empirically measured bootstrap noise floor for 120-row batches
  (max clean-batch |Δ| ≈ 0.11). EO is **reported but not alarmed** in
  monitoring: it requires true labels (which arrive with delay in real
  credit) and its per-cell sample sizes (~15–40) produced Δ up to 0.30 on
  *clean* batches. Alarming on it would train operators to ignore alarms.

## Monitoring behaviour under injected drift

Scenario from batch 4 onward: applicant pool skews younger **and** young
applicants request ~60% larger loans. Observed: the Evidently data-drift
alarm fires on batch 4 (leading indicator, 3 drifted columns); the
fairness-drift alarm confirms on batch 5 (ΔDP_sex = 0.16). Notably the
*age* parity gap barely moved under an age-targeted shock — the
equalized-odds constraint absorbed it, and the residual disparity surfaced
on the correlated `sex` margin. Constraining one attribute does not
inoculate its neighbours; monitor all of them.

## Limitations & ethical considerations

1. **Small groups, wide error bars.** 96 female / 41 young test rows: the
   gate's point estimates carry sampling error of a few points; values near
   threshold (DP_age = 0.096) should be read as "consistent with the
   threshold", not "safely below it". Reproducible to the digit via seeds +
   DVC-pinned data, so CI comparisons remain meaningful.
2. **Two audited attributes.** Intersections (young women) and other proxies
   (`foreign_worker`) are not separately gated — group cells become too
   small. A larger dataset should gate intersectionally.
3. **Dataset era.** Labels reflect 1970s German lending; "good credit" as
   ground truth inherits any historical bias in those decisions.
4. **Metric choice is policy.** DP and EO can be mutually incompatible;
   equalizing them does not guarantee individual fairness or calibration
   parity. The 0.10 threshold is a demonstration default, not a legal
   standard.
