"""Fairness & drift monitoring dashboard + live what-if predictor.

Data: reads ONLY artifacts the pipeline already produces — committed JSONs
(fetched from the public GitHub repo when not running inside the repo) plus
a copy of the pinned MLflow model bundled at deploy time by
`scripts/deploy_dashboard.py` (models live in MLflow, not git).

Tabs:
  📊 Monitoring   — gate status + drift charts with an interactive POLICY
                    EXPLORER: move the thresholds, watch alarms recompute.
  🧪 What-if      — live predictions from the actual mitigated model; the
                    form is generated from the fitted preprocessor, so it
                    can never drift from the model's schema.
  📇 Model card   — the governance document, rendered from the repo.
  ⚙️ CI status    — recent GitHub Actions runs of the gated pipeline.
"""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):  # Space layout (src/ beside app) / repo layout
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPO = "Ameen-GAMAL/credit-risk-mlops-fairness"
GITHUB_RAW = f"https://raw.githubusercontent.com/{REPO}/main"
LOCAL_ROOT = HERE.parent

LINKS = {
    "GitHub repo": f"https://github.com/{REPO}",
    "MLflow (DagsHub)": "https://dagshub.com/s-amin.mohamed/credit-risk-mlops-fairness.mlflow",
    "CI: gates green": f"https://github.com/{REPO}/actions/runs/29173438514",
    "CI: gate blocking deploy": f"https://github.com/{REPO}/actions/runs/29173416727",
}

FORM_DEFAULTS = {
    "checking_status": "<0", "duration": 24, "credit_history": "existing paid",
    "purpose": "radio/tv", "credit_amount": 3500, "savings_status": "<100",
    "employment": "1<=X<4", "installment_commitment": 3, "other_parties": "none",
    "residence_since": 2, "property_magnitude": "car", "other_payment_plans": "none",
    "housing": "own", "existing_credits": 1, "job": "skilled", "num_dependents": 1,
    "own_telephone": "yes", "foreign_worker": "yes",
}


# ── loaders ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_json(rel_path):
    local = LOCAL_ROOT / rel_path
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    resp = requests.get(f"{GITHUB_RAW}/{rel_path}", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def load_text(rel_path):
    local = LOCAL_ROOT / rel_path
    if local.exists():
        return local.read_text(encoding="utf-8")
    resp = requests.get(f"{GITHUB_RAW}/{rel_path}", timeout=20)
    resp.raise_for_status()
    return resp.text


@st.cache_resource
def load_model():
    """Unpickle the bundled FairPipeline. Requires src/ importable (the
    deploy script copies it next to this file for the Space)."""
    model_dir = HERE / "model"
    if not (model_dir / "model.pkl").exists():
        return None, None
    import src.features.transformers  # noqa: F401 — class referenced by the pickle
    with open(model_dir / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    run_id = (model_dir / "RUN_ID").read_text(encoding="utf-8").strip() \
        if (model_dir / "RUN_ID").exists() else None
    return model, run_id


@st.cache_data(ttl=120)
def github_runs():
    resp = requests.get(
        f"https://api.github.com/repos/{REPO}/actions/runs?per_page=6", timeout=15
    )
    resp.raise_for_status()
    return resp.json()["workflow_runs"]


@st.cache_data(ttl=120)
def github_jobs(run_id):
    resp = requests.get(
        f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/jobs", timeout=15
    )
    resp.raise_for_status()
    return resp.json()["jobs"]


# ── page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk — Fairness Monitor", page_icon="⚖️", layout="wide")
st.title("⚖️ Credit Risk Model — Fairness & Drift Monitor")
st.caption(
    "A fairness-gated MLOps pipeline: deployment is blocked in CI unless a "
    "Fairlearn audit passes, and monitoring watches *fairness drift* next to "
    "ordinary data drift. " + " · ".join(f"[{k}]({v})" for k, v in LINKS.items())
)

summary = load_json("monitoring/reports/summary.json")
gate = load_json("metrics/fairness_report.json")
evaluation = load_json("metrics/evaluation.json")
batches = [b["batch"] for b in summary["batches"]]
fairness_by_batch = {b: load_json(f"monitoring/reports/{b}_fairness_drift.json") for b in batches}
drift_by_batch = {b: load_json(f"monitoring/reports/{b}_drift.json") for b in batches}


def dataset_drift_stats(batch):
    for m in drift_by_batch[batch].get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            r = m.get("result", {})
            return int(r.get("number_of_drifted_columns", 0)), int(r.get("number_of_columns", 24))
    return 0, 24


tab_mon, tab_whatif, tab_card, tab_ci = st.tabs(
    ["📊 Monitoring", "🧪 What-if predictor", "📇 Model card", "⚙️ CI status"]
)

# ═════════════════════════════════ MONITORING ═══════════════════════════════
with tab_mon:
    with st.container(border=True):
        st.subheader("🎚️ Policy explorer")
        st.caption(
            "Thresholds are policy decisions, not constants of nature. Move them and "
            "watch the gate verdict and monitoring alarms recompute — too strict alarms "
            "on sampling noise, too loose misses the injected fairness breach."
        )
        s1, s2, s3 = st.columns(3)
        gate_thr = s1.slider("CI gate: max |DP| & |EO| per group", 0.01, 0.30,
                             float(gate["thresholds"]["demographic_parity_threshold"]), 0.01)
        drift_thr = s2.slider("Monitoring: max ΔDP vs reference", 0.01, 0.30,
                              float(summary["fairness_drift_threshold"]), 0.01)
        share_thr = s3.slider("Data drift: share of drifted columns", 0.05, 0.60, 0.10, 0.05)

    st.header("Shipped model — CI gate status")
    left, right = st.columns([1, 2])
    gate_rows, gate_ok = [], True
    for attr, m in gate["metrics"].items():
        dp, eo = m["demographic_parity_difference"], m["equalized_odds_difference"]
        gate_ok &= dp <= gate_thr and eo <= gate_thr
        gate_rows.append({
            "protected attribute": attr,
            "DP difference": round(dp, 4), "DP ok": "✅" if dp <= gate_thr else "🚨",
            "EO difference": round(eo, 4), "EO ok": "✅" if eo <= gate_thr else "🚨",
            "group sizes": ", ".join(f"{k}: {v}" for k, v in m["group_counts"].items()),
        })
    with left:
        st.metric("Test accuracy", f"{evaluation['test_accuracy']:.3f}")
        st.metric("Test F1", f"{evaluation['test_f1']:.3f}")
        st.metric("ROC-AUC", f"{evaluation['test_roc_auc']:.3f}")
        st.markdown(
            (f"✅ **Gate PASSES at threshold {gate_thr:.2f}**" if gate_ok
             else f"🚨 **Gate FAILS at threshold {gate_thr:.2f}** — deploy would be blocked")
            + f"  \nMLflow run `{gate['run_id'][:12]}…`"
        )
    with right:
        st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Held-out test set, audited with the same `compute_fairness_metrics` the CI "
            "gate and this monitor use — one metric definition, three consumers."
        )

    st.header("Fairness drift across production batches")
    fair_flags = {}
    attr_cols = st.columns(len(summary["reference_fairness"]))
    for col, attr in zip(attr_cols, summary["reference_fairness"]):
        ref_dp = summary["reference_fairness"][attr]["demographic_parity_difference"]
        dp_cur = [fairness_by_batch[b][attr]["demographic_parity_difference"]["current"] for b in batches]
        flags = [abs(v - ref_dp) > drift_thr for v in dp_cur]
        fair_flags[attr] = flags
        eo_cur = [fairness_by_batch[b][attr]["equalized_odds_difference"]["current"] for b in batches]

        fig = go.Figure()
        fig.add_hrect(y0=max(ref_dp - drift_thr, 0), y1=ref_dp + drift_thr,
                      fillcolor="green", opacity=0.08, line_width=0)
        fig.add_hline(y=ref_dp, line_dash="dash", line_color="green",
                      annotation_text="reference DP", annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=batches, y=dp_cur, mode="lines+markers", name="DP difference (alarmed)",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=12, color=["#d62728" if f else "#1f77b4" for f in flags],
                        symbol=["x" if f else "circle" for f in flags]),
        ))
        fig.add_trace(go.Scatter(
            x=batches, y=eo_cur, mode="lines+markers", name="EO difference (reported only)",
            line=dict(color="#7f7f7f", dash="dot"), marker=dict(size=7, color="#7f7f7f"),
        ))
        fig.update_layout(title=attr, height=340, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_title="metric value", legend=dict(orientation="h", y=-0.25))
        col.plotly_chart(fig, use_container_width=True)
    st.caption(
        "EO is charted but deliberately not alarmed: it needs true labels (delayed in "
        "real credit) and its per-group TPR/FPR cells are too small at this batch size — "
        "clean batches showed EO deltas up to 0.30. Alarming on it trains people to "
        "ignore alarms."
    )

    st.header("Data drift & alarm rollup")
    c1, c2 = st.columns(2)
    ds_flags = []
    with c1:
        n_drift, totals = zip(*[dataset_drift_stats(b) for b in batches])
        ds_flags = [n / t > share_thr for n, t in zip(n_drift, totals)]
        fig = go.Figure(go.Bar(
            x=batches, y=list(n_drift),
            marker_color=["#d62728" if f else "#2ca02c" for f in ds_flags],
            text=[("DRIFT" if f else "ok") for f in ds_flags], textposition="outside",
        ))
        fig.add_hline(y=share_thr * totals[0], line_dash="dot", line_color="#d62728",
                      annotation_text=f"alarm above {share_thr:.0%} of {totals[0]} cols")
        fig.update_layout(title="Evidently: drifted columns per batch", height=340,
                          margin=dict(l=10, r=10, t=40, b=10), yaxis_title="# drifted columns")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rows = []
        for i, b in enumerate(batches):
            fair = any(fair_flags[a][i] for a in fair_flags)
            rows.append({
                "batch": b, "data drift": "🚨" if ds_flags[i] else "—",
                "drifted cols": n_drift[i],
                "fairness drift": "🚨" if fair else "—",
                "ALARM": "🚨 ALARM" if (ds_flags[i] or fair) else "✅ ok",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Recomputed live against the sliders above · monitoring generated "
            f"{summary['generated_at'][:19]}Z · model run `{summary['model_run_id'][:12]}…`"
        )

    with st.expander("🔬 Full Evidently report (batch_05 — the both-alarms batch)"):
        if st.checkbox("Load report (~3 MB)"):
            components.html(load_text("monitoring/reports/batch_05_drift.html"),
                            height=700, scrolling=True)

    with st.expander("What am I looking at? (the simulation design)"):
        st.markdown(
            """
Batches 1–3 are bootstrap resamples of the monitoring pool — statistically
indistinguishable from the reference set, so a calibrated detector should stay
quiet. From **batch 4**, a synthetic scenario is injected: *the applicant pool
skews younger, and young applicants request ~60% larger loans*. The injection
is one clearly-named function (`src/data/make_reference_and_batches.py::inject_drift`)
and the batches are DVC-versioned, so this exact alarm pattern reproduces from
a fresh clone.

The data-drift alarm fires first (batch 4 — leading indicator); the
fairness-drift alarm confirms on batch 5 as the shock persists. Notably the
*age* parity gap barely moves under an age-targeted shock — the equalized-odds
training constraint absorbs it — and the residual disparity surfaces on the
correlated *sex* margin instead. Constraining one attribute does not inoculate
its neighbours; monitor all of them.
            """
        )

# ═════════════════════════════ WHAT-IF PREDICTOR ════════════════════════════
with tab_whatif:
    model, bundled_run = load_model()
    if model is None:
        st.info(
            "Model not bundled with this deployment. Run "
            "`python scripts/deploy_dashboard.py` from the repo to bundle the pinned "
            "MLflow model and redeploy."
        )
    else:
        st.header("Try the actual shipped model")
        st.markdown(
            f"Live inference from the fairness-mitigated pipeline (MLflow run "
            f"`{(bundled_run or '?')[:12]}…`). Notice what the form **doesn't** ask: "
            "sex, age, or marital status — protected attributes are never model "
            "inputs. The fairness audit exists because that alone doesn't prevent "
            "disparate outcomes via correlated proxies."
        )
        pre = model.preprocessor
        num_cols = list(pre.transformers_[0][2])
        cat_cols = list(pre.transformers_[1][2])
        categories = dict(zip(cat_cols, pre.named_transformers_["cat"].categories_))

        with st.form("whatif"):
            cols = st.columns(3)
            values = {}
            for i, feat in enumerate(num_cols + cat_cols):
                target = cols[i % 3]
                if feat in num_cols:
                    values[feat] = target.number_input(
                        feat, min_value=0.0, value=float(FORM_DEFAULTS.get(feat, 1)), step=1.0
                    )
                else:
                    opts = [str(c) for c in categories[feat]]
                    default = str(FORM_DEFAULTS.get(feat, opts[0]))
                    values[feat] = target.selectbox(
                        feat, opts, index=opts.index(default) if default in opts else 0
                    )
            submitted = st.form_submit_button("Assess credit risk", type="primary")

        if submitted:
            frame = pd.DataFrame([values])
            proba_good = float(model.predict_proba(frame)[0][1])
            decision = proba_good >= 0.5
            r1, r2 = st.columns([1, 2])
            r1.metric("Decision", "✅ GOOD credit" if decision else "🚨 BAD credit")
            r1.metric("P(good credit)", f"{proba_good:.1%}")
            r2.progress(proba_good, text=f"model confidence that this applicant repays: {proba_good:.1%}")
            r2.caption(
                "Probability is the weight-averaged output of the ExponentiatedGradient "
                "ensemble (14 constrained logistic regressions), thresholded at 0.5 — "
                "identical to the FastAPI/Kubernetes serving path."
            )

# ═══════════════════════════════ MODEL CARD ═════════════════════════════════
with tab_card:
    try:
        st.markdown(load_text("docs/model_card.md"))
    except Exception:
        st.warning("Could not fetch docs/model_card.md from the repo right now.")

# ═══════════════════════════════ CI STATUS ══════════════════════════════════
with tab_ci:
    st.header("Pipeline runs (GitHub Actions)")
    badge = {"success": "✅", "failure": "🚨", "cancelled": "⚪", "skipped": "⏭️",
             None: "🔄", "": "🔄"}
    try:
        runs = github_runs()
        latest = runs[0]
        st.subheader(f"Latest: {latest['display_title']}")
        st.markdown(
            f"{badge.get(latest['conclusion'], '🔄')} `{latest['name']}` · "
            f"{latest['status']} · [view on GitHub]({latest['html_url']})"
        )
        try:
            jobs = github_jobs(latest["id"])
            st.markdown(" → ".join(
                f"{badge.get(j['conclusion'], '🔄')} {j['name']}" for j in jobs
            ))
        except Exception:
            pass
        st.divider()
        rows = [{
            "run": f"[{r['display_title'][:60]}]({r['html_url']})",
            "workflow": r["name"], "status": badge.get(r["conclusion"], "🔄"),
            "when": r["created_at"][:16].replace("T", " "),
        } for r in runs]
        st.markdown(pd.DataFrame(rows).to_markdown(index=False))
        st.caption(
            "The fairness gate is a required jobs-graph edge: `deploy` needs "
            "`fairness_audit` to succeed, so a red audit leaves deploy *skipped*."
        )
    except Exception:
        st.warning("GitHub API unavailable (rate limit?) — try again in a minute.")
