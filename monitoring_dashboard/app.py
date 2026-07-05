"""Fairness & drift monitoring dashboard.

Reads ONLY the JSON artifacts the pipeline already produces (gate report,
evaluation metrics, per-batch monitoring output) — no model, no MLflow, no
fairlearn imports. Runs in two modes transparently:

  * locally, inside the repo: reads the files straight from disk
  * on Hugging Face Spaces: fetches the same files from the public GitHub
    repo's raw URLs, so the Space always reflects `main` with zero secrets
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

REPO = "Ameen-GAMAL/credit-risk-mlops-fairness"
GITHUB_RAW = f"https://raw.githubusercontent.com/{REPO}/main"
LOCAL_ROOT = Path(__file__).resolve().parents[1]

LINKS = {
    "GitHub repo": f"https://github.com/{REPO}",
    "MLflow (DagsHub)": "https://dagshub.com/s-amin.mohamed/credit-risk-mlops-fairness.mlflow",
    "CI: all gates green": f"https://github.com/{REPO}/actions/runs/28720895645",
    "CI: gate blocking deploy": f"https://github.com/{REPO}/actions/runs/28721066495",
}


@st.cache_data(ttl=300)
def load_json(rel_path):
    local = LOCAL_ROOT / rel_path
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    resp = requests.get(f"{GITHUB_RAW}/{rel_path}", timeout=15)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Credit Risk — Fairness Monitor", page_icon="⚖️", layout="wide")

st.title("⚖️ Credit Risk Model — Fairness & Drift Monitor")
st.caption(
    "A fairness-gated MLOps pipeline: deployment is blocked in CI unless a "
    "Fairlearn audit passes, and production monitoring watches *fairness "
    "drift* next to ordinary data drift. "
    + " · ".join(f"[{k}]({v})" for k, v in LINKS.items())
)

summary = load_json("monitoring/reports/summary.json")
gate = load_json("metrics/fairness_report.json")
evaluation = load_json("metrics/evaluation.json")

# ── Panel 1: the shipped model and its gate ─────────────────────────────────
st.header("Shipped model — CI gate status")
left, right = st.columns([1, 2])
with left:
    st.metric("Test accuracy", f"{evaluation['test_accuracy']:.3f}")
    st.metric("Test F1", f"{evaluation['test_f1']:.3f}")
    st.metric("ROC-AUC", f"{evaluation['test_roc_auc']:.3f}")
    st.markdown(
        ("✅ **Fairness gate: PASSED**" if gate["passed"] else "🚨 **Fairness gate: FAILED**")
        + f"  \nMLflow run `{gate['run_id'][:12]}…`"
    )
with right:
    dp_thr = gate["thresholds"]["demographic_parity_threshold"]
    eo_thr = gate["thresholds"]["equalized_odds_threshold"]
    rows = []
    for attr, m in gate["metrics"].items():
        rows.append(
            {
                "protected attribute": attr,
                "DP difference": m["demographic_parity_difference"],
                f"DP limit ({dp_thr})": "✅" if m["demographic_parity_difference"] <= dp_thr else "🚨",
                "EO difference": m["equalized_odds_difference"],
                f"EO limit ({eo_thr})": "✅" if m["equalized_odds_difference"] <= eo_thr else "🚨",
                "group sizes": ", ".join(f"{k}: {v}" for k, v in m["group_counts"].items()),
            }
        )
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Audited on the held-out test set with the same `compute_fairness_metrics` "
        "implementation the CI gate and this monitor use — one metric definition, "
        "three consumers."
    )

# ── Panel 2: fairness drift across simulated production batches ────────────
st.header("Fairness drift across production batches")
batches = [b["batch"] for b in summary["batches"]]
threshold = summary["fairness_drift_threshold"]
fairness_by_batch = {b: load_json(f"monitoring/reports/{b}_fairness_drift.json") for b in batches}

attr_cols = st.columns(len(summary["reference_fairness"]))
for col, attr in zip(attr_cols, summary["reference_fairness"]):
    ref_dp = summary["reference_fairness"][attr]["demographic_parity_difference"]
    dp_cur = [fairness_by_batch[b][attr]["demographic_parity_difference"]["current"] for b in batches]
    dp_flag = [fairness_by_batch[b][attr]["demographic_parity_difference"]["drift_flagged"] for b in batches]
    eo_cur = [fairness_by_batch[b][attr]["equalized_odds_difference"]["current"] for b in batches]

    fig = go.Figure()
    fig.add_hrect(
        y0=max(ref_dp - threshold, 0), y1=ref_dp + threshold,
        fillcolor="green", opacity=0.08, line_width=0,
    )
    fig.add_hline(y=ref_dp, line_dash="dash", line_color="green",
                  annotation_text="reference DP", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=batches, y=dp_cur, mode="lines+markers", name="DP difference (alarmed)",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=12, color=["#d62728" if f else "#1f77b4" for f in dp_flag],
                    symbol=["x" if f else "circle" for f in dp_flag]),
    ))
    fig.add_trace(go.Scatter(
        x=batches, y=eo_cur, mode="lines+markers", name="EO difference (reported only)",
        line=dict(color="#7f7f7f", dash="dot"), marker=dict(size=7, color="#7f7f7f"),
    ))
    fig.update_layout(
        title=f"{attr}", height=340, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="metric value", legend=dict(orientation="h", y=-0.25),
    )
    col.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Red ✕ = |ΔDP vs reference| > {threshold} (threshold set above the measured "
    "bootstrap noise floor for 120-row batches). EO is charted but deliberately "
    "not alarmed: it needs true labels — which arrive with delay in real credit — "
    "and its per-group TPR/FPR cells are too small at this batch size to alarm on "
    "without alarming on noise."
)

# ── Panel 3: data drift + alarm rollup ──────────────────────────────────────
st.header("Data drift & alarm rollup")
c1, c2 = st.columns([1, 1])
with c1:
    n_drift = [b["n_drifted_columns"] for b in summary["batches"]]
    ds_flag = [b["dataset_drift"] for b in summary["batches"]]
    fig = go.Figure(go.Bar(
        x=batches, y=n_drift,
        marker_color=["#d62728" if f else "#2ca02c" for f in ds_flag],
        text=[("DRIFT" if f else "ok") for f in ds_flag], textposition="outside",
    ))
    fig.update_layout(
        title="Evidently: drifted columns per batch (red = dataset-level drift)",
        height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="# drifted columns",
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    table = pd.DataFrame(summary["batches"]).rename(columns={
        "batch": "batch", "dataset_drift": "data drift", "n_drifted_columns": "drifted cols",
        "prediction_drift": "prediction drift", "fairness_drift": "fairness drift", "any_alarm": "ALARM",
    })
    for col_name in ("data drift", "prediction drift", "fairness drift"):
        table[col_name] = table[col_name].map({True: "🚨", False: "—"})
    table["ALARM"] = table["ALARM"].map({True: "🚨 ALARM", False: "✅ ok"})
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Monitoring generated {summary['generated_at'][:19]}Z · model run `{summary['model_run_id'][:12]}…`")

with st.expander("What am I looking at? (the simulation design)"):
    st.markdown(
        """
Batches 1–3 are bootstrap resamples of the monitoring pool — statistically
indistinguishable from the reference set, so a well-calibrated detector should
stay quiet. From **batch 4**, a synthetic scenario is injected: *the applicant
pool skews younger, and young applicants request ~60% larger loans*. The
injection lives in one clearly-named function
(`src/data/make_reference_and_batches.py::inject_drift`), and the batches are
versioned with DVC, so this exact alarm pattern reproduces from a fresh clone.

What the detectors show: the **data-drift alarm fires first** (batch 4 — the
leading indicator), and the **fairness-drift alarm confirms on batch 5** as the
shock persists. Notably, the *age* parity gap barely moves under an
age-targeted shock — the model's equalized-odds training constraint absorbs
it — and the residual disparity surfaces on the correlated *sex* margin
instead. Constraining one attribute does not inoculate its neighbours;
monitor all of them.
        """
    )
