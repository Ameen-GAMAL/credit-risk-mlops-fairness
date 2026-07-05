---
title: Credit Risk Fairness Monitor
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Fairness-drift monitoring for a gated MLOps pipeline
---

# Credit Risk — Fairness & Drift Monitoring Dashboard

Live monitoring view for
[Ameen-GAMAL/credit-risk-mlops-fairness](https://github.com/Ameen-GAMAL/credit-risk-mlops-fairness) —
an end-to-end MLOps pipeline where a Fairlearn audit **gates deployment in
CI** and production monitoring watches **fairness drift** alongside ordinary
data drift.

The dashboard reads the pipeline's committed JSON artifacts directly from the
GitHub repo at runtime, so it always reflects `main` and holds no secrets.
