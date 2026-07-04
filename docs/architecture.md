# Architecture

```mermaid
flowchart LR
    subgraph DVC["DVC pipeline (data versioned on DagsHub)"]
        A[fetch_data<br/>OpenML credit-g] --> B[preprocess<br/>derive sex / age_binary<br/>train / test / monitoring_pool]
        B --> C[make_reference_and_batches<br/>reference + 5 simulated batches<br/>drift injected from batch 4]
        B --> D[train<br/>sklearn Pipeline -> MLflow]
        D --> E[evaluate<br/>accuracy / F1 / ROC-AUC]
        D --> F[fairness_audit<br/>Fairlearn DP + EO per group]
    end

    subgraph CI["GitHub Actions: pipeline.yml"]
        L[linter] --> V[validate<br/>dvc pull + retrain + floors]
        V --> G{{fairness gate<br/>check_fairness_gate.py}}
        G -->|pass| H[deploy<br/>docker build --build-arg RUN_ID]
        G -->|fail| X[deploy skipped]
    end

    subgraph Serve["Serving"]
        H --> I[(Docker image<br/>model baked at build)]
        I --> J[Kubernetes<br/>2 replicas + /health probes]
        J --> K[FastAPI<br/>/predict /health /model-info /metrics]
    end

    subgraph Mon["Monitoring: monitoring.yml (weekly cron)"]
        C --> M[Evidently<br/>data + prediction drift]
        C --> N[Fairness drift<br/>same compute_fairness_metrics as the gate]
        M --> O[summary.json / summary.md]
        N --> O
    end

    D -. run_id .-> H
    F -. fairness_report.json .-> G
```

## Design decisions worth knowing

1. **Separation of storage concerns.** DVC versions *data* (raw, splits,
   simulated batches) on DagsHub; MLflow versions *models* and metrics. The
   only cross-reference is `model_info.txt` carrying the Run ID through the
   pipeline and into the Docker build — nothing is tracked twice.

2. **One fairness implementation, three consumers.**
   `src/fairness/metrics.py::compute_fairness_metrics` is called by the DVC
   audit stage, the CI gate, and the monitoring fairness-drift check.
   Training-time fairness and production-time fairness are therefore
   *provably* the same computation — a comparison between them is meaningful.

3. **Fairness through unawareness, then audit anyway.** The model never sees
   `personal_status`/`age` (or the derived `sex`/`age_binary`). The audit
   exists precisely because unawareness does not prevent disparate outcomes
   via correlated proxies (housing, employment, ...) — that is the entire
   reason outcome auditing is a thing.

4. **The gate is an exit code, deliberately.** `check_fairness_gate.py`
   exiting 1 fails the `fairness_audit` job; `deploy` requires it to succeed,
   so GitHub shows deploy as **skipped** — semantically "never attempted",
   not "broken". Simple, visible in the Actions graph, and promotable to a
   required status check via branch protection.

5. **Model baked at image build, not fetched at start.** Containers are
   self-contained: readiness probes measure *model readiness*, not network
   luck, and pods start with zero external dependencies. The cost — a new
   image per model version — is exactly the traceability you want anyway
   (image tag = MLflow Run ID).

6. **Monitoring is deterministic.** Simulated batches are generated once by a
   DVC stage and versioned, not sampled at monitoring time; anyone cloning
   the repo reproduces the exact alarm pattern (clean, clean, clean, ALARM,
   ALARM).
