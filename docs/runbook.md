# Runbook — reproduce this project end to end

Everything below assumes the repo root as working directory.

## 0. Prerequisites

| Tool | Used for | Notes |
|---|---|---|
| Python 3.12 | everything | 3.13 not yet supported by the pinned `evidently` |
| Git | version control | |
| Docker Desktop (with BuildKit) | serving image | default builder since 2023 |
| Minikube **or** Kind + kubectl | local Kubernetes | no cloud account needed |
| DagsHub account (free) | hosted MLflow + DVC remote | |

## 1. Local setup (fully offline-capable)

```bash
py -3.12 -m venv .venv               # Windows; python3.12 -m venv .venv elsewhere
.venv\Scripts\activate               # source .venv/bin/activate on Linux/macOS
pip install -r requirements-dev.txt
```

Without any env vars set, MLflow tracks to local `./mlruns` — the entire DVC
pipeline runs offline:

```bash
dvc repro          # fetch_data -> preprocess -> batches -> train -> evaluate -> fairness_audit
pytest             # unit + API tests
flake8 src scripts tests
```

Inspect results: `metrics/*.json`, or `mlflow ui` and open http://localhost:5000.

## 2. DagsHub hookup (hosted MLflow + DVC remote)

1. Create a DagsHub repo named `credit-risk-mlops-fairness` (Connect → New repo).
2. Mint a **fresh** token: dagshub.com → Settings → Tokens. Do **not** reuse
   tokens from older projects (any token that ever sat in plaintext on disk
   should be treated as burned).
3. Wire credentials (stored in `.dvc/config.local`, which is gitignored):

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user s-amin.mohamed
dvc remote modify origin --local password <FRESH_TOKEN>
dvc push                             # upload versioned data
```

4. For MLflow tracking against DagsHub, set (or put in `.env`):

```bash
set MLFLOW_TRACKING_URI=https://dagshub.com/s-amin.mohamed/credit-risk-mlops-fairness.mlflow
set MLFLOW_TRACKING_USERNAME=s-amin.mohamed
set MLFLOW_TRACKING_PASSWORD=<FRESH_TOKEN>
dvc repro --force train evaluate fairness_audit   # runs now land in DagsHub
```

## 3. GitHub hookup (CI/CD)

1. Create `github.com/Ameen-GAMAL/credit-risk-mlops-fairness`, push `main`.
2. Repo → Settings → Secrets and variables → Actions → add:
   - `DAGSHUB_TOKEN` — the fresh token
   - `MLFLOW_TRACKING_URI` — the `.mlflow` URI above
3. Trigger the full pipeline: commit to `main` with `[run-train]` in the
   message (or Actions → run `workflow_dispatch`). Plain commits run only the
   linter — that's the conditional-execution gate.
4. **Get the fairness-gate-blocking screenshot** (worth more than the gate
   merely existing): on a throwaway branch/commit, lower
   `fairness.demographic_parity_threshold` in `params.yaml` to `0.01`, push
   with `[run-train]`, and watch `fairness_audit` fail → `deploy` shows
   **skipped**. Revert.

## 4. Serving locally (no Docker)

```bash
python -m src.serving.model_loader --run-id <RUN_ID> --out artifacts/model_download
set MODEL_DIR=artifacts/model_download/model
set RUN_ID=<RUN_ID>
uvicorn src.serving.app:app --port 8000
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"checking_status\": \"<0\", \"duration\": 24, \"credit_history\": \"existing paid\", \"purpose\": \"radio/tv\", \"credit_amount\": 3500, \"savings_status\": \"<100\", \"employment\": \"1<=X<4\", \"installment_commitment\": 3, \"other_parties\": \"none\", \"residence_since\": 2, \"property_magnitude\": \"car\", \"other_payment_plans\": \"none\", \"housing\": \"own\", \"existing_credits\": 1, \"job\": \"skilled\", \"num_dependents\": 1, \"own_telephone\": \"yes\", \"foreign_worker\": \"yes\"}"
```

## 5. Docker image (model baked by Run ID)

```bash
echo <FRESH_TOKEN> > mlflow_token.txt
docker build ^
  --secret id=mlflow_token,src=mlflow_token.txt ^
  --build-arg RUN_ID=<RUN_ID> ^
  --build-arg MLFLOW_TRACKING_URI=https://dagshub.com/s-amin.mohamed/credit-risk-mlops-fairness.mlflow ^
  -t credit-risk-api:local -f docker/Dockerfile .
del mlflow_token.txt
docker run -p 8000:8000 credit-risk-api:local
```

Why a BuildKit secret and not a plain `ARG`: an ARG value is recorded in the
image layer history (`docker history`), so a token passed that way ships with
the image. The secret mount exists only during the one `RUN` that needs it.

**Local-only variant (no DagsHub yet):** serve your local `mlruns` briefly
(`mlflow server --backend-store-uri ./mlruns --port 5000`) and build with
`--build-arg MLFLOW_TRACKING_URI=http://host.docker.internal:5000` and a dummy
token file.

## 6. Kubernetes (Minikube or Kind)

Minikube (image built inside its Docker daemon — no registry needed):

```bash
minikube start
minikube docker-env | Invoke-Expression        # PowerShell (eval $(minikube docker-env) on bash)
docker build ... -t credit-risk-api:local -f docker/Dockerfile .   # as in §5
kubectl apply -k k8s/
kubectl get pods,svc -n credit-risk-mlops
minikube service credit-risk-api-service -n credit-risk-mlops
```

Kind:

```bash
kind create cluster
docker build ... -t credit-risk-api:local -f docker/Dockerfile .
kind load docker-image credit-risk-api:local
kubectl apply -k k8s/
kubectl port-forward svc/credit-risk-api-service 8000:80 -n credit-risk-mlops
```

Probe verification (proves the probes are load-bearing): temporarily change
the readiness path in `k8s/deployment.yaml` to `/nope`, re-apply, and watch
`kubectl get pods` report `0/1 READY` while the old ReplicaSet keeps serving.

## 7. Monitoring

```bash
python -m src.monitoring.run_monitoring
```

Outputs land in `monitoring/reports/`: per-batch Evidently HTML/JSON,
per-batch fairness-drift JSON, and the rollup `summary.json` / `summary.md`.
Expected shape: batches 1–3 clean, batches 4–5 alarm on BOTH data drift and
fairness drift (that's where `make_reference_and_batches.py` injects the
synthetic age/credit-amount shift). In CI, the same run happens weekly via
`.github/workflows/monitoring.yml` and prints the table to the workflow
summary page.

## Troubleshooting

- **`dvc pull` 403** — token wrong/expired, or `--local` remote creds not set
  (they live in `.dvc/config.local`, which is never committed).
- **MLflow 401 against DagsHub** — all three `MLFLOW_*` env vars must be set;
  the password is the token itself.
- **Evidently import errors** — you're on Python 3.13 or evidently ≥0.5
  (breaking API change); use Python 3.12 and the pinned requirements.
- **OneDrive** — this repo lives under OneDrive; if sync churn on `.venv/`
  becomes annoying, either mark `.venv` as "always keep on this device /
  don't sync" or move the clone outside the synced tree. Functionally
  everything works either way.
