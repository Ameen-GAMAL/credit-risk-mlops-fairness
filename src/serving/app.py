"""FastAPI serving app.

Endpoints:
  GET  /health      liveness/readiness target — 200 only once the model is in
                    memory, so Kubernetes never routes traffic to a pod that
                    booted but hasn't finished loading.
  POST /predict     credit-risk inference on raw applicant fields.
  GET  /model-info  which MLflow Run ID is actually serving (demo of the
                    Docker ARG-threading story).
  GET  /metrics     Prometheus scrape target (request counts by outcome,
                    prediction latency histogram).

Model resolution order at startup:
  1. MODEL_DIR env — local artifact dir (baked into the Docker image).
  2. RUN_ID env    — download from MLFLOW_TRACKING_URI on the fly (dev mode).
  3. Neither       — app starts DEGRADED: /health returns 503, /predict 503.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from src.serving.model_loader import download_model, load_model
from src.serving.schemas import PredictRequest, PredictResponse

logger = logging.getLogger("credit-risk-api")

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total predictions served", ["outcome"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)


def _resolve_model():
    """Returns (model, run_id) or (None, run_id) if nothing is configured."""
    run_id = os.environ.get("RUN_ID")
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir and Path(model_dir).exists():
        logger.info("Loading model from baked dir %s", model_dir)
        return load_model(model_dir), run_id
    if run_id:
        logger.info("Downloading model for run %s from MLflow", run_id)
        target = Path(os.environ.get("MODEL_CACHE_DIR", "artifacts/model_download"))
        target.mkdir(parents=True, exist_ok=True)
        model_path = download_model(run_id, str(target))
        return load_model(str(model_path)), run_id
    logger.warning("No MODEL_DIR or RUN_ID set — starting degraded (no model)")
    return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model, app.state.run_id = _resolve_model()
    except Exception:  # pragma: no cover - startup resilience
        logger.exception("Model load failed — starting degraded")
        app.state.model, app.state.run_id = None, None
    yield


app = FastAPI(
    title="Credit Risk API",
    description="Credit-risk classifier served from an MLflow-tracked, "
    "fairness-gated training pipeline.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    if getattr(app.state, "model", None) is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_run_id": getattr(app.state, "run_id", None),
        "model_loaded": getattr(app.state, "model", None) is not None,
        "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    frame = payload.to_dataframe()
    prediction = int(model.predict(frame)[0])
    probability_good = float(model.predict_proba(frame)[0][1])
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    label = "good" if prediction == 1 else "bad"
    PREDICTIONS_TOTAL.labels(outcome=label).inc()

    return PredictResponse(
        prediction=prediction,
        label=label,
        probability_good=probability_good,
        model_run_id=getattr(app.state, "run_id", None),
    )
