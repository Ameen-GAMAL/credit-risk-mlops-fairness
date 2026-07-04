# syntax=docker/dockerfile:1
# Serving image: bakes the model for a SPECIFIC MLflow Run ID in at build
# time, so containers are self-contained (no live MLflow/network dependency
# at startup — that's what makes the K8s readiness probe meaningful).
#
# Build (BuildKit secret keeps the DagsHub token out of image layers):
#   echo $DAGSHUB_TOKEN > mlflow_token.txt
#   docker build \
#     --secret id=mlflow_token,src=mlflow_token.txt \
#     --build-arg RUN_ID=<mlflow_run_id> \
#     --build-arg MLFLOW_TRACKING_URI=https://dagshub.com/s-amin.mohamed/credit-risk-mlops-fairness.mlflow \
#     -t credit-risk-api:<run_id> -f docker/Dockerfile .
FROM python:3.12-slim

ARG RUN_ID
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME=s-amin.mohamed

ENV RUN_ID=${RUN_ID} \
    MODEL_DIR=/app/model_artifact/model \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py src/__init__.py
COPY src/serving/ src/serving/
# FairPipeline is pickled by reference into the MLflow artifact — its module
# must be importable at model-load time.
COPY src/features/ src/features/

# Pull the tracked model at BUILD time. The token is mounted as a BuildKit
# secret: available to this RUN only, never written into a layer.
RUN --mount=type=secret,id=mlflow_token \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME} \
    MLFLOW_TRACKING_PASSWORD=$(cat /run/secrets/mlflow_token) \
    python -m src.serving.model_loader --run-id ${RUN_ID} --out /app/model_artifact

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
