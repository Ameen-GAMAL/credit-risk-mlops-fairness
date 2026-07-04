import numpy as np
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.schemas import PredictRequest


class StubPipeline:
    """Stands in for the MLflow-loaded sklearn pipeline."""

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


SAMPLE = PredictRequest.model_config["json_schema_extra"]["example"]


def test_health_degraded_then_ready():
    # No MODEL_DIR/RUN_ID in the test env -> app starts degraded
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=SAMPLE).status_code == 503

        app.state.model = StubPipeline()
        app.state.run_id = "test-run"
        assert client.get("/health").status_code == 200


def test_predict_returns_valid_schema():
    with TestClient(app) as client:
        app.state.model = StubPipeline()
        app.state.run_id = "test-run"

        resp = client.post("/predict", json=SAMPLE)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 1
        assert body["label"] == "good"
        assert body["probability_good"] == 0.8
        assert body["model_run_id"] == "test-run"


def test_model_info_and_metrics_endpoints():
    with TestClient(app) as client:
        app.state.model = StubPipeline()
        app.state.run_id = "test-run"

        info = client.get("/model-info").json()
        assert info["model_run_id"] == "test-run"
        assert info["model_loaded"] is True

        client.post("/predict", json=SAMPLE)
        metrics = client.get("/metrics/").text
        assert "predictions_total" in metrics
