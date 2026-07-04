"""Shared sklearn preprocessing, used by training and (via the logged MLflow
pipeline) by serving. The whole preprocessor + model is logged as ONE sklearn
Pipeline, so serving never re-implements feature handling."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            # handle_unknown="ignore": serving must not 500 on a category
            # value the training split happened not to contain.
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",  # protected attrs/target in monitoring frames are ignored
    )


def build_pipeline(model, df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    numeric, categorical = split_feature_types(df, feature_cols)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("model", model),
        ]
    )


class FairPipeline:
    """Serving-compatible wrapper around (fitted preprocessor, fitted
    fairlearn ExponentiatedGradient mitigator).

    Exposes the same predict/predict_proba surface as a plain sklearn
    Pipeline, so evaluation, the fairness audit, monitoring, and the FastAPI
    app treat mitigated and unmitigated models identically. Probabilities are
    the weight-averaged predict_proba over the reduction's inner ensemble
    (`predictors_`/`weights_` are fairlearn's public attributes).

    NOTE: instances are pickled by reference into the MLflow artifact, so
    this module must be importable wherever the model is loaded (the Docker
    image COPYs src/features/ for exactly this reason).
    """

    def __init__(self, preprocessor, mitigator):
        self.preprocessor = preprocessor
        self.mitigator = mitigator

    def _ensemble_proba(self, Xt):
        import numpy as np

        weights = self.mitigator.weights_.to_numpy()
        weights = weights / weights.sum()
        proba = np.zeros((Xt.shape[0], 2))
        for w, clf in zip(weights, self.mitigator.predictors_):
            if w > 0:
                proba += w * clf.predict_proba(Xt)
        return proba

    def predict(self, X):
        proba = self._ensemble_proba(self.preprocessor.transform(X))
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self._ensemble_proba(self.preprocessor.transform(X))
