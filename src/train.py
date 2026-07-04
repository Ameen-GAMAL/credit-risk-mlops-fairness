"""DVC stage `train`: fit the credit-risk model and log everything to MLflow.

The trained artifact (preprocessor + model as one sklearn Pipeline) lives in
MLflow only — DVC versions data, MLflow versions models; nothing is tracked
twice. `model_info.txt` carries the MLflow run id downstream (evaluate,
fairness audit, Docker build) exactly like the course Assign-5 pattern.
"""
from __future__ import annotations

import json

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.config import (
    METRICS_DIR,
    MLFLOW_EXPERIMENT,
    MODEL_INFO_PATH,
    TARGET_COL,
    TRAIN_PATH,
    feature_columns,
    load_params,
    tracking_uri,
)
from src.features.transformers import (
    FairPipeline,
    build_pipeline,
    build_preprocessor,
    split_feature_types,
)


def build_model(train_params: dict):
    model_type = train_params["model_type"]
    if model_type == "logreg":
        cfg = train_params["logreg"]
        return LogisticRegression(
            C=cfg["C"],
            max_iter=cfg["max_iter"],
            class_weight=cfg.get("class_weight"),
            random_state=train_params["random_state"],
        )
    if model_type == "xgboost":
        cfg = train_params["xgboost"]
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=train_params["random_state"],
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")


def fit_unmitigated(train_params: dict, df: pd.DataFrame, feature_cols: list[str], X, y):
    """Plain sklearn Pipeline + 5-fold CV."""
    pipeline = build_pipeline(build_model(train_params), df, feature_cols)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    pipeline.fit(X, y)
    extra_metrics = {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
    }
    return pipeline, extra_metrics


def fit_exponentiated_gradient(
    train_params: dict, fairness_params: dict, df: pd.DataFrame, feature_cols: list[str], X, y
):
    """Fairlearn in-processing reduction, constrained on the protected
    attributes jointly (intersectional groups). The mitigator only needs the
    sensitive features at TRAINING time; the returned FairPipeline serves
    from raw features alone."""
    from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient

    constraints = {
        "equalized_odds": EqualizedOdds,
        "demographic_parity": DemographicParity,
    }
    constraint_cls = constraints[train_params["eg_constraint"]]

    numeric, categorical = split_feature_types(df, feature_cols)
    preprocessor = build_preprocessor(numeric, categorical)
    Xt = preprocessor.fit_transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    mitigator = ExponentiatedGradient(
        estimator=build_model(train_params),
        constraints=constraint_cls(difference_bound=train_params["eg_eps"]),
    )
    sensitive = df[fairness_params["protected_attributes"]]
    mitigator.fit(Xt, y, sensitive_features=sensitive)

    extra_metrics = {
        "eg_best_gap": float(mitigator.best_gap_),
        "eg_n_predictors": int(len(mitigator.predictors_)),
    }
    return FairPipeline(preprocessor, mitigator), extra_metrics


def main() -> None:
    params = load_params()
    train_params = params["train"]
    mitigation = train_params.get("mitigation", "none")

    df = pd.read_csv(TRAIN_PATH)
    feature_cols = feature_columns(df)
    X, y = df[feature_cols], df[TARGET_COL]

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        if mitigation == "exponentiated_gradient":
            pipeline, extra_metrics = fit_exponentiated_gradient(
                train_params, params["fairness"], df, feature_cols, X, y
            )
        else:
            pipeline, extra_metrics = fit_unmitigated(train_params, df, feature_cols, X, y)
        y_pred = pipeline.predict(X)

        metrics = {
            **extra_metrics,
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "train_f1": float(f1_score(y, y_pred)),
        }

        mlflow.log_params(
            {
                "model_type": train_params["model_type"],
                **{f"model_{k}": v for k, v in train_params[train_params["model_type"]].items()},
                "mitigation": mitigation,
                "eg_constraint": train_params.get("eg_constraint"),
                "eg_eps": train_params.get("eg_eps"),
                "random_state": train_params["random_state"],
                "n_features": len(feature_cols),
                "n_train_rows": len(df),
                "protected_attrs_excluded_from_features": True,
            }
        )
        mlflow.set_tags(
            {
                "dataset": "openml credit-g v1",
                "project": "credit-risk-mlops-fairness",
                "stage": "training",
            }
        )
        mlflow.log_metrics(metrics)

        signature = infer_signature(X.head(5), pipeline.predict(X.head(5)))
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", signature=signature, input_example=X.head(2)
        )

        run_id = run.info.run_id

    MODEL_INFO_PATH.write_text(run_id, encoding="utf-8")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, **metrics}, fh, indent=2)

    print(f"run_id={run_id}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
