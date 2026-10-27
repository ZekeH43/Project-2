"""Train the original three classifiers and report correctly oriented metrics."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.data import DataSplit, split_data

MODEL_NAMES = ("baseline", "knn", "gradient_boosting", "xgboost")


@dataclass
class Experiment:
    """Expose results and fitted models for the CLI and notebook walkthrough."""

    split: DataSplit
    models: dict[str, Any]
    metrics: pd.DataFrame
    details: dict[str, dict]


def build_models(seed: int = 42) -> dict[str, Any]:
    """Build fresh estimators without fitting or accessing any data.

    KNN uses distances, so its scaler is part of the fitted pipeline. Trees do
    not need scaling. XGBoost is a separate comparison model, not a check that
    another model has avoided overfitting. All models run on the CPU.
    """
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "knn": Pipeline(
            [("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=3))]
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=seed,
        ),
    }


def classification_metrics(y_true, y_pred, fraud_probability) -> dict:
    """Evaluate class 1 (fraud); matrix rows are actual, columns are predicted.

    Actual labels MUST be the first argument. Reversing the arguments swaps
    the interpretation of precision and recall, even if accuracy looks right.
    Average precision and ROC AUC use scores, not hard class predictions.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, fraud_probability)),
        "roc_auc": float(roc_auc_score(y_true, fraud_probability)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["not_fraud", "fraud"],
            output_dict=True,
            zero_division=0,
        ),
    }


def run_experiment(
    frame: pd.DataFrame,
    *,
    sample_size: int | None = 10_000,
    test_size: float = 0.25,
    seed: int = 42,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> Experiment:
    """Fit each selected model on one shared split, without holdout tuning."""
    if not model_names or set(model_names) - set(MODEL_NAMES):
        raise ValueError("Choose at least one model from: " + ", ".join(MODEL_NAMES))
    split = split_data(frame, sample_size=sample_size, test_size=test_size, seed=seed)
    models = {name: model for name, model in build_models(seed).items() if name in model_names}
    details = {}
    for name, model in models.items():
        # Pipeline.fit learns KNN scaling from X_train only.
        model.fit(split.X_train, split.y_train)
        predicted = model.predict(split.X_test)
        fraud_column = list(model.classes_).index(1)
        probability = model.predict_proba(split.X_test)[:, fraud_column]
        details[name] = classification_metrics(split.y_test, predicted, probability)
    metrics = pd.DataFrame.from_dict(
        {
            name: {key: value for key, value in result.items() if isinstance(value, float)}
            for name, result in details.items()
        },
        orient="index",
    ).rename_axis("model")
    return Experiment(split, models, metrics, details)
