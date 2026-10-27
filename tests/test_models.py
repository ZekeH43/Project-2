"""Regression tests for the original metric-order bug and actual model fitting."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models import MODEL_NAMES, classification_metrics, run_experiment


def test_precision_recall_and_support_use_actual_labels_first():
    # TP=1, FP=2, FN=1, TN=1: precision and recall must differ.
    result = classification_metrics([0, 0, 0, 1, 1], [1, 1, 0, 1, 0], [0.9, 0.8, 0.1, 0.7, 0.2])
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 2)
    assert result["confusion_matrix"] == [[1, 2], [1, 1]]
    assert result["classification_report"]["fraud"]["support"] == 2
    assert result["classification_report"]["not_fraud"]["support"] == 3


def test_average_precision_uses_ranking_not_hard_predictions():
    result = classification_metrics([0, 1, 0, 1], [0, 0, 0, 0], [0.1, 0.4, 0.2, 0.3])
    assert result["recall"] == 0
    assert result["average_precision"] == 1
    assert result["roc_auc"] == 1


def test_all_models_fit_and_baseline_exposes_misleading_accuracy(transactions):
    experiment = run_experiment(transactions)
    assert tuple(experiment.models) == MODEL_NAMES
    assert np.isfinite(experiment.metrics.to_numpy()).all()
    assert experiment.metrics.loc["baseline", "accuracy"] == pytest.approx(0.8)
    assert experiment.metrics.loc["baseline", "recall"] == 0
    assert experiment.metrics.loc["baseline", "average_precision"] == pytest.approx(0.2)
    for result in experiment.details.values():
        assert sum(map(sum, result["confusion_matrix"])) == 75


def test_scaler_is_fitted_only_on_training_data(transactions):
    experiment = run_experiment(transactions, sample_size=200, model_names=("knn",))
    scaler = experiment.models["knn"].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, experiment.split.X_train.mean())
    assert scaler.n_samples_seen_ == len(experiment.split.X_train)
    assert not np.allclose(scaler.mean_, transactions.drop(columns="fraud").mean())


def test_model_results_are_repeatable(transactions):
    first = run_experiment(transactions)
    second = run_experiment(transactions)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)


@pytest.mark.parametrize("models", [(), ("typo",)])
def test_invalid_model_selection(transactions, models):
    with pytest.raises(ValueError, match="Choose at least one"):
        run_experiment(transactions, model_names=models)
