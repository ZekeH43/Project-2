"""Protect data validation, class balance, and train/test separation."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import FEATURES, load_data, split_data, validate_data


def test_bom_csv_and_reordered_columns(transactions, tmp_path):
    path = tmp_path / "transactions.csv"
    transactions.iloc[:, ::-1].to_csv(path, index=False, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(load_data(path), transactions, check_exact=False)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("distance_from_home", np.nan, "Missing or infinite"),
        ("distance_from_home", np.inf, "Missing or infinite"),
        ("distance_from_home", -1, "cannot be negative"),
        ("ratio_to_median_purchase_price", -0.1, "cannot be negative"),
        ("used_chip", 2, "must be 0 or 1"),
        ("fraud", 0.5, "must be 0 or 1"),
        ("fraud", 2, "must be 0 or 1"),
    ],
)
def test_rejects_invalid_values(transactions, column, value, message):
    transactions[column] = transactions[column].astype(float)
    transactions.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_data(transactions)


def test_rejects_text_values(transactions):
    transactions["distance_from_home"] = "unknown"
    with pytest.raises(ValueError, match="numeric"):
        validate_data(transactions)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_rejects_wrong_schema(transactions, change):
    if change == "missing":
        transactions = transactions.drop(columns="fraud")
    elif change == "extra":
        transactions["account_id"] = 1
    else:
        transactions = pd.concat([transactions, transactions[["fraud"]]], axis=1)
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_data(transactions)


def test_rejects_duplicate_rows(transactions):
    duplicate = pd.concat([transactions, transactions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate transaction"):
        validate_data(duplicate)


@pytest.mark.parametrize("fraud_rows", [0, 1])
def test_rejects_insufficient_classes(transactions, fraud_rows):
    transactions["fraud"] = 0
    transactions.loc[: fraud_rows - 1, "fraud"] = 1
    with pytest.raises(ValueError, match="Both fraud classes"):
        validate_data(transactions)


def test_rejects_too_few_rows(transactions):
    with pytest.raises(ValueError, match="at least 20"):
        validate_data(transactions.iloc[:19])


@pytest.mark.parametrize("sample_size", [0, -1, 19, 20.5, True])
def test_rejects_invalid_sample_size(transactions, sample_size):
    with pytest.raises(ValueError, match="Sample size"):
        split_data(transactions, sample_size=sample_size)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1, float("nan")])
def test_rejects_invalid_test_size(transactions, test_size):
    with pytest.raises(ValueError, match="Test size"):
        split_data(transactions, test_size=test_size)


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_rejects_invalid_seed(transactions, seed):
    with pytest.raises(ValueError, match="Seed"):
        split_data(transactions, seed=seed)


def test_split_is_disjoint_stratified_and_repeatable(transactions):
    first = split_data(transactions, sample_size=200)
    second = split_data(transactions, sample_size=200)
    assert set(first.X_train.index).isdisjoint(first.X_test.index)
    assert len(first.X_train) == 150
    assert len(first.X_test) == 50
    assert first.y_train.sum() == 30
    assert first.y_test.sum() == 10
    assert tuple(first.X_train.columns) == FEATURES
    assert first.X_train.index.equals(first.y_train.index)
    assert first.X_test.index.equals(first.y_test.index)
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)
    # Sampling is spread over the source CSV, not just its first N rows.
    assert max(first.X_train.index.max(), first.X_test.index.max()) >= 200


def test_small_csv_uses_every_available_row(transactions):
    split = split_data(transactions, sample_size=10_000)
    assert len(split.X_train) + len(split.X_test) == len(transactions)


def test_all_rows_option_matches_oversized_sample(transactions):
    pd.testing.assert_frame_equal(
        split_data(transactions, sample_size=None).X_test,
        split_data(transactions, sample_size=10_000).X_test,
    )


def test_cannot_leave_a_class_out_of_holdout(transactions):
    transactions["fraud"] = 0
    transactions.loc[:1, "fraud"] = 1
    with pytest.raises(ValueError, match="both classes|stratified"):
        split_data(transactions, test_size=0.01)
