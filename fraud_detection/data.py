"""Load the transaction CSV and make a reproducible, stratified data split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "Resources" / "credit card data.csv"
CONTINUOUS_FEATURES = (
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
)
BINARY_FEATURES = ("repeat_retailer", "used_chip", "used_pin_number", "online_order")
FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES
TARGET = "fraud"


@dataclass
class DataSplit:
    """Keep features and labels together without including the target in features."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric data in a fixed column order; explain invalid inputs early.

    Exact duplicate rows are rejected so they cannot appear on both sides of a
    random split. Missing values are rejected rather than silently imputed.
    """
    expected = [*FEATURES, TARGET]
    if frame.columns.has_duplicates or set(frame.columns) != set(expected):
        raise ValueError("CSV columns must be exactly: " + ", ".join(expected))
    if len(frame) < 20:
        raise ValueError("Provide at least 20 transaction rows.")
    try:
        data = frame.loc[:, expected].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All transaction columns must contain numeric values.") from exc
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite values are not supported.")
    if (data.loc[:, list(CONTINUOUS_FEATURES)] < 0).any().any():
        raise ValueError("Distances and purchase-price ratios cannot be negative.")
    if not data.loc[:, [*BINARY_FEATURES, TARGET]].isin([0, 1]).all().all():
        raise ValueError("Flags and fraud labels must be 0 or 1.")
    if data.duplicated().any():
        raise ValueError("Resolve duplicate transaction rows before splitting the data.")
    data = data.astype({column: int for column in (*BINARY_FEATURES, TARGET)})
    counts = data[TARGET].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Both fraud classes need at least two rows.")
    return data.reset_index(drop=True)


def load_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load a local CSV, including the BOM used by the smaller legacy CSV."""
    return validate_data(pd.read_csv(path, encoding="utf-8-sig"))


def split_data(
    frame: pd.DataFrame,
    *,
    sample_size: int | None = 10_000,
    test_size: float = 0.25,
    seed: int = 42,
) -> DataSplit:
    """Sample across the whole CSV, then reserve the same holdout for every model.

    Both operations preserve class proportions as closely as integer counts
    allow. A fixed seed makes the split repeatable. None uses the entire CSV;
    a requested sample larger than the dataset also uses the entire dataset.
    """
    if sample_size is not None and (
        isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 20
    ):
        raise ValueError("Sample size must be an integer of at least 20, or None.")
    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1, excluding the endpoints.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 through 4294967295.")
    data = validate_data(frame)
    try:
        if sample_size is not None and sample_size < len(data):
            data, _ = train_test_split(
                data, train_size=sample_size, stratify=data[TARGET], random_state=seed
            )
        X_train, X_test, y_train, y_test = train_test_split(
            data.loc[:, list(FEATURES)],
            data[TARGET],
            test_size=test_size,
            stratify=data[TARGET],
            random_state=seed,
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot create a stratified split. Increase the sample size or adjust test size."
        ) from exc
    if len(X_train) < 3 or y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "The split needs both classes in train and test, and at least three training rows."
        )
    return DataSplit(X_train, X_test, y_train, y_test)
