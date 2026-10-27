"""Small, synthetic transaction data keeps regression tests fast and offline."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import BINARY_FEATURES, CONTINUOUS_FEATURES


@pytest.fixture
def transactions():
    rng = np.random.default_rng(15)
    rows = 300
    data = {name: rng.lognormal(size=rows) for name in CONTINUOUS_FEATURES}
    data.update({name: rng.integers(0, 2, size=rows) for name in BINARY_FEATURES})
    # Deliberately unrelated labels: this fixture tests behavior, not model quality.
    data["fraud"] = (np.arange(rows) % 5 == 0).astype(int)
    return pd.DataFrame(data)
