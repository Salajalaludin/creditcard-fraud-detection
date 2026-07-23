"""Unit tests untuk loading, cleaning, dan splitting."""

import numpy as np
import pandas as pd

from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.data import (
    chronological_train_validation_test_split,
    clean_transactions,
    stratified_train_validation_test_split,
)


def synthetic_frame(rows: int = 200) -> pd.DataFrame:
    """Buat dataset kecil dengan minimal fraud untuk pengujian split."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["Time"] = np.arange(rows)
    frame["Amount"] = np.abs(frame["Amount"])
    frame["Class"] = 0
    frame.loc[::20, "Class"] = 1
    return frame


def test_clean_transactions_removes_exact_duplicates() -> None:
    frame = synthetic_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    clean, report = clean_transactions(duplicated)
    assert len(clean) == len(frame)
    assert report["duplicates_removed"] == 1


def test_stratified_split_is_disjoint_and_preserves_rows() -> None:
    frame = synthetic_frame()
    parts = stratified_train_validation_test_split(frame)
    features, targets = parts[:3], parts[3:]
    assert sum(map(len, features)) == len(frame)
    assert all(x.index.equals(y.index) for x, y in zip(features, targets))
    assert all(y.sum() > 0 for y in targets)


def test_chronological_split_respects_time_order() -> None:
    parts = chronological_train_validation_test_split(synthetic_frame())
    train, validation, test = parts[:3]
    assert train["Time"].max() < validation["Time"].min()
    assert validation["Time"].max() < test["Time"].min()
