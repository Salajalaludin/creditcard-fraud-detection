"""Unit tests untuk feature engineering, drift, dan inference validation."""

import io

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.feature_engineering import add_engineered_features
from fraud_detection.monitoring import drift_level, population_stability_index
from fraud_detection.prediction import load_inference_data


def test_feature_engineering_does_not_mutate_input() -> None:
    frame = pd.DataFrame({"Time": [0, 43_200], "Amount": [0, 100]})
    result = add_engineered_features(frame)
    assert "Amount_Log" not in frame.columns
    assert {"Amount_Log", "Hour", "Time_Period", "Amount_Group"}.issubset(result.columns)


def test_psi_detects_large_shift() -> None:
    reference = pd.Series(np.arange(100))
    current = pd.Series(np.arange(100) + 1_000)
    psi = population_stability_index(reference, current)
    assert psi > 0.25
    assert drift_level(psi) == "Investigate"


def test_inference_rejects_missing_feature() -> None:
    frame = pd.DataFrame(np.zeros((2, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame = frame.drop(columns=["V28"])
    with pytest.raises(ValueError, match="V28"):
        load_inference_data(io.StringIO(frame.to_csv(index=False)))

