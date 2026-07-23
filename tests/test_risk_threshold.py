"""Unit tests untuk threshold, business cost, dan risk scoring."""

import numpy as np
import pandas as pd

from fraud_detection.risk import risk_level_boundaries, score_transactions
from fraud_detection.threshold import add_business_costs, build_threshold_table, select_best_threshold
from fraud_detection.policy_reports import BUSINESS_SCENARIOS, build_policy_recommendations


def test_threshold_table_confusion_counts() -> None:
    table = build_threshold_table(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.4, 0.9]),
        np.array([10, 20, 30, 40]),
        thresholds=np.array([0.5]),
    )
    row = table.iloc[0]
    assert (row.true_positive, row.false_positive, row.false_negative, row.true_negative) == (1, 1, 1, 1)


def test_cost_selection_enforces_minimum_recall() -> None:
    base = pd.DataFrame(
        {"threshold": [0.2, 0.8], "recall": [0.9, 0.5], "alerts": [10, 2], "precision": [0.5, 1.0], "false_positive": [5, 0], "false_negative": [1, 5], "missed_fraud_amount": [10.0, 50.0]}
    )
    cost = add_business_costs(base, 1, 1, 10)
    assert select_best_threshold(cost, 0.8).threshold == 0.2


def test_policy_recommendations_cover_every_scenario() -> None:
    table = build_threshold_table(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]), np.ones(4))
    recommendations, cost_tables = build_policy_recommendations(table)
    assert set(cost_tables) == set(BUSINESS_SCENARIOS)
    assert {f"business_{name}" for name in BUSINESS_SCENARIOS} <= set(recommendations["strategy"])


def test_all_alerts_are_at_least_medium_risk() -> None:
    threshold = 0.4
    frame = pd.DataFrame({"Time": [1, 2], "Amount": [10, 20]})
    scored = score_transactions(frame, np.array([0.39, 0.40]), threshold)
    assert scored.loc[0, "risk_level"] == "Low"
    assert scored.loc[1, "risk_level"] == "Medium"
    assert risk_level_boundaries(threshold)["low"][1] == 40
