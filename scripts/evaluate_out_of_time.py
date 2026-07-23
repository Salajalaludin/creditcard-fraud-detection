"""Evaluasi pseudo out-of-time memakai transaksi awal untuk training dan akhir untuk test."""

from __future__ import annotations

import json

from sklearn.ensemble import ExtraTreesClassifier

from fraud_detection.config import DEFAULT_DATA_PATH, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from fraud_detection.data import chronological_train_validation_test_split, clean_transactions, load_transactions  # noqa: E402
from fraud_detection.evaluation import classification_metrics, ranking_metrics_at_k  # noqa: E402
from tune_advanced_models import optimal_f1_threshold  # noqa: E402


def main() -> None:
    """Fit parameter yang sudah dibekukan lalu nilai satu pseudo-future period."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    x_train, x_validation, x_test, y_train, y_validation, y_test = chronological_train_validation_test_split(frame)
    model = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(x_train, y_train)
    validation_score = model.predict_proba(x_validation)[:, 1]
    threshold, _ = optimal_f1_threshold(y_validation.to_numpy(), validation_score)
    test_score = model.predict_proba(x_test)[:, 1]
    payload = {
        "protocol": "pseudo_out_of_time_70_15_15",
        "model": "extra_trees_leaf1_frozen_parameters",
        "threshold_selected_on": "chronological_validation",
        "threshold": threshold,
        "class_counts": {
            "train_fraud": int(y_train.sum()),
            "validation_fraud": int(y_validation.sum()),
            "test_fraud": int(y_test.sum()),
        },
        "validation": classification_metrics(y_validation.to_numpy(), validation_score, threshold),
        "test": {
            **classification_metrics(y_test.to_numpy(), test_score, threshold),
            **ranking_metrics_at_k(y_test.to_numpy(), test_score),
        },
        "caveat": "Hyperparameters were informed by earlier random-split experiments; external future data is still required.",
    }
    (REPORTS_DIR / "out_of_time_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
