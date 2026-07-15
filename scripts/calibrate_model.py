"""Kalibrasi model aktif dengan validation subset terpisah dan audit pada test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_detection.config import DEFAULT_DATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from fraud_detection.data import clean_transactions, load_transactions, stratified_train_validation_test_split  # noqa: E402
from fraud_detection.evaluation import classification_metrics  # noqa: E402
from tune_advanced_models import optimal_f1_threshold  # noqa: E402


def main() -> None:
    """Pisahkan validation menjadi calibration dan threshold subsets."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    _, x_validation, x_test, _, y_validation, y_test = stratified_train_validation_test_split(frame)
    x_calibration, x_threshold, y_calibration, y_threshold = train_test_split(
        x_validation,
        y_validation,
        test_size=0.5,
        stratify=y_validation,
        random_state=RANDOM_STATE,
    )
    base_model = joblib.load(MODELS_DIR / "fraud_detection_model.joblib")
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(x_calibration, y_calibration)

    threshold_score = calibrated.predict_proba(x_threshold)[:, 1]
    threshold, _ = optimal_f1_threshold(y_threshold.to_numpy(), threshold_score)
    test_score = calibrated.predict_proba(x_test)[:, 1]
    payload = {
        "base_model": "extra_trees_leaf1",
        "method": "sigmoid",
        "calibration_rows": len(x_calibration),
        "threshold_rows": len(x_threshold),
        "threshold": threshold,
        "threshold_metrics": classification_metrics(y_threshold.to_numpy(), threshold_score, threshold),
        "test_metrics": classification_metrics(y_test.to_numpy(), test_score, threshold),
        "calibration_metrics": {
            "test_brier_score": float(brier_score_loss(y_test, test_score)),
            "test_log_loss": float(log_loss(y_test, test_score)),
        },
        "note": "Calibrated candidate is stored separately and not auto-promoted.",
    }
    joblib.dump(calibrated, MODELS_DIR / "calibrated_fraud_detection_model.joblib")
    (MODELS_DIR / "calibrated_threshold_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (REPORTS_DIR / "calibration_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

