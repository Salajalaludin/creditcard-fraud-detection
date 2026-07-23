"""Latih Isolation Forest sebagai pembanding unsupervised fraud detection."""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import DEFAULT_DATA_PATH, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from fraud_detection.data import clean_transactions, load_transactions, stratified_train_validation_test_split  # noqa: E402


def main() -> None:
    """Fit Isolation Forest pada transaksi normal train dan nilai validation/test."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    x_train, x_validation, x_test, y_train, y_validation, y_test = stratified_train_validation_test_split(frame)

    # Anomaly model belajar hanya dari pola normal; sample membatasi waktu training.
    normal_train = x_train.loc[y_train == 0].sample(n=min(100_000, int((y_train == 0).sum())), random_state=RANDOM_STATE)
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(normal_train)
    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(normal_scaled)

    results: dict[str, dict[str, float]] = {}
    for name, features, target in (
        ("validation", x_validation, y_validation),
        ("test", x_test, y_test),
    ):
        # Nilai dibalik agar score lebih besar berarti lebih mencurigakan.
        anomaly_score = -model.decision_function(scaler.transform(features))
        results[name] = {
            "pr_auc": float(average_precision_score(target, anomaly_score)),
            "roc_auc": float(roc_auc_score(target, anomaly_score)),
            "score_mean": float(np.mean(anomaly_score)),
        }

    payload = {
        "model": "IsolationForest",
        "training_rows": len(normal_train),
        "training_class": "normal_only",
        "results": results,
        "note": "Unsupervised benchmark; not promoted as production classifier.",
    }
    (REPORTS_DIR / "anomaly_detection_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
