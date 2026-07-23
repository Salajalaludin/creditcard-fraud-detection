"""Bandingkan distribusi training reference dan test/current menggunakan PSI."""

from __future__ import annotations

import json

import pandas as pd

from fraud_detection.config import DEFAULT_DATA_PATH, FEATURE_COLUMNS, REPORTS_DIR  # noqa: E402
from fraud_detection.data import clean_transactions, load_transactions, stratified_train_validation_test_split  # noqa: E402
from fraud_detection.monitoring import drift_level, population_stability_index  # noqa: E402


def main() -> None:
    """Hitung PSI setiap fitur dan simpan ringkasan untuk dashboard."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    x_train, _, x_test, _, _, _ = stratified_train_validation_test_split(frame)
    rows = []
    for feature in FEATURE_COLUMNS:
        psi = population_stability_index(x_train[feature], x_test[feature])
        rows.append({"feature": feature, "psi": psi, "drift_level": drift_level(psi)})
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    report.to_csv(REPORTS_DIR / "drift_report.csv", index=False)
    summary = {
        "reference": "random_split_train",
        "current": "random_split_test",
        "features": len(report),
        "investigate": int((report["drift_level"] == "Investigate").sum()),
        "monitor": int((report["drift_level"] == "Monitor").sum()),
        "max_psi": float(report["psi"].max()),
        "note": "Replace current data with production batches for operational monitoring.",
    }
    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report.head(15).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
