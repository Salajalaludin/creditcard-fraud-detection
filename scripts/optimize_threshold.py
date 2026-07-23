"""Optimasi threshold, simulasi biaya, risk scoring, dan investigation queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Simpan cache Matplotlib di workspace agar tidak membutuhkan akses home folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")

from fraud_detection.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    MODELS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    ensure_output_directories,
)
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import classification_metrics  # noqa: E402
from fraud_detection.policy_reports import BUSINESS_SCENARIOS, build_policy_recommendations  # noqa: E402
from fraud_detection.risk import risk_level_boundaries, score_transactions  # noqa: E402
from fraud_detection.threshold import build_threshold_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Baca lokasi dataset, model, dan skenario threshold yang direkomendasikan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "fraud_detection_model.joblib",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(BUSINESS_SCENARIOS),
        default="balanced",
    )
    return parser.parse_args()


def main() -> None:
    """Pilih threshold di validation dan evaluasi hasil final pada test set."""
    args = parse_args()
    ensure_output_directories()

    # Reproduksi split yang sama; train set tidak dibutuhkan karena model sudah fit.
    frame, _ = clean_transactions(load_transactions(args.data))
    _, x_validation, x_test, _, y_validation, y_test = (
        stratified_train_validation_test_split(frame)
    )
    model = joblib.load(args.model)

    # Semua pemilihan threshold dilakukan menggunakan validation set.
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold_table = build_threshold_table(
        y_validation.to_numpy(),
        validation_probabilities,
        x_validation["Amount"].to_numpy(),
    )
    recommendations, _ = build_policy_recommendations(threshold_table)

    # Skenario yang dipilih melalui CLI menjadi threshold operasional utama.
    selected_strategy = f"business_{args.scenario}"
    selected_row = recommendations.loc[recommendations["strategy"] == selected_strategy].iloc[0]
    selected_threshold = float(selected_row["threshold"])

    threshold_config = {
        # Gunakan nama class sebagai provenance dasar untuk pipeline non-tuned.
        "selected_model": model.__class__.__name__,
        "selected_scenario": args.scenario,
        "selected_strategy": selected_strategy,
        "threshold": selected_threshold,
        "validation_metrics": {
            key: float(selected_row[key])
            for key in ("precision", "recall", "f1", "alerts", "total_cost")
        },
        "business_assumptions": BUSINESS_SCENARIOS[args.scenario],
        # Risk-level bands mengikuti threshold agar semua alert minimal Medium.
        "risk_level_boundaries": risk_level_boundaries(selected_threshold),
        "score_note": "Risk score = model output x 100; score belum dikalibrasi sebagai probabilitas absolut.",
    }
    (MODELS_DIR / "threshold_config.json").write_text(
        json.dumps(threshold_config, indent=2), encoding="utf-8"
    )

    # Test set baru disentuh setelah model dan threshold sudah dipilih.
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = classification_metrics(
        y_test.to_numpy(), test_probabilities, threshold=selected_threshold
    )
    test_scored = score_transactions(
        x_test,
        test_probabilities,
        threshold=selected_threshold,
        actual_class=y_test,
    )
    detected_mask = (test_scored["predicted_class"] == 1) & (test_scored["actual_class"] == 1)
    missed_mask = (test_scored["predicted_class"] == 0) & (test_scored["actual_class"] == 1)
    test_metrics.update(
        {
            "detected_fraud_amount": float(test_scored.loc[detected_mask, "Amount"].sum()),
            "missed_fraud_amount": float(test_scored.loc[missed_mask, "Amount"].sum()),
            "selected_scenario": args.scenario,
        }
    )
    (REPORTS_DIR / "threshold_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )

    # Simpan seluruh test scoring dan queue alert yang sudah diurutkan prioritas.
    test_scored.sort_values("risk_score", ascending=False).to_csv(
        PREDICTIONS_DIR / "test_risk_scores.csv", index=False
    )
    investigation_queue = test_scored.loc[test_scored["predicted_class"] == 1].sort_values(
        ["risk_score", "Amount"], ascending=[False, False]
    )
    investigation_queue.insert(0, "queue_rank", range(1, len(investigation_queue) + 1))
    investigation_queue.to_csv(PREDICTIONS_DIR / "investigation_queue.csv", index=False)

    # Refresh menambahkan model_id/hash dan memastikan dashboard tidak memakai artefak stale.
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "refresh_active_policy.py")],
        check=True,
    )

    print(recommendations[["strategy", "threshold", "precision", "recall", "alerts"]].to_string(index=False))
    print(f"\nThreshold terpilih ({args.scenario}): {selected_threshold:.4f}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
