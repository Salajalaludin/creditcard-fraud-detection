"""Sinkronkan analisis threshold dan metadata dengan model produksi aktif."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")

from fraud_detection.artifacts import sha256_file
from fraud_detection.config import DEFAULT_DATA_PATH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_output_directories
from fraud_detection.data import clean_transactions, load_transactions, stratified_train_validation_test_split
from fraud_detection.policy_reports import build_policy_recommendations, save_policy_figures
from fraud_detection.threshold import build_threshold_table


def main() -> None:
    """Bangun ulang tabel, plot, dan provenance untuk active production policy."""
    ensure_output_directories()
    model_path = MODELS_DIR / "fraud_detection_model.joblib"
    config = json.loads((MODELS_DIR / "threshold_config.json").read_text(encoding="utf-8"))
    model_id = config["selected_model"]
    active_threshold = float(config["threshold"])
    model = joblib.load(model_path)

    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    _, x_validation, _, _, y_validation, _ = stratified_train_validation_test_split(frame)
    probabilities = model.predict_proba(x_validation)[:, 1]
    table = build_threshold_table(y_validation.to_numpy(), probabilities, x_validation["Amount"].to_numpy())
    table.insert(0, "model_id", model_id)
    table.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    recommendations, cost_tables = build_policy_recommendations(table)
    recommendations["model_id"] = model_id
    recommendations.to_csv(REPORTS_DIR / "threshold_recommendations.csv", index=False)
    save_policy_figures(table, cost_tables, active_threshold, FIGURES_DIR, model_id)

    metadata = {
        "model_id": model_id,
        "model_sha256": sha256_file(model_path),
        "threshold": active_threshold,
        "split_protocol": "stratified_random_70_15_15_seed_42",
        "selection_warning": config.get("validation_warning"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (REPORTS_DIR / "active_policy_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
