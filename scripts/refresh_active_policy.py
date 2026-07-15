"""Sinkronkan analisis threshold dan metadata dengan model produksi aktif."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Gunakan cache lokal agar Matplotlib tidak menulis ke home directory pengguna.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fraud_detection.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_output_directories,
)
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.threshold import add_business_costs, build_threshold_table, select_best_threshold  # noqa: E402

# Semua asumsi biaya ditulis eksplisit agar dapat diganti dengan angka bisnis aktual.
BUSINESS_SCENARIOS = {
    "aggressive": dict(investigation_cost=5.0, false_positive_cost=5.0, fixed_false_negative_cost=500.0, fraud_amount_multiplier=1.0, minimum_recall=0.90),
    "balanced": dict(investigation_cost=10.0, false_positive_cost=10.0, fixed_false_negative_cost=300.0, fraud_amount_multiplier=1.0, minimum_recall=0.80),
    "customer_friendly": dict(investigation_cost=20.0, false_positive_cost=30.0, fixed_false_negative_cost=200.0, fraud_amount_multiplier=1.0, minimum_recall=0.60),
}


def sha256_file(path: Path) -> str:
    """Hitung fingerprint model agar artefak turunan dapat diaudit."""
    # Pembacaan per blok menjaga pemakaian memory tetap kecil untuk model besar.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_statistical_rows(table: pd.DataFrame) -> list[dict]:
    """Pilih rekomendasi statistik dari validation set model aktif."""
    # Tie-breaker dibuat deterministik agar rerun menghasilkan pilihan yang sama.
    max_f1 = table.sort_values(["f1", "precision", "alerts"], ascending=[False, False, True]).iloc[0]
    recall_80 = table.loc[table["recall"] >= 0.80].sort_values(["precision", "alerts"], ascending=[False, True]).iloc[0]
    cap_500 = table.loc[table["alerts"] <= 500].sort_values(["recall", "precision"], ascending=False).iloc[0]
    return [
        {"strategy": "max_f1", **max_f1.to_dict()},
        {"strategy": "minimum_recall_80", **recall_80.to_dict()},
        {"strategy": "maximum_500_alerts", **cap_500.to_dict()},
    ]


def main() -> None:
    """Bangun ulang tabel, plot, dan provenance untuk active production policy."""
    ensure_output_directories()
    model_path = MODELS_DIR / "fraud_detection_model.joblib"
    config_path = MODELS_DIR / "threshold_config.json"

    # Konfigurasi produksi adalah sumber kebenaran untuk identitas dan threshold aktif.
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_id = config["selected_model"]
    active_threshold = float(config["threshold"])
    model = joblib.load(model_path)

    # Reproduksi validation split yang digunakan oleh pipeline random-split.
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    _, x_validation, _, _, y_validation, _ = stratified_train_validation_test_split(frame)
    probabilities = model.predict_proba(x_validation)[:, 1]
    table = build_threshold_table(y_validation.to_numpy(), probabilities, x_validation["Amount"].to_numpy())
    table.insert(0, "model_id", model_id)
    table.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    # Gabungkan rekomendasi statistik dan simulasi biaya dalam satu schema.
    rows = choose_statistical_rows(table)
    cost_tables: dict[str, pd.DataFrame] = {}
    for name, assumptions in BUSINESS_SCENARIOS.items():
        cost_args = {key: value for key, value in assumptions.items() if key != "minimum_recall"}
        cost_table = add_business_costs(table, **cost_args)
        cost_tables[name] = cost_table
        selected = select_best_threshold(cost_table, assumptions["minimum_recall"])
        rows.append({"strategy": f"business_{name}", **selected.to_dict(), **{f"assumption_{key}": value for key, value in assumptions.items()}})
    recommendations = pd.DataFrame(rows)
    recommendations["model_id"] = model_id
    recommendations.to_csv(REPORTS_DIR / "threshold_recommendations.csv", index=False)

    # Plot selalu memakai probabilitas model aktif dan menandai policy threshold aktif.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric in ("precision", "recall", "f1"):
        axes[0].plot(table["threshold"], table[metric], label=metric.title())
    axes[0].axvline(active_threshold, color="black", linestyle="--", label="Active policy")
    axes[0].set(title=f"Validation Metrics — {model_id}", xlabel="Threshold", ylabel="Metric")
    axes[0].legend()
    axes[1].plot(table["threshold"], table["alerts"], color="#c96f3b")
    axes[1].axvline(active_threshold, color="black", linestyle="--")
    axes[1].set(title="Validation Alert Volume", xlabel="Threshold", ylabel="Alerts")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_tradeoff.png", dpi=160)
    plt.close(fig)

    # Sensitivity chart membandingkan seluruh skenario biaya pada grid yang sama.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, cost_table in cost_tables.items():
        ax.plot(cost_table["threshold"], cost_table["total_cost"], label=name)
    ax.axvline(active_threshold, color="black", linestyle="--", label="Active policy")
    ax.set(title=f"Business Cost Sensitivity — {model_id}", xlabel="Threshold", ylabel="Estimated total cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "business_cost_sensitivity.png", dpi=160)
    plt.close(fig)

    # Metadata mengikat model, threshold, data protocol, dan waktu generasi.
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
