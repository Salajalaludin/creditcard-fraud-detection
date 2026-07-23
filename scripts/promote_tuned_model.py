"""Promosikan Extra Trees tuned dan perbarui artefak operasional/dashboard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
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
    PREDICTIONS_DIR,
    REPORTS_DIR,
)
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import classification_metrics, ranking_metrics_at_k  # noqa: E402
from fraud_detection.risk import risk_level_boundaries, score_transactions  # noqa: E402


def main() -> None:
    """Salin model tuned ke production path dan bangun ulang seluruh output test."""
    # Model ini dipilih sebagai peningkatan F1 holdout terbaik pada eksperimen
    # advanced. Hasil tetap perlu dikonfirmasi dengan future out-of-time data.
    tuned_model_path = MODELS_DIR / "tuned_fraud_detection_model.joblib"
    tuned_config = json.loads(
        (MODELS_DIR / "tuned_threshold_config.json").read_text(encoding="utf-8")
    )
    model = joblib.load(tuned_model_path)
    threshold = float(tuned_config["threshold"])

    # Simpan ulang melalui joblib agar production artifact berdiri sendiri.
    joblib.dump(model, MODELS_DIR / "fraud_detection_model.joblib")

    # Reproduksi test partition untuk memperbarui metrik dan investigation queue.
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    _, _, x_test, _, _, y_test = stratified_train_validation_test_split(frame)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        **classification_metrics(y_test.to_numpy(), probabilities, threshold),
        **ranking_metrics_at_k(y_test.to_numpy(), probabilities),
        "model": tuned_config["selected_model"],
        "selected_scenario": "f1_optimized",
    }

    # Buat test scoring dan hitung fraud amount yang berhasil ditangkap.
    scored = score_transactions(
        x_test,
        probabilities,
        threshold=threshold,
        actual_class=y_test,
    )
    detected = (scored["predicted_class"] == 1) & (scored["actual_class"] == 1)
    missed = (scored["predicted_class"] == 0) & (scored["actual_class"] == 1)
    metrics["detected_fraud_amount"] = float(scored.loc[detected, "Amount"].sum())
    metrics["missed_fraud_amount"] = float(scored.loc[missed, "Amount"].sum())

    (REPORTS_DIR / "threshold_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    scored.sort_values("risk_score", ascending=False).to_csv(
        PREDICTIONS_DIR / "test_risk_scores.csv", index=False
    )
    queue = scored.loc[scored["predicted_class"] == 1].sort_values(
        ["risk_score", "Amount"], ascending=[False, False]
    )
    queue.insert(0, "queue_rank", range(1, len(queue) + 1))
    queue.to_csv(PREDICTIONS_DIR / "investigation_queue.csv", index=False)

    # Production config mempertahankan schema yang dipakai CLI dan dashboard.
    production_config = {
        "selected_model": tuned_config["selected_model"],
        "selected_scenario": "f1_optimized",
        "selected_strategy": "advanced_validation_max_f1",
        "threshold": threshold,
        "validation_metrics": {
            "precision": tuned_config["validation_precision"],
            "recall": tuned_config["validation_recall"],
            "f1": tuned_config["validation_f1"],
            "pr_auc": tuned_config["validation_pr_auc"],
        },
        "test_metrics": metrics,
        "risk_level_boundaries": risk_level_boundaries(threshold),
        "score_note": "Risk score belum dikalibrasi sebagai probabilitas absolut.",
        "validation_warning": (
            "Model dipromosikan setelah iterative experimentation pada split yang sama; "
            "konfirmasi pada future out-of-time holdout diperlukan."
        ),
    }
    (MODELS_DIR / "threshold_config.json").write_text(
        json.dumps(production_config, indent=2), encoding="utf-8"
    )

    # Simpan hasil tuning aktif untuk visual ringkas pada dashboard dan laporan.
    comparison = pd.read_csv(REPORTS_DIR / "advanced_tuning_validation.csv")
    comparison.to_csv(REPORTS_DIR / "tuning_comparison.csv", index=False)
    top_plot = (
        comparison.sort_values("f1", ascending=False)
        .drop_duplicates("model")
        .head(15)
        .sort_values("f1")
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=top_plot, x="f1", y="model", hue="type", ax=ax)
    ax.axvline(0.90, color="red", linestyle="--", label="Target F1 0.90")
    ax.set(title="Advanced Tuning — Validation F1", xlabel="F1", ylabel="Model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "advanced_tuning_f1.png", dpi=160)
    plt.close(fig)

    # Bangun ulang analisis threshold agar seluruh dashboard terikat ke model baru.
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "refresh_active_policy.py")],
        check=True,
    )

    print(json.dumps(metrics, indent=2))
    print("Production model updated:", MODELS_DIR / "fraud_detection_model.joblib")


if __name__ == "__main__":
    main()
