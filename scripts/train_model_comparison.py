"""Bandingkan model class weighting dan resampling pada validation set."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Arahkan cache plotting ke workspace agar script aman di environment terbatas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

# Tambahkan package lokal ketika script dijalankan tanpa `pip install -e .`.
from fraud_detection.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_output_directories,
)
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import (  # noqa: E402
    classification_metrics,
    ranking_metrics_at_k,
)
from fraud_detection.modeling import build_model_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Baca lokasi dataset dan jumlah kandidat opsional untuk debug cepat."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Batasi jumlah kandidat; gunakan hanya untuk debug, bukan laporan final.",
    )
    return parser.parse_args()


def main() -> None:
    """Latih kandidat, pilih dengan validation PR-AUC, lalu evaluasi sekali di test."""
    args = parse_args()
    ensure_output_directories()

    # Cleaning dan split memakai fungsi yang sama dengan baseline agar hasil adil.
    frame, quality = clean_transactions(load_transactions(args.data))
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        stratified_train_validation_test_split(frame)
    )

    # max-models mempermudah smoke test tanpa mengubah definisi kandidat.
    candidate_items = list(build_model_candidates().items())
    if args.max_models is not None:
        candidate_items = candidate_items[: args.max_models]

    fitted_models: dict[str, object] = {}
    validation_probabilities: dict[str, object] = {}
    metric_rows: list[dict] = []

    # Semua kandidat hanya melihat training set saat proses fit.
    for name, model in candidate_items:
        print(f"Training {name} ...", flush=True)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        # Model comparison dilakukan pada validation set, bukan test set.
        probabilities = model.predict_proba(x_validation)[:, 1]
        fitted_models[name] = model
        validation_probabilities[name] = probabilities
        metric_rows.append(
            {
                "model": name,
                "split": "validation",
                "fit_seconds": fit_seconds,
                **classification_metrics(y_validation.to_numpy(), probabilities, threshold=0.5),
                **ranking_metrics_at_k(y_validation.to_numpy(), probabilities),
            }
        )

    validation_metrics = pd.DataFrame(metric_rows).sort_values("pr_auc", ascending=False)

    # Model dengan validation PR-AUC tertinggi menjadi satu-satunya model yang
    # boleh dievaluasi pada test set untuk mengurangi test-set peeking.
    best_name = str(validation_metrics.iloc[0]["model"])
    best_model = fitted_models[best_name]
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_row = {
        "model": best_name,
        "split": "test",
        "fit_seconds": float(validation_metrics.iloc[0]["fit_seconds"]),
        **classification_metrics(y_test.to_numpy(), test_probabilities, threshold=0.5),
        **ranking_metrics_at_k(y_test.to_numpy(), test_probabilities),
    }
    all_metrics = pd.concat([validation_metrics, pd.DataFrame([test_row])], ignore_index=True)

    # Simpan model terbaik dan tabel metrik untuk dipakai tahap berikutnya.
    joblib.dump(best_model, MODELS_DIR / "fraud_detection_model.joblib")
    all_metrics.to_csv(REPORTS_DIR / "model_comparison_metrics.csv", index=False)

    # Metadata mencatat konteks eksperimen dan menegaskan aturan pemilihan model.
    metadata = {
        "selected_model": best_name,
        "selection_metric": "validation_pr_auc",
        "default_threshold": 0.5,
        "random_state": RANDOM_STATE,
        "candidate_models": [name for name, _ in candidate_items],
        "data_quality": quality,
        "split": {
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "train_frauds": int(y_train.sum()),
            "validation_frauds": int(y_validation.sum()),
            "test_frauds": int(y_test.sum()),
        },
        "probability_note": (
            "Score berasal dari model berbobot/resampling dan belum dikalibrasi; "
            "gunakan untuk ranking dan threshold tervalidasi."
        ),
    }
    (REPORTS_DIR / "model_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Plot kurva PR seluruh kandidat pada validation set.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in validation_metrics.itertuples(index=False):
        precision, recall, _ = precision_recall_curve(
            y_validation,
            validation_probabilities[row.model],
        )
        ax.plot(recall, precision, label=f"{row.model} (AP={row.pr_auc:.3f})")
    ax.axhline(y_validation.mean(), color="gray", linestyle="--", label="Class prevalence")
    ax.set(title="Model Comparison — Validation PR Curve", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_pr_curve.png", dpi=160)
    plt.close(fig)

    # Plot PR-AUC dan alert volume bersama-sama untuk memperlihatkan trade-off.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_data = validation_metrics.sort_values("pr_auc", ascending=True)
    sns.barplot(data=plot_data, x="pr_auc", y="model", ax=axes[0], color="#35618f")
    axes[0].set_title("Validation PR-AUC")
    sns.barplot(data=plot_data, x="alerts", y="model", ax=axes[1], color="#c96f3b")
    axes[1].set_title("Alert Volume pada Threshold 0.5")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_summary.png", dpi=160)
    plt.close(fig)

    print(all_metrics.to_string(index=False))
    print(f"\nModel terpilih: {best_name}")


if __name__ == "__main__":
    main()
