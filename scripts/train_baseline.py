"""Melatih dan mengevaluasi baseline fraud classifier tanpa data leakage."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Cari root project dari lokasi script dan simpan cache Matplotlib di workspace.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.pipeline import Pipeline

# Memungkinkan import package lokal ketika script dijalankan langsung.
from fraud_detection.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_output_directories,
)
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import classification_metrics  # noqa: E402
from fraud_detection.features import build_baseline_preprocessor  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Baca opsi command line agar lokasi dataset mudah diganti."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    return parser.parse_args()


def candidates() -> dict[str, object]:
    """Definisikan seluruh model baseline yang akan dibandingkan.

    Untuk menambah model baseline, masukkan pasangan `nama: estimator` baru ke
    dictionary ini. Estimator harus menyediakan `fit` dan `predict_proba`.
    """
    return {
        # Dummy prior selalu mengeluarkan probabilitas sesuai prevalensi kelas.
        # Model ini menjadi batas bawah performa yang wajib dikalahkan.
        "dummy_prior": DummyClassifier(strategy="prior", random_state=RANDOM_STATE),

        # Logistic Regression standar: preprocessing berada di dalam Pipeline
        # agar scaler hanya belajar dari training set.
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_baseline_preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=1_000, solver="lbfgs", random_state=RANDOM_STATE),
                ),
            ]
        ),
        # Class weight balanced memberi penalti lebih besar jika model salah
        # memprediksi kelas fraud yang jumlahnya sangat sedikit.
        "logistic_regression_balanced": Pipeline(
            [
                ("preprocessor", build_baseline_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    """Jalankan seluruh alur baseline dari data mentah sampai model tersimpan."""
    # Siapkan argumen, folder output, lalu validasi dan bersihkan dataset.
    args = parse_args()
    ensure_output_directories()
    frame, quality = clean_transactions(load_transactions(args.data))

    # Split dilakukan sebelum fitting model untuk menjaga validation dan test
    # tetap independen dari proses pembelajaran.
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        stratified_train_validation_test_split(frame)
    )

    # Cache berikut menyimpan model, probability, dan metrik agar fitting tidak
    # perlu diulang saat membuat tabel atau visualisasi.
    models: dict[str, object] = {}
    prediction_cache: dict[tuple[str, str], object] = {}
    rows: list[dict] = []

    # Latih setiap kandidat hanya pada train set.
    for name, model in candidates().items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        models[name] = model

        # Validation dipakai untuk pemilihan model. Test dihitung sebagai audit
        # akhir dan tidak ikut menentukan kandidat terbaik.
        for split_name, features, target in (
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ):
            # Kolom index 1 adalah probabilitas untuk kelas positif/fraud.
            probabilities = model.predict_proba(features)[:, 1]
            prediction_cache[(name, split_name)] = probabilities
            metrics = classification_metrics(target.to_numpy(), probabilities, threshold=0.5)
            rows.append(
                {
                    "model": name,
                    "split": split_name,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )

    # Satukan metrik seluruh model menjadi tabel long format.
    metrics_frame = pd.DataFrame(rows)

    # Visualisasi kurva Precision–Recall memakai validation set karena PR curve
    # lebih informatif daripada accuracy pada class imbalance ekstrem.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in models:
        probabilities = prediction_cache[(name, "validation")]
        precision, recall, _ = precision_recall_curve(y_validation, probabilities)
        pr_auc = metrics_frame.loc[
            (metrics_frame["model"] == name) & (metrics_frame["split"] == "validation"),
            "pr_auc",
        ].iloc[0]
        ax.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")
    ax.axhline(y_validation.mean(), color="gray", linestyle="--", label="Class prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation Precision–Recall Curve")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_validation_pr_curve.png", dpi=160)
    plt.close(fig)

    # Confusion matrix pada threshold 0.5 memperlihatkan langsung trade-off
    # false positive dan false negative dari setiap kandidat.
    fig, axes = plt.subplots(1, len(models), figsize=(12, 3.8))
    for ax, name in zip(axes, models):
        probabilities = prediction_cache[(name, "validation")]
        ConfusionMatrixDisplay.from_predictions(
            y_validation,
            (probabilities >= 0.5).astype(int),
            labels=[0, 1],
            display_labels=["Normal", "Fraud"],
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name.replace("_", "\n"), fontsize=9)
    fig.suptitle("Validation Confusion Matrices — Threshold 0.5")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_validation_confusion_matrices.png", dpi=160)
    plt.close(fig)

    # Pilih model hanya berdasarkan PR-AUC validation; jangan memakai test set.
    validation_metrics = metrics_frame.loc[metrics_frame["split"] == "validation"]
    best_name = str(validation_metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"])
    best_model = models[best_name]

    # Simpan pipeline utuh (scaler + model), sehingga preprocessing inference
    # selalu identik dengan preprocessing saat training.
    joblib.dump(best_model, MODELS_DIR / "baseline_best.joblib")
    metrics_frame.to_csv(REPORTS_DIR / "baseline_metrics.csv", index=False)

    # Buat output probability pada test set untuk audit dan tahap risk scoring.
    # Index hasil cleaning digunakan sebagai transaction_id sementara karena
    # dataset asli tidak menyediakan identifier transaksi.
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    predictions = x_test[["Time", "Amount"]].copy()
    predictions.insert(0, "transaction_id", predictions.index)
    predictions["actual_class"] = y_test
    predictions["fraud_probability"] = test_probabilities
    predictions["predicted_class_at_0_5"] = (test_probabilities >= 0.5).astype(int)
    predictions.sort_values("fraud_probability", ascending=False).to_csv(
        PREDICTIONS_DIR / "baseline_test_predictions.csv", index=False
    )

    # Metadata merekam model terpilih, seed, kualitas data, serta komposisi split
    # supaya eksperimen dapat direproduksi dan diaudit di kemudian hari.
    metadata = {
        "selected_model": best_name,
        "selection_metric": "validation_pr_auc",
        "threshold": 0.5,
        "random_state": RANDOM_STATE,
        "data_quality": quality,
        "split": {
            "train_rows": len(x_train),
            "validation_rows": len(x_validation),
            "test_rows": len(x_test),
            "train_frauds": int(y_train.sum()),
            "validation_frauds": int(y_validation.sum()),
            "test_frauds": int(y_test.sum()),
        },
        "note": "Test metrics are reported for audit only; model selection uses validation PR-AUC.",
    }
    (REPORTS_DIR / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Tampilkan hasil ringkas di terminal setelah seluruh artefak tersimpan.
    print(metrics_frame.to_string(index=False))
    print(f"\nBaseline terpilih: {best_name}")
    print(f"Model: {MODELS_DIR / 'baseline_best.joblib'}")


if __name__ == "__main__":
    main()
