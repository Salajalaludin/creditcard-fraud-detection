"""Metrik klasifikasi yang sesuai untuk data fraud sangat imbalanced."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Hitung metrik statistik dan operasional pada threshold tertentu.

    PR-AUC dan ROC-AUC dihitung dari probability, sedangkan precision, recall,
    F1, confusion matrix, dan jumlah alert bergantung pada threshold.
    """
    # Konversi probability menjadi keputusan kelas sesuai threshold bisnis.
    predictions = (probabilities >= threshold).astype(int)

    # labels=[0, 1] menjaga bentuk confusion matrix tetap 2x2 walau salah satu
    # kelas tidak pernah diprediksi oleh model seperti Dummy Classifier.
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    # Kembalikan tipe Python standar agar hasil mudah diserialisasi ke JSON/CSV.
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "alerts": int(fp + tp),
    }


def ranking_metrics_at_k(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k_values: tuple[int, ...] = (100, 500, 1_000),
) -> dict[str, float | int]:
    """Hitung precision dan recall jika investigator hanya memeriksa top-K.

    Metrik ini menghubungkan ranking model dengan kapasitas investigasi harian.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    # Urutkan index dari risk score tertinggi menuju terendah satu kali saja.
    ranked_index = np.argsort(-probabilities)
    total_fraud = int(y_true.sum())
    result: dict[str, float | int] = {}

    for requested_k in k_values:
        # Jika dataset lebih kecil dari K, gunakan seluruh baris yang tersedia.
        actual_k = min(requested_k, len(y_true))
        fraud_found = int(y_true[ranked_index[:actual_k]].sum())
        result[f"fraud_found_at_{requested_k}"] = fraud_found
        result[f"precision_at_{requested_k}"] = fraud_found / actual_k if actual_k else 0.0
        result[f"recall_at_{requested_k}"] = fraud_found / total_fraud if total_fraud else 0.0

    return result
