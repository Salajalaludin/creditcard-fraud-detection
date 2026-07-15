"""Perhitungan metrik threshold dan simulasi biaya bisnis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_threshold_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hitung metrik klasifikasi dan nominal fraud untuk banyak threshold.

    Threshold default menggunakan grid 0,001–0,999. Grid membuat ukuran tabel
    stabil dan cukup detail untuk analisis tanpa menghitung puluhan ribu nilai.
    """
    # Ubah input menjadi array dengan tipe konsisten agar operasi vektor aman.
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    # Validasi panjang input untuk mencegah Amount tertukar dengan transaksi lain.
    if not (len(y_true) == len(probabilities) == len(amounts)):
        raise ValueError("y_true, probabilities, dan amounts harus sama panjang")

    # Sertakan threshold 0,5 secara eksplisit sebagai pembanding default.
    if thresholds is None:
        thresholds = np.unique(np.r_[np.linspace(0.001, 0.999, 999), 0.5])

    rows: list[dict[str, float | int]] = []
    actual_fraud = y_true == 1
    total_fraud_amount = float(amounts[actual_fraud].sum())

    # Iterasi grid threshold dan hitung confusion matrix secara langsung.
    for threshold in thresholds:
        predicted_fraud = probabilities >= threshold
        tp_mask = predicted_fraud & actual_fraud
        fp_mask = predicted_fraud & ~actual_fraud
        fn_mask = ~predicted_fraud & actual_fraud
        tn_mask = ~predicted_fraud & ~actual_fraud

        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(fn_mask.sum())
        tn = int(tn_mask.sum())
        alerts = tp + fp

        # Pembagian memakai kondisi eksplisit agar tidak menghasilkan NaN.
        precision = tp / alerts if alerts else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        detected_amount = float(amounts[tp_mask].sum())

        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "alerts": alerts,
                "alert_rate": alerts / len(y_true),
                "detected_fraud_amount": detected_amount,
                "missed_fraud_amount": total_fraud_amount - detected_amount,
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def add_business_costs(
    threshold_table: pd.DataFrame,
    investigation_cost: float,
    false_positive_cost: float,
    fixed_false_negative_cost: float,
    fraud_amount_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Tambahkan expected cost dengan asumsi biaya yang dapat dikonfigurasi."""
    result = threshold_table.copy()

    # Cost terdiri dari biaya review semua alert, gangguan false positive,
    # penalti tetap fraud yang lolos, dan nominal fraud yang tidak terdeteksi.
    result["investigation_cost_total"] = result["alerts"] * investigation_cost
    result["false_positive_cost_total"] = result["false_positive"] * false_positive_cost
    result["false_negative_cost_total"] = (
        result["false_negative"] * fixed_false_negative_cost
        + result["missed_fraud_amount"] * fraud_amount_multiplier
    )
    result["total_cost"] = (
        result["investigation_cost_total"]
        + result["false_positive_cost_total"]
        + result["false_negative_cost_total"]
    )
    return result


def select_best_threshold(
    cost_table: pd.DataFrame,
    minimum_recall: float,
) -> pd.Series:
    """Pilih threshold berbiaya terendah dengan batas minimum recall."""
    # Terapkan constraint recall agar optimasi biaya tidak memilih model yang
    # murah hanya karena hampir tidak pernah mengeluarkan alert.
    eligible = cost_table.loc[cost_table["recall"] >= minimum_recall]
    if eligible.empty:
        raise ValueError(f"Tidak ada threshold dengan recall >= {minimum_recall:.2f}")

    # Jika cost sama, prioritaskan alert lebih sedikit lalu precision lebih tinggi.
    return eligible.sort_values(
        ["total_cost", "alerts", "precision"],
        ascending=[True, True, False],
    ).iloc[0]

