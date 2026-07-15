"""Konversi model score menjadi risk level dan tindakan operasional."""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_level_boundaries(decision_threshold: float) -> dict[str, list[float]]:
    """Hitung batas kategori yang selaras dengan threshold operasional.

    Low berakhir tepat sebelum decision threshold. Rentang sisanya dibagi tiga
    agar setiap alert selalu berlevel Medium, High, atau Critical.
    """
    low_upper = float(decision_threshold * 100)
    step = (100 - low_upper) / 3
    return {
        "low": [0.0, low_upper],
        "medium": [low_upper, low_upper + step],
        "high": [low_upper + step, low_upper + 2 * step],
        "critical": [low_upper + 2 * step, 100.0],
    }


def assign_risk_level(
    risk_score: pd.Series,
    decision_threshold: float,
) -> pd.Categorical:
    """Kelompokkan skor 0–100 memakai batas yang mengikuti decision threshold."""
    boundaries = risk_level_boundaries(decision_threshold)

    # Batas tak hingga menjaga skor ekstrem tetap terklasifikasi dan right=False
    # membuat skor tepat pada threshold masuk ke kategori Medium.
    return pd.cut(
        risk_score,
        bins=[
            -np.inf,
            boundaries["low"][1],
            boundaries["medium"][1],
            boundaries["high"][1],
            np.inf,
        ],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
        right=False,
    )


def recommended_action(risk_level: pd.Series) -> pd.Series:
    """Petakan risk level ke tindakan human-in-the-loop yang disarankan."""
    action_map = {
        "Low": "Process normally",
        "Medium": "Enhanced monitoring",
        "High": "Manual review",
        "Critical": "Priority investigation",
    }
    return risk_level.astype("object").map(action_map)


def score_transactions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    actual_class: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Tambahkan model score, risk level, keputusan, dan rekomendasi tindakan."""
    if len(frame) != len(probabilities):
        raise ValueError("Jumlah transaksi dan probabilities harus sama")

    # Pertahankan identifier jika tersedia; jika tidak, gunakan index sebagai ID.
    result = pd.DataFrame(index=frame.index)
    result["transaction_id"] = frame.get("Transaction_ID", pd.Series(frame.index, index=frame.index))
    result["Time"] = frame["Time"].to_numpy()
    result["Amount"] = frame["Amount"].to_numpy()

    # Probability dari model dikalikan 100 sesuai brief. Pada model berbobot atau
    # resampling, nilai ini adalah model risk score dan belum tentu terkalibrasi.
    result["fraud_probability"] = np.asarray(probabilities, dtype=float)
    result["risk_score"] = result["fraud_probability"] * 100
    result["risk_level"] = assign_risk_level(result["risk_score"], threshold)
    result["predicted_class"] = (result["fraud_probability"] >= threshold).astype(int)
    result["recommended_action"] = recommended_action(result["risk_level"])

    # Actual class bersifat opsional agar fungsi dapat dipakai untuk data baru.
    if actual_class is not None:
        result["actual_class"] = np.asarray(actual_class, dtype=int)

    return result.reset_index(drop=True)
