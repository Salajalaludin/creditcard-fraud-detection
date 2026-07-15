"""Feature engineering transparan untuk analisis dan eksperimen lanjutan."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan transformasi Amount dan Time tanpa mengubah DataFrame input."""
    required = {"Time", "Amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom feature engineering tidak lengkap: {missing}")

    result = frame.copy()
    result["Amount_Log"] = np.log1p(result["Amount"].clip(lower=0))
    result["Hour"] = (result["Time"] / 3600) % 24
    result["Time_Period"] = pd.cut(
        result["Hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
        include_lowest=True,
        right=False,
    )
    result["Amount_Group"] = pd.cut(
        result["Amount"],
        bins=[-np.inf, 10, 50, 100, 500, 1_000, np.inf],
        labels=["Very Low", "Low", "Medium", "High", "Very High", "Extreme"],
    )
    return result

