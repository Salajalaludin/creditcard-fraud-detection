"""Validasi data inference dan helper prediction yang dapat dipakai ulang."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURE_COLUMNS, TARGET


def load_inference_data(source: str | Path | object) -> pd.DataFrame:
    """Baca CSV untuk inference dan validasi seluruh fitur model.

    `source` dapat berupa path atau file-like object dari Streamlit uploader.
    Kolom Class boleh ada untuk evaluasi, tetapi tidak wajib untuk data baru.
    """
    frame = pd.read_csv(source)

    # Prediction tidak boleh berjalan jika salah satu fitur model hilang.
    missing_columns = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Kolom inference tidak lengkap: {missing_columns}")

    # Konversi eksplisit memberikan error yang jelas jika ada teks di fitur.
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    # Model sklearn tidak menerima missing value pada pipeline yang digunakan.
    missing_cells = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    if missing_cells:
        raise ValueError(f"Data inference memiliki {missing_cells} missing cells")

    # Jika Class tersedia, validasi sebagai label biner untuk evaluasi opsional.
    if TARGET in frame.columns:
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
        invalid_targets = sorted(set(frame[TARGET].unique()) - {0, 1})
        if invalid_targets:
            raise ValueError(f"Nilai Class tidak valid: {invalid_targets}")

    return frame

