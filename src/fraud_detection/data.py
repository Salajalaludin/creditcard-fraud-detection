"""Fungsi untuk loading, validasi, cleaning, dan splitting dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Baca CSV dan validasi schema transaksi kartu kredit.

    Fungsi sengaja gagal lebih awal jika file, kolom, tipe data, atau nilai
    target tidak sesuai. Ini mencegah training berjalan dengan data yang salah.
    """
    # Normalisasi input menjadi Path dan pastikan file benar-benar tersedia.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    # Muat seluruh dataset; ukurannya masih aman untuk diproses di memory.
    frame = pd.read_csv(path)

    # Dataset harus memiliki tepat 30 fitur dan satu kolom target.
    expected = [*FEATURE_COLUMNS, TARGET]
    missing_columns = sorted(set(expected) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(expected))
    if missing_columns or extra_columns:
        raise ValueError(
            f"Schema tidak sesuai. Missing={missing_columns}, extra={extra_columns}"
        )

    # Susun ulang kolom secara konsisten dan paksa semuanya menjadi numerik.
    frame = frame[expected].copy()
    for column in expected:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame[TARGET] = frame[TARGET].astype("int8")

    # Binary classification hanya menerima label normal (0) atau fraud (1).
    invalid_targets = sorted(set(frame[TARGET].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Nilai target tidak valid: {invalid_targets}")
    return frame


def clean_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Hapus exact duplicates dan hasilkan laporan audit.

    Duplikat dihapus sebelum split agar transaksi identik tidak masuk ke dua
    partisi berbeda. DataFrame input dan CSV asli tidak pernah diubah.
    """
    # Simpan statistik mentah sebelum melakukan perubahan apa pun.
    missing_cells = int(frame.isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    raw_counts = frame[TARGET].value_counts().reindex([0, 1], fill_value=0)

    # Hanya exact duplicates yang dihapus; outlier tetap dipertahankan karena
    # dalam fraud detection sebuah outlier justru dapat membawa informasi.
    clean = frame.drop_duplicates().reset_index(drop=True)
    clean_counts = clean[TARGET].value_counts().reindex([0, 1], fill_value=0)

    # Gabungkan statistik sebelum dan sesudah cleaning dalam satu objek audit.
    report = {
        "rows_raw": len(frame),
        "columns": frame.shape[1],
        "missing_cells": missing_cells,
        "exact_duplicates": duplicate_count,
        "fraud_count_raw": int(raw_counts[1]),
        "normal_count_raw": int(raw_counts[0]),
        "fraud_rate_raw": float(raw_counts[1] / len(frame)),
        "rows_clean": len(clean),
        "duplicates_removed": duplicate_count,
        "fraud_count_clean": int(clean_counts[1]),
        "normal_count_clean": int(clean_counts[0]),
        "fraud_rate_clean": float(clean_counts[1] / len(clean)),
    }
    return clean, report


def stratified_train_validation_test_split(
    frame: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Buat split 70/15/15 yang reproducible dan mempertahankan rasio kelas.

    Urutan hasil: X_train, X_validation, X_test, y_train, y_validation, y_test.
    Validation digunakan untuk pemilihan model; test hanya untuk evaluasi akhir.
    """
    # Tolak proporsi yang tidak membentuk 100% dataset.
    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size harus sama dengan 1")

    # Pisahkan fitur dan target sebelum melakukan stratified split pertama.
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET]
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        train_size=train_size,
        random_state=random_state,
        stratify=target,
    )
    # Pecah temporary set menjadi validation dan test. Ukuran validation harus
    # dihitung relatif terhadap total temporary set, bukan terhadap data awal.
    relative_validation_size = validation_size / (validation_size + test_size)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        train_size=relative_validation_size,
        random_state=random_state,
        stratify=y_temp,
    )
    # Mengembalikan semua partisi secara eksplisit memudahkan audit dan testing.
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def chronological_train_validation_test_split(
    frame: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Buat pseudo out-of-time split berdasarkan urutan kolom Time.

    Split ini tidak melakukan stratification karena tujuan utamanya meniru kondisi
    deployment: model belajar dari masa lalu dan dinilai pada periode berikutnya.
    """
    if abs(train_size + validation_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size harus sama dengan 1")

    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    train_end = int(len(ordered) * train_size)
    validation_end = train_end + int(len(ordered) * validation_size)

    train = ordered.iloc[:train_end]
    validation = ordered.iloc[train_end:validation_end]
    test = ordered.iloc[validation_end:]
    return (
        train[FEATURE_COLUMNS],
        validation[FEATURE_COLUMNS],
        test[FEATURE_COLUMNS],
        train[TARGET],
        validation[TARGET],
        test[TARGET],
    )
