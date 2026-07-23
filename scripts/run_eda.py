"""Membuat laporan data quality dan artefak EDA secara reproducible."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Cari root project dari lokasi script, lalu arahkan cache Matplotlib ke dalam
# project supaya tidak membutuhkan izin menulis ke home directory pengguna.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Tambahkan folder src agar package lokal dapat dipakai tanpa instalasi editable.
from fraud_detection.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    TARGET,
    ensure_output_directories,
)
from fraud_detection.data import clean_transactions, load_transactions  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Baca opsi command line untuk lokasi dataset dan ekspor data bersih."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Lokasi dataset dapat diganti tanpa mengedit isi script.
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)

    # Opsi ini berguna jika disk terbatas karena clean CSV berukuran sekitar 150 MB.
    parser.add_argument(
        "--skip-clean-export",
        action="store_true",
        help="Do not export the deduplicated CSV.",
    )
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    """Simpan dictionary sebagai JSON UTF-8 yang rapi dan mudah dibaca."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Buat tabel agregasi utama untuk menjawab pertanyaan EDA pada brief."""
    # Ringkas jumlah transaksi serta statistik Amount untuk setiap kelas.
    class_summary = (
        frame.groupby(TARGET, observed=True)
        .agg(
            transactions=(TARGET, "size"),
            amount_total=("Amount", "sum"),
            amount_mean=("Amount", "mean"),
            amount_median=("Amount", "median"),
            amount_max=("Amount", "max"),
        )
        .reset_index()
    )
    class_summary["transaction_share"] = class_summary["transactions"] / len(frame)

    # Kelompok nominal dibuat eksplisit supaya batasnya mudah diubah kemudian.
    amount_edges = [-np.inf, 10, 50, 100, 500, 1_000, np.inf]
    amount_labels = ["<=10", "10-50", "50-100", "100-500", "500-1000", ">1000"]
    amount_group = pd.cut(frame["Amount"], bins=amount_edges, labels=amount_labels)
    amount_summary = (
        frame.assign(amount_group=amount_group)
        .groupby("amount_group", observed=False)
        .agg(transactions=(TARGET, "size"), frauds=(TARGET, "sum"), amount_total=("Amount", "sum"))
        .reset_index()
    )
    amount_summary["fraud_rate"] = amount_summary["frauds"] / amount_summary["transactions"]

    # Time adalah detik sejak transaksi pertama. Modulo 24 menghasilkan jam
    # relatif, bukan jam kalender sebenarnya karena tanggal asli tidak tersedia.
    hourly = (
        frame.assign(hour=((frame["Time"] / 3600).astype(int) % 24))
        .groupby("hour", observed=True)
        .agg(transactions=(TARGET, "size"), frauds=(TARGET, "sum"), amount_total=("Amount", "sum"))
        .reset_index()
    )
    hourly["fraud_rate"] = hourly["frauds"] / hourly["transactions"]

    # Korelasi dipakai sebagai screening awal fitur, bukan bukti hubungan kausal.
    correlations = (
        frame.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .rename("correlation_with_class")
        .to_frame()
    )
    correlations["absolute_correlation"] = correlations["correlation_with_class"].abs()
    correlations = correlations.sort_values("absolute_correlation", ascending=False).reset_index(
        names="feature"
    )

    # Nama dictionary juga menjadi nama file CSV di fungsi main.
    return {
        "class_summary": class_summary,
        "amount_group_summary": amount_summary,
        "hourly_summary": hourly,
        "feature_target_correlations": correlations,
    }


def build_figures(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    """Bangun visualisasi EDA dan simpan sebagai PNG tanpa membuka GUI."""
    # Terapkan tema yang konsisten untuk seluruh grafik.
    sns.set_theme(style="whitegrid", context="notebook")

    # Grafik 1: class imbalance memakai skala log agar bar fraud masih terlihat.
    counts = frame[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=["Normal", "Fraud"], y=counts.values, ax=ax, hue=["Normal", "Fraud"], legend=False)
    ax.set_yscale("log")
    ax.set_title("Distribusi Kelas Transaksi (Skala Log)")
    ax.set_ylabel("Jumlah transaksi")
    for index, value in enumerate(counts.values):
        ax.text(index, value * 1.08, f"{value:,}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    # Grafik 2: sample maksimal 50 ribu transaksi normal agar plotting ringan.
    # Seluruh transaksi fraud tetap dipakai karena jumlahnya sangat sedikit.
    normal_sample = frame.loc[frame[TARGET] == 0, "Amount"].sample(
        n=min(50_000, int((frame[TARGET] == 0).sum())), random_state=42
    )
    plot_amount = pd.concat(
        [
            pd.DataFrame({"Amount_Log": np.log1p(normal_sample), "Class": "Normal"}),
            pd.DataFrame(
                {
                    "Amount_Log": np.log1p(frame.loc[frame[TARGET] == 1, "Amount"]),
                    "Class": "Fraud",
                }
            ),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(
        data=plot_amount,
        x="Amount_Log",
        hue="Class",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        ax=ax,
    )
    ax.set_title("Distribusi log(1 + Amount) per Kelas")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_distribution.png", dpi=160)
    plt.close(fig)

    # Grafik 3: perubahan fraud rate menurut jam relatif.
    hourly = tables["hourly_summary"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=hourly, x="hour", y="fraud_rate", marker="o", ax=ax)
    ax.set_title("Fraud Rate berdasarkan Jam Relatif")
    ax.set_ylabel("Fraud rate")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hourly_fraud_rate.png", dpi=160)
    plt.close(fig)

    # Grafik 4: tampilkan 15 fitur paling berhubungan dengan target secara linear.
    top_correlations = tables["feature_target_correlations"].head(15).sort_values(
        "correlation_with_class"
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(
        data=top_correlations,
        x="correlation_with_class",
        y="feature",
        hue="correlation_with_class",
        palette="vlag",
        legend=False,
        ax=ax,
    )
    ax.set_title("15 Fitur dengan Korelasi Absolut Terbesar terhadap Class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_feature_correlations.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Orkestrasi loading, cleaning, tabel EDA, visualisasi, dan ekspor."""
    # Siapkan parameter dan folder output sebelum membaca dataset.
    args = parse_args()
    ensure_output_directories()

    # Validasi dataset mentah, lalu lakukan deduplikasi tanpa mengubah CSV asli.
    raw = load_transactions(args.data)
    clean, quality = clean_transactions(raw)

    # Data-quality report selalu disimpan; clean CSV bersifat opsional.
    save_json(quality, REPORTS_DIR / "data_quality.json")
    if not args.skip_clean_export:
        clean.to_csv(PROCESSED_DIR / "creditcard_clean.csv", index=False)

    # Buat seluruh tabel dan gunakan key dictionary sebagai nama file.
    tables = build_tables(clean)
    for name, table in tables.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", index=False)
    # Buat grafik setelah tabel agar perhitungan agregasi tidak diulang.
    build_figures(clean, tables)

    # Cetak ringkasan agar pengguna mendapat feedback saat menjalankan script.
    print(json.dumps(quality, indent=2))
    print(f"EDA selesai. Output: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
