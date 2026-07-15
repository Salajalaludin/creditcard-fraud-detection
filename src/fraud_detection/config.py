"""Konfigurasi pusat agar nilai penting tidak tersebar di banyak file."""

from pathlib import Path

# Seed tunggal membuat data split dan training dapat direproduksi.
RANDOM_STATE = 42

# Nama target dan daftar fitur resmi sesuai schema dataset Kaggle.
TARGET = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

# Semua path diturunkan dari lokasi file ini agar script dapat dijalankan dari
# terminal maupun notebook tanpa bergantung pada current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "Data" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_output_directories() -> None:
    """Buat folder output jika belum ada tanpa mengubah dataset mentah."""
    # parents=True juga membuat parent folder; exist_ok=True aman untuk rerun.
    for path in (PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
