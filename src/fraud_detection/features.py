"""Preprocessing baseline yang aman dari data leakage."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS


def build_baseline_preprocessor() -> ColumnTransformer:
    """Buat transformer untuk standardisasi semua fitur numerik.

    Transformer ini harus berada di dalam Pipeline. Dengan begitu nilai mean
    dan standard deviation hanya dipelajari dari training set, bukan dari
    validation atau test set.
    """
    # `remainder="drop"` memastikan model hanya menerima fitur resmi.
    # Nama fitur dipertahankan tanpa awalan `numeric__` agar mudah dibaca.
    return ColumnTransformer(
        transformers=[("numeric", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
