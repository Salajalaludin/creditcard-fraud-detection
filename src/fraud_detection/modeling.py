"""Definisi kandidat model untuk imbalance handling dan advanced tuning."""

from __future__ import annotations

import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import build_baseline_preprocessor


class ProbabilityBlend:
    """Gabungkan probability beberapa model fit menggunakan weighted average.

    Class sederhana ini dapat disimpan dengan joblib dan tetap menyediakan API
    `predict_proba` yang dibutuhkan prediction CLI serta dashboard.
    """

    def __init__(self, models: list[object], weights: list[float], names: list[str]) -> None:
        """Simpan model fit, bobot ensemble, dan nama komponen untuk audit."""
        if len(models) != len(weights) or len(models) != len(names):
            raise ValueError("models, weights, dan names harus sama panjang")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Jumlah weights harus sama dengan 1")
        self.models = models
        self.weights = weights
        self.names = names

    def predict_proba(self, features: object) -> np.ndarray:
        """Hitung weighted average probability kelas normal dan fraud."""
        component_probabilities = [model.predict_proba(features) for model in self.models]
        return np.average(component_probabilities, axis=0, weights=self.weights)


def build_model_candidates() -> dict[str, object]:
    """Buat model pembanding dengan strategi imbalance yang berbeda.

    Seluruh preprocessing dan resampling berada di dalam pipeline. Karena itu,
    operasi tersebut hanya di-fit pada training set dan tidak dapat melihat
    validation maupun test set.
    """
    return {
        # Baseline cost-sensitive: semua data dipakai dan kelas fraud diberi
        # bobot lebih besar secara otomatis.
        "logistic_class_weight": Pipeline(
            steps=[
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

        # Undersampling mengurangi kelas normal hingga rasio fraud:normal 1:10.
        # Metode ini cepat, tetapi berisiko membuang pola transaksi normal.
        "logistic_under_1_10": ImbalancedPipeline(
            steps=[
                ("preprocessor", build_baseline_preprocessor()),
                (
                    "resampler",
                    RandomUnderSampler(sampling_strategy=0.10, random_state=RANDOM_STATE),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # SMOTE membuat sampel fraud sintetis sampai rasio fraud:normal 1:10.
        # Resampling hanya terjadi saat fit; data validation/test tetap asli.
        "logistic_smote_1_10": ImbalancedPipeline(
            steps=[
                ("preprocessor", build_baseline_preprocessor()),
                (
                    "resampler",
                    SMOTE(sampling_strategy=0.10, random_state=RANDOM_STATE, k_neighbors=5),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # Random Forest menangkap hubungan non-linear. balanced_subsample
        # menghitung ulang class weight pada setiap bootstrap sample.
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Histogram Gradient Boosting menjadi pembanding boosting yang tersedia
        # langsung di scikit-learn dan efisien untuk dataset berukuran besar.
        "hist_gradient_boosting_balanced": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
