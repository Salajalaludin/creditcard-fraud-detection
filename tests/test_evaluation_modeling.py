"""Unit test untuk metrik evaluasi dan ensemble probability."""

import numpy as np
import pytest

from fraud_detection.evaluation import classification_metrics, ranking_metrics_at_k
from fraud_detection.modeling import ProbabilityBlend, build_model_candidates


class FixedModel:
    """Model test minimal yang selalu mengembalikan probability tersimpan."""

    def __init__(self, probabilities: list[list[float]]) -> None:
        """Simpan matriks probability untuk dipakai oleh predict_proba."""
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, features: object) -> np.ndarray:
        """Kembalikan probability tanpa menggunakan dummy features."""
        return self.probabilities


def test_classification_metrics_matches_known_confusion_matrix() -> None:
    """Metrik threshold harus cocok dengan kasus kecil yang dapat dihitung manual."""
    result = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.9, 0.2]), 0.5)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_ranking_metrics_handles_k_larger_than_dataset() -> None:
    """Top-K harus memakai panjang aktual ketika K lebih besar dari dataset."""
    result = ranking_metrics_at_k(np.array([0, 1, 0]), np.array([0.2, 0.9, 0.1]), (2, 10))
    assert result["fraud_found_at_2"] == 1
    assert result["precision_at_2"] == pytest.approx(0.5)
    assert result["precision_at_10"] == pytest.approx(1 / 3)
    assert result["recall_at_10"] == pytest.approx(1.0)


def test_probability_blend_uses_declared_weights() -> None:
    """ProbabilityBlend harus menghasilkan weighted average setiap komponen."""
    first = FixedModel([[0.8, 0.2], [0.2, 0.8]])
    second = FixedModel([[0.4, 0.6], [0.6, 0.4]])
    blend = ProbabilityBlend([first, second], [0.75, 0.25], ["first", "second"])
    result = blend.predict_proba(np.zeros((2, 1)))
    np.testing.assert_allclose(result, np.array([[0.7, 0.3], [0.3, 0.7]]))


def test_probability_blend_rejects_invalid_configuration() -> None:
    """Ensemble harus gagal cepat jika jumlah bobot atau totalnya tidak valid."""
    model = FixedModel([[0.5, 0.5]])
    with pytest.raises(ValueError):
        ProbabilityBlend([model], [0.5], ["model"])
    with pytest.raises(ValueError):
        ProbabilityBlend([model], [1.0, 0.0], ["model"])


def test_model_candidates_cover_weighting_and_resampling() -> None:
    """Factory wajib menyediakan strategi class weight, undersampling, dan SMOTE."""
    names = set(build_model_candidates())
    assert "logistic_class_weight" in names
    assert "logistic_under_1_10" in names
    assert "logistic_smote_1_10" in names
