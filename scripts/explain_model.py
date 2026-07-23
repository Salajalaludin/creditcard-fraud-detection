"""Buat global feature importance untuk model aktif."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from fraud_detection.config import DEFAULT_DATA_PATH, FEATURE_COLUMNS, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from fraud_detection.data import clean_transactions, load_transactions, stratified_train_validation_test_split  # noqa: E402


def main() -> None:
    """Hitung impurity dan permutation importance pada validation sample."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    _, x_validation, _, _, y_validation, _ = stratified_train_validation_test_split(frame)
    model = joblib.load(MODELS_DIR / "fraud_detection_model.joblib")
    config = json.loads((MODELS_DIR / "threshold_config.json").read_text(encoding="utf-8"))

    fraud_index = y_validation.loc[y_validation == 1].index
    normal_index = y_validation.loc[y_validation == 0].sample(n=5_000, random_state=RANDOM_STATE).index
    sample_index = fraud_index.union(normal_index)
    sample_x = x_validation.loc[sample_index]
    sample_y = y_validation.loc[sample_index]

    permutation = permutation_importance(
        model,
        sample_x,
        sample_y,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
        }
    )
    if hasattr(model, "feature_importances_"):
        importance["model_feature_importance"] = model.feature_importances_
    importance = importance.sort_values("permutation_importance_mean", ascending=False)
    importance.insert(0, "model_id", config["selected_model"])
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    sns.set_theme(style="whitegrid")
    top = importance.head(15).sort_values("permutation_importance_mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=top, x="permutation_importance_mean", y="feature", color="#3D8DFF", ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.4f", label_type="center", color="white", fontsize=8, fontweight="bold")
    ax.set(title="Active Model — Permutation Importance", xlabel="Mean decrease in validation PR-AUC", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
