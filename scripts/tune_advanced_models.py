"""Advanced tuning untuk memaksimalkan validation F1 tanpa melihat test label."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Batasi deteksi core joblib yang bermasalah pada sebagian environment Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from fraud_detection.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    ensure_output_directories,
)
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import (  # noqa: E402
    classification_metrics,
    ranking_metrics_at_k,
)
from fraud_detection.modeling import ProbabilityBlend  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Baca lokasi dataset dan jumlah kandidat opsional untuk smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--max-models", type=int, default=None)
    return parser.parse_args()


def build_tuning_candidates() -> dict[str, object]:
    """Buat kandidat dengan variasi kompleksitas dan positive-class weight."""
    # Parameter bersama XGBoost menjaga eksperimen konsisten dan efisien.
    common_xgb = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    return {
        # Extra Trees memakai random split yang lebih agresif daripada Random
        # Forest dan sering meningkatkan ranking pada fitur PCA.
        "extra_trees_leaf1": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees_leaf2": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features=0.70,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Random Forest lebih dalam menguji apakah batas max_depth=12 pada model
        # sebelumnya terlalu membatasi pola fraud yang kompleks.
        "random_forest_deep": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "random_forest_depth18": RandomForestClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=1,
            max_features=0.50,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # scale_pos_weight sengaja divariasikan dari 1 sampai 10. Nilai penuh
        # sekitar 599 cenderung meningkatkan recall tetapi merusak precision/F1.
        "xgb_d4_w1": XGBClassifier(
            **common_xgb,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=2.0,
            scale_pos_weight=1.0,
        ),
        "xgb_d4_w3": XGBClassifier(
            **common_xgb,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=2.0,
            scale_pos_weight=3.0,
        ),
        "xgb_d5_w5": XGBClassifier(
            **common_xgb,
            n_estimators=600,
            learning_rate=0.04,
            max_depth=5,
            min_child_weight=3,
            gamma=0.10,
            reg_alpha=0.10,
            reg_lambda=3.0,
            scale_pos_weight=5.0,
        ),
        "xgb_d6_w10": XGBClassifier(
            **common_xgb,
            n_estimators=600,
            learning_rate=0.035,
            max_depth=6,
            min_child_weight=5,
            gamma=0.15,
            reg_alpha=0.15,
            reg_lambda=4.0,
            scale_pos_weight=10.0,
        ),
    }


def optimal_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Cari threshold exact yang memaksimalkan F1 pada validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)

    # Elemen terakhir precision/recall tidak memiliki pasangan threshold.
    denominator = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index]), float(f1_scores[best_index])


def main() -> None:
    """Tune kandidat, uji ensemble, lalu evaluasi pemenang satu kali pada test."""
    args = parse_args()
    ensure_output_directories()

    # Gunakan split identik dengan eksperimen sebelumnya untuk perbandingan adil.
    frame, _ = clean_transactions(load_transactions(args.data))
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        stratified_train_validation_test_split(frame)
    )

    candidate_items = list(build_tuning_candidates().items())
    if args.max_models is not None:
        candidate_items = candidate_items[: args.max_models]

    fitted_models: dict[str, object] = {}
    validation_scores: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    # Fit model pada train dan optimalkan threshold pada validation.
    for name, model in candidate_items:
        print(f"Tuning candidate: {name}", flush=True)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(x_validation)[:, 1]
        threshold, best_f1 = optimal_f1_threshold(y_validation.to_numpy(), probabilities)

        fitted_models[name] = model
        validation_scores[name] = probabilities
        rows.append(
            {
                "model": name,
                "type": "single_model",
                "fit_seconds": fit_seconds,
                **classification_metrics(y_validation.to_numpy(), probabilities, threshold),
                **ranking_metrics_at_k(y_validation.to_numpy(), probabilities),
                "optimized_f1": best_f1,
            }
        )

    single_metrics = pd.DataFrame(rows).sort_values("f1", ascending=False)

    # Blend tiga model dengan PR-AUC tertinggi. Bobot divariasikan agar ensemble
    # tidak otomatis menganggap dua model memiliki kontribusi yang sama.
    top_names = single_metrics.sort_values("pr_auc", ascending=False).head(3)["model"].tolist()
    blend_rows: list[dict] = []
    blend_objects: dict[str, ProbabilityBlend] = {}
    weight_options = [(0.50, 0.30, 0.20), (0.60, 0.25, 0.15), (0.40, 0.35, 0.25)]
    for weights in weight_options:
        name = "blend_" + "_".join(str(int(weight * 100)) for weight in weights)
        probabilities = np.average(
            [validation_scores[component] for component in top_names],
            axis=0,
            weights=weights,
        )
        threshold, best_f1 = optimal_f1_threshold(y_validation.to_numpy(), probabilities)
        blend = ProbabilityBlend(
            models=[fitted_models[component] for component in top_names],
            weights=list(weights),
            names=top_names,
        )
        blend_objects[name] = blend
        validation_scores[name] = probabilities
        blend_rows.append(
            {
                "model": name,
                "type": "weighted_ensemble",
                "fit_seconds": float(single_metrics.loc[single_metrics["model"].isin(top_names), "fit_seconds"].sum()),
                **classification_metrics(y_validation.to_numpy(), probabilities, threshold),
                **ranking_metrics_at_k(y_validation.to_numpy(), probabilities),
                "optimized_f1": best_f1,
                "components": ",".join(top_names),
                "weights": ",".join(map(str, weights)),
            }
        )

    all_validation = pd.concat([single_metrics, pd.DataFrame(blend_rows)], ignore_index=True)
    all_validation = all_validation.sort_values(["f1", "pr_auc"], ascending=False)

    # Model/ensemble teratas dipilih murni dari validation F1.
    best_row = all_validation.iloc[0]
    best_name = str(best_row["model"])
    best_model = blend_objects.get(best_name, fitted_models.get(best_name))
    if best_model is None:
        raise RuntimeError(f"Model terpilih tidak ditemukan: {best_name}")
    best_threshold = float(best_row["threshold"])

    # Test set hanya dievaluasi setelah kandidat dan threshold dibekukan.
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    test_metrics = {
        "model": best_name,
        "split": "test",
        **classification_metrics(y_test.to_numpy(), test_probabilities, best_threshold),
        **ranking_metrics_at_k(y_test.to_numpy(), test_probabilities),
    }

    # Simpan kandidat tuned secara terpisah; promosi dilakukan setelah hasil
    # dibandingkan dengan model produksi saat ini.
    joblib.dump(best_model, MODELS_DIR / "tuned_fraud_detection_model.joblib")
    all_validation.to_csv(REPORTS_DIR / "advanced_tuning_validation.csv", index=False)
    (REPORTS_DIR / "advanced_tuning_test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    tuned_config = {
        "selected_model": best_name,
        "selection_objective": "validation_f1",
        "threshold": best_threshold,
        "validation_f1": float(best_row["f1"]),
        "validation_precision": float(best_row["precision"]),
        "validation_recall": float(best_row["recall"]),
        "validation_pr_auc": float(best_row["pr_auc"]),
        "test_metrics": test_metrics,
        "probability_note": "Model output belum dikalibrasi sebagai probabilitas absolut.",
    }
    (MODELS_DIR / "tuned_threshold_config.json").write_text(
        json.dumps(tuned_config, indent=2), encoding="utf-8"
    )

    print(all_validation[["model", "type", "threshold", "precision", "recall", "f1", "pr_auc", "alerts"]].to_string(index=False))
    print("\nSelected tuned candidate:", best_name)
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
