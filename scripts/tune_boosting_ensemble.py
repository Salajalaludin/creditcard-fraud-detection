"""Putaran tuning LightGBM, CatBoost, dan ensemble lintas-algoritma."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from fraud_detection.config import DEFAULT_DATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import classification_metrics, ranking_metrics_at_k  # noqa: E402
from fraud_detection.modeling import ProbabilityBlend  # noqa: E402

# Pakai helper threshold yang sama agar perbandingan dengan putaran pertama adil.
from tune_advanced_models import optimal_f1_threshold  # noqa: E402


def build_boosting_candidates() -> dict[str, object]:
    """Buat variasi LightGBM dan CatBoost dengan bobot positif moderat."""
    common_lgbm = {
        "objective": "binary",
        "n_estimators": 700,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return {
        "lgbm_leaves15_w1": LGBMClassifier(
            **common_lgbm,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=20,
            reg_alpha=0.05,
            scale_pos_weight=1.0,
        ),
        "lgbm_leaves31_w3": LGBMClassifier(
            **common_lgbm,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            reg_alpha=0.10,
            scale_pos_weight=3.0,
        ),
        "lgbm_leaves31_w5": LGBMClassifier(
            **common_lgbm,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            reg_alpha=0.20,
            scale_pos_weight=5.0,
        ),
        "lgbm_leaves63_w10": LGBMClassifier(
            **common_lgbm,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=30,
            reg_alpha=0.30,
            scale_pos_weight=10.0,
        ),

        # CatBoost memakai ordered boosting internal dan regularisasi berbeda
        # sehingga error-nya berpotensi melengkapi model bagging/LightGBM.
        "catboost_d5_plain": CatBoostClassifier(
            iterations=700,
            learning_rate=0.04,
            depth=5,
            loss_function="Logloss",
            eval_metric="PRAUC",
            l2_leaf_reg=4.0,
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        ),
        "catboost_d6_sqrt_balanced": CatBoostClassifier(
            iterations=700,
            learning_rate=0.04,
            depth=6,
            loss_function="Logloss",
            eval_metric="PRAUC",
            l2_leaf_reg=5.0,
            auto_class_weights="SqrtBalanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        ),
        "catboost_d7_w5": CatBoostClassifier(
            iterations=650,
            learning_rate=0.035,
            depth=7,
            loss_function="Logloss",
            eval_metric="PRAUC",
            l2_leaf_reg=6.0,
            class_weights=[1.0, 5.0],
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
        ),
    }


def main() -> None:
    """Bandingkan booster, gabungkan model terbaik, dan simpan pemenang final."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        stratified_train_validation_test_split(frame)
    )

    models: dict[str, object] = {}
    probabilities: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    # Masukkan model terbaik sebelumnya sebagai anchor ensemble tanpa refit.
    anchor_paths = {
        "extra_trees_anchor": MODELS_DIR / "tuned_fraud_detection_model.joblib",
        "random_forest_anchor": MODELS_DIR / "fraud_detection_model.joblib",
    }
    for name, path in anchor_paths.items():
        model = joblib.load(path)
        score = model.predict_proba(x_validation)[:, 1]
        threshold, best_f1 = optimal_f1_threshold(y_validation.to_numpy(), score)
        models[name] = model
        probabilities[name] = score
        rows.append(
            {
                "model": name,
                "type": "anchor",
                "fit_seconds": 0.0,
                **classification_metrics(y_validation.to_numpy(), score, threshold),
                **ranking_metrics_at_k(y_validation.to_numpy(), score),
                "optimized_f1": best_f1,
            }
        )

    # Fit booster hanya pada training set dan cari threshold di validation.
    for name, model in build_boosting_candidates().items():
        print(f"Tuning booster: {name}", flush=True)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        score = model.predict_proba(x_validation)[:, 1]
        threshold, best_f1 = optimal_f1_threshold(y_validation.to_numpy(), score)
        models[name] = model
        probabilities[name] = score
        rows.append(
            {
                "model": name,
                "type": "single_model",
                "fit_seconds": fit_seconds,
                **classification_metrics(y_validation.to_numpy(), score, threshold),
                **ranking_metrics_at_k(y_validation.to_numpy(), score),
                "optimized_f1": best_f1,
            }
        )

    single_metrics = pd.DataFrame(rows)
    top_names = single_metrics.sort_values("pr_auc", ascending=False).head(4)["model"].tolist()

    # Uji campuran model terbaik dengan grid bobot kecil yang sudah ditentukan.
    ensemble_specs = [
        ((0.40, 0.30, 0.20, 0.10), "blend_40_30_20_10"),
        ((0.50, 0.25, 0.15, 0.10), "blend_50_25_15_10"),
        ((0.30, 0.30, 0.25, 0.15), "blend_30_30_25_15"),
    ]
    ensembles: dict[str, ProbabilityBlend] = {}
    ensemble_rows: list[dict] = []
    for weights, name in ensemble_specs:
        score = np.average([probabilities[item] for item in top_names], axis=0, weights=weights)
        threshold, best_f1 = optimal_f1_threshold(y_validation.to_numpy(), score)
        ensemble = ProbabilityBlend(
            [models[item] for item in top_names],
            list(weights),
            top_names,
        )
        ensembles[name] = ensemble
        ensemble_rows.append(
            {
                "model": name,
                "type": "weighted_ensemble",
                "fit_seconds": float(single_metrics.loc[single_metrics["model"].isin(top_names), "fit_seconds"].sum()),
                **classification_metrics(y_validation.to_numpy(), score, threshold),
                **ranking_metrics_at_k(y_validation.to_numpy(), score),
                "optimized_f1": best_f1,
                "components": ",".join(top_names),
                "weights": ",".join(map(str, weights)),
            }
        )

    all_validation = pd.concat([single_metrics, pd.DataFrame(ensemble_rows)], ignore_index=True)
    all_validation = all_validation.sort_values(["f1", "pr_auc"], ascending=False)
    best_row = all_validation.iloc[0]
    best_name = str(best_row["model"])
    best_model = ensembles.get(best_name, models.get(best_name))
    if best_model is None:
        raise RuntimeError(f"Model final tidak ditemukan: {best_name}")

    # Evaluasi final test memakai model dan threshold yang dipilih dari validation.
    threshold = float(best_row["threshold"])
    test_score = best_model.predict_proba(x_test)[:, 1]
    test_metrics = {
        "model": best_name,
        "split": "test",
        **classification_metrics(y_test.to_numpy(), test_score, threshold),
        **ranking_metrics_at_k(y_test.to_numpy(), test_score),
    }

    joblib.dump(best_model, MODELS_DIR / "final_tuned_model.joblib")
    all_validation.to_csv(REPORTS_DIR / "boosting_ensemble_validation.csv", index=False)
    (REPORTS_DIR / "boosting_ensemble_test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    (MODELS_DIR / "final_tuned_threshold.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "selection_objective": "validation_f1",
                "threshold": threshold,
                "validation_metrics": {
                    key: float(best_row[key])
                    for key in ("precision", "recall", "f1", "pr_auc", "alerts")
                },
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(all_validation[["model", "type", "threshold", "precision", "recall", "f1", "pr_auc", "alerts"]].to_string(index=False))
    print("\nFinal selected candidate:", best_name)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()

