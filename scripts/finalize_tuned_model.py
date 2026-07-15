"""Finalisasi model dengan out-of-fold threshold tuning pada development set."""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from fraud_detection.config import DEFAULT_DATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR  # noqa: E402
from fraud_detection.data import (  # noqa: E402
    clean_transactions,
    load_transactions,
    stratified_train_validation_test_split,
)
from fraud_detection.evaluation import classification_metrics, ranking_metrics_at_k  # noqa: E402
from fraud_detection.risk import risk_level_boundaries  # noqa: E402
from tune_advanced_models import optimal_f1_threshold  # noqa: E402


def finalist_candidates() -> dict[str, object]:
    """Buat ulang dua kandidat teratas dengan parameter hasil tuning."""
    return {
        "extra_trees_leaf1": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
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


def out_of_fold_probabilities(
    model: object,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[np.ndarray, float]:
    """Hasilkan probability OOF sehingga setiap baris dinilai model yang tidak melatihnya."""
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.zeros(len(features), dtype=float)
    started = time.perf_counter()

    # Fit clone baru pada setiap fold untuk mencegah state model bocor antar-fold.
    for fold_number, (fit_index, validation_index) in enumerate(
        splitter.split(features, target), start=1
    ):
        print(f"  fold {fold_number}/3", flush=True)
        # CatBoost memodifikasi representasi internal class_weights sehingga
        # sklearn.clone gagal; deep copy pada estimator yang belum fit aman untuk
        # membuat instance independen dengan parameter yang sama.
        fold_model = deepcopy(model)
        fold_model.fit(features.iloc[fit_index], target.iloc[fit_index])
        probabilities[validation_index] = fold_model.predict_proba(
            features.iloc[validation_index]
        )[:, 1]

    return probabilities, time.perf_counter() - started


def main() -> None:
    """Pilih finalis dengan OOF F1, refit development data, dan audit di test."""
    frame, _ = clean_transactions(load_transactions(DEFAULT_DATA_PATH))
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        stratified_train_validation_test_split(frame)
    )

    # Train dan validation digabung menjadi development set. Threshold dipilih
    # dari OOF predictions, sehingga seluruh label development dapat digunakan
    # tanpa menilai baris dengan model yang pernah melatih baris tersebut.
    x_development = pd.concat([x_train, x_validation], ignore_index=True)
    y_development = pd.concat([y_train, y_validation], ignore_index=True)

    rows: list[dict] = []
    candidates = finalist_candidates()
    for name, model in candidates.items():
        print(f"OOF finalist: {name}", flush=True)
        oof_score, fit_seconds = out_of_fold_probabilities(model, x_development, y_development)
        threshold, best_f1 = optimal_f1_threshold(y_development.to_numpy(), oof_score)
        rows.append(
            {
                "model": name,
                "threshold": threshold,
                "fit_seconds_3fold": fit_seconds,
                **classification_metrics(y_development.to_numpy(), oof_score, threshold),
                **ranking_metrics_at_k(y_development.to_numpy(), oof_score),
                "optimized_f1": best_f1,
            }
        )

    oof_metrics = pd.DataFrame(rows).sort_values(["f1", "pr_auc"], ascending=False)
    selected = oof_metrics.iloc[0]
    selected_name = str(selected["model"])
    selected_threshold = float(selected["threshold"])

    # Refit kandidat terpilih pada seluruh development data agar model final
    # belajar dari 402 fraud, bukan hanya 331 fraud pada training split awal.
    final_model = candidates[selected_name]
    final_model.fit(x_development, y_development)

    # Test tetap menjadi holdout dan tidak ikut menentukan model atau threshold.
    test_score = final_model.predict_proba(x_test)[:, 1]
    test_metrics = {
        "model": selected_name,
        "split": "test",
        **classification_metrics(y_test.to_numpy(), test_score, selected_threshold),
        **ranking_metrics_at_k(y_test.to_numpy(), test_score),
    }

    # Simpan model final dan config yang kompatibel dengan prediction pipeline.
    joblib.dump(final_model, MODELS_DIR / "oof_tuned_model.joblib")
    oof_metrics.to_csv(REPORTS_DIR / "oof_finalist_metrics.csv", index=False)
    final_config = {
        "selected_model": selected_name,
        "selection_objective": "3fold_oof_f1",
        "threshold": selected_threshold,
        "development_rows": len(x_development),
        "development_frauds": int(y_development.sum()),
        "oof_metrics": {
            key: float(selected[key])
            for key in ("precision", "recall", "f1", "pr_auc", "alerts")
        },
        "test_metrics": test_metrics,
        "risk_level_boundaries": risk_level_boundaries(selected_threshold),
        "score_note": "Risk score belum dikalibrasi sebagai probabilitas absolut.",
    }
    (MODELS_DIR / "oof_tuned_threshold.json").write_text(
        json.dumps(final_config, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "oof_final_test.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )

    print(oof_metrics[["model", "threshold", "precision", "recall", "f1", "pr_auc", "alerts"]].to_string(index=False))
    print("\nOOF-selected model:", selected_name)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
