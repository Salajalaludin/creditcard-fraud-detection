"""Paketkan model dan dashboard artifacts ke ZIP yang tidak masuk Git."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from fraud_detection.artifacts import sha256_file  # noqa: E402

# Daftar eksplisit menghindari dataset mentah atau output sensitif ikut terpaket.
REQUIRED_FILES = (
    "models/fraud_detection_model.joblib",
    "models/threshold_config.json",
    "data/predictions/test_risk_scores.csv",
    "data/predictions/investigation_queue.csv",
    "reports/threshold_test_metrics.json",
    "reports/threshold_recommendations.csv",
    "reports/model_comparison_metrics.csv",
    "reports/figures/model_comparison_pr_curve.png",
    "reports/figures/threshold_tradeoff.png",
    "reports/figures/business_cost_sensitivity.png",
)

# Artifact monitoring bersifat opsional agar bundle tetap dapat dibuat setelah pipeline inti.
OPTIONAL_FILES = (
    "reports/tuning_comparison.csv",
    "reports/feature_importance.csv",
    "reports/figures/feature_importance.png",
    "reports/calibration_report.json",
    "reports/out_of_time_evaluation.json",
    "reports/anomaly_detection_metrics.json",
    "reports/drift_report.csv",
    "reports/active_policy_metadata.json",
)


def main() -> None:
    """Validasi artifact, buat ZIP deterministik, dan cetak SHA-256 deployment."""
    output = PROJECT_ROOT / "outputs" / "streamlit_artifacts.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Dashboard tidak boleh dirilis dengan bundle yang kehilangan file wajib.
    missing = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Artifact wajib belum tersedia: " + ", ".join(missing))

    included = [*REQUIRED_FILES, *(name for name in OPTIONAL_FILES if (PROJECT_ROOT / name).is_file())]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative_name in included:
            archive.write(PROJECT_ROOT / relative_name, arcname=relative_name)

    metadata = {
        "path": str(output),
        "size_mb": round(output.stat().st_size / (1024 * 1024), 2),
        "sha256": sha256_file(output),
        "files": len(included),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
