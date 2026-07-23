"""CLI untuk menghasilkan risk score dari CSV transaksi baru."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from fraud_detection.config import FEATURE_COLUMNS, MODELS_DIR, TARGET  # noqa: E402
from fraud_detection.prediction import load_inference_data  # noqa: E402
from fraud_detection.risk import score_transactions  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Baca lokasi input, output, model, dan konfigurasi threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV transaksi yang akan dinilai.")
    parser.add_argument("--output", type=Path, required=True, help="Lokasi CSV hasil scoring.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "fraud_detection_model.joblib",
    )
    parser.add_argument(
        "--threshold-config",
        type=Path,
        default=MODELS_DIR / "threshold_config.json",
    )
    return parser.parse_args()


def main() -> None:
    """Validasi input, jalankan model, lalu simpan hasil risk scoring."""
    args = parse_args()

    # Load pipeline utuh dan threshold hasil validasi, bukan threshold hard-coded.
    model = joblib.load(args.model)
    threshold_config = json.loads(args.threshold_config.read_text(encoding="utf-8"))
    threshold = float(threshold_config["threshold"])

    # Loader menerima data dengan atau tanpa kolom actual Class.
    frame = load_inference_data(args.input)
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    actual_class = frame[TARGET] if TARGET in frame.columns else None
    scored = score_transactions(frame, probabilities, threshold, actual_class=actual_class)

    # Buat parent folder otomatis agar output path mudah dikustomisasi.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.sort_values("risk_score", ascending=False).to_csv(args.output, index=False)

    print(f"Scoring selesai: {len(scored):,} transaksi")
    print(f"Threshold: {threshold:.4f}")
    print(f"Alerts: {int(scored['predicted_class'].sum()):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
