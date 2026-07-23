"""Dashboard Streamlit untuk fraud risk, queue, dan model monitoring."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Path absolut membuat dashboard aman dijalankan dari working directory mana pun.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Konfigurasi halaman wajib menjadi perintah Streamlit pertama.
st.set_page_config(page_title="Fraud Risk Monitor", page_icon="🛡️", layout="wide")


def deployment_setting(name: str) -> str:
    """Baca setting dari environment atau Streamlit Secrets tanpa memaksa file lokal."""
    # Environment variable memudahkan container deployment di luar Community Cloud.
    if value := os.getenv(name):
        return value
    try:
        return str(st.secrets.get(name, ""))
    except FileNotFoundError:
        return ""


# Jika URL deployment tersedia, arahkan seluruh generated artifact ke cache runtime.
ARTIFACT_URL = deployment_setting("ARTIFACT_URL")
ARTIFACT_SHA256 = deployment_setting("ARTIFACT_SHA256")
if ARTIFACT_URL:
    os.environ.setdefault("FRAUD_ARTIFACT_ROOT", str(PROJECT_ROOT / ".runtime_artifacts"))

from fraud_detection.artifacts import download_artifact_bundle  # noqa: E402
from fraud_detection.config import ARTIFACT_ROOT, FEATURE_COLUMNS, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, TARGET  # noqa: E402
from fraud_detection.prediction import load_inference_data  # noqa: E402
from fraud_detection.risk import score_transactions  # noqa: E402

# Kontrak minimal ZIP memastikan aplikasi tidak menerima release yang salah.
REQUIRED_ARTIFACT_MEMBERS = (
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


@st.cache_resource(show_spinner="Downloading verified model artifacts...")
def bootstrap_deployment_artifacts() -> str | None:
    """Unduh artifact satu kali ketika deployment tidak memiliki model lokal."""
    # Local development langsung memakai file pipeline yang sudah tersedia.
    if (MODELS_DIR / "fraud_detection_model.joblib").is_file():
        return None
    if not ARTIFACT_URL:
        return None
    return download_artifact_bundle(
        ARTIFACT_URL,
        ARTIFACT_ROOT,
        expected_sha256=ARTIFACT_SHA256 or None,
        required_members=REQUIRED_ARTIFACT_MEMBERS,
    )


@st.cache_resource
def load_model_and_threshold() -> tuple[object, dict]:
    """Load model dan policy sekali, lalu cache antarrerun dashboard."""
    model = joblib.load(MODELS_DIR / "fraud_detection_model.joblib")
    config = json.loads((MODELS_DIR / "threshold_config.json").read_text(encoding="utf-8"))
    return model, config


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    """Cache CSV agar filter dan perpindahan tab tetap responsif."""
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    """Cache laporan JSON kecil yang dipakai pada KPI."""
    return json.loads(path.read_text(encoding="utf-8"))


def render_overview(scored: pd.DataFrame, metrics: dict) -> None:
    """Tampilkan KPI eksekutif dan distribusi level risiko."""
    st.subheader("Executive overview")
    columns = st.columns(5)
    columns[0].metric("Test transactions", f"{len(scored):,}")
    columns[1].metric("Alerts", f"{int(metrics['alerts']):,}")
    columns[2].metric("Precision", f"{metrics['precision']:.1%}")
    columns[3].metric("Recall", f"{metrics['recall']:.1%}")
    columns[4].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    # KPI nominal menghubungkan metrik model dengan dampak transaksi.
    amount_columns = st.columns(4)
    amount_columns[0].metric("Detected fraud amount", f"{metrics['detected_fraud_amount']:,.2f}")
    amount_columns[1].metric("Missed fraud amount", f"{metrics['missed_fraud_amount']:,.2f}")
    amount_columns[2].metric("Average risk score", f"{scored['risk_score'].mean():.2f}")
    amount_columns[3].metric("Critical transactions", f"{int((scored['risk_level'] == 'Critical').sum()):,}")

    # Bar horizontal dan skala symlog menjaga level kecil tetap terlihat.
    order = ["Low", "Medium", "High", "Critical"]
    risk_summary = pd.DataFrame(
        {
            "risk_level": order,
            "transactions": scored["risk_level"].value_counts().reindex(order, fill_value=0).to_numpy(),
            "average_amount": scored.groupby("risk_level", observed=False)["Amount"].mean().reindex(order).fillna(0).to_numpy(),
        }
    )

    def risk_chart(field: str, title: str, axis_title: str, scale: str, number_format: str) -> dict:
        encoding = {
            "x": {"field": field, "type": "quantitative", "scale": {"type": scale}, "axis": {"title": axis_title}},
            "y": {"field": "risk_level", "type": "ordinal", "sort": order, "axis": {"title": None}},
        }
        return {
            "title": title,
            "height": 280,
            "layer": [
                {"mark": {"type": "bar", "color": "#0E6FCA", "cornerRadiusEnd": 3}, "encoding": encoding},
                {
                    "mark": {"type": "text", "align": "right", "dx": -6, "color": "white", "fontWeight": "bold"},
                    "encoding": {**encoding, "text": {"field": field, "type": "quantitative", "format": number_format}},
                },
            ],
        }

    left, right = st.columns(2)
    left.vega_lite_chart(risk_summary, risk_chart("transactions", "Transaction volume by risk level", "Transactions (symlog scale)", "symlog", ",.0f"), width="stretch")
    right.vega_lite_chart(risk_summary, risk_chart("average_amount", "Average transaction amount by risk level", "Average amount", "linear", ",.2f"), width="stretch")


def render_queue(queue: pd.DataFrame) -> None:
    """Tampilkan investigation queue dengan filter operasional dan unduhan."""
    st.subheader("Investigation queue")
    filters = st.columns(3)
    levels = ["Low", "Medium", "High", "Critical"]
    selected = filters[0].multiselect("Risk level", levels, default=levels)
    minimum = filters[1].slider("Minimum risk score", 0.0, 100.0, 0.0, 1.0)
    maximum = filters[2].number_input("Maximum rows", 10, 5_000, 500, 10)

    # Sortasi dari pipeline dipertahankan; filter hanya mempersempit scope investigator.
    filtered = queue.loc[queue["risk_level"].isin(selected) & (queue["risk_score"] >= minimum)].head(int(maximum))
    visible = ["queue_rank", "transaction_id", "Time", "Amount", "fraud_probability", "risk_score", "risk_level", "predicted_class", "actual_class", "recommended_action"]
    st.dataframe(filtered[visible], width="stretch", hide_index=True)
    st.download_button("Download filtered queue", filtered.to_csv(index=False).encode("utf-8"), "filtered_investigation_queue.csv", "text/csv")


def render_model_performance(config: dict) -> None:
    """Tampilkan pencarian model dan threshold yang terikat ke model aktif."""
    st.subheader("Model performance and threshold")
    metrics = load_csv(REPORTS_DIR / "model_comparison_metrics.csv")
    recommendations = load_csv(REPORTS_DIR / "threshold_recommendations.csv")
    active_model = str(config["selected_model"])

    # Provenance guard mencegah grafik/tabel lama ditampilkan untuk model baru.
    artifact_models = set(recommendations["model_id"].dropna().astype(str))
    if artifact_models != {active_model}:
        st.error("Threshold artifact tidak cocok dengan model aktif. Jalankan scripts/refresh_active_policy.py.")
        return

    st.info(f"Active model: **{active_model}** · active threshold: **{float(config['threshold']):.4f}**")
    st.caption("Historical model search — validation candidates and selected-model test result")
    columns = ["model", "split", "precision", "recall", "f1", "pr_auc", "roc_auc", "alerts"]
    st.dataframe(metrics[columns], width="stretch", hide_index=True)

    st.caption(f"Threshold recommendations for {active_model} on validation set")
    threshold_columns = ["strategy", "threshold", "precision", "recall", "f1", "alerts"]
    st.dataframe(recommendations[threshold_columns], width="stretch", hide_index=True)

    # Tuning table bersifat historis; policy aktif selalu ditampilkan terpisah di atas.
    tuning_path = REPORTS_DIR / "tuning_comparison.csv"
    if tuning_path.exists():
        tuning = load_csv(tuning_path)
        tuning_columns = ["model", "type", "threshold", "precision", "recall", "f1", "pr_auc", "alerts"]
        st.caption("Advanced tuning — historical validation experiments")
        st.dataframe(tuning.sort_values(["f1", "pr_auc"], ascending=False)[tuning_columns].head(20), width="stretch", hide_index=True)

    # Semua figures dibuat oleh pipeline sehingga dashboard tidak melakukan training ulang.
    figures = st.columns(2)
    figures[0].image(REPORTS_DIR / "figures" / "model_comparison_pr_curve.png")
    figures[1].image(REPORTS_DIR / "figures" / "threshold_tradeoff.png")
    st.image(REPORTS_DIR / "figures" / "business_cost_sensitivity.png")


def render_monitoring() -> None:
    """Tampilkan explainability, calibration, pseudo-OOT, anomaly, dan drift."""
    st.subheader("Explainability & monitoring")
    importance_path = REPORTS_DIR / "feature_importance.csv"
    if importance_path.exists():
        importance = load_csv(importance_path)
        st.caption("Permutation importance — relative contribution, not causality")
        st.dataframe(importance.head(15), width="stretch", hide_index=True)
        figure = REPORTS_DIR / "figures" / "feature_importance.png"
        if figure.exists():
            st.image(figure)

    # Tiga protokol evaluasi ditampilkan terpisah agar hasilnya tidak tertukar.
    summary = st.columns(3)
    calibration_path = REPORTS_DIR / "calibration_report.json"
    oot_path = REPORTS_DIR / "out_of_time_evaluation.json"
    anomaly_path = REPORTS_DIR / "anomaly_detection_metrics.json"
    if calibration_path.exists():
        result = load_json(calibration_path)
        summary[0].metric("Calibrated test F1", f"{result['test_metrics']['f1']:.1%}")
        summary[0].caption("Candidate only; not promoted")
    if oot_path.exists():
        result = load_json(oot_path)
        summary[1].metric("Pseudo OOT F1", f"{result['test']['f1']:.1%}")
        summary[1].caption("Chronological 70/15/15 check")
    if anomaly_path.exists():
        result = load_json(anomaly_path)
        summary[2].metric("Anomaly PR-AUC", f"{result['results']['test']['pr_auc']:.3f}")
        summary[2].caption("Isolation Forest benchmark")

    # PSI random-split hanya sanity check; baseline produksi harus berbasis waktu.
    drift_path = REPORTS_DIR / "drift_report.csv"
    if drift_path.exists():
        drift = load_csv(drift_path)
        st.caption("Feature drift (PSI) — random-split sanity check, not production monitoring")
        st.dataframe(drift.sort_values("psi", ascending=False), width="stretch", hide_index=True)


def render_batch_prediction(model: object, config: dict) -> None:
    """Terima CSV, validasi schema, lalu jalankan scoring dengan policy aktif."""
    st.subheader("Batch prediction")
    st.info("Upload CSV dengan Time, V1–V28, dan Amount. Class bersifat opsional.")
    uploaded = st.file_uploader("Transaction CSV", type=["csv"])
    if uploaded is None:
        return
    try:
        # Gunakan modul inference yang sama dengan CLI untuk menjaga konsistensi schema.
        frame = load_inference_data(uploaded)
        probabilities = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
        actual = frame[TARGET] if TARGET in frame.columns else None
        scored = score_transactions(frame, probabilities, float(config["threshold"]), actual).sort_values("risk_score", ascending=False)
    except Exception as error:
        st.error(f"Prediction gagal: {error}")
        return
    st.success(f"{len(scored):,} transaksi berhasil dinilai.")
    st.dataframe(scored.head(500), width="stretch", hide_index=True)
    st.download_button("Download scored transactions", scored.to_csv(index=False).encode("utf-8"), "scored_transactions.csv", "text/csv")


def main() -> None:
    """Susun dashboard dan hubungkan tab ke artefak pipeline tervalidasi."""
    st.title("🛡️ Fraud Detection & Transaction Risk Scoring")
    st.caption("Human-in-the-loop decision support for transaction investigation")

    # Bootstrap berjalan sebelum pengecekan required files pada deployment cloud.
    try:
        downloaded_checksum = bootstrap_deployment_artifacts()
    except Exception as error:
        st.error(f"Deployment artifact gagal dimuat: {error}")
        st.info("Periksa ARTIFACT_URL dan ARTIFACT_SHA256 pada Streamlit Secrets.")
        st.stop()
    if downloaded_checksum:
        st.caption(f"Verified deployment artifact: `{downloaded_checksum[:12]}…`")

    # Fail fast memberikan instruksi yang jelas jika pipeline belum selesai.
    required = [MODELS_DIR / "fraud_detection_model.joblib", MODELS_DIR / "threshold_config.json", PREDICTIONS_DIR / "test_risk_scores.csv", PREDICTIONS_DIR / "investigation_queue.csv", REPORTS_DIR / "threshold_test_metrics.json", REPORTS_DIR / "threshold_recommendations.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        st.error("Artefak pipeline belum lengkap:\n" + "\n".join(missing))
        if not ARTIFACT_URL:
            st.info(
                "Local: jalankan pipeline. Cloud: set ARTIFACT_URL dan "
                "ARTIFACT_SHA256 melalui Streamlit Secrets."
            )
        st.stop()

    model, config = load_model_and_threshold()
    scored = load_csv(PREDICTIONS_DIR / "test_risk_scores.csv")
    queue = load_csv(PREDICTIONS_DIR / "investigation_queue.csv")
    metrics = load_json(REPORTS_DIR / "threshold_test_metrics.json")

    # Sidebar membuat policy operasional selalu terlihat pada seluruh tab.
    with st.sidebar:
        st.header("Active policy")
        st.write(f"Model: **{config['selected_model']}**")
        st.metric("Decision threshold", f"{config['threshold']:.3f}")
        st.write(f"Scenario: **{config['selected_scenario']}**")
        st.warning("Risk score is not a calibrated absolute fraud probability.")

    overview, queue_tab, performance, monitoring, prediction = st.tabs(["Overview", "Investigation Queue", "Model & Threshold", "Explainability & Monitoring", "Batch Prediction"])
    with overview:
        render_overview(scored, metrics)
    with queue_tab:
        render_queue(queue)
    with performance:
        render_model_performance(config)
    with monitoring:
        render_monitoring()
    with prediction:
        render_batch_prediction(model, config)


if __name__ == "__main__":
    main()
