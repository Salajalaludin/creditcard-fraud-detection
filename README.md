# Fraud Detection & Transaction Risk Scoring

End-to-end machine learning untuk mendeteksi fraud kartu kredit, menghasilkan risk score, dan menyusun investigation queue. Sistem ini adalah **human-in-the-loop decision support**, bukan pemblokiran otomatis.

Repository: [Salajalaludin/creditcard-detection](https://github.com/Salajalaludin/creditcard-detection)

## Hasil utama

Model aktif: **Extra Trees** (600 trees, `class_weight="balanced"`) dengan threshold **0,3783**.

| Metric | Validation | Test | Pseudo OOT |
|---|---:|---:|---:|
| Precision | 100,00% | **94,83%** | 92,68% |
| Recall | 78,87% | **77,46%** | 73,08% |
| F1 | 88,19% | **85,27%** | 81,72% |
| PR-AUC | 82,97% | **82,25%** | 77,77% |
| Alerts | 56 | **58** | 41 |

Test random-split menghasilkan 55 TP, 3 FP, 16 FN, dan 42.485 TN. Top-100 queue menemukan 58 dari 71 fraud (81,69%). Target F1 >90% **belum tercapai secara valid**. Kandidat terkalibrasi mencapai F1 90,91% pada threshold-validation subset, tetapi test tetap 85,27%; angka 90,91% tidak diklaim sebagai generalisasi.

> **Audit note:** test random-split pernah diamati selama eksperimen lanjutan, sehingga 85,27% adalah hasil holdout terbaik yang teramati, bukan estimasi test yang sepenuhnya untouched. Pseudo out-of-time memberi pemeriksaan lebih konservatif, tetapi data masa depan eksternal tetap diperlukan.

## Data dan metodologi

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Letakkan sebagai `Data/creditcard.csv`.

- 284.807 transaksi mentah dan 492 fraud;
- tidak ada missing value;
- 1.081 exact duplicate dihapus sebelum split;
- 283.726 transaksi bersih dan 473 fraud (0,1667%).

```text
Raw CSV → schema check + deduplication → train/validation/test
        → class weighting/resampling → model comparison + tuning
        → threshold + business cost → risk score + investigation queue
        → explainability + calibration + drift monitoring → dashboard/CLI
```

Split utama stratified 70/15/15 (`random_state=42`): train 198.608/331 fraud, validation 42.559/71 fraud, test 42.559/71 fraud. Resampling dan preprocessing hanya di-fit pada train. Metrik utama: PR-AUC, precision, recall, F1, alert volume, top-K recall, dan fraud amount.

Model yang dibandingkan mencakup Logistic Regression, class weighting, undersampling, SMOTE, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost, weighted ensemble, serta Isolation Forest sebagai benchmark anomaly detection.

## Setup

Prasyarat: Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Gunakan `requirements.txt` bila ingin rentang versi yang fleksibel. Untuk development/test: `python -m pip install -r requirements-dev.txt`.

## Project guide

Jalankan pipeline inti dari root repository:

```powershell
# 1–3: validation, cleaning, EDA, baseline
.\.venv\Scripts\python.exe scripts\run_eda.py
.\.venv\Scripts\python.exe scripts\train_baseline.py

# 4–6: imbalance strategies dan model comparison
.\.venv\Scripts\python.exe scripts\train_model_comparison.py

# 7–9: threshold, cost, risk score, queue, dan policy provenance
.\.venv\Scripts\python.exe scripts\optimize_threshold.py
.\.venv\Scripts\python.exe scripts\refresh_active_policy.py
```

Advanced/diagnostic pipeline:

```powershell
# Tuning (resource intensive)
.\.venv\Scripts\python.exe scripts\tune_advanced_models.py
.\.venv\Scripts\python.exe scripts\tune_boosting_ensemble.py
.\.venv\Scripts\python.exe scripts\finalize_tuned_model.py
.\.venv\Scripts\python.exe scripts\promote_tuned_model.py

# Audit-ready diagnostics
.\.venv\Scripts\python.exe scripts\calibrate_model.py
.\.venv\Scripts\python.exe scripts\evaluate_out_of_time.py
.\.venv\Scripts\python.exe scripts\train_anomaly_detection.py
.\.venv\Scripts\python.exe scripts\explain_model.py
.\.venv\Scripts\python.exe scripts\monitor_drift.py
```

Sesudah model dipromosikan, selalu jalankan `refresh_active_policy.py` agar tabel/plot threshold memiliki `model_id` dan hash yang sama dengan model aktif.

### Prediction CLI

CSV wajib memiliki `Time`, `V1`–`V28`, dan `Amount`; `Class` opsional.

```powershell
.\.venv\Scripts\python.exe scripts\predict_transactions.py `
  --input Data\creditcard.csv `
  --output data\predictions\scored_transactions.csv
```

### Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Dashboard memiliki Overview, Investigation Queue, Model & Threshold, Explainability & Monitoring, serta Batch Prediction. Provenance guard menolak artefak threshold yang berasal dari model berbeda.

### Test dan notebook

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest --cov=fraud_detection --cov-report=term-missing
.\.venv\Scripts\python.exe -m jupyter lab
```

CI GitHub Actions menjalankan test dan coverage pada setiap push/pull request.

## Risk policy

`Risk Score = model output × 100`. Score aktif belum merupakan probabilitas fraud absolut; gunakan untuk ranking relatif.

| Level | Batas aktif | Action |
|---|---:|---|
| Low | 0–37,83 | Process normally |
| Medium | 37,83–58,56 | Enhanced monitoring |
| High | 58,56–79,28 | Manual review |
| Critical | 79,28–100 | Priority investigation |

Simulasi biaya:

```text
Total Cost = review semua alert + gangguan false positive
           + penalti fraud lolos + nominal fraud yang terlewat
```

Ubah asumsi di `BUSINESS_SCENARIOS` pada `scripts/optimize_threshold.py` atau `scripts/refresh_active_policy.py` menggunakan angka operasional aktual.

## Ringkasan eksperimen dan audit

- Class weighting meningkatkan recall tetapi Logistic Regression balanced menghasilkan terlalu banyak alert pada threshold 0,5.
- Random Forest memberi validation PR-AUC awal 84,66%; Extra Trees dipromosikan karena kombinasi F1 holdout terbaik yang teramati.
- XGBoost, LightGBM, CatBoost, ensemble, dan OOF tuning tidak menghasilkan test F1 >90% tanpa praktik leakage.
- Calibration sigmoid memperbaiki interpretasi score (test Brier 0,000455; log loss 0,003506), tetapi tidak menaikkan test F1.
- Isolation Forest hanya benchmark: test PR-AUC 9,46%, jauh di bawah supervised model.
- Pseudo OOT: threshold 0,3033, F1 81,72%, PR-AUC 77,77%.
- Permutation importance teratas: V14, V4, V17, V16, V11, V12, dan V10. Fitur anonim membatasi penjelasan bisnis dan importance bukan kausalitas.
- PSI seluruh 30 fitur berstatus Stable pada random-split sanity check; monitoring produksi tetap harus memakai reference/current window berbasis waktu.

## Struktur dan artefak

```text
├── Data/                     # raw dataset (tidak masuk Git)
├── data/predictions/         # risk scores dan queue
├── dashboard/app.py          # Streamlit dashboard
├── models/                   # model + policy config
├── notebooks/                # notebook 01–09
├── reports/                  # CSV/JSON/figures untuk mesin/dashboard
├── scripts/                  # executable pipelines
├── src/fraud_detection/      # reusable modules
├── tests/                    # unit tests
└── README.md                 # setup, guide, laporan, executive summary
```

Artefak utama: `models/fraud_detection_model.joblib`, `models/threshold_config.json`, `reports/active_policy_metadata.json`, `data/predictions/test_risk_scores.csv`, dan `data/predictions/investigation_queue.csv`. Generated data/model/report tidak disimpan ke Git; bangun ulang melalui pipeline.

## Executive summary dan rekomendasi

Extra Trees memberikan queue yang sangat presisi pada data ini: 58 alert test berisi 55 fraud. Namun recall 77,46% berarti 16 fraud masih lolos. Gunakan output untuk prioritas review, bukan automatic decline.

Sebelum produksi:

1. validasi pada future out-of-time dataset yang benar-benar baru;
2. tetapkan biaya dan kapasitas alert aktual;
3. promosikan model terkalibrasi hanya setelah acceptance test terpisah;
4. monitor drift, latency, alert yield, investigator outcome, dan subgroup impact;
5. retrain dengan fitur customer, merchant, device, location, serta transaction history.

## Limitasi

- Dataset hanya sekitar dua hari dan fitur `V1`–`V28` anonim.
- Tidak ada fitur entitas/history untuk menangkap pola berulang.
- Test random-split tidak lagi pristine setelah eksperimen iteratif.
- Asumsi biaya belum berasal dari data bisnis aktual.
- Pseudo OOT bukan pengganti external future validation.

Seluruh fungsi dan blok kode utama memiliki docstring/comment agar mudah dimodifikasi.
