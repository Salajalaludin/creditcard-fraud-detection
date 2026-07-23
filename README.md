# 🛡️ Fraud Detection & Transaction Risk Scoring

A practical machine learning project for spotting suspicious credit card transactions, assigning risk scores, and building a review queue for investigators.

The goal is not to automatically block payments. Think of it as a second pair of eyes that helps a fraud team decide what to review first.

## ✨ What this project does

- Cleans and checks highly imbalanced transaction data.
- Compares class weighting, undersampling, SMOTE, tree models, and boosting models.
- Tunes the decision threshold instead of relying on the default `0.5`.
- Turns model output into simple risk levels and an investigation queue.
- Includes a batch prediction tool and a Streamlit dashboard.
- Adds calibration checks, drift monitoring, explainability, tests, and CI.

## 📊 Results at a glance

The active model is an **Extra Trees classifier** with a decision threshold of **0.3783**.

| Metric | Test result |
|---|---:|
| Precision | **94.83%** |
| Recall | **77.46%** |
| F1 score | **85.27%** |
| PR-AUC | **82.25%** |
| Alerts | **58** |

Out of 42,559 test transactions, the model produced 58 alerts: 55 were real fraud cases and 3 were false alarms. It still missed 16 fraud cases, so the model should support human review rather than make final decisions on its own.

The project explored several ways to reach an F1 score above 90%. A calibrated candidate passed 90% on a validation subset, but it did not repeat that result on the test set. This README reports the more honest test result instead.

## 💳 Dataset

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Download `creditcard.csv` and place it here:

```text
Data/creditcard.csv
```

The raw data contains 284,807 transactions and only 492 fraud cases. Exact duplicates are removed before splitting the data to reduce leakage.

## 🚀 Quick start

You need Python 3.10 or newer. Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Build the main project artifacts:

```powershell
.\.venv\Scripts\python.exe scripts\run_eda.py
.\.venv\Scripts\python.exe scripts\train_baseline.py
.\.venv\Scripts\python.exe scripts\train_model_comparison.py
.\.venv\Scripts\python.exe scripts\optimize_threshold.py
```

## 🖥️ Run the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard includes:

- A quick performance overview.
- A filterable investigation queue.
- Model and threshold comparisons.
- Feature importance and drift checks.
- CSV upload for batch predictions.

## 📁 Score a CSV file

Your CSV needs `Time`, `V1` through `V28`, and `Amount`. The `Class` column is optional.

```powershell
.\.venv\Scripts\python.exe scripts\predict_transactions.py `
  --input Data\creditcard.csv `
  --output data\predictions\scored_transactions.csv
```

Each transaction receives a model score, risk score, risk level, predicted class, and suggested action.

## 🚦 Risk levels

The model score is multiplied by 100 to make it easier to read. It is a relative risk score, not a guaranteed fraud probability.

| Risk level | Score | Suggested action |
|---|---:|---|
| Low | 0–37.83 | Process normally |
| Medium | 37.83–58.56 | Monitor more closely |
| High | 58.56–79.28 | Manual review |
| Critical | 79.28–100 | Priority investigation |

## ☁️ Deploy on Streamlit Community Cloud

The trained model and generated reports are intentionally kept out of Git. Package them into one deployment bundle:

```powershell
.\.venv\Scripts\python.exe scripts\package_streamlit_artifacts.py
```

This creates:

```text
outputs/streamlit_artifacts.zip
```

Upload that ZIP as an asset in a GitHub Release. Do not use the release page or GitHub's automatic source-code ZIP—the dashboard needs the direct asset download URL.

Create an app at [Streamlit Community Cloud](https://share.streamlit.io) with:

```text
Repository : Salajalaludin/creditcard-fraud-detection
Branch     : main
Main file  : dashboard/app.py
Python     : 3.12
```

Add these values under **Advanced settings → Secrets**:

```toml
ARTIFACT_URL = "https://github.com/Salajalaludin/creditcard-fraud-detection/releases/download/v1.0.0/streamlit_artifacts.zip"
ARTIFACT_SHA256 = "SHA256_PRINTED_BY_THE_PACKAGE_SCRIPT"
```

The dashboard downloads the bundle at startup and verifies its checksum before loading the model.

## ✅ Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The current test suite has 20 passing tests. GitHub Actions runs the tests again after each push or pull request.

## 🗂️ Project structure

```text
dashboard/              Streamlit app
notebooks/              Project notebooks 01–09
scripts/                Training, evaluation, and deployment commands
src/fraud_detection/    Reusable Python code
tests/                  Unit tests
models/                 Local model and threshold files
reports/                Local metrics and charts
data/predictions/       Local scores and investigation queue
```

Large datasets, trained models, generated reports, deployment ZIP files, and presentation outputs are ignored by Git.

## 🧪 Extra experiments

The repository also includes optional scripts for:

- Advanced model tuning and ensembles.
- Pseudo out-of-time evaluation.
- Isolation Forest anomaly detection.
- Permutation feature importance.
- Population Stability Index monitoring.

These scripts are useful for exploration, but they are not required just to run the dashboard with an existing artifact bundle.

## ⚠️ Limitations

- The dataset covers only about two days of transactions.
- Most features are anonymous PCA components, so business explanations are limited.
- Customer, merchant, device, location, and transaction-history features are missing.
- The current business costs are example assumptions.
- The model still needs validation on truly new future data before real-world use.

## 📝 Final note

This is a portfolio and learning project, not a production fraud-blocking system. Its safest use is to rank transactions for human investigation and make the review process more focused.
