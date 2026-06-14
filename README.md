# Credit Risk Scoring Engine

An end-to-end machine learning system that predicts the probability of a loan applicant defaulting, with per-prediction explainability using SHAP. Built on the Home Credit Default Risk dataset (300k+ applicants).

🔗 **Live Demo:** [credit-risk-scoring-engine-adarsh.streamlit.app](https://credit-risk-scoring-engine-adarsh.streamlit.app)

## Overview

Lenders often need to assess credit risk for applicants with little to no traditional credit history. This project builds a complete pipeline — from raw, messy multi-table data to a deployed interactive dashboard — that predicts default risk and explains *why* the model made its decision for any given applicant.

## Features

- **ETL Pipeline** — merges and aggregates 3 separate data tables (application, bureau, previous applications) into a single feature set using groupby aggregations
- **Feature Engineering** — derived ratio features (loan-to-income, annuity-to-income) and bureau credit history aggregations
- **Class Imbalance Handling** — addressed an 8.1% default rate using `class_weight='balanced'` and `scale_pos_weight`
- **Model Comparison** — Logistic Regression, Decision Tree, Random Forest, and XGBoost evaluated on AUC-ROC, F1, Precision, and Recall
- **Explainability** — SHAP values for global feature importance and per-applicant local explanations
- **Interactive Dashboard** — Streamlit app with 4 tabs: Predict, EDA, Model Performance, and Correlation

## Results

| Model | AUC-ROC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.6119 | 0.1876 | 0.1148 | 0.5132 |
| Decision Tree | 0.7153 | 0.2418 | 0.1489 | 0.6425 |
| Random Forest | 0.7247 | 0.0032 | 0.8000 | 0.0016 |
| **XGBoost (final model)** | **0.7458** | **0.2738** | **0.1772** | **0.6020** |

XGBoost was selected as the final model — it achieved the best AUC-ROC while maintaining strong recall, meaning it catches the majority of actual defaulters without sacrificing overall ranking quality.

## 📂 Project Structure

```text
credit-risk-scoring-engine/
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Cleaning, Merging, Feature Engineering
│   ├── 03_modeling.ipynb         # Model Training & Comparison
│   └── 04_shap_analysis.ipynb    # SHAP Explainability
│
├── app/
│   └── streamlit_app.py          # Interactive Dashboard
│
├── models/
│   └── xgb_model.pkl             # Final Trained Model
│
├── outputs/                      # Saved Plots and Sample Data
│
├── requirements.txt
└── README.md
```
## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn

## Dataset

[Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk/data)

Tables used:
- `application_train.csv` / `application_test.csv` — main applicant data
- `bureau.csv` — applicant's past loans from other financial institutions
- `previous_application.csv` — applicant's previous loan applications with Home Credit

## Setup & Run Locally

```bash
# Clone the repo
git clone https://github.com/adarshsiingh/credit-risk-scoring-engine.git
cd credit-risk-scoring-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Run the dashboard
cd app
streamlit run streamlit_app.py
```

## How It Works

1. **EDA** — analyzed class imbalance, missing values, feature distributions, and correlations with the target variable
2. **Preprocessing** — dropped columns with >50% missing data, fixed anomalies (e.g. `DAYS_EMPLOYED` placeholder values), imputed missing values, one-hot encoded categorical features
3. **Feature Engineering** — created `LOAN_TO_INCOME` and `ANNUITY_TO_INCOME` ratios, aggregated bureau and previous application history per applicant
4. **Modeling** — trained and compared 4 classifiers using stratified train-validation split
5. **Explainability** — used SHAP TreeExplainer to compute global feature importance and generate per-applicant explanations
6. **Deployment** — built a 4-tab Streamlit dashboard and deployed on Streamlit Community Cloud
