import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
import os

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Credit Risk Scorer", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgb_model.pkl'))
    return model

@st.cache_data
def load_data():
    X_val = pd.read_csv(os.path.join(BASE_DIR, 'outputs', 'X_val_sample.csv'))
    y_val = pd.read_csv(os.path.join(BASE_DIR, 'outputs', 'y_val_sample.csv')).squeeze()
    X_val.columns = X_val.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    return X_val, y_val

model        = load_model()
X_val, y_val = load_data()
explainer    = shap.TreeExplainer(model)


tab1, tab2, tab3, tab4 = st.tabs([
    "Predict", "EDA", "Model Performance", "Correlation"
])



# TAB 1 — PREDICT

with tab1:
    st.title("Credit Risk Prediction")
    st.write("Fill in the applicant details and click Predict.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Applicant Details")

        age            = st.slider("Age", 18, 70, 35)
        income         = st.number_input("Annual Income", value=150000, step=10000)
        credit         = st.number_input("Loan Amount", value=500000, step=10000)
        annuity        = st.number_input("Monthly Annuity", value=25000, step=1000)
        years_employed = st.slider("Years Employed", 0, 40, 5)
        goods_price    = st.number_input("Goods Price", value=450000, step=10000)
        ext_source_2   = st.slider("External Score 2 (0 to 1)", 0.0, 1.0, 0.5)
        ext_source_3   = st.slider("External Score 3 (0 to 1)", 0.0, 1.0, 0.5)

        predict_btn = st.button("Run Prediction")

    with col2:
        if predict_btn:

            if income == 0:
                st.error("Annual Income cannot be zero. Please enter a valid income.")

            else:
                loan_to_income    = credit / income
                annuity_to_income = annuity / income

                sample_row = X_val.iloc[[0]].copy()

                if 'AGE' in sample_row.columns:
                    sample_row['AGE'] = age
                if 'AMT_INCOME_TOTAL' in sample_row.columns:
                    sample_row['AMT_INCOME_TOTAL'] = income
                if 'AMT_CREDIT' in sample_row.columns:
                    sample_row['AMT_CREDIT'] = credit
                if 'AMT_ANNUITY' in sample_row.columns:
                    sample_row['AMT_ANNUITY'] = annuity
                if 'YEARS_EMPLOYED' in sample_row.columns:
                    sample_row['YEARS_EMPLOYED'] = years_employed
                if 'AMT_GOODS_PRICE' in sample_row.columns:
                    sample_row['AMT_GOODS_PRICE'] = goods_price
                if 'EXT_SOURCE_2' in sample_row.columns:
                    sample_row['EXT_SOURCE_2'] = ext_source_2
                if 'EXT_SOURCE_3' in sample_row.columns:
                    sample_row['EXT_SOURCE_3'] = ext_source_3
                if 'LOAN_TO_INCOME' in sample_row.columns:
                    sample_row['LOAN_TO_INCOME'] = loan_to_income
                if 'ANNUITY_TO_INCOME' in sample_row.columns:
                    sample_row['ANNUITY_TO_INCOME'] = annuity_to_income

                prob     = model.predict_proba(sample_row)[0][1]
                prob_pct = round(prob * 100, 1)

                st.subheader("Prediction Result")

                if prob > 0.5:
                    st.error(f"Default Probability: {prob_pct}%")
                    st.error("Verdict: HIGH RISK — Loan not recommended")
                else:
                    st.success(f"Default Probability: {prob_pct}%")
                    st.success("Verdict: LOW RISK — Loan can be considered")

                st.progress(int(prob_pct))
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Loan-to-Income Ratio", round(loan_to_income, 2))
                with m2:
                    st.metric("Annuity-to-Income Ratio", round(annuity_to_income, 2))
                    
                st.subheader("Why this prediction? (SHAP)")

                shap_vals = explainer.shap_values(sample_row)[0]

                shap_df = pd.DataFrame({
                    'Feature':    X_val.columns.tolist(),
                    'SHAP Value': shap_vals
                })

                shap_df['Abs'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values('Abs', ascending=False)
                shap_df = shap_df.head(12)

                colors = []
                for v in shap_df['SHAP Value']:
                    if v > 0:
                        colors.append('#e74c3c')
                    else:
                        colors.append('#2ecc71')

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(shap_df['Feature'][::-1], shap_df['SHAP Value'][::-1], color=colors[::-1])
                ax.axvline(x=0, color='black', linewidth=0.8)
                ax.set_title('Top Factors Influencing This Prediction')
                ax.set_xlabel('SHAP Value (Red = increases risk, Green = reduces risk)')
                plt.tight_layout()
                st.pyplot(fig)



# TAB 2 — EDA

with tab2:
    st.title("Exploratory Data Analysis")

    st.subheader("Class Imbalance")
    st.image(os.path.join(BASE_DIR, 'outputs', 'class_imbalance.png'))

    st.subheader("Age Distribution")
    st.image(os.path.join(BASE_DIR, 'outputs', 'age_distribution.png'))

    st.subheader("Income Distribution")
    st.image(os.path.join(BASE_DIR, 'outputs', 'income_distribution.png'))

    st.subheader("Default Rate by Employment Type")
    st.image(os.path.join(BASE_DIR, 'outputs', 'default_by_employment.png'))

    st.subheader("Default Rate by Education Level")
    st.image(os.path.join(BASE_DIR, 'outputs', 'default_by_education.png'))



# TAB 3 — MODEL PERFORMANCE

with tab3:
    st.title("Model Performance Comparison")

    results = {
    'Model':     ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
    'AUC-ROC':   [0.6119, 0.7153, 0.7247, 0.7458],
    'F1 Score':  [0.1876, 0.2418, 0.0032, 0.2738],
    'Precision': [0.1148, 0.1489, 0.8000, 0.1772],
    'Recall':    [0.5132, 0.6425, 0.0016, 0.6020]
    }

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Model Comparison Chart")
    st.image(os.path.join(BASE_DIR, 'outputs', 'model_comparison.png'))

    st.subheader("ROC Curves")
    st.image(os.path.join(BASE_DIR, 'outputs', 'roc_curves.png'))

    st.subheader("Confusion Matrix — XGBoost")
    st.image(os.path.join(BASE_DIR, 'outputs', 'confusion_matrix.png'))



# TAB 4 — CORRELATION

with tab4:
    st.title("Feature Correlation with Default")
    st.image(os.path.join(BASE_DIR, 'outputs', 'top_correlations.png'))
   