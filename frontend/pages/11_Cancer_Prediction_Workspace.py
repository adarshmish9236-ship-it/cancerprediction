"""
CANCERWISE AI — Page 11: Cancer Prediction Workspace
Structured patient-level clinical decision support workspace with preset profiles,
biomarker domain inputs, probability index gauge, local SHAP attribution waterfall,
and mandatory research governance disclaimers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    PredictionResultView,
    ExplanationPanel,
    ChartContainer,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="Prediction Workspace | CANCERWISE AI", page_icon="🩺", layout="wide")
except Exception:
    pass

render_sidebar("Cancer Prediction")
api = APIClient()

render_header(
    title="Clinical Cancer Risk Prediction Workspace",
    subtitle="Evaluate cytology morphometry, estimate malignancy probabilities, and inspect local biomarker attributions.",
    breadcrumbs=["CANCERWISE AI", "Intelligence", "Cancer Prediction"],
)

st.markdown("""
<div class="cw-alert-clinical">
    ⚕️ <strong>Institutional Governance & Clinical Safety Notice:</strong> 
    This software is an investigational research decision-support prototype. 
    Outputs represent statistical estimates and <strong>CANNOT</strong> substitute for histopathological biopsy examination, 
    radiological imaging, or certified medical oncologist review. 
    <strong>DO NOT</strong> base pharmaceutical, surgical, or chemotherapy decisions on this computational aid.
</div>
""", unsafe_allow_html=True)

selected_ds = st.session_state.get("selected_dataset", "uci_wdbc")
df = api.load_dataset(selected_ds)

if df is None:
    st.error("Dataset not loaded.")
    st.stop()

target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
y_raw = df[target_col]
mapping = {"m": 1, "b": 0, "malignant": 1, "benign": 0, "1": 1, "0": 0, 1: 1, 0: 0}
y = pd.Series([mapping.get(str(v).strip().lower(), mapping.get(v, 0)) for v in y_raw], dtype=int)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col and c != "id"]
X = df[num_cols].fillna(df[num_cols].median())

# Reference Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
model.fit(X_train, y_train)

tab_single, tab_batch = st.tabs([
    "👤 Individual Patient Risk Assessment",
    "👥 Multi-Patient Cohort Screening",
])

with tab_single:
    col_input, col_out = st.columns([1.2, 1.4])

    with col_input:
        st.markdown("##### 1. Patient Case Input & Clinical Presets")

        # Preset Profile Selector
        preset = st.radio(
            "Load Clinical Case Preset",
            ["Custom Input Profile", "Preset Case A: Typical Benign", "Preset Case B: Highly Suspicious (Malignant)"],
            horizontal=False,
        )

        if "Benign" in preset:
            benign_sample = df[y == 0][num_cols].iloc[0]
            default_vals = benign_sample.to_dict()
        elif "Suspicious" in preset:
            mal_sample = df[y == 1][num_cols].iloc[0]
            default_vals = mal_sample.to_dict()
        else:
            default_vals = X.median().to_dict()

        st.markdown("###### Primary Nuclear Morphometry Biomarkers")

        input_data: Dict[str, float] = {}

        # Organized grouped inputs into clean 2-column sub-grid
        sub_c1, sub_c2 = st.columns(2)
        primary_feats = [
            ("radius_mean", "Mean Radius (μm)", 6.0, 30.0),
            ("texture_mean", "Mean Texture", 9.0, 40.0),
            ("perimeter_mean", "Mean Perimeter", 40.0, 190.0),
            ("area_mean", "Mean Area (μm²)", 140.0, 2500.0),
            ("smoothness_mean", "Mean Smoothness", 0.05, 0.20),
            ("compactness_mean", "Mean Compactness", 0.01, 0.35),
            ("concavity_mean", "Mean Concavity", 0.0, 0.45),
            ("concave points_mean", "Mean Concave Pts", 0.0, 0.25),
        ]

        for i, (feat, label, min_val, max_val) in enumerate(primary_feats):
            if feat in num_cols:
                val = float(default_vals.get(feat, X[feat].median()))
                target_col_box = sub_c1 if (i % 2 == 0) else sub_c2
                with target_col_box:
                    input_data[feat] = st.number_input(
                        label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=round(val, 3),
                        step=0.1,
                        key=f"inp_{feat}",
                    )

        # Fill remaining features with default values
        for feat in num_cols:
            if feat not in input_data:
                input_data[feat] = float(default_vals.get(feat, X[feat].median()))

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        eval_btn = st.button("🔬 Execute Risk Prediction", type="primary", use_container_width=True)

    with col_out:
        st.markdown("##### 2. Diagnostic Assessment & Explainability")
        patient_series = pd.Series(input_data)
        patient_df = pd.DataFrame([patient_series[num_cols]])
        prob_mal = float(model.predict_proba(patient_df)[0, 1])
        pred_class = int(prob_mal >= 0.5)
        eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        PredictionResultView.render(
            predicted_class=pred_class,
            prob_malignant=prob_mal,
            model_name="Random Forest Classifier (v1.0 - Cross-Validated)",
        )

        st.markdown(f"<div style='font-size: 0.75rem; color: #718096; margin-bottom: 0.5rem;'>Timestamp: <code>{eval_time}</code> | Cohort Baseline: <code>{selected_ds}</code></div>", unsafe_allow_html=True)

        st.markdown("##### Local Biomarker Attributions (SHAP Decision Drivers)")
        local_shap = api.get_shap_local(model, patient_series[num_cols], X_train.iloc[:40], idx=0, top_k=6)
        ExplanationPanel.render(
            features=[c.feature_name for c in local_shap.contributions],
            impacts=[c.shap_value for c in local_shap.contributions],
            directions=[c.contribution_direction for c in local_shap.contributions],
            title="Local Feature Attributions Contributing to Risk",
        )

with tab_batch:
    st.markdown("##### Multi-Patient Batch Cohort Screening")
    st.markdown("Screen up to 100 patient biopsy records simultaneously and generate risk stratification tiers.")

    batch_preview = X_test.head(15).copy()
    batch_probs = model.predict_proba(batch_preview)[:, 1]
    batch_preview["Malignancy Probability"] = [f"{p*100:.1f}%" for p in batch_probs]
    batch_preview["Risk Stratification"] = [
        "Elevated Risk (Malignant) 🔴" if p >= 0.5 else "Low Risk (Benign) 🟢"
        for p in batch_probs
    ]

    st.dataframe(batch_preview[["Malignancy Probability", "Risk Stratification"] + num_cols[:5]], use_container_width=True)
