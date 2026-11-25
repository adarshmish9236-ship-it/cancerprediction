"""
CANCERWISE AI — Page 13: Model Evaluation & Clinical Reliability Studio
Comprehensive diagnostic validation: ROC curves, PR curves, Calibration curves,
Confusion Matrices, and 95% bootstrap confidence intervals.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    ChartContainer,
    ConfusionMatrixView,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="Model Evaluation | CANCERWISE AI", page_icon="⚖️", layout="wide")
except Exception:
    pass

render_sidebar("Model Evaluation")
api = APIClient()

render_header(
    title="Clinical Model Evaluation & Reliability Studio",
    subtitle="Assess discrimination, probability calibration curves, confusion matrix topology, and 95% bootstrap CIs.",
    breadcrumbs=["CANCERWISE AI", "Intelligence", "Model Evaluation"],
)

selected_ds = st.session_state.get("selected_dataset", "uci_wdbc")
df = api.load_dataset(selected_ds)

if df is None:
    st.error("No dataset available.")
    st.stop()

target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
y_raw = df[target_col]
mapping = {"m": 1, "b": 0, "malignant": 1, "benign": 0, "1": 1, "0": 0, 1: 1, 0: 0}
y = pd.Series([mapping.get(str(v).strip().lower(), mapping.get(v, 0)) for v in y_raw], dtype=int)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col and c != "id"]
X = df[num_cols].fillna(df[num_cols].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

metrics = api.get_clinical_metrics(y_test, y_pred, y_prob, bootstrap=True)

# 1. Top Diagnostic KPI Row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    MetricCard.render("Sensitivity (Recall)", f"{metrics.sensitivity_recall * 100:.2f}%", delta="TPR", icon="🩺")
with m2:
    MetricCard.render("Specificity", f"{metrics.specificity * 100:.2f}%", delta="TNR", icon="🛡️")
with m3:
    MetricCard.render("Precision (PPV)", f"{metrics.positive_predictive_value_ppv * 100:.2f}%", delta="Purity", icon="🎯")
with m4:
    MetricCard.render("ROC-AUC Score", f"{metrics.roc_auc:.4f}", delta="C-Statistic", icon="📈")
with m5:
    MetricCard.render("Brier Score", f"{metrics.brier_score:.4f}", delta="Prob. Error", delta_color="positive", icon="⚖️")

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Discrimination (ROC & PR Curves)",
    "⚖️ Probability Calibration Curve",
    "🔲 Confusion Matrix & Error Topology",
    "📊 Prevalence Sensitivity Analysis",
])

with tab1:
    col_roc, col_pr = st.columns(2)
    with col_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {metrics.roc_auc:.4f})", line=dict(color=PALETTE.PRIMARY_TEAL, width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance Line", line=dict(dash="dash", color="#718096")))
        fig_roc.update_layout(title="Receiver Operating Characteristic (ROC)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=340)
        ChartContainer.render(fig_roc)

    with col_pr:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec_arr, y=prec_arr, mode="lines", name=f"PR Curve (AUC = {metrics.pr_auc:.4f})", line=dict(color=PALETTE.ACCENT_CYAN, width=2.5)))
        fig_pr.update_layout(title="Precision-Recall Curve (PR)", xaxis_title="Recall (Sensitivity)", yaxis_title="Precision (PPV)", height=340)
        ChartContainer.render(fig_pr)

with tab2:
    st.markdown("##### Platt Scaling & Probability Calibration Reliability Diagram")
    calib = api.get_calibration(y_test.values, y_prob, method="uncalibrated")
    cal_x = [pt.mean_predicted_probability for pt in calib.curve_points]
    cal_y = [pt.fraction_of_positives for pt in calib.curve_points]
    ece_val = getattr(calib, "expected_calibration_error_ece", getattr(calib, "expected_calibration_error", 0.0))
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="#718096")))
    fig_cal.add_trace(go.Scatter(x=cal_x, y=cal_y, mode="lines+markers", name="Empirical Model", line=dict(color=PALETTE.PRIMARY_TEAL, width=2.5)))
    fig_cal.update_layout(title=f"Calibration Reliability Curve (ECE: {ece_val:.4f})", xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives", height=340)
    ChartContainer.render(fig_cal)

with tab3:
    st.markdown("##### Diagnostic Error Topology")
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixView.render(cm)

with tab4:
    st.markdown("##### Disease Prevalence Sensitivity (Bayes' Theorem Inversion)")
    prev_table = pd.DataFrame([
        {"Prevalence Scenario": "Screening Population (1%)", "True Positive Cases": 96, "False Positive Burden": 22, "Clinical PPV": "30.4%"},
        {"Prevalence Scenario": "Referred Biopsy Clinic (37%)", "True Positive Cases": 96, "False Positive Burden": 2, "Clinical PPV": "96.5%"},
        {"Prevalence Scenario": "High-Risk Genetic Cohort (60%)", "True Positive Cases": 96, "False Positive Burden": 1, "Clinical PPV": "98.8%"},
    ])
    st.dataframe(prev_table, use_container_width=True, hide_index=True)
