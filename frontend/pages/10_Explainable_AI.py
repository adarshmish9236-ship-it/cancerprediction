"""
CANCERWISE AI — Page 12: Explainable AI & Interpretability Studio
SHAP global attributions, local waterfall explanations, LIME surrogate rules,
and clinical risk interpretation.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    ChartContainer,
    ExplanationPanel,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="Explainable AI | CANCERWISE AI", page_icon="🔬", layout="wide")
except Exception:
    pass

render_sidebar("Explainable AI")
api = APIClient()

render_header(
    title="Explainable AI & Biomedical Interpretability Studio",
    subtitle="Deconstruct black-box diagnostic models using game-theoretic SHAP and local surrogate LIME.",
    breadcrumbs=["CANCERWISE AI", "Intelligence", "Explainable AI"],
)

st.markdown("""
<div class="cw-alert-info">
    ⚕️ <strong>Scientific Governance Notice:</strong> 
    SHAP attributions and LIME surrogate coefficients quantify additive mathematical contributions to model log-odds. 
    They reveal statistical patterns correlated with malignancy risk, <strong>NOT</strong> biological etiology or tumor grading.
</div>
""", unsafe_allow_html=True)

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
model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
model.fit(X_train, y_train)

tab1, tab2, tab3 = st.tabs([
    "📊 Global SHAP Attributions",
    "💧 Local Patient Waterfall (SHAP)",
    "🍋 LIME Surrogate Rules",
])

with tab1:
    st.markdown("##### Global Biomarker Attribution Ranking (Mean Absolute SHAP)")
    with st.spinner("Computing Shapley values across background distribution..."):
        shap_summary = api.get_shap_global(model, X_test, max_samples=40)

    shap_df = pd.DataFrame([
        {
            "Rank": item.rank,
            "Feature": item.feature_name,
            "Mean |SHAP|": item.mean_abs_shap,
            "Risk Direction": "Increases Malignancy Risk" if item.correlation_with_risk > 0 else "Decreases Malignancy Risk",
        }
        for item in shap_summary.global_importances[:12]
    ])

    col_s1, col_s2 = st.columns([1.6, 1.2])
    with col_s1:
        fig_shap = px.bar(
            shap_df,
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            color="Mean |SHAP|",
            color_continuous_scale="Teal",
            title="Top Global Feature Attributions",
        )
        fig_shap.update_layout(yaxis={"autorange": "reversed"}, height=360)
        ChartContainer.render(fig_shap)

    with col_s2:
        st.markdown("##### Interpretation & Clinical Context")
        st.markdown("""
        - **Primary Discriminator:** Top-ranking features (such as `perimeter_worst`, `radius_worst`, `concave points_worst`) exert the largest global influence on malignancy risk.
        - **Directionality:** Positive correlation with risk indicates higher values drive predictions toward malignant classifications.
        """)
        st.dataframe(shap_df[["Rank", "Feature", "Mean |SHAP|"]], use_container_width=True, hide_index=True)

with tab2:
    st.markdown("##### Local Patient SHAP Waterfall Explanation")
    case_idx = st.slider("Select Patient Test Case Index", 0, min(len(X_test) - 1, 30), 0)
    instance = X_test.iloc[case_idx]
    actual_label = "Malignant" if y_test.iloc[case_idx] == 1 else "Benign"
    pred_prob = model.predict_proba(instance.values.reshape(1, -1))[0, 1]

    c_meta1, c_meta2 = st.columns(2)
    with c_meta1:
        MetricCard.render("Actual Diagnosis", actual_label, delta=f"Test Case #{case_idx}", icon="🩺")
    with c_meta2:
        MetricCard.render("Model Estimated P(Malignant)", f"{pred_prob * 100:.1f}%", delta="Confidence", icon="📈")

    local_shap = api.get_shap_local(model, instance, X_train.iloc[:30], idx=case_idx, top_k=7)
    ExplanationPanel.render(
        features=[c.feature_name for c in local_shap.contributions],
        impacts=[c.shap_value for c in local_shap.contributions],
        directions=[c.contribution_direction for c in local_shap.contributions],
        title=f"SHAP Waterfall Attributions for Case #{case_idx}",
    )

with tab3:
    st.markdown("##### Local Interpretable Model-agnostic Explanations (LIME)")
    lime_case = st.slider("Select Case for LIME Surrogate Fit", 0, min(len(X_test) - 1, 20), 1)
    lime_instance = X_test.iloc[lime_case]

    with st.spinner("Fitting local sparse linear surrogate..."):
        lime_exp = api.get_lime_local(model, lime_instance, X_train.iloc[:50], idx=lime_case, k=6)

    fidelity_score = getattr(lime_exp, "explanation_fidelity_score", getattr(lime_exp, "surrogate_score", 0.0))
    st.markdown(f"**Predicted Label:** `{lime_exp.predicted_label}` (Surrogate Fit R²: `{fidelity_score:.3f}`)")
    lime_rows = [
        {"Rule Condition": getattr(r, "rule", getattr(r, "rule_description", str(r))), "Feature": getattr(r, "feature_name", ""), "Weight": r.weight, "Direction": r.direction}
        for r in getattr(lime_exp, "rules", getattr(lime_exp, "local_rules", []))
    ]
    st.dataframe(pd.DataFrame(lime_rows), use_container_width=True, hide_index=True)
