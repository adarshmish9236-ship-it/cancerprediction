"""
CANCERWISE AI — Page 6: Feature Engineering Studio
Biomarker interaction synthesis, polynomial expansions, non-linear transforms, and PCA.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    ChartContainer,
    DataPreview,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="Feature Engineering | CANCERWISE AI", page_icon="⚙️", layout="wide")
except Exception:
    pass

render_sidebar("Feature Engineering")
api = APIClient()

render_header(
    title="Feature Engineering & Biomarker Synthesis Studio",
    subtitle="Extract domain-specific cytology ratios, interaction polynomials, and reduce dimensionality.",
    breadcrumbs=["CANCERWISE AI", "Data Analytics", "Feature Engineering"],
)

selected_ds = st.session_state.get("selected_dataset", "uci_wdbc")
df = api.load_dataset(selected_ds)

if df is None:
    st.error("No dataset loaded.")
    st.stop()

target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
y_raw = df[target_col]
mapping = {"m": 1, "b": 0, "malignant": 1, "benign": 0, "1": 1, "0": 0, 1: 1, 0: 0}
y = pd.Series([mapping.get(str(v).strip().lower(), mapping.get(v, 0)) for v in y_raw], dtype=int)
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col and c != "id"]
X = df[num_cols].fillna(df[num_cols].median())

tab1, tab2, tab3 = st.tabs([
    "🧬 Biomarker Transformations & Morphometric Ratios",
    "📐 Polynomial & Non-Linear Expansions",
    "🌌 PCA & Manifold Projections",
])

with tab1:
    st.markdown("##### Domain-Informed Morphometric Ratios")
    st.markdown("""
    Synthesizes cytopathological ratios (e.g. *Area-to-Perimeter*, *Compactness-to-Radius*) 
    clinically correlated with invasive cell boundary irregularities.
    """)

    c_btn, c_info = st.columns([1, 2])
    with c_btn:
        gen_btn = st.button("Synthesize Cytology Ratios", type="primary", use_container_width=True)

    if gen_btn or "X_derived" in st.session_state:
        X_derived = X.copy()
        if "area_mean" in X.columns and "perimeter_mean" in X.columns:
            X_derived["ratio_area_perimeter"] = X_derived["area_mean"] / (X_derived["perimeter_mean"] + 1e-5)
        if "compactness_mean" in X_derived.columns and "concavity_mean" in X_derived.columns:
            X_derived["interaction_compact_concave"] = X_derived["compactness_mean"] * X_derived["concavity_mean"]
        if "radius_worst" in X_derived.columns and "radius_mean" in X_derived.columns:
            X_derived["ratio_worst_to_mean_radius"] = X_derived["radius_worst"] / (X_derived["radius_mean"] + 1e-5)

        st.session_state["X_derived"] = X_derived
        st.success(f"Successfully generated 3 clinical interaction biomarkers! Feature space: {X.shape[1]} → {X_derived.shape[1]} features.")

        st.markdown("##### Preview Derived Cytology Biomarkers")
        derived_cols = ["ratio_area_perimeter", "interaction_compact_concave", "ratio_worst_to_mean_radius"]
        st.dataframe(X_derived[derived_cols].head(10), use_container_width=True)

with tab2:
    st.markdown("##### Non-Linear & Polynomial Transformations")
    poly_col = st.selectbox("Select Feature for Non-Linear Expansion", num_cols[:8])
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_orig = px.histogram(X, x=poly_col, title=f"Original Distribution: `{poly_col}`", color_discrete_sequence=[PALETTE.SECONDARY_NAVY])
        ChartContainer.render(fig_orig, height=280)
    with col_t2:
        log_transformed = np.log1p(np.maximum(X[poly_col], 0))
        fig_log = px.histogram(log_transformed, x=poly_col, title=f"Log1p Transformed: `{poly_col}`", color_discrete_sequence=[PALETTE.PRIMARY_TEAL])
        ChartContainer.render(fig_log, height=280)

with tab3:
    st.markdown("##### Principal Component Analysis (PCA)")
    n_components = st.slider("PCA Target Dimensions", 2, 5, 2)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_
    st.markdown(f"**Cumulative Explained Variance:** `{np.sum(var_explained) * 100:.2f}%` (PC1: `{var_explained[0]*100:.1f}%`, PC2: `{var_explained[1]*100:.1f}%`)")

    pca_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
    pca_df["Diagnosis"] = y.map({1: "Malignant", 0: "Benign"})

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Diagnosis",
        color_discrete_map={"Malignant": PALETTE.ERROR, "Benign": PALETTE.PRIMARY_TEAL},
        title="2D Principal Component Manifold Projection",
    )
    fig_pca.update_layout(height=400)
    ChartContainer.render(fig_pca)
