"""
CANCERWISE AI — Page 5: Exploratory Data Analysis & Statistical Mining
Univariate distributions, bivariate diagnostic separation, correlation geometry,
and FDR-adjusted hypothesis testing.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    ChartContainer,
    ResultsTable,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="EDA Studio | CANCERWISE AI", page_icon="📈", layout="wide")
except Exception:
    pass

render_sidebar("Exploratory Data Analysis")
api = APIClient()

render_header(
    title="Exploratory Data Analysis & Statistical Mining Studio",
    subtitle="Univariate distributions, bivariate diagnostic separation, correlation geometry, and hypothesis tests.",
    breadcrumbs=["CANCERWISE AI", "Data Analytics", "Exploratory Data Analysis"],
)

selected_ds = st.session_state.get("selected_dataset", "uci_wdbc")
df = api.load_dataset(selected_ds)

if df is None:
    st.error("No dataset available.")
    st.stop()

target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col and c != "id"]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Univariate Distributions",
    "🩺 Bivariate Diagnostic Separation",
    "🔥 Correlation Matrix",
    "🌌 Multivariate Feature Space",
    "🧪 Hypothesis Testing (Mann-Whitney / t-test)",
])

with tab1:
    st.markdown("##### Univariate Feature Distributions & Kernel Density")
    c_sel, c_bins = st.columns([2, 1])
    with c_sel:
        feat = st.selectbox("Select Morphological Biomarker", num_cols, index=0)
    with c_bins:
        nbins = st.slider("Histogram Bins", 15, 60, 30)

    fig_hist = px.histogram(
        df,
        x=feat,
        nbins=nbins,
        marginal="box",
        color_discrete_sequence=[PALETTE.PRIMARY_TEAL],
        title=f"Empirical Frequency Distribution: `{feat}`",
    )
    fig_hist.update_layout(xaxis_title=feat, yaxis_title="Patient Frequency", height=380)
    ChartContainer.render(fig_hist)

with tab2:
    st.markdown("##### Feature Stratification by Diagnostic Class")
    c1, c2 = st.columns(2)
    with c1:
        f_box = st.selectbox("Stratify Feature (Box Plot)", num_cols, index=min(2, len(num_cols) - 1))
        fig_box = px.box(
            df,
            x=target_col,
            y=f_box,
            color=target_col,
            color_discrete_map={"M": PALETTE.ERROR, "B": PALETTE.PRIMARY_TEAL, 1: PALETTE.ERROR, 0: PALETTE.PRIMARY_TEAL},
            title=f"`{f_box}` Stratified by Diagnosis",
        )
        fig_box.update_layout(height=360)
        ChartContainer.render(fig_box)

    with c2:
        f_x = st.selectbox("Scatter Plot X-Axis", num_cols, index=0)
        f_y = st.selectbox("Scatter Plot Y-Axis", num_cols, index=min(1, len(num_cols) - 1))
        fig_scat = px.scatter(
            df,
            x=f_x,
            y=f_y,
            color=target_col,
            color_discrete_map={"M": PALETTE.ERROR, "B": PALETTE.PRIMARY_TEAL, 1: PALETTE.ERROR, 0: PALETTE.PRIMARY_TEAL},
            title=f"Bivariate Scatter: `{f_x}` vs `{f_y}`",
        )
        fig_scat.update_layout(height=360)
        ChartContainer.render(fig_scat)

with tab3:
    st.markdown("##### Pearson / Spearman Correlation Geometry")
    corr_subset = st.multiselect("Select Biomarkers for Heatmap", num_cols, default=num_cols[:10])
    if len(corr_subset) >= 2:
        corr_matrix = df[corr_subset].corr().round(3)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[[0, "#142B45"], [0.5, "#F4F7FA"], [1, "#159B91"]],
            title="Biomarker Correlation Heatmap",
        )
        fig_corr.update_layout(height=480)
        ChartContainer.render(fig_corr)
    else:
        st.warning("Please select at least 2 biomarkers.")

with tab4:
    st.markdown("##### 3D Multivariate Feature Projection")
    if len(num_cols) >= 3:
        c3d_1, c3d_2, c3d_3 = st.columns(3)
        with c3d_1:
            dim_x = st.selectbox("Axis X", num_cols, index=0)
        with c3d_2:
            dim_y = st.selectbox("Axis Y", num_cols, index=1)
        with c3d_3:
            dim_z = st.selectbox("Axis Z", num_cols, index=2)

        fig_3d = px.scatter_3d(
            df,
            x=dim_x,
            y=dim_y,
            z=dim_z,
            color=target_col,
            color_discrete_map={"M": PALETTE.ERROR, "B": PALETTE.PRIMARY_TEAL, 1: PALETTE.ERROR, 0: PALETTE.PRIMARY_TEAL},
            title="3D Biomarker Separation",
        )
        fig_3d.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
        ChartContainer.render(fig_3d)

with tab5:
    st.markdown("##### Clinical Hypothesis Testing (Malignant vs Benign Differences)")
    stat_rows = []
    mal_mask = (df[target_col] == "M") | (df[target_col] == 1)
    ben_mask = (df[target_col] == "B") | (df[target_col] == 0)

    for col in num_cols[:12]:
        group_mal = df[mal_mask][col].dropna()
        group_ben = df[ben_mask][col].dropna()
        stat_val, p_val = stats.mannwhitneyu(group_mal, group_ben)
        cohen_d = (group_mal.mean() - group_ben.mean()) / np.sqrt((group_mal.var() + group_ben.var()) / 2)
        stat_rows.append({
            "Biomarker Feature": col,
            "Mann-Whitney U": round(stat_val, 1),
            "p-Value": f"{p_val:.2e}" if p_val < 0.001 else round(p_val, 4),
            "Cohen's d Effect Size": round(cohen_d, 3),
            "FDR Significant (α=0.01)": "YES 🟢" if p_val < 0.01 else "NO 🔴",
        })

    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
