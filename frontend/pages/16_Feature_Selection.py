"""
CANCERWISE AI — Page 7: Feature Selection & Consensus Ranking Studio
Filter-based selection (Mutual Information, ANOVA), wrapper RFE, embedded tree importances,
and Borda count consensus ranking.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier

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
    st.set_page_config(page_title="Feature Selection | CANCERWISE AI", page_icon="🎯", layout="wide")
except Exception:
    pass

render_sidebar("Feature Selection")
api = APIClient()

render_header(
    title="Feature Selection & Consensus Ranking Studio",
    subtitle="Evaluate multi-method feature ranking (Filter, Embedded, Borda consensus) to isolate optimal diagnostic biomarkers.",
    breadcrumbs=["CANCERWISE AI", "Data Analytics", "Feature Selection"],
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
    "🏆 Consensus Borda Ranking",
    "📊 Filter Methods (Mutual Info & ANOVA)",
    "🌲 Embedded Tree Importance",
])

with tab1:
    st.markdown("##### Multi-Paradigm Consensus Ranking (Borda Count Aggregation)")
    st.write("Aggregates rankings across Mutual Information, ANOVA F-Score, and Random Forest Gini impurity to avoid single-method bias.")

    with st.spinner("Computing multi-method consensus rankings..."):
        mi_scores = mutual_info_classif(X, y, random_state=42)
        f_scores, _ = f_classif(X, y)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        rf_importances = rf.feature_importances_

        # Rank each (1 = best)
        rank_mi = pd.Series(mi_scores, index=num_cols).rank(ascending=False)
        rank_f = pd.Series(f_scores, index=num_cols).rank(ascending=False)
        rank_rf = pd.Series(rf_importances, index=num_cols).rank(ascending=False)
        borda_points = (len(num_cols) - rank_mi) + (len(num_cols) - rank_f) + (len(num_cols) - rank_rf)

        consensus_df = pd.DataFrame({
            "Biomarker Feature": num_cols,
            "Mutual Information": np.round(mi_scores, 4),
            "ANOVA F-Score": np.round(f_scores, 1),
            "Random Forest Gini": np.round(rf_importances, 4),
            "Borda Consensus Score": borda_points.values,
        }).sort_values(by="Borda Consensus Score", ascending=False).reset_index(drop=True)
        consensus_df["Consensus Rank"] = range(1, len(consensus_df) + 1)

    c1, c2 = st.columns([1.6, 1.2])
    with c1:
        fig_borda = px.bar(
            consensus_df.head(10),
            x="Borda Consensus Score",
            y="Biomarker Feature",
            orientation="h",
            color="Borda Consensus Score",
            color_continuous_scale="Teal",
            title="Top 10 Biomarkers by Borda Consensus Score",
        )
        fig_borda.update_layout(yaxis={"autorange": "reversed"}, height=360)
        ChartContainer.render(fig_borda)

    with c2:
        st.markdown("##### Top 5 Diagnostic Biomarkers")
        for i, row in consensus_df.head(5).iterrows():
            st.markdown(f"**#{row['Consensus Rank']} `{row['Biomarker Feature']}`** — Borda: {int(row['Borda Consensus Score'])} pts")

with tab2:
    st.markdown("##### Filter Ranking: Mutual Information vs ANOVA F-Statistic")
    fig_scatter_filters = px.scatter(
        consensus_df,
        x="Mutual Information",
        y="ANOVA F-Score",
        text="Biomarker Feature",
        color="Borda Consensus Score",
        color_continuous_scale="Viridis",
        title="Information Theoretic vs Parametric Linear Correlation",
    )
    fig_scatter_filters.update_traces(textposition='top right')
    fig_scatter_filters.update_layout(height=420)
    ChartContainer.render(fig_scatter_filters)

with tab3:
    st.markdown("##### Full Multi-Method Ranking Table")
    st.dataframe(consensus_df, use_container_width=True, hide_index=True)
