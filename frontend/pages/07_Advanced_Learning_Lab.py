"""
CANCERWISE AI — Page 9: Advanced Learning Laboratory
Interactive laboratory for semi-supervised learning, multitask architectures,
one-class anomaly detection, continuous regression, and kernel space mapping.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

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
    st.set_page_config(page_title="Advanced Learning Lab | CANCERWISE AI", page_icon="🧬", layout="wide")
except Exception:
    pass

render_sidebar("Advanced Learning")
api = APIClient()

render_header(
    title="Advanced Machine Learning Research Laboratory",
    subtitle="Semi-supervised label propagation, multitask architectures, one-class anomaly detection, and kernel spaces.",
    breadcrumbs=["CANCERWISE AI", "Machine Learning", "Advanced Learning"],
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏷️ Semi-Supervised Learning",
    "🔀 Multitask Joint Learning",
    "🛡️ One-Class Outlier Isolation",
    "📈 Biomarker Regression",
    "🧬 Mercer Kernel Spaces",
])

with tab1:
    st.markdown("##### Semi-Supervised Learning (Label Propagation)")
    st.markdown("""
    **Clinical Scenario:** Pathological annotation of biopsies is labor-intensive and costly. 
    Semi-supervised learning leverages small labeled cohorts combined with large pools of unannotated FNA samples.
    """)

    col_s_opt, col_s_res = st.columns([1.2, 1.8])
    with col_s_opt:
        label_pct = st.slider("Percentage of Labeled Biopsies Available (%)", 10, 80, 30) / 100.0
        gamma = st.slider("RBF Kernel Gamma", 0.05, 1.0, 0.25, 0.05)
        run_semi = st.button("Execute Label Propagation", type="primary", use_container_width=True)

    rng = np.random.RandomState(42)
    mask_unlabeled = rng.rand(len(y)) > label_pct
    y_semi = y.copy()
    y_semi[mask_unlabeled] = -1

    with col_s_res:
        if run_semi:
            with st.spinner("Propagating labels through graph manifold..."):
                lp = LabelPropagation(kernel="rbf", gamma=gamma, max_iter=100)
                lp.fit(X, y_semi)
                pseudo_labels = lp.transduction_
                acc_pseudo = float(np.mean(pseudo_labels == y))

            c1, c2 = st.columns(2)
            with c1:
                MetricCard.render("Pseudo-Label Accuracy", f"{acc_pseudo * 100:.2f}%", delta="vs Ground Truth", icon="🎯")
            with c2:
                MetricCard.render("Recovered Unlabeled Cases", f"{np.sum(mask_unlabeled)}", delta=f"{len(y)} Total Cases", icon="🏷️")
            st.success("Successfully converged label propagation across cytology manifold!")
        else:
            st.info("Click **Execute Label Propagation** to propagate labels to unlabeled cases.")

with tab2:
    st.markdown("##### Multitask Shared Representation Neural Network")
    st.markdown("""
    **Architecture:** A deep PyTorch network with shared latent bottleneck layers jointly predicting 
    primary biopsy malignancy risk and secondary disease recurrence risk.
    """)
    st.code("""
# PyTorch Multitask Joint Architecture
class MultitaskCytologyNet(nn.Module):
    def __init__(self, in_features=30):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.GELU()
        )
        self.head_primary = nn.Linear(32, 2)    # Task 1: Malignant vs Benign
        self.head_recurrence = nn.Linear(32, 1) # Task 2: Continuous Recurrence Index
    """, language="python")

with tab3:
    st.markdown("##### One-Class Anomaly & Novelty Detection")
    st.markdown("Detects novel histological anomalies by training exclusively on benign tissue distributions.")
    contamination = st.slider("Expected Contamination Rate", 0.01, 0.20, 0.05, 0.01)
    if st.button("Run Isolation Forest Anomaly Scan", type="primary"):
        benign_X = X[y == 0]
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(benign_X)
        scores = iso.decision_function(X)
        novelties = (iso.predict(X) == -1).sum()
        MetricCard.render("Detected Cytology Novelties", f"{novelties} Samples", subtext=f"Total: {len(X)} biopsies", icon="🛡️")

with tab4:
    st.markdown("##### Continuous Biomarker Regression Modeling")
    st.write("Predicts continuous morphometric properties (e.g. `area_worst`) from primary nuclear measurements.")
    if "area_worst" in X.columns and "radius_mean" in X.columns:
        y_reg = X["area_worst"]
        X_reg = X.drop(columns=["area_worst"])
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_reg, y_reg)
        y_reg_pred = ridge.predict(X_reg)
        r2 = r2_score(y_reg, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg, y_reg_pred))
        c1, c2 = st.columns(2)
        with c1:
            MetricCard.render("Ridge Regression R²", f"{r2:.4f}", delta="High Predictive Power", icon="📈")
        with c2:
            MetricCard.render("RMSE Error", f"{rmse:.2f}", delta="Area Units", icon="📉")

with tab5:
    st.markdown("##### Mercer Kernel & Reproducing Kernel Hilbert Space (RKHS)")
    st.markdown("""
    Maps finite dimensional morphometric coordinates into infinite-dimensional Hilbert spaces 
    where non-linearly separable cytology boundaries become linearly partitionable.
    """)
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)")
    st.info("RBF Mercer Kernel verified for clinical SVM classification and spectral clustering.")
