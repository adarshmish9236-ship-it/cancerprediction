"""
CANCERWISE AI — Key Platform Capabilities Component (Section 4)
Renders responsive 8-grid showcase with perfectly aligned interactive workspace navigation.
"""

from __future__ import annotations

import streamlit as st


def render_capabilities(is_dark: bool = True) -> None:
    """Renders Section 4: Key Platform Capabilities."""
    st.markdown('<div id="capabilities"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="lp-section">
            <div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
                <div class="lp-eyebrow">✦ CAPABILITY ECOSYSTEM</div>
                <h2 class="lp-heading">An Integrated Ecosystem for Cancer Data Analysis</h2>
                <p class="lp-subheading lp-subheading-center">
                    Explore eight interconnected modules engineered specifically for end-to-end oncology data science, 
                    from raw cytology ingestion to explainable clinical decision-support intelligence.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    caps = [
        {
            "icon": "📊",
            "title": "Executive Dashboard",
            "badge": "Core Telemetry",
            "desc": "Real-time research KPIs, verified cohort statistics, cross-validated model leaderboards, and interactive risk stratifiers.",
            "tags": ["KPIs", "Leaderboard", "Radar Profiles", "Cohort Telemetry"],
            "route": "pages/01_Overview_Dashboard.py",
            "btn_label": "Open Dashboard →",
        },
        {
            "icon": "📁",
            "title": "Dataset Explorer",
            "badge": "Active Service",
            "desc": "Full tabular dataset exploration, 30 continuous feature schemas, cytological data dictionaries, and class distributions.",
            "tags": ["UCI WDBC", "Parquet I/O", "Schema Audit", "Target Distribution"],
            "route": "pages/02_Dataset_Explorer.py",
            "btn_label": "Open Dataset Studio →",
        },
        {
            "icon": "🧹",
            "title": "Data Wrangling Studio",
            "badge": "Data Hygiene",
            "desc": "Automated missing-value imputation (KNN / Iterative), duplicate removal, IQR outlier winsorization, and pipeline export.",
            "tags": ["KNN Imputer", "Duplicate Audit", "IQR Outliers", "Scaling"],
            "route": "pages/03_Data_Wrangling_Studio.py",
            "btn_label": "Open Wrangling Studio →",
        },
        {
            "icon": "📈",
            "title": "Exploratory Analysis (EDA)",
            "badge": "Interactive",
            "desc": "Descriptive statistics, skewness/kurtosis, bivariate Pearson correlations, and interactive cytological distributions.",
            "tags": ["Correlations", "Pairwise Heatmaps", "Box Plots", "KDE Density"],
            "route": "pages/04_EDA_Studio.py",
            "btn_label": "Open EDA Studio →",
        },
        {
            "icon": "⚙️",
            "title": "Feature Engineering",
            "badge": "Feature Suite",
            "desc": "PCA dimensionality reduction, mutual information ranking, recursive feature elimination (RFE), and LASSO L1 selection.",
            "tags": ["PCA", "RFE", "LASSO (L1)", "Mutual Info"],
            "route": "pages/05_Feature_Engineering.py",
            "btn_label": "Open Feature Studio →",
        },
        {
            "icon": "🤖",
            "title": "Machine Learning Lab",
            "badge": "16 Algorithms",
            "desc": "Systematic training of Random Forest, XGBoost, LightGBM, SVM, and deep PyTorch MLPs with stratified 5-fold cross-validation.",
            "tags": ["Stratified K-Fold", "PyTorch MLP", "Ensemble Models", "Grid Search"],
            "route": "pages/06_Supervised_Learning_Lab.py",
            "btn_label": "Open ML Laboratory →",
        },
        {
            "icon": "🩺",
            "title": "Cancer Prediction",
            "badge": "Decision Support",
            "desc": "Calibrated inference with Platt scaling, 95% empirical bootstrap confidence intervals, and single-patient cytological risk scoring.",
            "tags": ["Calibrated Probabilities", "95% Bootstrap CI", "Biopsy Risk", "Inference"],
            "route": "pages/11_Cancer_Prediction_Workspace.py",
            "btn_label": "Open Prediction Workspace →",
        },
        {
            "icon": "🔬",
            "title": "Explainable AI (XAI)",
            "badge": "Clinical Audit",
            "desc": "Local and global interpretability using SHAP Tree/Kernel summary beeswarms, waterfall plots, and LIME local approximations.",
            "tags": ["SHAP Beeswarm", "Waterfall Plots", "LIME Explainer", "Governance"],
            "route": "pages/10_Explainable_AI.py",
            "btn_label": "Open Explainability Module →",
        },
    ]

    # Render in 2 rows of 4 columns with balanced heights and integrated action buttons
    for row_idx in range(2):
        cols = st.columns(4, gap="medium")
        for col_idx in range(4):
            item_idx = row_idx * 4 + col_idx
            cap = caps[item_idx]
            with cols[col_idx]:
                tags_html = "".join([f'<span class="lp-cap-tag">{t}</span>' for t in cap["tags"]])
                st.markdown(
                    f"""<div class="lp-cap-card">
    <div>
        <div class="lp-cap-icon-row">
            <div class="lp-cap-icon">{cap["icon"]}</div>
            <span class="lp-cap-badge">{cap["badge"]}</span>
        </div>
        <div class="lp-cap-title">{cap["title"]}</div>
        <div class="lp-cap-desc">{cap["desc"]}</div>
        <div class="lp-cap-tags">{tags_html}</div>
    </div>
</div>""",
                    unsafe_allow_html=True,
                )
                if st.button(cap["btn_label"], key=f"cap_btn_{item_idx}", use_container_width=True):
                    st.switch_page(cap["route"])

        st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
