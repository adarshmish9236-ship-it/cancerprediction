"""
CANCERWISE AI — How It Works Component (Section 5)
Renders the 4-phase connected clinical research workflow timeline from raw cytology to explainable intelligence.
"""

from __future__ import annotations

import streamlit as st


def render_how_it_works(is_dark: bool = True) -> None:
    """Renders Section 5: How It Works."""
    st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="lp-section">
<div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
    <div class="lp-eyebrow">✦ SYSTEMATIC RESEARCH PIPELINE</div>
    <h2 class="lp-heading">From Raw Cytology to Explainable Intelligence</h2>
    <p class="lp-subheading lp-subheading-center">
        CANCERWISE AI implements a rigorous, reproducible 4-phase scientific methodology ensuring 
        high statistical fidelity, full auditability, and clinical governance at every step.
    </p>
</div>

<div class="lp-timeline-track">
    <div class="lp-timeline-node">
        <div class="lp-step-number">PHASE 01 <span style="font-size: 0.9rem; color: #21976B;">●</span></div>
        <div class="lp-step-title">Ingestion &amp; Data Hygiene</div>
        <div class="lp-step-desc">
            Register authentic cancer cohorts (UCI WDBC), validate schema types, detect and remove duplicate records, 
            impute missing values via KNN, and winsorize extreme cytological outliers.
        </div>
        <div style="margin-top: 0.75rem; font-size: 0.74rem; font-weight: 600; color: var(--lp-ai-cyan);">
            Deliverable: Clean Baseline Cohort
        </div>
    </div>

    <div class="lp-timeline-node">
        <div class="lp-step-number">PHASE 02 <span style="font-size: 0.9rem; color: #43BFC4;">●</span></div>
        <div class="lp-step-title">Cytology Mining &amp; Features</div>
        <div class="lp-step-desc">
            Explore 30 nuclear morphological dimensions, evaluate pairwise Pearson correlations, perform PCA 
            dimensionality reduction, and rank features via mutual information and Recursive Feature Elimination (RFE).
        </div>
        <div style="margin-top: 0.75rem; font-size: 0.74rem; font-weight: 600; color: var(--lp-ai-cyan);">
            Deliverable: Ranked Feature Spaces
        </div>
    </div>

    <div class="lp-timeline-node">
        <div class="lp-step-number">PHASE 03 <span style="font-size: 0.9rem; color: #159B91;">●</span></div>
        <div class="lp-step-title">Multi-Model Training &amp; CV</div>
        <div class="lp-step-desc">
            Train and benchmark 16 diverse machine learning architectures including Random Forest, SVM, LightGBM, 
            and PyTorch MLPs using stratified 5-fold cross-validation with hyperparameter grid search.
        </div>
        <div style="margin-top: 0.75rem; font-size: 0.74rem; font-weight: 600; color: var(--lp-ai-cyan);">
            Deliverable: Validated Model Registry
        </div>
    </div>

    <div class="lp-timeline-node">
        <div class="lp-step-number">PHASE 04 <span style="font-size: 0.9rem; color: #21976B;">●</span></div>
        <div class="lp-step-title">XAI &amp; Clinical Governance</div>
        <div class="lp-step-desc">
            Generate Platt-calibrated malignancy probabilities with 95% bootstrap confidence intervals, compute SHAP 
            and LIME feature attributions, and export audit-ready research documentation.
        </div>
        <div style="margin-top: 0.75rem; font-size: 0.74rem; font-weight: 600; color: var(--lp-ai-cyan);">
            Deliverable: Explainable Decision Support
        </div>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
