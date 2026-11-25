"""
CANCERWISE AI — Why CANCERWISE AI Component (Section 8)
Renders 6 core value propositions focusing on transparency, data quality, and reproducibility.
"""

from __future__ import annotations

import streamlit as st


def render_why_cancerwise(is_dark: bool = True) -> None:
    """Renders Section 8: Why CANCERWISE AI."""
    st.markdown('<div id="why-cancerwise"></div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="lp-section">
<div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
    <div class="lp-eyebrow">✦ RESEARCH INTEGRITY</div>
    <h2 class="lp-heading">Designed for Transparent and Responsible AI Research</h2>
    <p class="lp-subheading lp-subheading-center">
        Bridging the divide between black-box statistical modeling and clinical auditability, 
        CANCERWISE AI prioritizes rigorous data quality, reproducibility, and explainability.
    </p>
</div>

<div class="lp-values-grid">
    <div class="lp-value-card">
        <div class="lp-value-title">
            <span style="color: var(--lp-healthcare-teal);">✦</span> 1. Integrated Workflow
        </div>
        <div class="lp-value-desc">
            A unified computational environment orchestrating data ingestion, cleaning, exploratory visualization, 
            feature selection, modeling, and XAI without disjointed toolchains.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">
            <span style="color: var(--lp-ai-cyan);">✦</span> 2. Model Transparency
        </div>
        <div class="lp-value-desc">
            Explainable AI methods (SHAP game-theoretic Shapley values and LIME local approximations) 
            interrogate feature contributions, ensuring no opaque decision boundaries.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">
            <span style="color: var(--lp-success);">✦</span> 3. Research-Oriented Design
        </div>
        <div class="lp-value-desc">
            Supports structured experimentation, stratified cross-validation, hyperparameter grid search, 
            and automated metric tracking across classical and modern deep learning models.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">
            <span style="color: var(--lp-healthcare-teal);">✦</span> 4. Data Quality Focus
        </div>
        <div class="lp-value-desc">
            Emphasizes schema validation, missing-value handling, outlier mitigation, and distribution checks 
            before any learning algorithm touches patient cohorts.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">
            <span style="color: var(--lp-ai-cyan);">✦</span> 5. Reproducible Analysis
        </div>
        <div class="lp-value-desc">
            Encourages documented experiments, serialized model artifacts with metadata, standardized random seeds, 
            and exportable research reports for peer scrutiny.
        </div>
    </div>

    <div class="lp-value-card" style="background: rgba(21, 155, 145, 0.08); border-color: rgba(21, 155, 145, 0.35);">
        <div class="lp-value-title">
            <span style="color: var(--lp-success);">🛡️</span> 6. Calibrated Uncertainty
        </div>
        <div class="lp-value-desc">
            All estimated probabilities are Platt-scaled or isotonic-calibrated, paired with 95% bootstrap 
            confidence intervals to communicate statistical uncertainty honestly.
        </div>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
