"""
CANCERWISE AI — Platform Introduction Component (Section 3)
Renders the 3 core pillars: Data Intelligence, Machine Learning, and Explainable AI.
"""

from __future__ import annotations

import streamlit as st


def render_intro(is_dark: bool = True) -> None:
    """Renders Section 3: Platform Introduction."""
    st.markdown(
        """<div id="about" class="lp-section">
<div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
    <div class="lp-eyebrow">✦ INTEGRATED ONCOLOGY RESEARCH PLATFORM</div>
    <h2 class="lp-heading">Where Healthcare Research Meets Artificial Intelligence</h2>
    <p class="lp-subheading lp-subheading-center">
        Transform complex fine-needle aspirate cytology datasets into structured, interpretable, 
        and research-ready insights through a unified machine learning and data mining environment.
    </p>
</div>

<div class="lp-pillars-grid">
    <div class="lp-pillar-card">
        <div class="lp-pillar-icon">📊</div>
        <h3 class="lp-pillar-title">Clinical Data Intelligence</h3>
        <p style="font-size: 0.88rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 1rem;">
            Rigorous clinical data ingestion, schema validation pipelines, missing-value imputation, and multi-stage wrangling.
        </p>
        <ul class="lp-pillar-list">
            <li>Structured dataset exploration &amp; preview</li>
            <li>KNN &amp; iterative missing-value imputation</li>
            <li>IQR outlier detection &amp; winsorization</li>
            <li>Multi-dimensional data quality auditing</li>
        </ul>
    </div>

    <div class="lp-pillar-card">
        <div class="lp-pillar-icon">🤖</div>
        <h3 class="lp-pillar-title">Multi-Paradigm Machine Learning</h3>
        <p style="font-size: 0.88rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 1rem;">
            High-precision supervised, semi-supervised, and deep neural network paradigms tailored for cellular oncology patterns.
        </p>
        <ul class="lp-pillar-list">
            <li>16 benchmarked supervised algorithms</li>
            <li>Deep PyTorch multi-layer perceptron (MLP)</li>
            <li>Semi-supervised &amp; anomaly detection models</li>
            <li>Platt &amp; isotonic calibrated risk scoring</li>
        </ul>
    </div>

    <div class="lp-pillar-card">
        <div class="lp-pillar-icon">🔬</div>
        <h3 class="lp-pillar-title">Explainable AI &amp; Governance</h3>
        <p style="font-size: 0.88rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 1rem;">
            Interpretable model behaviors delivering clinical transparency, auditability, and mathematical feature attribution.
        </p>
        <ul class="lp-pillar-list">
            <li>SHAP global beeswarm &amp; local force plots</li>
            <li>LIME decision boundary approximations</li>
            <li>Brier calibration curves &amp; ROC-AUC analysis</li>
            <li>Statutory research disclaimers &amp; audit trails</li>
        </ul>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
