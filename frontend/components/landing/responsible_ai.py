"""
CANCERWISE AI — Responsible AI & Data Privacy Component (Section 9)
Renders ethical principles, dataset limitations, privacy practices, and prominent clinical disclaimer.
"""

from __future__ import annotations

import streamlit as st


def render_responsible_ai(is_dark: bool = True) -> None:
    """Renders Section 9: Responsible AI and Data Privacy."""
    st.markdown('<div id="responsible-ai"></div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="lp-section">
<div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
    <div class="lp-eyebrow">✦ ETHICAL AI &amp; CLINICAL SAFETY</div>
    <h2 class="lp-heading">Responsible AI. Transparent Research. Informed Decisions.</h2>
    <p class="lp-subheading lp-subheading-center">
        Healthcare data mining requires the highest standard of accountability. CANCERWISE AI is built 
        with explicit awareness of statistical bias, model limitations, and clinical safety boundaries.
    </p>
</div>

<div class="lp-values-grid">
    <div class="lp-value-card">
        <div class="lp-value-title">🔒 Data Privacy Awareness</div>
        <div class="lp-value-desc">
            Datasets utilized within the platform (such as UCI WDBC) are fully de-identified and anonymized, 
            free from Protected Health Information (PHI) and direct patient identifiers.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">⚖️ Bias &amp; Cohort Limitations</div>
        <div class="lp-value-desc">
            Models reflect the demographic and morphological characteristics of their source cohorts. 
            Algorithms trained on FNA cytology should not be generalized to imaging or histopathology without validation.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">🩺 Human-in-the-Loop Oversight</div>
        <div class="lp-value-desc">
            Artificial intelligence must augment, never replace, trained clinical judgment. Algorithmic outputs 
            serve solely as decision-support heuristics to assist expert human review.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">🔬 Calibrated Uncertainty</div>
        <div class="lp-value-desc">
            Raw softmax outputs can suffer from overconfidence. The platform integrates Brier score calibration 
            and empirical bootstrap intervals to represent confidence intervals truthfully.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">📜 Audit Trails &amp; Versioning</div>
        <div class="lp-value-desc">
            Every dataset ingestion, feature transformation, and trained model artifact maintains persistent metadata, 
            SHA-256 checksums, and version tracking for complete provenance.
        </div>
    </div>

    <div class="lp-value-card">
        <div class="lp-value-title">🔍 Explainable Auditing</div>
        <div class="lp-value-desc">
            Physicians and researchers can interrogate both global feature rankings and individual prediction 
            force plots to understand exactly why a model flags elevated risk.
        </div>
    </div>
</div>

<div class="lp-disclaimer-box">
    <div style="display: flex; align-items: flex-start; gap: 12px;">
        <span style="font-size: 1.4rem;">⚠️</span>
        <div>
            <strong style="text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.85rem;">
                Statutory Medical &amp; Educational Disclaimer
            </strong>
            <p style="margin: 0.4rem 0 0 0; line-height: 1.55;">
                CANCERWISE AI is designed for educational, analytical, and research purposes. 
                Its predictions are not medical diagnoses and must not be used as the sole basis for clinical decisions. 
                Always consult qualified healthcare professionals and licensed oncologists for patient care.
            </p>
        </div>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
