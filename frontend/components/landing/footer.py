"""
CANCERWISE AI — Professional Footer Component (Section 12)
Renders the brand summary, categorized navigation links, documentation references,
academic copyright, and prominent medical disclaimer.
"""

from __future__ import annotations

import streamlit as st


def render_footer(is_dark: bool = True) -> None:
    """Renders Section 12: Professional Footer."""
    st.markdown('<div id="contact"></div>', unsafe_allow_html=True)

    st.markdown(
        """<footer class="lp-footer">
<div class="lp-footer-grid">
    <div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.85rem;">
            <span style="font-size: 1.6rem;">🧬</span>
            <span style="font-size: 1.25rem; font-weight: 800; color: var(--lp-text-primary); letter-spacing: -0.02em;">
                CANCERWISE AI
            </span>
        </div>
        <p style="font-size: 0.86rem; color: var(--lp-text-secondary); line-height: 1.6; max-width: 320px; margin-bottom: 1rem;">
            An AI-powered platform for cancer data analysis, machine learning experimentation, 
            and explainable clinical research insights.
        </p>
        <div style="font-size: 0.78rem; color: var(--lp-text-muted);">
            Clinical Oncology Data Mining Suite • Version 1.0.0
        </div>
    </div>

    <div>
        <div class="lp-footer-col-title">Platform Workspaces</div>
        <ul class="lp-footer-links">
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Executive Dashboard</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Dataset Explorer</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Data Wrangling Studio</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Exploratory Data Analysis</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Feature Engineering</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Supervised ML Lab</span></li>
        </ul>
    </div>

    <div>
        <div class="lp-footer-col-title">Intelligence &amp; Governance</div>
        <ul class="lp-footer-links">
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Cancer Risk Predictor</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Explainable AI (SHAP &amp; LIME)</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Model Evaluation Suite</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Research Reports</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Experiment History</span></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">System Telemetry</span></li>
        </ul>
    </div>

    <div>
        <div class="lp-footer-col-title">API &amp; Specifications</div>
        <ul class="lp-footer-links">
            <li><a href="http://localhost:8000/docs" target="_blank" class="lp-footer-link">FastAPI Swagger UI ↗</a></li>
            <li><a href="http://localhost:8000/redoc" target="_blank" class="lp-footer-link">ReDoc API Specifications ↗</a></li>
            <li><a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" target="_blank" class="lp-footer-link">UCI WDBC Repository ↗</a></li>
            <li><span style="color: var(--lp-text-secondary); font-size: 0.84rem;">Responsible AI Policy</span></li>
        </ul>
    </div>
</div>

<div class="lp-footer-bottom">
    <div>
        © 2026 <strong>CANCERWISE AI</strong>. Academic &amp; Scientific Oncology Data Mining Platform. All rights reserved.
    </div>
    <div style="max-width: 650px; text-align: right; font-size: 0.74rem; color: var(--lp-text-muted); line-height: 1.4;">
        <strong>Medical Disclaimer:</strong> Designed for educational and scientific research support. 
        Not a diagnostic medical device or a substitute for professional clinical advice.
    </div>
</div>
</footer>""",
        unsafe_allow_html=True,
    )
