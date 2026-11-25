"""
CANCERWISE AI — Final Call to Action Component (Section 11)
Renders high-impact navy-to-teal banner with dual action buttons.
"""

from __future__ import annotations

import streamlit as st


def render_final_cta(is_dark: bool = True) -> None:
    """Renders Section 11: Final Call to Action."""
    st.markdown('<div id="cta"></div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="lp-section">
<div class="lp-final-cta">
    <div style="display: inline-flex; align-items: center; gap: 6px; padding: 0.35rem 0.85rem; border-radius: 9999px; background: rgba(255, 255, 255, 0.15); border: 1px solid rgba(255, 255, 255, 0.3); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.25rem;">
        ✦ ADVANCED RESEARCH ENVIRONMENT
    </div>
    <h2 class="lp-final-cta-title">Turn Complex Cancer Data Into Meaningful Research Insights.</h2>
    <p class="lp-final-cta-desc">
        Explore the capabilities of CANCERWISE AI and experience an integrated environment for data mining, 
        machine learning, and explainable analytics.
    </p>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    col_s1, col_b1, col_b2, col_s2 = st.columns([1.0, 1.4, 1.4, 1.0])
    with col_b1:
        if st.button("Launch Executive Dashboard 📊", type="primary", key="final_launch_btn", use_container_width=True):
            st.switch_page("pages/01_Overview_Dashboard.py")
    with col_b2:
        if st.button("Run Cancer Risk Predictor 🩺", key="final_pred_btn", use_container_width=True):
            st.switch_page("pages/11_Cancer_Prediction_Workspace.py")

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
