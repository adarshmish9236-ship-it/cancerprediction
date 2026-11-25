"""
CANCERWISE AI — Landing Page Navigation Bar Component
Renders the high-fidelity clinical header bar with brand identity, live gateway telemetry,
and immediate workspace access actions.
"""

from __future__ import annotations

import streamlit as st
from frontend.utils.api_client import APIClient


def render_navbar(is_dark: bool = True) -> None:
    """Renders the top clinical navigation and gateway status bar."""
    api = APIClient()
    is_online = st.session_state.get("backend_online", api.is_backend_online())
    api_badge = (
        '<span style="color: #21976B; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">'
        '<span style="font-size: 0.6rem;">●</span> API Gateway: Online (FastAPI v1)</span>'
        if is_online
        else '<span style="color: #D99A38; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">'
        '<span style="font-size: 0.6rem;">●</span> API Gateway: Local Fallback</span>'
    )

    c_brand, c_status, c_cta = st.columns([2.5, 2.0, 1.3], gap="medium")

    with c_brand:
        st.markdown(
            """<div style="display: flex; align-items: center; gap: 12px;">
    <div class="lp-nav-logo-icon">🧬</div>
    <div>
        <div class="lp-nav-brand-title">CANCERWISE AI</div>
        <div class="lp-nav-brand-tag">Intelligent Oncology Research &amp; Decision Support</div>
    </div>
</div>""",
            unsafe_allow_html=True,
        )

    with c_status:
        st.markdown(
            f"""<div style="display: flex; align-items: center; gap: 10px; height: 100%; justify-content: flex-end; padding-right: 8px;">
    <span class="lp-nav-badge">📁 UCI WDBC (569 Cases)</span>
    <span class="lp-nav-badge">{api_badge}</span>
</div>""",
            unsafe_allow_html=True,
        )

    with c_cta:
        if st.button("Launch Dashboard 📊", type="primary", key="nav_explore_cta", use_container_width=True):
            st.switch_page("pages/01_Overview_Dashboard.py")

    st.markdown("<hr style='border: 0; border-top: 1px solid var(--lp-border); margin: 0.75rem 0 1.5rem 0;'>", unsafe_allow_html=True)
