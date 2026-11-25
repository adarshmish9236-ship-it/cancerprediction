"""
CANCERWISE AI — Application Header & Breadcrumbs Component
Renders the standardized scientific header bar with breadcrumb trail,
cohort badge, and live API gateway status indicator.
"""

from __future__ import annotations

from typing import List, Optional
import streamlit as st


def render_header(
    title: str,
    subtitle: Optional[str] = None,
    breadcrumbs: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    show_status_badge: bool = True,
) -> None:
    """Renders professional top header with breadcrumbs and cohort badge."""
    theme_mode = st.session_state.get("sb_theme_choice", "Dark Navy")
    is_dark = "Dark" in theme_mode or "navy" in theme_mode.lower()
    crumb_color = "#64748B" if is_dark else "#718096"
    active_crumb_color = "#159B91"
    border_color = "#203D5C" if is_dark else "#E3EAF0"

    # 1. Breadcrumbs
    if breadcrumbs:
        crumbs_html = []
        for i, c in enumerate(breadcrumbs):
            is_last = (i == len(breadcrumbs) - 1)
            color = active_crumb_color if is_last else crumb_color
            weight = "600" if is_last else "400"
            crumbs_html.append(f'<span style="color: {color}; font-weight: {weight};">{c}</span>')
        crumb_str = f' <span style="color: {crumb_color}; opacity: 0.6; margin: 0 4px;">/</span> '.join(crumbs_html)
        st.markdown(
            f'<div style="font-size: 0.78rem; margin-bottom: 0.35rem; display: flex; align-items: center;">{crumb_str}</div>',
            unsafe_allow_html=True,
        )

    # 2. Main Title Row
    col_title, col_meta = st.columns([3.2, 1.3])
    with col_title:
        st.markdown(
            f'<h1 style="margin: 0; padding-bottom: 0.2rem; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em;">{title}</h1>',
            unsafe_allow_html=True,
        )
        if subtitle:
            st.markdown(
                f'<p style="color: {crumb_color}; font-size: 0.9rem; margin-top: 0.2rem; margin-bottom: 0.5rem; line-height: 1.4;">{subtitle}</p>',
                unsafe_allow_html=True,
            )

    with col_meta:
        if show_status_badge:
            ds = dataset_name or st.session_state.get("selected_dataset", "uci_wdbc")
            backend_online = st.session_state.get("backend_online", True)
            api_status = '<span class="cw-badge cw-badge-success">API: Online 🟢</span>' if backend_online else '<span class="cw-badge cw-badge-warning">API: Local 🟡</span>'
            
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; gap: 8px; margin-top: 0.35rem;">
                <span class="cw-badge cw-badge-primary">Cohort: {ds}</span>
                {api_status}
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"<hr style='border: 0; border-top: 1px solid {border_color}; margin: 0.65rem 0 1.25rem 0;'>", unsafe_allow_html=True)
