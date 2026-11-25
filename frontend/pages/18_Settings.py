"""
CANCERWISE AI — Page 18: Application Settings & Diagnostics
Configuration preferences, API endpoints, logging levels, and cache management.
"""

from __future__ import annotations

import streamlit as st

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    apply_theme,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="Settings | CANCERWISE AI", page_icon="⚙️", layout="wide")
except Exception:
    pass

render_sidebar("Settings")
api = APIClient()

render_header(
    title="Application Settings & Platform Preferences",
    subtitle="Manage API gateway endpoints, visual appearance themes, telemetry thresholds, and memory caches.",
    breadcrumbs=["CANCERWISE AI", "System", "Settings"],
)

tab1, tab2, tab3 = st.tabs([
    "🎨 Display & Appearance",
    "🌐 API Gateway & Network",
    "🧹 Memory & Cache Management",
])

with tab1:
    st.markdown("##### Visual Theme & Font Settings")
    current_theme = st.session_state.get("sb_theme_choice", "Dark Navy")
    theme_choice = st.radio("Active Theme Mode", ["Dark Navy", "Clean Light"], index=0 if current_theme == "Dark Navy" else 1, horizontal=True)
    if theme_choice != current_theme:
        st.session_state["sb_theme_choice"] = theme_choice
        apply_theme(theme_choice)
        st.rerun()

    st.markdown("""
    - **Font Family:** `Inter` (UI Primary), `JetBrains Mono` (Code & Data Tables)
    - **Palette Mode:** Clinical High-Contrast
    - **Card Elevation:** Rounded 10px with Subtle Border Accent
    """)

with tab2:
    st.markdown("##### Backend Microservice Configuration")
    api_host = st.text_input("FastAPI Host Address", value="http://localhost:8000")
    api_prefix = st.text_input("REST Endpoint Prefix", value="/api/v1")
    timeout_sec = st.slider("Request Timeout (Seconds)", 1.0, 15.0, 5.0)

    if st.button("Test Gateway Connection", type="primary"):
        online = api.is_backend_online()
        if online:
            st.success("API Gateway successfully reached! HTTP 200 OK.")
        else:
            st.warning("Could not reach FastAPI on specified port. Local Python service fallback active.")

with tab3:
    st.markdown("##### Memory & Cache Management")
    st.markdown("Clear session caches, temporary imputation buffers, and in-memory dataframes.")
    if st.button("Purge Streamlit Caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("All cached datasets, models, and figures cleared from memory.")
