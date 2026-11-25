"""
CANCERWISE AI — Application Sidebar Component
Provides brand identity, global dataset selection, active model tracking,
theme switching, live hardware telemetry, and clinical governance notes.
"""

from __future__ import annotations

from typing import List, Optional
import psutil
import streamlit as st

from backend.services.dataset_service import DatasetService
from data_mining.classification.serializer import ModelSerializer
from frontend.styles.theme import apply_theme

_dataset_service = DatasetService()
_serializer = ModelSerializer()


def render_sidebar(current_page: str = "Overview") -> str:
    """Renders the standard CANCERWISE AI sidebar and returns the selected theme mode."""
    with st.sidebar:
        # 1. Branding Header
        st.markdown("""
        <div style="padding: 0.25rem 0 0.85rem 0; border-bottom: 1px solid rgba(32, 61, 92, 0.4); margin-bottom: 0.85rem;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2.1rem; filter: drop-shadow(0 2px 4px rgba(21, 155, 145, 0.3));">🧬</span>
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15;">CANCERWISE AI</div>
                    <div style="font-size: 0.72rem; color: #159B91; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Oncology Data Mining Suite</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Global Dataset Selector
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;'>📁 Active Research Cohort</div>", unsafe_allow_html=True)
        registered_ds = _dataset_service.list_datasets()
        ds_ids = []
        for d in registered_ds:
            if hasattr(d, "dataset_id"):
                ds_ids.append(d.dataset_id)
            elif isinstance(d, dict):
                ds_ids.append(d.get("dataset_id", "uci_wdbc"))
        if not ds_ids:
            ds_ids = ["uci_wdbc"]
        if "uci_wdbc" not in ds_ids:
            ds_ids.insert(0, "uci_wdbc")

        current_ds = st.session_state.get("selected_dataset", "uci_wdbc")
        idx = ds_ids.index(current_ds) if current_ds in ds_ids else 0
        selected_ds = st.selectbox(
            "Select Working Cohort",
            ds_ids,
            index=idx,
            label_visibility="collapsed",
            key="sb_dataset_selector",
        )
        st.session_state["selected_dataset"] = selected_ds

        # 3. Active Model Indicator
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.65rem; margin-bottom: 0.25rem;'>🤖 Active Predictive Model</div>", unsafe_allow_html=True)
        saved_models = _serializer.list_models()
        model_names = [m.get("model_id") for m in saved_models if m.get("model_id")]
        if not model_names:
            model_names = ["Random Forest (Ensemble)", "Logistic Regression (L2)", "PyTorch MLP (Deep)"]

        current_m = st.session_state.get("active_model", model_names[0])
        m_idx = model_names.index(current_m) if current_m in model_names else 0
        selected_model = st.selectbox(
            "Active Predictive Model",
            model_names,
            index=m_idx,
            label_visibility="collapsed",
            key="sb_model_selector",
        )
        st.session_state["active_model"] = selected_model

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(32, 61, 92, 0.4); margin: 0.85rem 0;'>", unsafe_allow_html=True)

        # 4. Theme & Appearance
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;'>🎨 Visual Theme</div>", unsafe_allow_html=True)
        theme_choice = st.radio(
            "Theme Mode",
            ["Dark Navy", "Clean Light"],
            horizontal=True,
            label_visibility="collapsed",
            key="sb_theme_choice",
        )
        apply_theme(theme_choice)

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(32, 61, 92, 0.4); margin: 0.85rem 0;'>", unsafe_allow_html=True)

        # 5. System Resource Telemetry
        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;'>⚡ System Telemetry</div>", unsafe_allow_html=True)
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        st.markdown(f"""
        <div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span>CPU Host Utilization:</span>
            <strong style="color: {'#21976B' if cpu_pct < 75 else '#D9534F'};">{cpu_pct:.1f}%</strong>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span>RAM Committed:</span>
            <strong>{mem.percent:.1f}% ({mem.used / (1024**3):.1f} GB)</strong>
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between;">
            <span>Gateway Protocol:</span>
            <span class="cw-badge cw-badge-success" style="font-size: 0.68rem; padding: 0.1rem 0.45rem;">FastAPI / ASGI</span>
        </div>
        """, unsafe_allow_html=True)

        # 6. Governance Footer Notice
        st.markdown("""
        <div style="margin-top: 1.25rem; padding-top: 0.75rem; border-top: 1px solid rgba(32, 61, 92, 0.4); font-size: 0.7rem; color: #64748B; line-height: 1.35;">
            ⚕️ <strong>Research Prototype:</strong> Algorithmic decision support only. Requires clinical pathologist validation.
        </div>
        """, unsafe_allow_html=True)

    return theme_choice
