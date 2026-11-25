"""
CANCERWISE AI — Research and Dataset Component (Section 10)
Showcases genuine clinical datasets and connectors registered in the platform.
"""

from __future__ import annotations

import streamlit as st


def render_dataset_section(is_dark: bool = True) -> None:
    """Renders Section 10: Research and Dataset Section."""
    st.markdown('<div id="research"></div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="lp-section">
<div style="text-align: center; max-width: 800px; margin: 0 auto 2.5rem auto;">
    <div class="lp-eyebrow">✦ PEER-REVIEWED COHORTS</div>
    <h2 class="lp-heading">Built Around Structured Cancer Research Data</h2>
    <p class="lp-subheading lp-subheading-center">
        CANCERWISE AI operates on verified, peer-reviewed biomedical datasets with documented provenance, 
        standardized data dictionaries, and explicit attribution.
    </p>
</div>

<div class="lp-dataset-card">
    <div class="lp-dataset-header">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.35rem;">
                <span style="font-size: 1.4rem;">📁</span>
                <span style="font-size: 1.25rem; font-weight: 700; color: var(--lp-text-primary);">
                    Wisconsin Diagnostic Breast Cancer (WDBC)
                </span>
                <span class="lp-cap-badge" style="background: rgba(33, 151, 107, 0.15); color: #21976B;">
                    ● Verified Active Cohort
                </span>
            </div>
            <div style="font-size: 0.86rem; color: var(--lp-text-secondary);">
                Source: UCI Machine Learning Repository • University of Wisconsin Hospitals (Wolberg, Street, Mangasarian, 1995)
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="lp-meta-pill">Records: 569 Cases</span>
            <span class="lp-meta-pill">Features: 30 Continuous</span>
            <span class="lp-meta-pill">Target: Benign (357) / Malignant (212)</span>
            <span class="lp-meta-pill">License: CC BY 4.0</span>
        </div>
    </div>

    <div style="font-size: 0.88rem; color: var(--lp-text-secondary); line-height: 1.6; margin-bottom: 1.25rem;">
        Features are computed from digitized images of Fine Needle Aspirate (FNA) biopsies of breast masses. 
        They describe ten fundamental characteristics of cell nuclei: radius, texture, perimeter, area, smoothness, 
        compactness, concavity, concave points, symmetry, and fractal dimension, captured across mean, standard error, and worst (extreme) values.
    </div>

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.25rem;">
        <div style="background: var(--lp-surface-alt); padding: 0.85rem; border-radius: 8px; border: 1px solid var(--lp-border);">
            <div style="font-size: 0.72rem; color: var(--lp-text-muted); text-transform: uppercase;">Missing Values</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--lp-success);">0 (100% Complete)</div>
        </div>
        <div style="background: var(--lp-surface-alt); padding: 0.85rem; border-radius: 8px; border: 1px solid var(--lp-border);">
            <div style="font-size: 0.72rem; color: var(--lp-text-muted); text-transform: uppercase;">Class Balance</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--lp-ai-cyan);">62.7% B / 37.3% M</div>
        </div>
        <div style="background: var(--lp-surface-alt); padding: 0.85rem; border-radius: 8px; border: 1px solid var(--lp-border);">
            <div style="font-size: 0.72rem; color: var(--lp-text-muted); text-transform: uppercase;">Modality</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--lp-text-primary);">Tabular FNA Cytology</div>
        </div>
    </div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.75rem;">
    <div class="lp-dataset-card" style="margin-bottom: 0;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.45rem;">
            <span style="font-size: 1.2rem;">🧬</span>
            <span style="font-weight: 700; font-size: 1.05rem; color: var(--lp-text-primary);">TCGA-BRCA Genomic Connector</span>
        </div>
        <p style="font-size: 0.84rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 0.85rem;">
            The Cancer Genome Atlas (NCI GDC) RNA-seq expression connector supporting exploration of multi-omic breast invasive carcinoma profiles.
        </p>
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <span class="lp-meta-pill">Modality: Genomic Expression</span>
            <span class="lp-meta-pill">Source: NCI Genomic Data Commons</span>
        </div>
    </div>

    <div class="lp-dataset-card" style="margin-bottom: 0;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.45rem;">
            <span style="font-size: 1.2rem;">🔬</span>
            <span style="font-weight: 700; font-size: 1.05rem; color: var(--lp-text-primary);">BreakHis Histopathology Connector</span>
        </div>
        <p style="font-size: 0.84rem; color: var(--lp-text-secondary); line-height: 1.5; margin-bottom: 0.85rem;">
            Breast Cancer Histopathological Image connector referencing microscopic biopsy slide manifests at 40X, 100X, 200X, and 400X magnification.
        </p>
        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <span class="lp-meta-pill">Modality: Microscopic Imaging</span>
            <span class="lp-meta-pill">Benchmark: Peer-Reviewed</span>
        </div>
    </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    col_s1, col_btn, col_s2 = st.columns([1.2, 1.4, 1.2])
    with col_btn:
        if st.button("Explore Dataset Analysis Studio 📁", type="primary", key="btn_explore_datasets", use_container_width=True):
            st.switch_page("pages/02_Dataset_Explorer.py")

    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
