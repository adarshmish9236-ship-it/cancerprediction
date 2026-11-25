"""
CANCERWISE AI — Page 2: Dataset Explorer
Multi-modal dataset ingestion, schema inspection, statistical profiling, and download.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from frontend.components import (
    render_header,
    render_sidebar,
    MetricCard,
    DataPreview,
    DatasetUploader,
    ExportControl,
    ChartContainer,
    PALETTE,
)
from frontend.utils.api_client import APIClient

try:
    st.set_page_config(page_title="Dataset Explorer | CANCERWISE AI", page_icon="📁", layout="wide")
except Exception:
    pass

render_sidebar("Dataset Explorer")
api = APIClient()

render_header(
    title="Dataset Explorer & Provenance Inspector",
    subtitle="Ingest, validate, profile, and export multi-modal clinical oncology cohorts.",
    breadcrumbs=["CANCERWISE AI", "Data Management", "Dataset Explorer"],
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Active Cohort Overview & Preview",
    "📊 Feature Distributions & Schema",
    "📤 Ingest External Cohort",
    "📜 Clinical Metadata & Provenance",
])

selected_ds = st.session_state.get("selected_dataset", "uci_wdbc")
df = api.load_dataset(selected_ds)

with tab1:
    st.markdown(f"##### Working Cohort: `{selected_ds}`")
    if df is not None:
        target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
        n_rows, n_cols = df.shape
        missing_count = int(df.isna().sum().sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            MetricCard.render("Total Biopsy Records", f"{n_rows:,}", delta="De-identified", delta_color="positive", icon="👥")
        with c2:
            MetricCard.render("Morphometric Features", f"{n_cols - 1}", delta="Cytology", delta_color="positive", icon="🧬")
        with c3:
            MetricCard.render("Missing Value Cells", f"{missing_count}", delta="0.0% Missing", delta_color="positive", icon="🛡️")
        with c4:
            MetricCard.render("Storage Format", "Parquet", delta="Zero-Copy IPC", delta_color="positive", icon="💾")

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

        col_pv_head, col_pv_ctrl = st.columns([2, 1])
        with col_pv_head:
            st.markdown("##### Tabular Cohort Browser")
        with col_pv_ctrl:
            max_rows = st.select_slider("Preview Rows", options=[10, 25, 50, 100], value=10)

        DataPreview.render(df, max_rows=max_rows, show_stats=False)

        st.markdown("##### Descriptive Numerical Statistics")
        st.dataframe(df.describe().T.round(3), use_container_width=True)

        ExportControl.render(df, filename=f"{selected_ds}_exported", label=f"Export {selected_ds}")
    else:
        st.warning(f"Dataset `{selected_ds}` could not be loaded.")

with tab2:
    st.markdown("##### Feature Schema & Data Type Distribution")
    if df is not None:
        col_type_chart, col_target_chart = st.columns(2)
        
        with col_type_chart:
            dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ["Data Type", "Count"]
            fig_types = px.pie(
                dtype_counts,
                names="Data Type",
                values="Count",
                color_discrete_sequence=[PALETTE.PRIMARY_TEAL, PALETTE.ACCENT_CYAN, PALETTE.SECONDARY_NAVY],
                hole=0.4,
                title="Biomarker Column Data Types",
            )
            ChartContainer.render(fig_types, height=280)

        with col_target_chart:
            target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
            target_counts = df[target_col].value_counts().reset_index()
            target_counts.columns = ["Target Class", "Count"]
            fig_target = px.bar(
                target_counts,
                x="Target Class",
                y="Count",
                color="Target Class",
                color_discrete_sequence=[PALETTE.PRIMARY_TEAL, PALETTE.ERROR],
                title="Target Variable Distribution",
            )
            ChartContainer.render(fig_target, height=280)

        st.markdown("##### Detailed Feature Schema")
        schema_rows = []
        for col in df.columns:
            schema_rows.append({
                "Feature Name": col,
                "Data Type": str(df[col].dtype),
                "Missing Values": int(df[col].isna().sum()),
                "Unique Values": int(df[col].nunique()),
                "Domain Category": "Diagnosis Target" if col == target_col else "Nuclear Morphometry",
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

with tab3:
    st.markdown("##### Upload External Clinical Research Cohort")
    st.info("Supported formats: `.csv`, `.parquet`, `.xlsx`. All uploads undergo automated schema & privacy validation.")
    res = DatasetUploader.render(key="explorer_uploader")
    if res is not None:
        name, new_df = res
        st.markdown("##### Uploaded Cohort Preview")
        DataPreview.render(new_df, max_rows=10)

with tab4:
    st.markdown("##### Clinical Provenance & Metadata")
    st.markdown("""
    - **Cohort Identity:** Wisconsin Diagnostic Breast Cancer (WDBC)
    - **Origin:** University of Wisconsin Hospitals, Madison (Dr. William H. Wolberg)
    - **Biopsy Method:** Fine Needle Aspirate (FNA) cytology
    - **Imaging Modality:** Digitized nuclear imagery via Xcyt software
    - **De-identification:** Fully anonymized; patient IDs are decoupled from clinical outcomes
    - **Ethical Clearance:** Academic research cohort, publicly shared for biomedical data science
    """)
