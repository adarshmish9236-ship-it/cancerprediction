"""
CANCERWISE AI — Standardized Scientific UI Components
Provides accessible, responsive, and clinical-grade components:
- MetricCard
- ChartContainer
- DataPreview
- DatasetUploader
- ModelSelector
- AlgorithmConfigurator
- ExperimentProgress
- ResultsTable
- ConfusionMatrixView
- PredictionResultView
- ExplanationPanel
- ExportControl
"""

from __future__ import annotations

import io
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.styles.design_tokens import PALETTE
from frontend.styles.theme import get_plotly_template


class MetricCard:
    """Renders a styled scientific KPI card with title, value, delta, subtext, and theme awareness."""

    @staticmethod
    def render(
        title: str,
        value: Union[str, int, float],
        delta: Optional[str] = None,
        delta_color: str = "positive",
        subtext: Optional[str] = None,
        icon: Optional[str] = None,
    ) -> None:
        delta_class = f"delta-{delta_color}" if delta else ""
        delta_icon = "↑ " if delta_color == "positive" and delta and not delta.startswith(("-", "+", "↑", "↓")) else ""
        delta_html = f'<div class="cw-metric-delta {delta_class}">{delta_icon}{delta}</div>' if delta else ""
        sub_html = f'<div style="font-size: 0.75rem; color: #718096; margin-top: 0.35rem; line-height: 1.3;">{subtext}</div>' if subtext else ""
        icon_html = f'<span style="font-size: 1rem; margin-right: 4px;">{icon}</span>' if icon else ""

        html = f'<div class="cw-metric-card"><div class="cw-metric-title">{icon_html}{title}</div><div class="cw-metric-value">{value}</div>{delta_html}{sub_html}</div>'
        st.markdown(html, unsafe_allow_html=True)


class ChartContainer:
    """Wraps Plotly figures or charts in a standardized scientific card container."""

    @staticmethod
    def render(
        fig: Any,
        title: Optional[str] = None,
        description: Optional[str] = None,
        use_container_width: bool = True,
        height: Optional[int] = None,
    ) -> None:
        theme_mode = st.session_state.get("sb_theme_choice", "Dark Navy")
        template = get_plotly_template(theme_mode)

        if title:
            st.markdown(f"<div style='font-weight: 600; font-size: 1.05rem; margin-bottom: 0.2rem; display: flex; align-items: center; gap: 6px;'>{title}</div>", unsafe_allow_html=True)
        if description:
            st.markdown(f"<div style='font-size: 0.8rem; color: #718096; margin-bottom: 0.65rem;'>{description}</div>", unsafe_allow_html=True)

        # Apply synchronized template styling to Plotly figure
        if hasattr(fig, "update_layout"):
            fig.update_layout(
                paper_bgcolor=template["paper_bgcolor"],
                plot_bgcolor=template["plot_bgcolor"],
                font=template["font"],
                margin=template["margin"],
            )
            if height:
                fig.update_layout(height=height)

        st.plotly_chart(fig, use_container_width=use_container_width)


class DataPreview:
    """Interactive preview for tabular cohorts with dimensions, memory, and missing stats."""

    @staticmethod
    def render(
        df: pd.DataFrame,
        max_rows: int = 10,
        show_stats: bool = True,
    ) -> None:
        if df is None or df.empty:
            st.info("No records loaded to preview.")
            return

        if show_stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='cw-card' style='padding: 0.6rem 0.85rem; margin-bottom: 0.75rem;'><span style='font-size: 0.75rem; color: #718096;'>PATIENTS / ROWS</span><br><strong style='font-size: 1.1rem;'>{df.shape[0]:,}</strong></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='cw-card' style='padding: 0.6rem 0.85rem; margin-bottom: 0.75rem;'><span style='font-size: 0.75rem; color: #718096;'>BIOMARKERS</span><br><strong style='font-size: 1.1rem;'>{df.shape[1]}</strong></div>", unsafe_allow_html=True)
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            c3.markdown(f"<div class='cw-card' style='padding: 0.6rem 0.85rem; margin-bottom: 0.75rem;'><span style='font-size: 0.75rem; color: #718096;'>MEMORY</span><br><strong style='font-size: 1.1rem;'>{mem_mb:.2f} MB</strong></div>", unsafe_allow_html=True)
            missing_count = int(df.isna().sum().sum())
            missing_color = "#21976B" if missing_count == 0 else "#D9534F"
            c4.markdown(f"<div class='cw-card' style='padding: 0.6rem 0.85rem; margin-bottom: 0.75rem;'><span style='font-size: 0.75rem; color: #718096;'>MISSING CELLS</span><br><strong style='font-size: 1.1rem; color: {missing_color};'>{missing_count}</strong></div>", unsafe_allow_html=True)

        st.dataframe(df.head(max_rows), use_container_width=True)


class DatasetUploader:
    """Standardized drag-and-drop dataset upload control with validation."""

    @staticmethod
    def render(
        supported_types: Optional[List[str]] = None,
        key: str = "dataset_uploader",
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        if supported_types is None:
            supported_types = ["csv", "parquet", "xlsx"]

        uploaded_file = st.file_uploader(
            "Upload Oncology Cohort (CSV, Parquet, XLSX)",
            type=supported_types,
            key=key,
            help="Maximum supported file size: 100MB. Ensure patient privacy (de-identified only).",
        )
        if uploaded_file is not None:
            name = uploaded_file.name
            try:
                if name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif name.endswith(".parquet"):
                    df = pd.read_parquet(uploaded_file)
                elif name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format.")
                    return None
                st.success(f"Successfully parsed `{name}` ({df.shape[0]} rows, {df.shape[1]} columns)")
                return name, df
            except Exception as e:
                st.error(f"Failed to parse uploaded dataset: {e}")
                return None
        return None


class ModelSelector:
    """Dropdown control for selecting active machine learning models from registry."""

    @staticmethod
    def render(
        available_models: List[str],
        default_index: int = 0,
        key: str = "global_model_selector",
    ) -> str:
        if not available_models:
            available_models = ["Random Forest (Default)", "Logistic Regression", "PyTorch MLP"]
        selected = st.selectbox(
            "Active Predictive Model",
            available_models,
            index=min(default_index, len(available_models) - 1),
            key=key,
        )
        return selected


class AlgorithmConfigurator:
    """Dynamic form generating algorithm-specific hyperparameters with expandable settings."""

    @staticmethod
    def render(algorithm: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        if "Random Forest" in algorithm or "Extra Trees" in algorithm:
            c1, c2 = st.columns(2)
            params["n_estimators"] = c1.slider("Estimators", 20, 300, 100, 10)
            params["max_depth"] = c2.slider("Max Tree Depth", 2, 20, 6)
            with st.expander("⚙️ Advanced Tree Hyperparameters", expanded=False):
                ca1, ca2 = st.columns(2)
                params["min_samples_split"] = ca1.selectbox("Min Samples Split", [2, 5, 10])
                params["criterion"] = ca2.selectbox("Split Criterion", ["gini", "entropy", "log_loss"])

        elif "Logistic" in algorithm or "SGD" in algorithm:
            c1, c2 = st.columns(2)
            params["C"] = c1.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0, 0.05)
            params["max_iter"] = c2.slider("Max Solver Iterations", 200, 2000, 1000, 100)
            with st.expander("⚙️ Advanced Solver Hyperparameters", expanded=False):
                params["solver"] = st.selectbox("Optimization Algorithm", ["lbfgs", "saga", "liblinear"])

        elif "SVM" in algorithm:
            c1, c2 = st.columns(2)
            params["C"] = c1.slider("Penalty Parameter (C)", 0.1, 10.0, 1.0)
            params["kernel"] = c2.selectbox("Kernel Function", ["rbf", "linear", "poly"])
            with st.expander("⚙️ Advanced Kernel Hyperparameters", expanded=False):
                params["gamma"] = st.selectbox("Gamma Setting", ["scale", "auto"])

        elif "XGBoost" in algorithm or "Gradient Boosting" in algorithm:
            c1, c2 = st.columns(2)
            params["n_estimators"] = c1.slider("Boosting Stages", 30, 300, 100, 10)
            params["learning_rate"] = c2.slider("Learning Rate (η)", 0.01, 0.3, 0.1, 0.01)
            with st.expander("⚙️ Advanced Boosting Settings", expanded=False):
                params["max_depth"] = st.slider("Max Depth", 2, 10, 4)

        else:
            params["random_state"] = 42

        return params


class ExperimentProgress:
    """Progress tracker for multi-stage ML pipelines with stage labels."""

    @staticmethod
    def render(stage: str, percent: int) -> None:
        st.markdown(f"**Executing Stage:** `{stage}` ({percent}%)")
        st.progress(min(max(percent, 0), 100))


class ResultsTable:
    """Formatted comparison table with sorting, search, and rank badges."""

    @staticmethod
    def render(df: pd.DataFrame, highlight_col: Optional[str] = None) -> None:
        if df.empty:
            st.info("No evaluation results available.")
            return

        if highlight_col and highlight_col in df.columns:
            st.dataframe(
                df.style.highlight_max(subset=[highlight_col], color="rgba(33, 151, 107, 0.25)"),
                use_container_width=True,
            )
        else:
            st.dataframe(df, use_container_width=True)


class ConfusionMatrixView:
    """Interactive heatmap visualization of clinical diagnostic confusion matrix."""

    @staticmethod
    def render(
        cm: Union[List[List[int]], np.ndarray],
        labels: Optional[List[str]] = None,
        title: str = "Diagnostic Confusion Matrix",
    ) -> None:
        if labels is None:
            labels = ["Benign (0)", "Malignant (1)"]

        cm_arr = np.asarray(cm)
        fig = px.imshow(
            cm_arr,
            labels=dict(x="Predicted Diagnosis", y="True Diagnosis", color="Patients"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=[[0, "#EBF5FB"], [0.5, "#43BFC4"], [1, "#142B45"]],
        )
        fig.update_layout(
            title=title,
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#718096"),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


class PredictionResultView:
    """Displays single patient classification results, probability gauge, and warnings."""

    @staticmethod
    def render(
        predicted_class: int,
        prob_malignant: float,
        model_name: str = "Random Forest",
    ) -> None:
        is_mal = predicted_class == 1
        label = "Malignant (Elevated Risk)" if is_mal else "Benign (Non-Malignant)"
        color = PALETTE.ERROR if is_mal else PALETTE.SUCCESS
        badge_type = "cw-badge-danger" if is_mal else "cw-badge-success"

        c1, c2 = st.columns([1.1, 1.3])
        with c1:
            st.markdown(f"""
            <div class="cw-metric-card" style="border-left: 5px solid {color}; padding: 1.25rem 1.4rem;">
                <div class="cw-metric-title">Model Decision Assessment</div>
                <div class="cw-metric-value" style="color: {color}; font-size: 1.65rem;">{label}</div>
                <div style="margin-top: 0.65rem;">
                    <span class="cw-badge {badge_type}" style="font-size: 0.82rem; padding: 0.3rem 0.8rem;">
                        P(Malignancy) = {prob_malignant * 100:.2f}%
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #718096; margin-top: 0.85rem; line-height: 1.4;">
                    Evaluated by <strong>{model_name}</strong><br>
                    Classification Threshold: <strong>0.500</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob_malignant * 100, 2),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Biopsy Malignancy Probability Index (%)", 'font': {'size': 13, 'family': 'Inter', 'color': '#718096'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#718096", 'tickwidth': 1},
                    'bar': {'color': color, 'thickness': 0.28},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(33, 151, 107, 0.15)"},
                        {'range': [30, 70], 'color': "rgba(217, 154, 56, 0.15)"},
                        {'range': [70, 100], 'color': "rgba(217, 83, 79, 0.15)"},
                    ],
                    'threshold': {
                        'line': {'color': "#D9534F", 'width': 3},
                        'thickness': 0.8,
                        'value': 50.0,
                    }
                }
            ))
            fig.update_layout(
                height=210,
                margin=dict(l=25, r=25, t=35, b=15),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="cw-alert-clinical">
            ⚕️ <strong>Critical Governance & Safety Notice:</strong> 
            Computational probabilities are statistical decision-support aids for research only. 
            They do <strong>NOT</strong> constitute a definitive oncological diagnosis. 
            All cases require independent histological, radiological, and clinical oncologist verification.
        </div>
        """, unsafe_allow_html=True)


class ExplanationPanel:
    """Renders SHAP local waterfall or LIME local rule attributions."""

    @staticmethod
    def render(
        features: List[str],
        impacts: List[float],
        directions: List[str],
        title: str = "Local Feature Contributions (SHAP Attributions)",
    ) -> None:
        df = pd.DataFrame({
            "Biomarker Feature": features,
            "Attribution Weight": impacts,
            "Diagnostic Tendency": directions,
        })
        fig = px.bar(
            df,
            x="Attribution Weight",
            y="Biomarker Feature",
            orientation="h",
            color="Diagnostic Tendency",
            color_discrete_map={
                "Increases Risk (Malignant)": PALETTE.ERROR,
                "Decreases Risk (Benign)": PALETTE.SUCCESS,
                "Supports Malignant": PALETTE.ERROR,
                "Supports Benign": PALETTE.SUCCESS,
            },
            title=title,
        )
        fig.update_layout(
            yaxis={"autorange": "reversed"},
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#718096"),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


class ExportControl:
    """One-click export buttons for DataFrames, JSON, and Reports."""

    @staticmethod
    def render(
        data: Union[pd.DataFrame, dict, str],
        filename: str = "export",
        label: str = "Download Dataset",
    ) -> None:
        if isinstance(data, pd.DataFrame):
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"📥 {label} (CSV)",
                data=csv,
                file_name=f"{filename}.csv",
                mime="text/csv",
            )
        elif isinstance(data, dict):
            js = json.dumps(data, indent=2).encode("utf-8")
            st.download_button(
                label=f"📥 {label} (JSON)",
                data=js,
                file_name=f"{filename}.json",
                mime="application/json",
            )
        elif isinstance(data, str):
            st.download_button(
                label=f"📥 {label}",
                data=data,
                file_name=filename,
                mime="text/plain",
            )
