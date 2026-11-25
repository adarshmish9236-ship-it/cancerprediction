"""
CANCERWISE AI — Exploratory Data Analysis (EDA) Service
Orchestrates statistical hypothesis testing, univariate/multivariate profiling,
interactive visual chart generation, and publication-ready HTML/PDF report synthesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.core.exceptions import DatasetNotFoundError
from backend.core.logging_setup import get_logger
from backend.schemas.eda import (
    DatasetOverview,
    UnivariateStats,
    BivariateAnalysisResult,
    MultivariateStats,
    StatisticalTestResult,
    WDBCSpecificFindings,
)
from backend.services.dataset_management_service import DatasetManagementService
from data_mining.eda.eda_engine import EDAEngine
from data_mining.eda.eda_reporter import EDAReporter
from data_mining.eda.visualization_service import VisualizationService

logger = get_logger("services.eda")


class EDAService:
    """
    Dedicated service for exploratory data analysis, hypothesis testing, and reporting.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        eda_engine: Optional[EDAEngine] = None,
        eda_reporter: Optional[EDAReporter] = None,
        vis_service: Optional[VisualizationService] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.eda_engine = eda_engine or EDAEngine()
        self.eda_reporter = eda_reporter or EDAReporter()
        self.vis_service = vis_service or VisualizationService()

    def get_overview(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> DatasetOverview:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        meta = self.dataset_mgmt.get_metadata(dataset_id)
        return self.eda_engine.get_overview(
            df,
            dataset_id=dataset_id,
            dataset_name=meta.name,
            target_column=target_column,
        )

    def get_univariate_profile(
        self,
        dataset_id: str,
        feature_name: str,
    ) -> UnivariateStats:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        return self.eda_engine.get_univariate_profile(df, feature_name=feature_name)

    def get_bivariate_profile(
        self,
        dataset_id: str,
        feature_x: str,
        feature_y: str,
        target_column: Optional[str] = "diagnosis",
    ) -> BivariateAnalysisResult:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        return self.eda_engine.get_bivariate_profile(
            df,
            x_col=feature_x,
            y_col=feature_y,
            target_column=target_column,
        )

    def get_multivariate_profile(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> MultivariateStats:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        return self.eda_engine.get_multivariate_profile(df, target_column=target_column)

    def run_statistical_tests(
        self,
        dataset_id: str,
        target_column: str = "diagnosis",
    ) -> List[StatisticalTestResult]:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        return self.eda_engine.run_statistical_tests(df, target_column=target_column)

    def analyze_cancer_morphometrics(
        self,
        dataset_id: str,
        target_column: str = "diagnosis",
    ) -> WDBCSpecificFindings:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        return self.eda_engine.analyze_cancer_wdbc(df, target_column=target_column)

    def run_comprehensive_analysis(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> Dict[str, Any]:
        """Runs unified end-to-end EDA analysis across univariate, bivariate, multivariate, and hypothesis tests."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        overview = self.get_overview(dataset_id, target_column=target_column)
        multivariate = self.get_multivariate_profile(dataset_id, target_column=target_column)
        stat_tests = self.run_statistical_tests(dataset_id, target_column=target_column or "diagnosis")

        return {
            "dataset_id": dataset_id,
            "overview": overview.model_dump() if hasattr(overview, "model_dump") else overview,
            "multivariate": multivariate.model_dump() if hasattr(multivariate, "model_dump") else multivariate,
            "top_statistical_tests": [
                t.model_dump() if hasattr(t, "model_dump") else t for t in stat_tests[:10]
            ],
            "total_significant_features": sum(1 for t in stat_tests if getattr(t, "significant_fdr_adjusted", getattr(t, "is_significant", True))),

        }

    def get_visualizations(
        self,
        dataset_id: str,
        chart_type: str = "correlation_heatmap",
        feature_x: Optional[str] = None,
        feature_y: Optional[str] = None,
        target_column: Optional[str] = "diagnosis",
    ) -> Dict[str, Any]:
        """Generates interactive Plotly JSON / chart definitions."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        specs: Dict[str, Any] = {"dataset_id": dataset_id, "chart_type": chart_type}

        if chart_type == "correlation_heatmap":
            fig = self.vis_service.create_correlation_heatmap(df)
            specs["figure"] = fig.to_dict() if hasattr(fig, "to_dict") else str(fig)
        elif chart_type == "distribution":
            col = feature_x or df.select_dtypes(include=["number"]).columns[0]
            fig = self.vis_service.create_histogram(df, column=col, target_column=target_column)
            specs["figure"] = fig.to_dict() if hasattr(fig, "to_dict") else str(fig)
        elif chart_type == "pca_scatter":
            fig, _ = self.vis_service.create_pca_projection(df, target_column=target_column)
            specs["figure"] = fig.to_dict() if hasattr(fig, "to_dict") else str(fig)
        elif chart_type == "scatter":
            col_x = feature_x or df.select_dtypes(include=["number"]).columns[0]
            col_y = feature_y or df.select_dtypes(include=["number"]).columns[1]
            fig = self.vis_service.create_scatter_plot(df, x_col=col_x, y_col=col_y, target_column=target_column)
            specs["figure"] = fig.to_dict() if hasattr(fig, "to_dict") else str(fig)
        else:
            specs["available_charts"] = ["correlation_heatmap", "distribution", "pca_scatter", "scatter"]


        return specs

    def generate_report(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> Dict[str, Any]:
        """Generates standalone HTML report and returns file path and metadata."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        report_path, _ = self.eda_reporter.generate_html_report(
            df,
            dataset_id=dataset_id,
            target_column=target_column,
        )
        file_size_kb = round(Path(report_path).stat().st_size / 1024.0, 2) if Path(report_path).exists() else 0.0
        return {
            "dataset_id": dataset_id,
            "status": "SUCCESS",
            "report_path": str(report_path),
            "report_size_kb": file_size_kb,
        }
