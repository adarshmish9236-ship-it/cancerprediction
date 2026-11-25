"""
CANCERWISE AI — Exploratory Data Analysis REST Endpoints
Provides interactive routes for overview, distributions, correlations,
hypothesis testing, visualizations, and automated HTML reporting.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.core.logging_setup import get_logger
from backend.schemas.eda import (
    DatasetOverview,
    UnivariateStats,
    BivariateAnalysisResult,
    MultivariateStats,
    StatisticalTestResult,
    WDBCSpecificFindings,
)
from backend.schemas.phase9_schemas import (
    EDAAnalyzeRequest,
    EDAAnalyzeResponse,
    EDAVisualizationResponse,
    EDAReportResponse,
)
from backend.services.eda_service import EDAService

logger = get_logger("api.eda")
router = APIRouter(prefix="/eda", tags=["Exploratory Data Analysis & Statistical Mining"])

_eda_service = EDAService()


# ---------------------------------------------------------------------------
# Section 2 EDA APIs
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=EDAAnalyzeResponse, summary="Run Comprehensive EDA Analysis")
def analyze_eda(req: EDAAnalyzeRequest):
    """Executes full exploratory data analysis, multivariate correlation, and hypothesis tests."""
    return _eda_service.run_comprehensive_analysis(dataset_id=req.dataset_id, target_column=req.target_column)


@router.get("/visualizations", response_model=EDAVisualizationResponse, summary="Get EDA Visualizations")
def get_eda_visualizations(
    dataset_id: str = Query("uci_wdbc", description="Dataset identifier"),
    chart_type: str = Query("correlation_heatmap", description="Chart type: correlation_heatmap, distribution, pca_scatter, scatter"),
    feature_x: Optional[str] = Query(None, description="Feature for x axis"),
    feature_y: Optional[str] = Query(None, description="Feature for y axis"),
    target_column: Optional[str] = Query("diagnosis", description="Target label column"),
):
    """Generates interactive Plotly JSON visualization specs."""
    if hasattr(dataset_id, "default"): dataset_id = dataset_id.default
    if hasattr(chart_type, "default"): chart_type = chart_type.default
    if hasattr(feature_x, "default"): feature_x = feature_x.default
    if hasattr(feature_y, "default"): feature_y = feature_y.default
    if hasattr(target_column, "default"): target_column = target_column.default

    return _eda_service.get_visualizations(
        dataset_id=dataset_id,
        chart_type=chart_type,
        feature_x=feature_x,
        feature_y=feature_y,
        target_column=target_column,
    )


@router.get("/report", response_model=EDAReportResponse, summary="Generate EDA Report")
def get_eda_report(
    dataset_id: str = Query("uci_wdbc", description="Dataset identifier"),
    target_column: Optional[str] = Query("diagnosis", description="Target label column"),
):
    """Compiles publication-grade HTML EDA report and returns artifact location."""
    if hasattr(dataset_id, "default"): dataset_id = dataset_id.default
    if hasattr(target_column, "default"): target_column = target_column.default

    return _eda_service.generate_report(dataset_id=dataset_id, target_column=target_column)



# ---------------------------------------------------------------------------
# Backward Compatibility Endpoints
# ---------------------------------------------------------------------------
@router.get("/overview/{dataset_id}", response_model=DatasetOverview, summary="Dataset Structural Overview")
def get_dataset_overview(dataset_id: str, target_column: Optional[str] = "diagnosis"):
    return _eda_service.get_overview(dataset_id=dataset_id, target_column=target_column)


@router.get("/univariate/{dataset_id}/{feature_name}", response_model=UnivariateStats, summary="Univariate Feature Stats")
def get_feature_univariate(dataset_id: str, feature_name: str):
    return _eda_service.get_univariate_profile(dataset_id=dataset_id, feature_name=feature_name)


@router.get("/bivariate/{dataset_id}", response_model=BivariateAnalysisResult, summary="Bivariate Feature Stats")
def get_feature_bivariate(
    dataset_id: str,
    feature_x: str = Query(..., description="First feature name"),
    feature_y: str = Query(..., description="Second feature name"),
    target_column: Optional[str] = "diagnosis",
):
    return _eda_service.get_bivariate_profile(
        dataset_id=dataset_id,
        feature_x=feature_x,
        feature_y=feature_y,
        target_column=target_column,
    )


@router.get("/multivariate/{dataset_id}", response_model=MultivariateStats, summary="Multivariate Correlation & PCA")
def get_multivariate_summary(dataset_id: str, target_column: Optional[str] = "diagnosis"):
    return _eda_service.get_multivariate_profile(dataset_id=dataset_id, target_column=target_column)


@router.get("/statistical-tests/{dataset_id}", response_model=List[StatisticalTestResult], summary="Hypothesis Testing")
def run_statistical_hypothesis_tests(dataset_id: str, target_column: str = "diagnosis"):
    return _eda_service.run_statistical_tests(dataset_id=dataset_id, target_column=target_column)


@router.get("/cancer-findings/{dataset_id}", response_model=WDBCSpecificFindings, summary="Cancer Morphometrics Findings")
def get_cancer_morphometrics_findings(dataset_id: str, target_column: str = "diagnosis"):
    return _eda_service.analyze_cancer_morphometrics(dataset_id=dataset_id, target_column=target_column)


@router.post("/generate-report/{dataset_id}", summary="Generate HTML Report (POST)")
def generate_eda_report_post(dataset_id: str, target_column: Optional[str] = "diagnosis"):
    res = _eda_service.generate_report(dataset_id=dataset_id, target_column=target_column)
    return {"status": "SUCCESS", "report_path": res["report_path"]}


# Alias for backwards compatibility
generate_eda_report = generate_eda_report_post

