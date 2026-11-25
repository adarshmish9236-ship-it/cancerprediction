"""
CANCERWISE AI — Exploratory Data Analysis & Statistical Mining Schemas
Pydantic schemas for dataset overviews, univariate/bivariate/multivariate profiling,
hypothesis testing, cancer-specific morphometric findings, and automated reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataset Overview Schemas
# ---------------------------------------------------------------------------
class DatasetOverview(BaseModel):
    dataset_id: str
    dataset_name: str
    total_records: int
    total_features: int
    numerical_features_count: int
    categorical_features_count: int
    numerical_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    missing_value_cells: int
    missing_value_percentage: float
    duplicate_records_count: int
    duplicate_records_percentage: float
    target_column: Optional[str] = None
    target_distribution: Dict[str, int] = Field(default_factory=dict)
    target_proportions: Dict[str, float] = Field(default_factory=dict)
    memory_utilization_bytes: int
    memory_utilization_mb: float
    summary_text: str


# ---------------------------------------------------------------------------
# Univariate Profiling Schemas
# ---------------------------------------------------------------------------
class UnivariateStats(BaseModel):
    feature_name: str
    dtype: str
    count: int
    missing_count: int
    missing_percentage: float
    distinct_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    iqr: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    normality_test_name: str = "Shapiro-Wilk"
    normality_statistic: Optional[float] = None
    normality_p_value: Optional[float] = None
    is_normally_distributed: Optional[bool] = None
    top_categories: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bivariate Profiling Schemas
# ---------------------------------------------------------------------------
class BivariateAnalysisResult(BaseModel):
    feature_x: str
    feature_y: str
    pearson_r: Optional[float] = None
    pearson_p: Optional[float] = None
    spearman_rho: Optional[float] = None
    spearman_p: Optional[float] = None
    kendall_tau: Optional[float] = None
    kendall_p: Optional[float] = None
    relationship_strength: str = "None"
    scatter_sample: List[Dict[str, Any]] = Field(default_factory=list)
    group_statistics: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    @property
    def pearson_correlation(self) -> Optional[float]:
        return self.pearson_r

    @property
    def spearman_correlation(self) -> Optional[float]:
        return self.spearman_rho


# ---------------------------------------------------------------------------
# Multivariate Profiling Schemas
# ---------------------------------------------------------------------------
class MultivariateStats(BaseModel):
    features_included: List[str] = Field(default_factory=list)
    correlation_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    pca_explained_variance_ratio: List[float] = Field(default_factory=list)
    pca_total_variance_explained: float = 0.0
    top_correlated_pairs: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistical Hypothesis Testing Schemas
# ---------------------------------------------------------------------------
class StatisticalTestResult(BaseModel):
    test_name: str  # Mann-Whitney U, ANOVA, Chi-Square, Kruskal-Wallis, Mutual Information
    feature_name: str
    target_column: str
    statistic: float
    p_value: Optional[float] = None
    p_value_corrected: Optional[float] = None
    correction_method: str = "None"
    is_significant: bool = False
    effect_size_name: Optional[str] = None  # e.g. Cohen's d, Rank Biserial, Cramér's V, eta-squared
    effect_size: Optional[float] = None
    assumptions_checked: List[str] = Field(default_factory=list)
    assumption_warnings: List[str] = Field(default_factory=list)
    interpretation: str = ""


# ---------------------------------------------------------------------------
# Cancer Dataset Specific Findings (WDBC)
# ---------------------------------------------------------------------------
class WDBCFeatureDifference(BaseModel):
    feature: str
    benign_mean: float
    malignant_mean: float
    mean_difference_percent: float
    cohens_d: float
    mann_whitney_u: float
    p_value: float
    significance_rank: int


class WDBCSpecificFindings(BaseModel):
    dataset_name: str = "Wisconsin Diagnostic Breast Cancer"
    total_samples: int = 569
    benign_count: int = 357
    malignant_count: int = 212
    imbalance_ratio: float = 1.68  # 357 / 212
    top_differentiating_features: List[WDBCFeatureDifference] = Field(default_factory=list)
    collinear_feature_clusters: List[Dict[str, Any]] = Field(default_factory=list)
    morphometric_observations: List[str] = Field(default_factory=list)
    non_causal_disclaimer: str = (
        "Statistical associations, hypothesis test significance, and effect sizes in this exploratory analysis "
        "indicate diagnostic discriminative utility and cytological morphology differences. "
        "They DO NOT establish biological etiology, disease causality, or oncogenic progression mechanisms."
    )


# ---------------------------------------------------------------------------
# Automated Report Schemas
# ---------------------------------------------------------------------------
class EDAReportMetadata(BaseModel):
    report_id: str
    dataset_id: str
    dataset_name: str
    created_at: str = Field(default_factory=_utc_now_str)
    report_html_path: str
    total_records: int
    total_features: int
    findings_count: int
