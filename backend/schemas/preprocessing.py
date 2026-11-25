"""
CANCERWISE AI — Preprocessing and Data Wrangling Schemas
Pydantic schemas for data quality profiling, missing values, duplicates,
outliers, transformations, pipelines, and validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ImputationStrategy(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"


class OutlierMethod(str, Enum):
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ROBUST_COVARIANCE = "robust_covariance"


class ScalingStrategy(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NORMALIZER = "normalizer"
    POWER_YEO_JOHNSON = "power_yeo_johnson"
    POWER_BOX_COX = "power_box_cox"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    LOG1P = "log1p"
    PASSTHROUGH = "passthrough"


class EncodingStrategy(str, Enum):
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    PASSTHROUGH = "passthrough"


# ---------------------------------------------------------------------------
# Data Quality Schemas
# ---------------------------------------------------------------------------
class ColumnQualityIssue(BaseModel):
    column: str
    issue_type: str
    severity: str = "warning"  # info, warning, critical
    details: str
    affected_count: int = 0
    affected_percentage: float = 0.0


class DataQualityReport(BaseModel):
    dataset_name: str
    timestamp: str = Field(default_factory=_utc_now_str)
    total_records: int
    total_columns: int
    overall_quality_score: float = Field(..., ge=0.0, le=100.0)
    score_interpretation: str
    clinical_disclaimer: str = (
        "This score measures statistical and structural data hygiene only. "
        "It is NOT a clinical quality, diagnostic reliability, or oncological specimen assessment."
    )
    issues: List[ColumnQualityIssue] = Field(default_factory=list)
    missing_summary: Dict[str, Any] = Field(default_factory=dict)
    duplicate_rows_count: int = 0
    duplicate_identifiers_count: int = 0
    constant_columns: List[str] = Field(default_factory=list)
    near_zero_variance_columns: List[str] = Field(default_factory=list)
    outlier_summary: Dict[str, int] = Field(default_factory=dict)
    high_cardinality_columns: List[str] = Field(default_factory=list)
    inconsistent_dtype_columns: List[str] = Field(default_factory=list)
    potential_leakage_columns: List[str] = Field(default_factory=list)
    rules_applied: Dict[str, Any] = Field(default_factory=dict)

    @property
    def overall_cleanliness_score(self) -> float:
        return self.overall_quality_score

    @property
    def missingness(self):
        return type("Missingness", (), {"total_missing_cells": 0})()




# ---------------------------------------------------------------------------
# Missing Value Schemas
# ---------------------------------------------------------------------------
class ImputationRequest(BaseModel):
    dataset_id: str
    strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    columns: Optional[List[str]] = None
    fill_value: Optional[Union[float, int, str]] = None
    knn_neighbors: int = 5
    max_iter: int = 10
    add_indicator: bool = False
    target_column: Optional[str] = None


class ImputationComparisonItem(BaseModel):
    strategy: str
    column: str
    mean_before: Optional[float] = None
    mean_after: Optional[float] = None
    std_before: Optional[float] = None
    std_after: Optional[float] = None
    median_before: Optional[float] = None
    median_after: Optional[float] = None
    missing_before: int = 0
    missing_after: int = 0


# ---------------------------------------------------------------------------
# Duplicate Schemas
# ---------------------------------------------------------------------------
class DuplicateAuditReport(BaseModel):
    total_records: int
    exact_duplicates_count: int
    duplicate_identifier_count: int
    identifier_column: Optional[str] = None
    duplicate_record_indices: List[int] = Field(default_factory=list)
    duplicate_identifier_groups: Dict[str, List[int]] = Field(default_factory=dict)
    similar_records_count: int = 0
    clinical_preservation_warning: str = (
        "Records with identical features must not be purged automatically without identifier verification, "
        "as they may represent distinct clinical patients or serial biopsy encounters."
    )


# ---------------------------------------------------------------------------
# Outlier Schemas
# ---------------------------------------------------------------------------
class OutlierReport(BaseModel):
    method: OutlierMethod
    total_outliers: int
    outlier_percentage: float
    outlier_indices: List[int] = Field(default_factory=list)
    column_outlier_counts: Dict[str, int] = Field(default_factory=dict)
    threshold_params: Dict[str, Any] = Field(default_factory=dict)
    biological_extreme_disclaimer: str = (
        "Extreme values in oncological biomarkers and nuclear cytology morphometrics may indicate "
        "aggressive malignant phenotypes or rare biological variants rather than measurement errors. "
        "Do not purge without clinical review."
    )


# ---------------------------------------------------------------------------
# Transformation & Pipeline Schemas
# ---------------------------------------------------------------------------
class TransformationStepConfig(BaseModel):
    name: str
    step_type: str  # scaler, encoder, imputer, power, custom
    strategy: str
    columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingRecipe(BaseModel):
    name: str = "cancerwise_pipeline"
    version: str = "1.0.0"
    created_at: str = Field(default_factory=_utc_now_str)
    description: Optional[str] = None
    steps: List[TransformationStepConfig] = Field(default_factory=list)
    target_column: Optional[str] = None
    reversible: bool = True


class TransformationLogEntry(BaseModel):
    operation_id: str
    operation_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utc_now_str)
    input_version: str = "v1.0.0"
    output_version: str = "v1.1.0"
    input_shape: List[int] = Field(default_factory=list)
    output_shape: List[int] = Field(default_factory=list)
    records_affected: int = 0
    columns_affected: List[str] = Field(default_factory=list)
    execution_status: str = "SUCCESS"  # SUCCESS or FAILED
    duration_ms: float = 0.0
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Validation Schemas
# ---------------------------------------------------------------------------
class SchemaValidationReport(BaseModel):
    is_valid: bool
    timestamp: str = Field(default_factory=_utc_now_str)
    total_checks: int
    passed_checks: int
    failed_checks: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    column_validations: Dict[str, bool] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
