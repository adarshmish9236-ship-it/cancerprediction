"""
CANCERWISE AI — Feature Engineering & Selection Schemas
Pydantic models for derived features, selection results, model comparison,
explainable importance rankings, and dimensionality reduction.
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
class FilterSelectionMethod(str, Enum):
    VARIANCE_THRESHOLD = "variance_threshold"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    CHI2 = "chi2"
    ANOVA_F = "anova_f"
    MUTUAL_INFO = "mutual_info"


class WrapperSelectionMethod(str, Enum):
    RFE = "rfe"
    RFECV = "rfecv"
    SFS_FORWARD = "sfs_forward"
    SFS_BACKWARD = "sfs_backward"


class EmbeddedSelectionMethod(str, Enum):
    L1_LOGISTIC = "l1_logistic"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"


class DimensionalityMethod(str, Enum):
    PCA = "pca"
    KERNEL_PCA = "kernel_pca"
    TRUNCATED_SVD = "truncated_svd"
    UMAP = "umap"


# ---------------------------------------------------------------------------
# Feature Engineering Schemas
# ---------------------------------------------------------------------------
class DerivedFeatureDefinition(BaseModel):
    feature_name: str
    base_features: List[str]
    transformation_type: str  # polynomial, ratio, domain, statistical, log
    formula: str
    domain_rationale: str
    created_at: str = Field(default_factory=_utc_now_str)


class FeatureEngineeringResult(BaseModel):
    status: str = "SUCCESS"
    dataset_name: str
    original_feature_count: int
    derived_feature_count: int
    total_feature_count: int
    derived_definitions: List[DerivedFeatureDefinition] = Field(default_factory=list)
    execution_time_seconds: float = 0.0

    @property
    def engineered_feature_count(self) -> int:
        return self.derived_feature_count

    @property
    def new_features_created(self) -> List[str]:
        return self.derived_feature_names

    @property
    def derived_feature_names(self) -> List[str]:
        return [d.feature_name for d in self.derived_definitions]



# ---------------------------------------------------------------------------
# Feature Selection Schemas
# ---------------------------------------------------------------------------
class FeatureRankingItem(BaseModel):
    feature_name: str
    rank: int
    score: float
    method: str
    is_selected: bool = True


class SelectionComparisonItem(BaseModel):
    method: str
    category: str  # filter, wrapper, embedded
    original_feature_count: int
    selected_feature_count: int
    selected_features: List[str] = Field(default_factory=list)
    execution_time_seconds: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_f1_mean: float
    cv_f1_std: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    data_leakage_prevented: bool = True


# ---------------------------------------------------------------------------
# Explainability Schemas
# ---------------------------------------------------------------------------
class FeatureImportanceSummary(BaseModel):
    model_type: str
    importance_type: str  # Permutation, SHAP, Native
    rankings: List[FeatureRankingItem] = Field(default_factory=list)
    consensus_rankings: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dimensionality Reduction Schemas
# ---------------------------------------------------------------------------
class DimensionalityReductionResult(BaseModel):
    method: DimensionalityMethod
    original_dimensions: int
    reduced_dimensions: int
    explained_variance_ratio: List[float] = Field(default_factory=list)
    total_explained_variance: float = 0.0
    execution_time_seconds: float = 0.0
    downstream_cv_accuracy: float = 0.0
    downstream_cv_roc_auc: float = 0.0
    scree_plot_data: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def n_components(self) -> int:
        return self.reduced_dimensions

