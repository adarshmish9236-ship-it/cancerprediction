"""
CANCERWISE AI — Phase 9 API Request & Response Schemas
Provides validated Pydantic models for all v1 REST endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from backend.schemas.modeling import AlgorithmType, ResamplingStrategy, TuningStrategy
from backend.schemas.advanced_learning import (
    SemiSupervisedMethod,
    MulticlassStrategy,
    OneClassMethod,
    RegressionMethod,
)


# ---------------------------------------------------------------------------
# Data Wrangling Schemas
# ---------------------------------------------------------------------------
class WranglingAnalyzeRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    target_column: Optional[str] = Field("diagnosis", description="Target diagnostic column.")


class WranglingCleanRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    imputation_strategy: str = Field("median", description="Imputation strategy: 'mean', 'median', 'mode', 'knn'.")
    drop_duplicates: bool = Field(True, description="Whether to drop exact duplicate observations.")
    target_column: Optional[str] = Field("diagnosis", description="Target diagnostic column.")


class WranglingCleanResponse(BaseModel):
    dataset_id: str
    status: str
    initial_shape: List[int]
    cleaned_shape: List[int]
    duplicates_removed: int
    imputation_strategy: str
    missing_stats: Dict[str, Any]
    log_id: str


class WranglingTransformRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    scaling_strategy: str = Field("standard", description="Scaling method: 'standard', 'minmax', 'robust'.")
    encoding_strategy: str = Field("onehot", description="Encoding method: 'onehot', 'ordinal'.")
    target_column: Optional[str] = Field("diagnosis", description="Target diagnostic column.")


class WranglingTransformResponse(BaseModel):
    dataset_id: str
    status: str
    original_features: int
    transformed_features: int
    scaling_applied: str
    encoding_applied: str
    log_id: str


# ---------------------------------------------------------------------------
# EDA Schemas
# ---------------------------------------------------------------------------
class EDAAnalyzeRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    target_column: Optional[str] = Field("diagnosis", description="Target column for hypothesis testing.")


class EDAAnalyzeResponse(BaseModel):
    dataset_id: str
    overview: Dict[str, Any]
    multivariate: Dict[str, Any]
    top_statistical_tests: List[Dict[str, Any]]
    total_significant_features: int


class EDAVisualizationResponse(BaseModel):
    dataset_id: str
    chart_type: str
    figure: Optional[Dict[str, Any]] = None
    available_charts: Optional[List[str]] = None


class EDAReportResponse(BaseModel):
    dataset_id: str
    status: str
    report_path: str
    report_size_kb: float


# ---------------------------------------------------------------------------
# Feature Engineering & Selection Schemas
# ---------------------------------------------------------------------------
class FeatureEngineerRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    apply_domain_features: bool = Field(True, description="Compute cytology shape and border indices.")
    apply_ratios: bool = Field(True, description="Compute perimeter-to-area and dynamic expansion ratios.")
    apply_polynomials: bool = Field(False, description="Generate polynomial interaction terms.")
    apply_log_transforms: bool = Field(False, description="Apply log1p transformations to skewed features.")
    apply_aggregations: bool = Field(True, description="Compute composite row-wise statistics.")


class FeatureSelectRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    method: str = Field("mutual_info", description="Selection algorithm: 'mutual_info', 'chi2', 'f_classif', 'rfe', 'lasso'.")
    k: int = Field(10, ge=1, le=100, description="Number of top features to retain.")
    target_column: str = Field("diagnosis", description="Target diagnostic label column.")


class FeatureSelectResponse(BaseModel):
    dataset_id: str
    method: str
    target_column: str
    original_feature_count: int
    selected_feature_count: int
    selected_features: List[str]
    rankings: Dict[str, float]


class FeatureReduceRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    method: str = Field("pca", description="Reduction method: 'pca', 'kpca', 'svd', 'umap'.")
    n_components: int = Field(2, ge=2, le=20, description="Target latent dimension.")
    target_column: Optional[str] = Field("diagnosis", description="Optional label column for coloring.")


# ---------------------------------------------------------------------------
# Machine Learning Schemas
# ---------------------------------------------------------------------------
class ModelTrainRequest(BaseModel):
    dataset_id: str = Field("uci_wdbc", description="Registered dataset identifier.")
    algorithm: AlgorithmType = Field(AlgorithmType.RANDOM_FOREST, description="Classification algorithm.")
    target_column: str = Field("diagnosis", description="Target class column.")
    resampling: ResamplingStrategy = Field(ResamplingStrategy.SMOTE, description="In-fold imbalance treatment.")
    tuning: TuningStrategy = Field(TuningStrategy.NONE, description="Hyperparameter search strategy.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters.")
    cv_folds: int = Field(5, ge=2, le=10, description="Cross-validation folds.")
    async_job: bool = Field(False, description="Run training as a non-blocking background job.")


class ClinicalPredictionRequest(BaseModel):
    features: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="Patient feature observation dictionary or list of observations.",
        json_schema_extra={"example": {"mean radius": 14.5, "mean texture": 19.2, "mean perimeter": 92.4, "mean area": 654.3}},
    )


# ---------------------------------------------------------------------------
# Advanced Learning Schemas
# ---------------------------------------------------------------------------
class AdvancedSemiSupervisedRequest(BaseModel):
    method: SemiSupervisedMethod = Field(SemiSupervisedMethod.LABEL_SPREADING, description="Algorithm.")
    labeled_fractions: Optional[List[float]] = Field(None, description="Varying labeled data fractions (0.10 to 0.70).")
    async_job: bool = Field(False, description="Execute asynchronously as background job.")


class AdvancedMultitaskRequest(BaseModel):
    epochs: int = Field(40, ge=10, le=200, description="Training epochs.")
    learning_rate: float = Field(0.005, ge=0.0001, le=0.1, description="Adam optimizer learning rate.")
    async_job: bool = Field(False, description="Execute asynchronously as background job.")


class AdvancedMulticlassRequest(BaseModel):
    strategy: MulticlassStrategy = Field(MulticlassStrategy.MULTINOMIAL_LOGISTIC, description="Multiclass strategy.")
    async_job: bool = Field(False, description="Execute asynchronously as background job.")


class AdvancedOneClassRequest(BaseModel):
    method: OneClassMethod = Field(OneClassMethod.ISOLATION_FOREST, description="One-class novelty detector.")
    contamination: float = Field(0.05, ge=0.01, le=0.5, description="Expected normative contamination.")
    async_job: bool = Field(False, description="Execute asynchronously as background job.")


class AdvancedRegressionRequest(BaseModel):
    method: RegressionMethod = Field(RegressionMethod.RIDGE, description="Continuous regression algorithm.")
    async_job: bool = Field(False, description="Execute asynchronously as background job.")


class AdvancedKernelRequest(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Explainability Schemas
# ---------------------------------------------------------------------------
class ExplainabilitySHAPRequest(BaseModel):
    model_id: str = Field("baseline_rf", description="Saved model identifier.")
    type: str = Field("global", description="'global' for mean abs SHAP, or 'local' for instance waterfall.")
    sample_index: int = Field(0, ge=0, description="Test sample index for local explanation.")
    num_features: int = Field(8, ge=1, le=30, description="Top features to report.")
    max_samples: int = Field(60, ge=10, le=300, description="Background sample budget.")


class ExplainabilityLIMERequest(BaseModel):
    model_id: str = Field("baseline_rf", description="Saved model identifier.")
    sample_index: int = Field(0, ge=0, description="Test sample index to explain.")
    num_features: int = Field(8, ge=1, le=30, description="Top surrogate rules to report.")
