"""
CANCERWISE AI — Explainability & Model Validation Schemas
Pydantic schemas for SHAP, LIME, Partial Dependence, Healthcare Metrics,
Calibration Curves, Threshold Analysis, Robustness Audits, and Model Governance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class CalibrationMethod(str, Enum):
    UNCALIBRATED = "uncalibrated"
    PLATT_SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


# ---------------------------------------------------------------------------
# SHAP Schemas
# ---------------------------------------------------------------------------
class SHAPFeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_direction: str  # "Increases Risk (Malignant)" or "Decreases Risk (Benign)"


class LocalSHAPExplanation(BaseModel):
    sample_index: int
    base_value: float
    predicted_probability: float
    predicted_class: int
    predicted_label: str
    contributions: List[SHAPFeatureContribution]
    governance_notice: str = (
        "SHAP values quantify additive mathematical feature contributions to the log-odds model output. "
        "They indicate predictive biomarker association, NOT direct biological etiology or disease causation."
    )

    @property
    def waterfall_features(self) -> List[SHAPFeatureContribution]:
        return self.contributions


class GlobalSHAPImportanceItem(BaseModel):
    feature_name: str
    rank: int
    mean_abs_shap: float
    correlation_with_risk: float


class SHAPExplanationSummary(BaseModel):
    model_id: str
    explainer_type: str
    num_samples_analyzed: int
    global_importances: List[GlobalSHAPImportanceItem]
    local_sample_explanation: Optional[LocalSHAPExplanation] = None
    governance_notice: str = (
        "SHAP values quantify additive mathematical feature contributions to the log-odds model output. "
        "They indicate predictive biomarker association, NOT direct biological etiology or disease causation."
    )



# ---------------------------------------------------------------------------
# LIME Schemas
# ---------------------------------------------------------------------------
class LIMEFeatureRule(BaseModel):
    rule: str
    feature_name: str
    weight: float
    direction: str  # "Supports Malignant" or "Supports Benign"


class LIMEExplanationResult(BaseModel):
    model_id: str
    sample_index: int
    predicted_class: int
    predicted_label: str
    probability_benign: float
    probability_malignant: float
    local_intercept: float
    rules: List[LIMEFeatureRule]
    explanation_fidelity_score: float  # R^2 of surrogate model


# ---------------------------------------------------------------------------
# Partial Dependence & ICE Schemas
# ---------------------------------------------------------------------------
class PartialDependenceResult(BaseModel):
    feature_name: str
    grid_values: List[float]
    average_partial_dependence: List[float]
    ice_curves: List[List[float]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Healthcare Validation & Prevalence Schemas
# ---------------------------------------------------------------------------
class PrevalenceImpactItem(BaseModel):
    hypothetical_prevalence_percent: float
    positive_predictive_value_ppv: float
    negative_predictive_value_npv: float
    clinical_interpretation: str


class HealthcareValidationMetrics(BaseModel):
    sensitivity_recall: float
    specificity: float
    false_negative_rate_fnr: float
    false_positive_rate_fpr: float
    positive_predictive_value_ppv: float
    negative_predictive_value_npv: float
    empirical_cohort_prevalence: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    matthews_corr_coef: float
    balanced_accuracy: float
    confusion_matrix: List[List[int]]
    confidence_intervals_95: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    prevalence_sensitivity_analysis: List[PrevalenceImpactItem] = Field(default_factory=list)

    @property
    def sensitivity(self) -> float:
        return self.sensitivity_recall


# ---------------------------------------------------------------------------
# Calibration & Threshold Schemas
# ---------------------------------------------------------------------------
class CalibrationPoint(BaseModel):
    mean_predicted_probability: float
    fraction_of_positives: float
    sample_count: int


class CalibrationResult(BaseModel):
    method: CalibrationMethod
    brier_score: float
    expected_calibration_error_ece: float
    curve_points: List[CalibrationPoint]


class ThresholdAnalysisItem(BaseModel):
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    fnr: float
    fpr: float
    f1_score: float
    balanced_accuracy: float


# ---------------------------------------------------------------------------
# Robustness Schemas
# ---------------------------------------------------------------------------
class RobustnessDegradationStep(BaseModel):
    perturbation_level: float
    accuracy: float
    roc_auc: float
    f1_score: float
    performance_drop_percent: float


class RobustnessAuditReport(BaseModel):
    model_id: str
    noise_sensitivity: List[RobustnessDegradationStep]
    missing_value_sensitivity: List[RobustnessDegradationStep]
    distribution_shift_gap: float
    adversarial_boundary_sensitivity: float
    resilience_rating: str  # "High Resilience", "Moderate Resilience", "Sensitive"
    medical_disclaimer: str = (
        "Robustness stress tests measure computational stability against artificial perturbations. "
        "They DO NOT prove real-world clinical generalizability or diagnostic immunity to hospital demographic shifts."
    )


# ---------------------------------------------------------------------------
# Governance & Report Metadata
# ---------------------------------------------------------------------------
class ModelGovernanceRecord(BaseModel):
    model_id: str
    model_version: str
    algorithm: str
    dataset_version: str
    dataset_hash: str
    feature_count: int
    selected_features: List[str]
    hyperparameters: Dict[str, Any]
    training_timestamp: str
    mlflow_run_id: Optional[str] = None
    software_dependencies: Dict[str, str]
    validation_metrics: Dict[str, float]


class ResearchReportMetadata(BaseModel):
    report_id: str
    model_id: str
    report_path_html: str
    report_path_md: str
    generated_at: str = Field(default_factory=_utc_now_str)
