"""
CANCERWISE AI — Explainability, Model Validation & Governance REST Endpoints
Provides interactive routes for SHAP global/local explanations, LIME surrogate rules,
PDP/ICE curves, clinical diagnostic metrics, probability calibration, threshold trade-offs,
perturbation robustness audits, and research report generation.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.core.logging_setup import get_logger
from backend.schemas.explainability import (
    CalibrationMethod,
    CalibrationResult,
    LocalSHAPExplanation,
    SHAPExplanationSummary,
    LIMEExplanationResult,
    PartialDependenceResult,
    HealthcareValidationMetrics,
    ThresholdAnalysisItem,
    RobustnessAuditReport,
    ModelGovernanceRecord,
    ResearchReportMetadata,
)
from backend.schemas.phase9_schemas import (
    ExplainabilitySHAPRequest,
    ExplainabilityLIMERequest,
)
from backend.services.explainability_service import ExplainabilityService
from backend.services.model_evaluation_service import ModelEvaluationService

logger = get_logger("api.explainability")
router = APIRouter(prefix="/explainability", tags=["Explainable AI, Clinical Validation & Governance"])

_xai_service = ExplainabilityService()
_eval_service = ModelEvaluationService()


# ---------------------------------------------------------------------------
# Section 2 Explainability APIs
# ---------------------------------------------------------------------------
@router.post("/shap", summary="Calculate SHAP Explanation")
def calculate_shap(
    req: ExplainabilitySHAPRequest,
) -> Union[SHAPExplanationSummary, LocalSHAPExplanation]:
    """
    Computes Shapley additive feature attributions.
    Supports 'global' mean absolute SHAP values and 'local' patient waterfall contributions.
    """
    if req.type.lower() == "local":
        return _xai_service.explain_local_shap(
            model_id=req.model_id,
            sample_index=req.sample_index,
            num_features=req.num_features,
        )
    return _xai_service.explain_global_shap(
        model_id=req.model_id,
        max_samples=req.max_samples,
    )


@router.post("/lime", response_model=LIMEExplanationResult, summary="Calculate LIME Local Surrogate Explanation")
def calculate_lime(req: ExplainabilityLIMERequest):
    """Fits an interpretable sparse linear surrogate model explaining an individual patient prediction."""
    return _xai_service.explain_local_lime(
        model_id=req.model_id,
        sample_index=req.sample_index,
        num_features=req.num_features,
    )


@router.get("/{model_id}", summary="Get Model Explainability & Governance Summary")
def get_model_explainability(model_id: str) -> Dict[str, Any]:
    """Retrieves explainability availability, top contributing SHAP features, and governance summary."""
    return _xai_service.get_model_explainability(model_id)


# ---------------------------------------------------------------------------
# Phase 8 Governance & Validation Endpoints (Preserved for compatibility)
# ---------------------------------------------------------------------------
class LocalExplanationRequest(BaseModel):
    sample_index: int = Field(0, description="Index of sample in test set to explain.")
    num_features: int = Field(8, description="Number of top features to report.")


class PDPRequest(BaseModel):
    feature_name: str = Field("mean radius", description="Feature name to compute PDP and ICE for.")
    grid_resolution: int = Field(25, description="Number of grid evaluation points.")


class CalibrationRequest(BaseModel):
    method: CalibrationMethod = Field(CalibrationMethod.PLATT_SIGMOID, description="Calibration method.")


@router.get("/governance/list", summary="List All Governed Model Audits")
def list_governance_records() -> List[Dict[str, Any]]:
    return _eval_service.list_governance_records()


@router.get("/governance/{model_id}", response_model=ModelGovernanceRecord, summary="Get Model Governance Record")
def get_governance_record(model_id: str):
    return _eval_service.get_governance_record(model_id)


@router.post("/shap-global/{model_id}", response_model=SHAPExplanationSummary, summary="Calculate Global SHAP (Legacy)")
def calculate_shap_global(
    model_id: str,
    max_samples: int = Query(80, description="Max background samples"),
):
    return _xai_service.explain_global_shap(model_id=model_id, max_samples=max_samples)


@router.post("/shap-local/{model_id}", response_model=LocalSHAPExplanation, summary="Calculate Local SHAP (Legacy)")
def calculate_shap_local(model_id: str, req: LocalExplanationRequest):
    return _xai_service.explain_local_shap(
        model_id=model_id,
        sample_index=req.sample_index,
        num_features=req.num_features,
    )


@router.post("/lime-local/{model_id}", response_model=LIMEExplanationResult, summary="Calculate LIME Local (Legacy)")
def calculate_lime_local(model_id: str, req: LocalExplanationRequest):
    return _xai_service.explain_local_lime(
        model_id=model_id,
        sample_index=req.sample_index,
        num_features=req.num_features,
    )


@router.post("/pdp-ice/{model_id}", response_model=PartialDependenceResult, summary="Compute Partial Dependence & ICE")
def compute_partial_dependence(model_id: str, req: PDPRequest):
    return _xai_service.compute_pdp(
        model_id=model_id,
        feature_name=req.feature_name,
        grid_resolution=req.grid_resolution,
    )


@router.post("/clinical-metrics/{model_id}", response_model=HealthcareValidationMetrics, summary="Evaluate Clinical Metrics")
def evaluate_clinical_metrics(
    model_id: str,
    compute_bootstrap: bool = Query(True),
):
    return _eval_service.evaluate_model(model_id=model_id, compute_bootstrap=compute_bootstrap)


@router.post("/calibration/{model_id}", response_model=CalibrationResult, summary="Probability Calibration")
def evaluate_calibration(model_id: str, req: CalibrationRequest):
    return _eval_service.evaluate_calibration(model_id=model_id, method=req.method)


@router.post("/thresholds/{model_id}", response_model=List[ThresholdAnalysisItem], summary="Decision Thresholds")
def evaluate_threshold_tradeoffs(model_id: str):
    return _eval_service.evaluate_decision_thresholds(model_id=model_id)


@router.post("/robustness-audit/{model_id}", response_model=RobustnessAuditReport, summary="Robustness Stress Test")
def audit_robustness(model_id: str):
    return _eval_service.audit_robustness(model_id=model_id)


@router.post("/generate-report/{model_id}", response_model=ResearchReportMetadata, summary="Generate Research Report")
def generate_research_report(model_id: str):
    return _eval_service.generate_research_report(model_id=model_id)
