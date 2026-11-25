"""
CANCERWISE AI — Data Wrangling REST Endpoints
Provides interactive routes for quality profiling, imputation, deduplication,
outlier management, scaling/encoding, and audit history.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.core.logging_setup import get_logger
from backend.schemas.preprocessing import (
    DataQualityReport,
    ImputationComparisonItem,
    DuplicateAuditReport,
    OutlierMethod,
    OutlierReport,
    TransformationLogEntry,
    SchemaValidationReport,
)
from backend.schemas.phase9_schemas import (
    WranglingAnalyzeRequest,
    WranglingCleanRequest,
    WranglingCleanResponse,
    WranglingTransformRequest,
    WranglingTransformResponse,
)
from backend.services.data_wrangling_service import DataWranglingService
from backend.services.data_validation_service import DataValidationService

logger = get_logger("api.wrangling")
router = APIRouter(prefix="/wrangling", tags=["Data Wrangling & Preprocessing"])

_wrangling_service = DataWranglingService()
_validation_service = DataValidationService()


# ---------------------------------------------------------------------------
# Section 2 Data Wrangling APIs
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=DataQualityReport, summary="Analyze Dataset Quality")
def analyze_dataset_quality(req: WranglingAnalyzeRequest):
    """Generates automated data hygiene profile measuring statistical cleanliness."""
    return _wrangling_service.analyze_quality(dataset_id=req.dataset_id, target_column=req.target_column)


@router.post("/clean", response_model=WranglingCleanResponse, summary="Clean Dataset")
def clean_dataset(req: WranglingCleanRequest):
    """
    Cleans missing values using designated imputation strategy and removes duplicate rows.
    Logs transformation audit trail.
    """
    return _wrangling_service.clean_data(
        dataset_id=req.dataset_id,
        imputation_strategy=req.imputation_strategy,
        drop_duplicates=req.drop_duplicates,
        target_column=req.target_column,
    )


@router.post("/transform", response_model=WranglingTransformResponse, summary="Transform Features")
def transform_dataset(req: WranglingTransformRequest):
    """Applies scaling and categorical encoding to features with parameter preservation."""
    return _wrangling_service.transform_data(
        dataset_id=req.dataset_id,
        scaling_strategy=req.scaling_strategy,
        encoding_strategy=req.encoding_strategy,
        target_column=req.target_column,
    )


@router.get("/history", response_model=List[TransformationLogEntry], summary="Get Transformation History")
def get_transformation_history(limit: int = Query(50, ge=1, le=500)):
    """Retrieves transformation audit log entries with UUIDs."""
    return _wrangling_service.get_history(limit=limit)


# ---------------------------------------------------------------------------
# Backward Compatibility Endpoints
# ---------------------------------------------------------------------------
@router.get("/quality-profile/{dataset_id}", response_model=DataQualityReport, summary="Quality Profile by Dataset ID")
def get_quality_profile(dataset_id: str, target_column: Optional[str] = "diagnosis"):
    return _wrangling_service.analyze_quality(dataset_id=dataset_id, target_column=target_column)


@router.post("/compare-imputation/{dataset_id}", response_model=List[ImputationComparisonItem], summary="Compare Imputation Strategies")
def compare_imputation_strategies(dataset_id: str, target_column: Optional[str] = "diagnosis"):
    return _wrangling_service.compare_imputation_strategies(dataset_id=dataset_id, target_column=target_column)


@router.get("/audit-duplicates/{dataset_id}", response_model=DuplicateAuditReport, summary="Audit Duplicates")
def audit_dataset_duplicates(dataset_id: str, id_column: Optional[str] = None):
    return _wrangling_service.audit_duplicates(dataset_id=dataset_id, id_column=id_column)


@router.get("/detect-outliers/{dataset_id}", response_model=OutlierReport, summary="Detect Outliers")
def detect_dataset_outliers(
    dataset_id: str,
    method: OutlierMethod = OutlierMethod.IQR,
    target_column: Optional[str] = "diagnosis",
):
    return _wrangling_service.detect_outliers(dataset_id=dataset_id, method=method, target_column=target_column)


@router.get("/validate-schema/{dataset_id}", response_model=SchemaValidationReport, summary="Validate Schema")
def validate_dataset_schema(dataset_id: str, target_column: Optional[str] = "diagnosis"):
    return _validation_service.validate_schema(dataset_id=dataset_id, target_column=target_column)
