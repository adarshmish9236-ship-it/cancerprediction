"""
CANCERWISE AI — Feature Engineering & Selection REST Endpoints
Provides interactive routes for domain feature synthesis, filter/wrapper/embedded selection,
cross-validation comparisons, and dimensionality reduction.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.core.logging_setup import get_logger
from backend.schemas.feature_engineering import (
    FeatureEngineeringResult,
    SelectionComparisonItem,
    DimensionalityReductionResult,
)
from backend.schemas.phase9_schemas import (
    FeatureEngineerRequest,
    FeatureSelectRequest,
    FeatureSelectResponse,
    FeatureReduceRequest,
)
from backend.services.feature_engineering_service import FeatureEngineeringService
from backend.services.feature_selection_service import FeatureSelectionService

logger = get_logger("api.features")
router = APIRouter(prefix="/features", tags=["Feature Engineering & Selection"])

_eng_service = FeatureEngineeringService()
_sel_service = FeatureSelectionService()


# ---------------------------------------------------------------------------
# Section 2 Feature APIs (Supporting both Request models and kwargs)
# ---------------------------------------------------------------------------
@router.post("/engineer", response_model=FeatureEngineeringResult, summary="Engineer Features")
def engineer_features(
    req: Optional[FeatureEngineerRequest] = None,
    dataset_id: Optional[str] = None,
    apply_domain_features: bool = True,
    apply_ratios: bool = True,
    apply_polynomials: bool = False,
    apply_log_transforms: bool = False,
    apply_aggregations: bool = True,
):
    """Applies domain cytological features, morphological ratios, and non-linear interactions."""
    ds_id = req.dataset_id if req is not None else dataset_id
    if ds_id is None:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    dom = req.apply_domain_features if req is not None else apply_domain_features
    rat = req.apply_ratios if req is not None else apply_ratios
    poly = req.apply_polynomials if req is not None else apply_polynomials
    log_t = req.apply_log_transforms if req is not None else apply_log_transforms
    agg = req.apply_aggregations if req is not None else apply_aggregations

    return _eng_service.engineer_features(
        dataset_id=ds_id,
        apply_domain_features=dom,
        apply_ratios=rat,
        apply_polynomials=poly,
        apply_log_transforms=log_t,
        apply_aggregations=agg,
    )


@router.post("/select", response_model=FeatureSelectResponse, summary="Select Features")
def select_features(
    req: Optional[FeatureSelectRequest] = None,
    dataset_id: Optional[str] = None,
    method: str = "mutual_info",
    k: int = 10,
    target_column: str = "diagnosis",
):
    """Executes feature selection algorithm (mutual_info, chi2, f_classif, rfe, lasso)."""
    ds_id = req.dataset_id if req is not None else dataset_id
    if ds_id is None:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    meth = req.method if req is not None else method
    k_val = req.k if req is not None else k
    tgt = req.target_column if req is not None else target_column

    return _sel_service.select_features(
        dataset_id=ds_id,
        method=meth,
        k=k_val,
        target_column=tgt,
    )


@router.post("/reduce", response_model=DimensionalityReductionResult, summary="Reduce Dimensionality")
def reduce_dimensions(
    req: Optional[FeatureReduceRequest] = None,
    dataset_id: Optional[str] = None,
    method: str = "pca",
    n_components: int = 2,
    target_column: str = "diagnosis",
):
    """Applies PCA, Kernel PCA, Truncated SVD, or UMAP dimensionality reduction."""
    ds_id = req.dataset_id if req is not None else dataset_id
    if ds_id is None:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    meth = req.method if req is not None else method
    n_comp = req.n_components if req is not None else n_components
    tgt = req.target_column if req is not None else target_column

    return _sel_service.reduce_dimensions(
        dataset_id=ds_id,
        method=meth,
        n_components=n_comp,
        target_column=tgt,
    )



# ---------------------------------------------------------------------------
# Backward Compatibility Endpoints
# ---------------------------------------------------------------------------
@router.post("/engineer/{dataset_id}", response_model=FeatureEngineeringResult, summary="Engineer Features by Dataset ID")
def engineer_features_by_id(
    dataset_id: str,
    apply_domain_features: bool = Query(True),
    apply_ratios: bool = Query(True),
    apply_polynomials: bool = Query(False),
    apply_log_transforms: bool = Query(False),
    apply_aggregations: bool = Query(True),
):
    return _eng_service.engineer_features(
        dataset_id=dataset_id,
        apply_domain_features=apply_domain_features,
        apply_ratios=apply_ratios,
        apply_polynomials=apply_polynomials,
        apply_log_transforms=apply_log_transforms,
        apply_aggregations=apply_aggregations,
    )


@router.post("/select/{dataset_id}", summary="Select Features by Dataset ID")
def select_features_by_id(
    dataset_id: str,
    method: str = Query("mutual_info"),
    k: int = Query(10, ge=1, le=100),
    target_column: str = Query("diagnosis"),
):
    return _sel_service.select_features(
        dataset_id=dataset_id,
        method=method,
        k=k,
        target_column=target_column,
    )


@router.post("/compare-selection/{dataset_id}", response_model=List[SelectionComparisonItem], summary="Compare Selection Strategies")
def compare_selection_strategies_by_id(
    dataset_id: str,
    k: int = Query(10, ge=2, le=50),
    cv_folds: int = Query(5, ge=2, le=10),
    target_column: str = Query("diagnosis"),
):
    return _sel_service.compare_selection_strategies(
        dataset_id=dataset_id,
        k=k,
        cv_folds=cv_folds,
        target_column=target_column,
    )


# Aliases and helpers for test compatibility
compare_selection_strategies = compare_selection_strategies_by_id


@router.post("/compare-reduction/{dataset_id}", summary="Compare Reduction Methods")
def compare_reduction_methods(dataset_id: str, target_column: str = "diagnosis"):
    return _sel_service.compare_reduction_methods(dataset_id=dataset_id, target_column=target_column)


@router.get("/consensus-importance/{dataset_id}", summary="Get Consensus Importance")
def get_feature_importance_consensus(dataset_id: str, target_column: str = "diagnosis"):
    from feature_engineering.ranking_service import FeatureRankingService
    from backend.services.dataset_service import DatasetService
    ds = DatasetService()
    df = ds.load_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    X = df.drop(columns=[col for col in ["id", target_column] if col in df.columns])
    y = df[target_column]
    srv = FeatureRankingService()
    return srv.compute_consensus_ranking(X, y)

