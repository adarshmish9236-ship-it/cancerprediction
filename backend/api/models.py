"""
CANCERWISE AI — Machine Learning & Model Management REST Endpoints
Provides routes for model training, registry lookups, comprehensive evaluation,
production inference, and artifact management.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, status

from backend.core.logging_setup import get_logger
from backend.schemas.modeling import (
    AlgorithmType,
    ResamplingStrategy,
    TuningStrategy,
    ModelTrainingConfig,
    ModelTrainingResult,
    ModelComparisonSummary,
    PredictionRequest,
    PredictionResponse,
)
from backend.schemas.explainability import HealthcareValidationMetrics
from backend.schemas.phase9_schemas import ModelTrainRequest, ClinicalPredictionRequest
from backend.services.supervised_learning_service import SupervisedLearningService
from backend.services.model_evaluation_service import ModelEvaluationService
from backend.services.prediction_service import (
    PredictionService,
    ClinicalPredictionResponse,
)
from backend.services.job_service import JobInfo

logger = get_logger("api.models")
router = APIRouter(prefix="/models", tags=["Supervised Learning & Cancer Prediction"])

_sup_service = SupervisedLearningService()
_eval_service = ModelEvaluationService()
_pred_service = PredictionService()


# ---------------------------------------------------------------------------
# Section 2 Machine Learning APIs
# ---------------------------------------------------------------------------
@router.get("", summary="List Registered Models")
def list_models() -> List[Dict[str, Any]]:
    """Lists all trained model manifests persisted in the internal model registry."""
    return _sup_service.list_models()


@router.post("/train", summary="Train Supervised Model")
def train_model(
    req: Optional[ModelTrainRequest] = None,
    dataset_id: Optional[str] = None,
    config: Optional[ModelTrainingConfig] = None,
) -> Union[ModelTrainingResult, JobInfo]:
    """
    Trains a supervised classification pipeline with optional resampling and hyperparameter tuning.
    Supports asynchronous background execution when `async_job=True`.
    """
    if req is not None:
        cfg = ModelTrainingConfig(
            dataset_id=req.dataset_id,
            algorithm=req.algorithm,
            target_column=req.target_column,
            resampling=req.resampling,
            tuning=req.tuning,
            hyperparameters=req.hyperparameters,
            cv_folds=req.cv_folds,
        )
        return _sup_service.train_model(config=cfg, async_job=req.async_job)
    elif config is not None:
        if dataset_id:
            config.dataset_id = dataset_id
        res = _sup_service.train_model(config=config, async_job=False)
        return res
    else:
        raise HTTPException(status_code=422, detail="Request payload or configuration required.")



@router.get("/{model_id}", summary="Get Model Details")
def get_model_details(model_id: str) -> Dict[str, Any]:
    """Retrieves full manifest metadata and metrics for a saved model artifact."""
    return _sup_service.get_model_manifest(model_id)


@router.get("/{model_id}/evaluate", response_model=HealthcareValidationMetrics, summary="Evaluate Model Clinically")
def evaluate_model(
    model_id: str,
    compute_bootstrap: bool = Query(True, description="Compute 95% bootstrap confidence intervals"),
):
    """
    Evaluates saved model on 12 clinical & statistical metrics:
    Sensitivity, Specificity, FNR, FPR, PPV, NPV, ROC-AUC, Brier score, and Bayes' prevalence analysis.
    """
    return _eval_service.evaluate_model(model_id=model_id, compute_bootstrap=compute_bootstrap)


@router.post("/{model_id}/predict", response_model=ClinicalPredictionResponse, summary="Clinical Inference")
def predict_model(
    model_id: str,
    payload: ClinicalPredictionRequest,
):
    """
    Runs model inference from internal registry.
    Validates feature names, ordering, and dtypes.
    Returns predicted class, probabilities, model version, prediction identifier,
    and explanation availability.
    """
    return _pred_service.predict(model_id=model_id, features=payload.features)


# ---------------------------------------------------------------------------
# Backward Compatibility Endpoints
# ---------------------------------------------------------------------------
@router.get("/algorithms", summary="List Supported Algorithms")
def list_supported_algorithms() -> List[Dict[str, Any]]:
    """Lists all registered classification algorithms with categories and hyperparameters."""
    return _sup_service.list_supported_algorithms()


@router.get("/saved", summary="List Saved Models (Legacy)")
def list_saved_models_legacy() -> List[Dict[str, Any]]:
    return _sup_service.list_models()


# Aliases for backwards compatibility
list_saved_models = list_saved_models_legacy



@router.get("/manifest/{model_id}", summary="Get Model Manifest (Legacy)")
def get_model_manifest_legacy(model_id: str) -> Dict[str, Any]:
    return _sup_service.get_model_manifest(model_id)


@router.post("/train/{dataset_id}", response_model=ModelTrainingResult, summary="Train Model (Legacy POST)")
def train_model_legacy(dataset_id: str, config: ModelTrainingConfig):
    config.dataset_id = dataset_id
    res = _sup_service.train_model(config=config, async_job=False)
    if isinstance(res, ModelTrainingResult):
        return res
    raise HTTPException(status_code=500, detail="Unexpected job return")


@router.post("/compare/{dataset_id}", response_model=ModelComparisonSummary, summary="Benchmark Models")
def compare_models(
    dataset_id: str,
    target_column: str = Query("diagnosis"),
    ranking_metric: str = Query("test_roc_auc"),
    algorithms: Optional[List[AlgorithmType]] = Query(None),
):
    return _sup_service.compare_models(
        dataset_id=dataset_id,
        target_column=target_column,
        ranking_metric=ranking_metric,
        algorithms=algorithms,
    )


@router.post("/predict/{model_id}", response_model=PredictionResponse, summary="Legacy Predict")
def predict_legacy(model_id: str, payload: PredictionRequest):
    from data_mining.classification.prediction_service import PredictionService as LegacyPredService
    legacy_svc = LegacyPredService()
    try:
        return legacy_svc.predict(model_id, payload.features)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' artifact not found.")


# Alias for backwards compatibility
predict = predict_legacy



@router.delete("/{model_id}", summary="Delete Model")
def delete_model(model_id: str):
    return _sup_service.delete_model(model_id)
