"""
CANCERWISE AI — Advanced Learning Laboratory REST Endpoints
Provides interactive routes for Semi-Supervised, Multitask (PyTorch), Multiclass Subtypes,
One-Class Novelty, Continuous Regression, Kernel Learning, and Structured Output.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, status

from backend.core.logging_setup import get_logger
from backend.schemas.advanced_learning import (
    SemiSupervisedMethod,
    MulticlassStrategy,
    OneClassMethod,
    RegressionMethod,
    SemiSupervisedExperimentResult,
    MultitaskExperimentResult,
    MulticlassExperimentResult,
    OneClassExperimentResult,
    RegressionExperimentResult,
    KernelLearningExperimentResult,
    StructuredOutputExperimentResult,
    AdvancedLabComparisonSummary,
)
from backend.schemas.phase9_schemas import (
    AdvancedSemiSupervisedRequest,
    AdvancedMultitaskRequest,
    AdvancedMulticlassRequest,
    AdvancedOneClassRequest,
    AdvancedRegressionRequest,
    AdvancedKernelRequest,
)
from backend.services.semi_supervised_service import SemiSupervisedLearningService
from backend.services.multitask_service import MultitaskLearningService
from backend.services.multiclass_service import MulticlassClassificationService
from backend.services.one_class_service import OneClassClassificationService
from backend.services.regression_service import RegressionLearningService
from data_mining.experiments.advanced_experiment_manager import AdvancedExperimentManager
from backend.services.job_service import JobInfo

logger = get_logger("api.advanced")
router = APIRouter(prefix="/advanced", tags=["Advanced Learning Laboratory"])

_semi_service = SemiSupervisedLearningService()
_multi_service = MultitaskLearningService()
_mc_service = MulticlassClassificationService()
_oc_service = OneClassClassificationService()
_reg_service = RegressionLearningService()
_exp_manager = AdvancedExperimentManager()


# ---------------------------------------------------------------------------
# Section 2 Advanced Learning APIs
# ---------------------------------------------------------------------------
@router.post("/semi-supervised/train", summary="Train Semi-Supervised Model")
def train_semi_supervised(req: AdvancedSemiSupervisedRequest) -> Union[SemiSupervisedExperimentResult, JobInfo]:
    """Executes semi-supervised label propagation/spreading across varying labeled partitions."""
    return _semi_service.train_semi_supervised(
        method=req.method,
        labeled_fractions=req.labeled_fractions,
        async_job=req.async_job,
    )


@router.post("/multitask/train", summary="Train Multitask Neural Network")
def train_multitask(req: AdvancedMultitaskRequest) -> Union[MultitaskExperimentResult, JobInfo]:
    """Executes PyTorch shared-representation multitask neural network training."""
    return _multi_service.train_multitask(
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        async_job=req.async_job,
    )


@router.post("/multiclass/train", summary="Train Multiclass Molecular Subtypes")
def train_multiclass(req: AdvancedMulticlassRequest) -> Union[MulticlassExperimentResult, JobInfo]:
    """Evaluates multiclass cancer molecular subtype classification."""
    return _mc_service.train_multiclass(
        strategy=req.strategy,
        async_job=req.async_job,
    )


@router.post("/one-class/train", summary="Train One-Class Novelty Detector")
def train_one_class(req: AdvancedOneClassRequest) -> Union[OneClassExperimentResult, JobInfo]:
    """Evaluates one-class novelty detection on normative benign cytology baseline."""
    return _oc_service.train_one_class(
        method=req.method,
        contamination=req.contamination,
        async_job=req.async_job,
    )


@router.post("/regression/train", summary="Train Continuous Regression Model")
def train_regression(req: AdvancedRegressionRequest) -> Union[RegressionExperimentResult, JobInfo]:
    """Evaluates continuous predictive modeling for genuine continuous targets."""
    return _reg_service.train_regression(
        method=req.method,
        async_job=req.async_job,
    )


@router.post("/kernel/train", response_model=KernelLearningExperimentResult, summary="Train & Benchmark Mercer Kernels")
def train_kernel(req: Optional[AdvancedKernelRequest] = None):
    """Benchmarks Linear, Polynomial, RBF, and Sigmoid Mercer kernels in Hilbert space."""
    return _exp_manager.run_kernel_learning()


# ---------------------------------------------------------------------------
# Backward Compatibility Endpoints
# ---------------------------------------------------------------------------
@router.post("/semi-supervised", response_model=SemiSupervisedExperimentResult, summary="Run Semi-Supervised (Legacy)")
def run_semi_supervised_experiment(
    method: SemiSupervisedMethod = Query(SemiSupervisedMethod.LABEL_SPREADING),
):
    return _exp_manager.run_semi_supervised(method=method)


@router.post("/multitask", response_model=MultitaskExperimentResult, summary="Run Multitask (Legacy)")
def run_multitask_experiment(
    epochs: int = Query(40, ge=10, le=200),
):
    return _exp_manager.run_multitask(epochs=epochs)


@router.post("/multiclass", response_model=MulticlassExperimentResult, summary="Run Multiclass (Legacy)")
def run_multiclass_experiment(
    strategy: MulticlassStrategy = Query(MulticlassStrategy.MULTINOMIAL_LOGISTIC),
):
    return _exp_manager.run_multiclass(strategy=strategy)


@router.post("/one-class", response_model=OneClassExperimentResult, summary="Run One-Class (Legacy)")
def run_one_class_novelty(
    method: OneClassMethod = Query(OneClassMethod.ISOLATION_FOREST),
):
    return _exp_manager.run_one_class(method=method)


@router.post("/regression", response_model=RegressionExperimentResult, summary="Run Regression (Legacy)")
def run_regression_experiment(
    method: RegressionMethod = Query(RegressionMethod.RIDGE),
):
    return _exp_manager.run_regression(method=method)


@router.post("/kernel-learning", response_model=KernelLearningExperimentResult, summary="Kernel Benchmark (Legacy)")
def run_kernel_benchmark():
    return _exp_manager.run_kernel_learning()


@router.post("/structured-output", response_model=StructuredOutputExperimentResult, summary="Structured Output (Legacy)")
def run_structured_output_experiment():
    return _exp_manager.run_structured_output()


@router.get("/suite-summary", response_model=AdvancedLabComparisonSummary, summary="Run Full Suite (Legacy)")
def run_full_advanced_suite():
    return _exp_manager.run_full_suite()
