"""
CANCERWISE AI — Large Scale Learning & Performance REST Endpoints
Provides interactive API routes for hardware telemetry, memory-aware downcasting,
Polars/PyArrow processing benchmarks, out-of-core incremental learning, and PyTorch mini-batch training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
import numpy as np
import pandas as pd

from backend.core.logging_setup import get_logger
from backend.services.dataset_service import DatasetService
from backend.schemas.large_scale import (
    ProcessingFramework,
    IncrementalAlgorithm,
    SystemResourceSnapshot,
    MemoryOptimizationResult,
    DataProcessingBenchmarkItem,
    IncrementalLearningResult,
    ScalableTrainingResult,
    ScalabilityBenchmarkingReport,
)
from data_mining.large_scale.performance_monitor import PerformanceMonitor
from data_mining.large_scale.data_processor import LargeScaleDataProcessor
from data_mining.large_scale.incremental_engine import IncrementalLearningEngine
from data_mining.large_scale.scalable_trainer import ScalableTrainingEngine
from data_mining.large_scale.benchmarking_service import BenchmarkingService

logger = get_logger("api.large_scale")
router = APIRouter(prefix="/scale", tags=["Large-Scale Learning & Optimization"])

_monitor = PerformanceMonitor()
_processor = LargeScaleDataProcessor(monitor=_monitor)
_incremental_engine = IncrementalLearningEngine(monitor=_monitor)
_scalable_trainer = ScalableTrainingEngine(monitor=_monitor)
_benchmarking_service = BenchmarkingService(
    processor=_processor,
    incremental_engine=_incremental_engine,
    scalable_trainer=_scalable_trainer,
    monitor=_monitor,
)
_dataset_service = DatasetService()


def _load_data_and_target(dataset_id: str, target_column: str = "diagnosis") -> tuple[pd.DataFrame, pd.Series]:
    df = _dataset_service.load_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    if target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    else:
        X = df.select_dtypes(include=[np.number])
        y = pd.Series([0] * len(df))
    return X, y


@router.get("/system-status", response_model=SystemResourceSnapshot)
def get_system_telemetry():
    """Captures current host CPU, physical RAM, process memory, and GPU VRAM status."""
    return _monitor.get_system_snapshot()


@router.post("/optimize-memory/{dataset_id}", response_model=MemoryOptimizationResult)
def optimize_dataset_memory(
    dataset_id: str,
):
    """Downcasts floating-point and integer types to minimal sufficient representations."""
    X, _ = _load_data_and_target(dataset_id)
    _, result = _processor.optimize_memory(X, dataset_name=dataset_id)
    return result


@router.post("/benchmark-processing/{dataset_id}", response_model=List[DataProcessingBenchmarkItem])
def benchmark_processing_frameworks(
    dataset_id: str,
):
    """Benchmarks Pandas Standard vs Chunked vs Polars Eager vs Polars Lazy vs PyArrow."""
    X, _ = _load_data_and_target(dataset_id)
    return _processor.benchmark_frameworks(X)


@router.post("/incremental-train/{dataset_id}", response_model=IncrementalLearningResult)
def run_incremental_learning(
    dataset_id: str,
    algorithm: IncrementalAlgorithm = Query(IncrementalAlgorithm.SGD_CLASSIFIER),
    chunk_size: int = Query(64, ge=16, le=512),
    target_column: str = Query("diagnosis"),
):
    """Executes out-of-core streaming training using partial_fit."""
    X, y = _load_data_and_target(dataset_id, target_column)
    try:
        return _incremental_engine.train_incremental_classification(
            X, y, algorithm=algorithm, chunk_size=chunk_size, dataset_name=dataset_id
        )
    except Exception as e:
        logger.error("Incremental training error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scalable-nn-train/{dataset_id}", response_model=ScalableTrainingResult)
def train_scalable_neural_net(
    dataset_id: str,
    batch_size: int = Query(64, ge=16, le=256),
    max_epochs: int = Query(30, ge=5, le=100),
    target_column: str = Query("diagnosis"),
):
    """Executes mini-batch PyTorch deep neural network training with learning-rate scheduling."""
    X, y = _load_data_and_target(dataset_id, target_column)
    try:
        return _scalable_trainer.train_scalable(
            X, y, batch_size=batch_size, max_epochs=max_epochs, model_name=f"{dataset_id}_ScalableNet"
        )
    except Exception as e:
        logger.error("Scalable NN training error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarking-report", response_model=ScalabilityBenchmarkingReport)
def generate_full_scalability_report():
    """Generates complete scalability stress benchmark across authentic and synthetic cohorts."""
    try:
        return _benchmarking_service.run_full_scalability_benchmark()
    except Exception as e:
        logger.error("Benchmarking report error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
