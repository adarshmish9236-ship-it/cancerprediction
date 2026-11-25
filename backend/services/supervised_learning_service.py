"""
CANCERWISE AI — Supervised Learning Service
Coordinates 16 supervised algorithms, in-fold SMOTE resampling, hyperparameter optimization,
multi-model benchmarking, and model artifact management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from backend.core.exceptions import DatasetNotFoundError, ModelNotFoundError, ProcessingError
from backend.core.logging_setup import get_logger
from backend.schemas.modeling import (
    AlgorithmType,
    ModelTrainingConfig,
    ModelTrainingResult,
    ModelComparisonSummary,
)
from backend.services.dataset_management_service import DatasetManagementService
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.classification.registry import ModelRegistry
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.comparison import ModelComparisonEngine
from data_mining.classification.serializer import ModelSerializer

logger = get_logger("services.supervised_learning")


class SupervisedLearningService:
    """
    Dedicated service for training, benchmarking, and persisting supervised learning models.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        registry: Optional[ModelRegistry] = None,
        serializer: Optional[ModelSerializer] = None,
        trainer: Optional[ModelTrainer] = None,
        comparison_engine: Optional[ModelComparisonEngine] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.registry = registry or ModelRegistry()
        self.serializer = serializer or ModelSerializer()
        self.trainer = trainer or ModelTrainer(registry=self.registry, serializer=self.serializer)
        self.comparison_engine = comparison_engine or ModelComparisonEngine(trainer=self.trainer)
        self.job_manager = get_job_manager()

    def _load_data_and_target(self, dataset_id: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        if target_column not in df.columns:
            raise ProcessingError(f"Target column '{target_column}' not found in dataset '{dataset_id}'.")
        y = df[target_column]
        cols_to_drop = [c for c in [target_column, "id"] if c in df.columns]
        X = df.drop(columns=cols_to_drop)

        return X, y

    def list_supported_algorithms(self) -> List[Dict[str, Any]]:
        """Lists all registered classification algorithms with hyperparameters."""
        return self.registry.list_available_algorithms()

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists all persisted models from the model registry."""
        return self.serializer.list_models()

    def get_model_manifest(self, model_id: str) -> Dict[str, Any]:
        """Retrieves manifest metadata for saved model or raises ModelNotFoundError."""
        manifest = self.serializer.get_manifest(model_id)
        if not manifest:
            raise ModelNotFoundError(model_id)
        return manifest

    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Deletes model artifacts and manifest."""
        deleted = self.serializer.delete_model(model_id)
        if not deleted:
            raise ModelNotFoundError(model_id)
        return {"status": "SUCCESS", "message": f"Model '{model_id}' deleted."}

    def train_model(
        self,
        config: ModelTrainingConfig,
        async_job: bool = False,
    ) -> ModelTrainingResult | JobInfo:
        """
        Trains supervised classifier. If async_job is True, dispatches to background worker.
        """
        if async_job:
            def _async_worker(ctx):
                ctx.set_progress(0.1, f"Loading dataset {config.dataset_id}...")
                X, y = self._load_data_and_target(config.dataset_id, config.target_column)
                ctx.set_progress(0.4, f"Training {config.algorithm.value} with {config.resampling.value}...")
                _, result = self.trainer.train(X, y, config)
                ctx.set_progress(0.9, "Serializing model artifacts...")
                return result

            job = self.job_manager.submit(
                job_type=f"train_model_{config.algorithm.value}",
                target_fn=_async_worker,
            )
            return job

        X, y = self._load_data_and_target(config.dataset_id, config.target_column)
        _, result = self.trainer.train(X, y, config)
        return result

    def compare_models(
        self,
        dataset_id: str,
        target_column: str = "diagnosis",
        ranking_metric: str = "test_roc_auc",
        algorithms: Optional[List[AlgorithmType]] = None,
    ) -> ModelComparisonSummary:
        """Benchmarks multiple models and ranks them by performance."""
        X, y = self._load_data_and_target(dataset_id, target_column)
        return self.comparison_engine.benchmark_models(
            X,
            y,
            dataset_id=dataset_id,
            algorithms=algorithms,
            ranking_metric=ranking_metric,
        )

