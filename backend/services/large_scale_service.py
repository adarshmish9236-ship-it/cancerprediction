"""
CANCERWISE AI — Large-Scale Learning Service
Coordinates hardware resource telemetry, memory-aware type downcasting,
Polars/PyArrow columnar benchmarks, out-of-core streaming partial_fit,
and scalable PyTorch mini-batch training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.core.exceptions import DatasetNotFoundError
from backend.core.logging_setup import get_logger
from backend.schemas.large_scale import (
    IncrementalAlgorithm,
    SystemResourceSnapshot,
    MemoryOptimizationResult,
    DataProcessingBenchmarkItem,
    IncrementalLearningResult,
    ScalableTrainingResult,
)
from backend.services.dataset_management_service import DatasetManagementService
from backend.services.job_service import get_job_manager, JobInfo
from data_mining.large_scale.performance_monitor import PerformanceMonitor
from data_mining.large_scale.data_processor import LargeScaleDataProcessor
from data_mining.large_scale.incremental_engine import IncrementalLearningEngine
from data_mining.large_scale.scalable_trainer import ScalableTrainingEngine
from data_mining.large_scale.benchmarking_service import BenchmarkingService

logger = get_logger("services.large_scale")


class LargeScaleLearningService:
    """
    Dedicated service for high-throughput, memory-bounded, and scalable ML operations.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        monitor: Optional[PerformanceMonitor] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.monitor = monitor or PerformanceMonitor()
        self.processor = LargeScaleDataProcessor(monitor=self.monitor)
        self.incremental_engine = IncrementalLearningEngine(monitor=self.monitor)
        self.scalable_trainer = ScalableTrainingEngine(monitor=self.monitor)
        self.benchmarking_service = BenchmarkingService(
            processor=self.processor,
            incremental_engine=self.incremental_engine,
            scalable_trainer=self.scalable_trainer,
            monitor=self.monitor,
        )
        self.job_manager = get_job_manager()

    def _load_data_and_target(
        self, dataset_id: str, target_column: str = "diagnosis"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
        else:
            X = df.select_dtypes(include=[np.number])
            y = pd.Series([0] * len(df))
        return X, y

    def get_system_telemetry(self) -> SystemResourceSnapshot:
        """Captures host CPU, RAM, process RSS, and GPU VRAM snapshot."""
        return self.monitor.get_system_snapshot()

    def optimize_dataset_memory(self, dataset_id: str) -> MemoryOptimizationResult:
        """Applies memory-aware downcasting to reduce RAM consumption."""
        X, _ = self._load_data_and_target(dataset_id)
        _, result = self.processor.optimize_memory(X, dataset_name=dataset_id)
        return result

    def benchmark_frameworks(self, dataset_id: str) -> List[DataProcessingBenchmarkItem]:
        """Runs throughput comparison between Pandas, Polars Eager/Lazy, and PyArrow."""
        X, _ = self._load_data_and_target(dataset_id)
        return self.processor.benchmark_frameworks(X)

    def train_incremental(
        self,
        dataset_id: str,
        algorithm: IncrementalAlgorithm = IncrementalAlgorithm.SGD_CLASSIFIER,
        chunk_size: int = 64,
        target_column: str = "diagnosis",
        async_job: bool = False,
    ) -> IncrementalLearningResult | JobInfo:
        """Performs out-of-core streaming learning."""
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.1, f"Loading streaming chunks for {algorithm.value}...")
                X, y = self._load_data_and_target(dataset_id, target_column)
                res = self.incremental_engine.train_incremental_classification(
                    X, y, algorithm=algorithm, chunk_size=chunk_size, dataset_name=dataset_id
                )
                ctx.set_progress(1.0, "Incremental learning complete.")
                return res

            return self.job_manager.submit(
                job_type=f"incremental_train_{algorithm.value}",
                target_fn=_worker,
            )

        X, y = self._load_data_and_target(dataset_id, target_column)
        return self.incremental_engine.train_incremental_classification(
            X, y, algorithm=algorithm, chunk_size=chunk_size, dataset_name=dataset_id
        )

    def train_scalable(
        self,
        dataset_id: str,
        epochs: int = 20,
        batch_size: int = 32,
        target_column: str = "diagnosis",
        async_job: bool = False,
    ) -> ScalableTrainingResult | JobInfo:
        """Executes mini-batch PyTorch scalable training."""
        if async_job:
            def _worker(ctx):
                ctx.set_progress(0.1, f"Initializing scalable PyTorch training ({epochs} epochs)...")
                X, y = self._load_data_and_target(dataset_id, target_column)
                res = self.scalable_trainer.train_scalable_pytorch(
                    X, y, epochs=epochs, batch_size=batch_size, dataset_name=dataset_id
                )
                ctx.set_progress(1.0, "Scalable training finished.")
                return res

            return self.job_manager.submit(
                job_type="scalable_pytorch_training",
                target_fn=_worker,
            )

        X, y = self._load_data_and_target(dataset_id, target_column)
        return self.scalable_trainer.train_scalable_pytorch(
            X, y, epochs=epochs, batch_size=batch_size, dataset_name=dataset_id
        )
