"""
CANCERWISE AI — Benchmarking Service
Empirically benchmarks data processing frameworks (Polars Lazy/Eager vs Pandas vs PyArrow),
batch size throughput scaling, and memory-saving downcasting across authentic and synthetic cohorts.

GOVERNANCE DIRECTIVE:
Synthetic scalability stress benchmarks are strictly separated from real clinical research datasets.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from backend.schemas.large_scale import (
    ProcessingFramework,
    DataProcessingBenchmarkItem,
    MemoryOptimizationResult,
    ScalabilityBenchmarkingReport,
)
from data_mining.large_scale.performance_monitor import PerformanceMonitor
from data_mining.large_scale.data_processor import LargeScaleDataProcessor
from data_mining.large_scale.incremental_engine import IncrementalLearningEngine
from data_mining.large_scale.scalable_trainer import ScalableTrainingEngine
from backend.services.dataset_service import DatasetService

logger = logging.getLogger("cancerwise.large_scale.benchmarking")


class BenchmarkingService:
    """
    Coordinates controlled computational performance and scalability benchmarks.
    """

    def __init__(
        self,
        processor: Optional[LargeScaleDataProcessor] = None,
        incremental_engine: Optional[IncrementalLearningEngine] = None,
        scalable_trainer: Optional[ScalableTrainingEngine] = None,
        monitor: Optional[PerformanceMonitor] = None,
    ):
        self.monitor = monitor or PerformanceMonitor()
        self.processor = processor or LargeScaleDataProcessor(monitor=self.monitor)
        self.incremental_engine = incremental_engine or IncrementalLearningEngine(monitor=self.monitor)
        self.scalable_trainer = scalable_trainer or ScalableTrainingEngine(monitor=self.monitor)
        self.dataset_service = DatasetService()

    @staticmethod
    def generate_synthetic_benchmark_dataset(
        n_samples: int = 25000,
        n_features: int = 20,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates synthetic large-scale benchmark data for computational stress testing.
        """
        n_inf = min(max(int(n_features * 0.7), 2), n_features - 2)
        n_red = min(max(int(n_features * 0.2), 1), n_features - n_inf - 1)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_inf,
            n_redundant=n_red,
            random_state=random_state,
        )
        cols = [f"biomarker_dim_{i+1}" for i in range(n_features)]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")

    def run_batch_size_benchmark(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        batch_sizes: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tests PyTorch mini-batch training throughput across batch sizes (32, 64, 128, 256).
        """
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256]

        results = []
        for bs in batch_sizes:
            res = self.scalable_trainer.train_scalable(
                X, y, batch_size=bs, max_epochs=12, patience=4, model_name=f"ScaleTest_BS{bs}"
            )
            results.append({
                "batch_size": bs,
                "epochs_trained": res.total_epochs,
                "throughput_samples_per_sec": res.average_throughput_samples_per_sec,
                "training_time_seconds": res.total_training_time_seconds,
                "val_accuracy": res.final_val_accuracy,
                "val_roc_auc": res.final_val_roc_auc,
            })
        return results

    def run_full_scalability_benchmark(self) -> ScalabilityBenchmarkingReport:
        """
        Executes complete scalability benchmark:
        1. Primary WDBC dataset processing benchmark
        2. Synthetic 25,000-sample processing stress test
        3. Memory optimization downcasting
        4. Incremental learning streaming verification
        5. Scalable PyTorch training run
        """
        # 1. Primary Dataset (UCI WDBC)
        df_wdbc = self.dataset_service.load_dataset("uci_wdbc")
        if df_wdbc is None:
            raise FileNotFoundError("Primary UCI WDBC dataset unavailable.")

        target_col = "diagnosis"
        X_wdbc = df_wdbc.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y_wdbc = df_wdbc[target_col]

        logger.info("Running framework benchmark on primary dataset...")
        wdbc_benchmarks = self.processor.benchmark_frameworks(X_wdbc)

        # 2. Synthetic Scaled Stress Dataset (25,000 records)
        logger.info("Generating synthetic 25,000-sample dataset...")
        X_synth, y_synth = self.generate_synthetic_benchmark_dataset(n_samples=25000, n_features=16)

        logger.info("Running framework benchmark on synthetic 25,000-sample dataset...")
        synth_benchmarks = self.processor.benchmark_frameworks(X_synth)

        # 3. Memory Optimization
        _, mem_summary = self.processor.optimize_memory(X_synth, dataset_name="Synthetic_25k_Stress")

        # 4. Batch Size Benchmark on WDBC
        batch_benchmarks = self.run_batch_size_benchmark(X_wdbc, y_wdbc, batch_sizes=[32, 64, 128])

        # 5. Incremental Learning Run
        incr_result = self.incremental_engine.train_incremental_classification(
            X_wdbc, y_wdbc, chunk_size=64, dataset_name="UCI WDBC Stream"
        )

        # 6. Scalable Neural Network Run
        scale_nn_result = self.scalable_trainer.train_scalable(
            X_wdbc, y_wdbc, batch_size=64, max_epochs=20
        )

        system_status = self.monitor.get_system_snapshot()

        return ScalabilityBenchmarkingReport(
            primary_dataset_benchmarks=wdbc_benchmarks,
            synthetic_scale_benchmarks=synth_benchmarks,
            batch_size_benchmarks=batch_benchmarks,
            memory_optimization_summary=mem_summary,
            incremental_performance=incr_result,
            scalable_training_summary=scale_nn_result,
            system_profile=system_status,
        )
