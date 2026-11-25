"""
CANCERWISE AI — Phase 7 Comprehensive Test Suite
Tests Performance Monitor, Large-Scale Data Processor (Polars & PyArrow),
Incremental Learning (partial_fit), Scalable PyTorch Mini-Batch Training,
Benchmarking Service, and REST Endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from backend.schemas.large_scale import (
    ProcessingFramework,
    IncrementalAlgorithm,
    SystemResourceSnapshot,
    MemoryOptimizationResult,
    IncrementalLearningResult,
    ScalableTrainingResult,
)
from data_mining.large_scale.performance_monitor import PerformanceMonitor
from data_mining.large_scale.data_processor import LargeScaleDataProcessor
from data_mining.large_scale.incremental_engine import IncrementalLearningEngine
from data_mining.large_scale.scalable_trainer import ScalableTrainingEngine
from data_mining.large_scale.benchmarking_service import BenchmarkingService
from backend.services.dataset_service import DatasetService
from backend.api.large_scale import (
    get_system_telemetry,
    optimize_dataset_memory,
    benchmark_processing_frameworks,
    run_incremental_learning,
)


@pytest.fixture
def sample_tabular_data():
    """Generates synthetic tabular data for large scale testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        f"col_{i}": np.random.randn(n).astype(np.float64) for i in range(10)
    })
    df["int_col"] = np.random.randint(0, 100, size=n, dtype=np.int64)
    df["diagnosis"] = np.random.choice([0, 1], size=n)
    return df


# ---------------------------------------------------------------------------
# 1. Performance Monitor Tests
# ---------------------------------------------------------------------------
class TestPerformanceMonitor:
    def test_system_snapshot(self):
        monitor = PerformanceMonitor()
        snap = monitor.get_system_snapshot()
        assert isinstance(snap, SystemResourceSnapshot)
        assert snap.cpu_cores_logical >= 1
        assert snap.ram_total_mb > 1000.0
        assert snap.process_rss_mb > 0.0

    def test_track_resources_context(self):
        monitor = PerformanceMonitor()
        with monitor.track_resources("test_operation") as perf:
            perf["num_records"] = 1000
            sum(range(10000))

        assert "elapsed_seconds" in perf
        assert perf["elapsed_seconds"] >= 0.0
        assert "throughput_records_per_sec" in perf


# ---------------------------------------------------------------------------
# 2. Large Scale Data Processor Tests
# ---------------------------------------------------------------------------
class TestLargeScaleDataProcessor:
    def test_memory_optimization_downcasting(self, sample_tabular_data):
        processor = LargeScaleDataProcessor()
        df_opt, res = processor.optimize_memory(sample_tabular_data, dataset_name="sample")

        assert isinstance(res, MemoryOptimizationResult)
        assert res.memory_reduced_mb >= 0.0
        assert res.memory_savings_percent >= 0.0
        assert len(res.downcasted_columns) > 0
        assert str(df_opt["col_0"].dtype) == "float32"

    def test_polars_processing(self, sample_tabular_data):
        processor = LargeScaleDataProcessor()
        # Eager
        df_eager, t_eager = processor.process_with_polars(sample_tabular_data, lazy=False)
        assert len(df_eager) == len(sample_tabular_data)
        assert t_eager > 0.0

        # Lazy
        df_lazy, t_lazy = processor.process_with_polars(sample_tabular_data, lazy=True)
        assert len(df_lazy) == len(sample_tabular_data)
        assert t_lazy > 0.0

    def test_benchmark_frameworks(self, sample_tabular_data):
        processor = LargeScaleDataProcessor()
        results = processor.benchmark_frameworks(sample_tabular_data)
        assert len(results) >= 4
        framework_names = [r.framework for r in results]
        assert ProcessingFramework.POLARS_EAGER in framework_names
        assert ProcessingFramework.PANDAS_STANDARD in framework_names


# ---------------------------------------------------------------------------
# 3. Incremental Learning Engine Tests
# ---------------------------------------------------------------------------
class TestIncrementalLearningEngine:
    def test_streaming_sgd_classifier(self, sample_tabular_data):
        engine = IncrementalLearningEngine(random_state=42)
        X = sample_tabular_data.drop(columns=["diagnosis"])
        y = sample_tabular_data["diagnosis"]

        res = engine.train_incremental_classification(
            X, y,
            algorithm=IncrementalAlgorithm.SGD_CLASSIFIER,
            chunk_size=50,
            dataset_name="test_stream",
        )

        assert isinstance(res, IncrementalLearningResult)
        assert res.total_chunks == 4
        assert res.total_records_streamed == 160  # 80% train of 200
        assert len(res.streaming_steps) == 4
        assert 0.0 <= res.final_test_accuracy <= 1.0

    def test_compatible_algorithms_catalog(self):
        engine = IncrementalLearningEngine()
        catalog = engine.get_compatible_algorithms()
        assert len(catalog) >= 4
        alg_names = [c["algorithm"] for c in catalog]
        assert "SGDClassifier" in alg_names
        assert "MiniBatchKMeans" in alg_names


# ---------------------------------------------------------------------------
# 4. Scalable PyTorch Training Engine Tests
# ---------------------------------------------------------------------------
class TestScalableTrainingEngine:
    def test_minibatch_training_and_checkpointing(self, sample_tabular_data, tmp_path):
        engine = ScalableTrainingEngine(checkpoint_dir=tmp_path, random_state=42)
        X = sample_tabular_data.drop(columns=["diagnosis"])
        y = sample_tabular_data["diagnosis"]

        res = engine.train_scalable(
            X, y, batch_size=32, max_epochs=6, model_name="TestScalableNet"
        )

        assert isinstance(res, ScalableTrainingResult)
        assert res.total_epochs > 0
        assert 0.0 <= res.final_val_accuracy <= 1.0
        assert res.average_throughput_samples_per_sec > 0.0
        assert Path(res.checkpoint_file).exists()


# ---------------------------------------------------------------------------
# 5. Benchmarking Service Tests
# ---------------------------------------------------------------------------
class TestBenchmarkingService:
    def test_synthetic_scaling_and_batch_sizes(self, sample_tabular_data):
        service = BenchmarkingService()
        X_synth, y_synth = service.generate_synthetic_benchmark_dataset(n_samples=300, n_features=8)
        assert len(X_synth) == 300

        batch_benchmarks = service.run_batch_size_benchmark(X_synth, y_synth, batch_sizes=[32, 64])
        assert len(batch_benchmarks) == 2
        for b in batch_benchmarks:
            assert b["throughput_samples_per_sec"] > 0.0


# ---------------------------------------------------------------------------
# 6. REST API Endpoints Tests
# ---------------------------------------------------------------------------
class TestScalableAPI:
    def test_api_routes(self):
        # 1. System status
        snap = get_system_telemetry()
        assert snap.cpu_cores_logical >= 1

        # 2. Memory optimization on WDBC
        dataset_id = "uci_wdbc"
        ds = DatasetService()
        if ds.load_dataset(dataset_id) is not None:
            mem_res = optimize_dataset_memory(dataset_id=dataset_id)
            assert mem_res.memory_savings_percent >= 0.0

            # 3. Benchmark processing
            bench_res = benchmark_processing_frameworks(dataset_id=dataset_id)
            assert len(bench_res) >= 3

            # 4. Incremental training
            inc_res = run_incremental_learning(
                dataset_id=dataset_id,
                algorithm=IncrementalAlgorithm.SGD_CLASSIFIER,
                chunk_size=64,
            )
            assert inc_res.total_chunks > 1
