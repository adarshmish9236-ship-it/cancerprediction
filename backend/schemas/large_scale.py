"""
CANCERWISE AI — Large-Scale Learning & Performance Optimization Schemas
Pydantic schemas for hardware telemetry, memory-aware processing, out-of-core incremental learning,
PyTorch scalable mini-batch training, and processing benchmarks.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ProcessingFramework(str, Enum):
    PANDAS_STANDARD = "pandas_standard"
    PANDAS_CHUNKED = "pandas_chunked"
    POLARS_EAGER = "polars_eager"
    POLARS_LAZY = "polars_lazy"
    PYARROW = "pyarrow"


class IncrementalAlgorithm(str, Enum):
    SGD_CLASSIFIER = "sgd_classifier"
    SGD_REGRESSOR = "sgd_regressor"
    MINIBATCH_KMEANS = "minibatch_kmeans"
    PASSIVE_AGGRESSIVE = "passive_aggressive"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


# ---------------------------------------------------------------------------
# Telemetry & Resource Snapshots
# ---------------------------------------------------------------------------
class SystemResourceSnapshot(BaseModel):
    cpu_cores_logical: int
    cpu_cores_physical: int
    cpu_percent: float
    ram_total_mb: float
    ram_used_mb: float
    ram_available_mb: float
    ram_percent: float
    process_rss_mb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_allocated_mb: Optional[float] = None
    timestamp: str = Field(default_factory=_utc_now_str)

    @property
    def process_memory_rss_mb(self) -> float:
        return self.process_rss_mb



# ---------------------------------------------------------------------------
# Data Processing Benchmarks
# ---------------------------------------------------------------------------
class MemoryOptimizationResult(BaseModel):
    dataset_name: str
    original_memory_mb: float
    optimized_memory_mb: float
    memory_reduced_mb: float
    memory_savings_percent: float
    downcasted_columns: Dict[str, str]  # col -> new_dtype


class DataProcessingBenchmarkItem(BaseModel):
    framework: ProcessingFramework
    dataset_rows: int
    dataset_columns: int
    execution_time_seconds: float
    throughput_records_per_sec: float
    peak_memory_mb: float
    memory_efficiency_score: float  # throughput / peak_memory


# ---------------------------------------------------------------------------
# Incremental Learning Schemas
# ---------------------------------------------------------------------------
class IncrementalChunkStep(BaseModel):
    chunk_index: int
    records_in_chunk: int
    cumulative_records: int
    chunk_training_time_ms: float
    running_test_accuracy: float
    running_test_roc_auc: float
    running_loss: float


class IncrementalLearningResult(BaseModel):
    algorithm: IncrementalAlgorithm
    dataset_name: str
    total_chunks: int
    total_records_streamed: int
    chunk_size: int
    total_training_time_seconds: float
    final_test_accuracy: float
    final_test_roc_auc: float
    final_test_f1: float
    streaming_steps: List[IncrementalChunkStep]
    supports_partial_fit: bool = True
    memory_peak_mb: float


# ---------------------------------------------------------------------------
# Scalable PyTorch Mini-Batch Schemas
# ---------------------------------------------------------------------------
class ScalableEpochMetric(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_duration_seconds: float
    throughput_samples_per_sec: float


class ScalableTrainingResult(BaseModel):
    model_name: str
    device_used: DeviceType
    batch_size: int
    total_epochs: int
    early_stopped_epoch: int
    best_val_loss: float
    final_val_accuracy: float
    final_val_roc_auc: float
    total_training_time_seconds: float
    average_throughput_samples_per_sec: float
    checkpoint_file: str
    epoch_history: List[ScalableEpochMetric]
    hardware_status: SystemResourceSnapshot


# ---------------------------------------------------------------------------
# Comprehensive Benchmarking Report
# ---------------------------------------------------------------------------
class ScalabilityBenchmarkingReport(BaseModel):
    primary_dataset_benchmarks: List[DataProcessingBenchmarkItem]
    synthetic_scale_benchmarks: List[DataProcessingBenchmarkItem]
    batch_size_benchmarks: List[Dict[str, Any]]
    memory_optimization_summary: MemoryOptimizationResult
    incremental_performance: IncrementalLearningResult
    scalable_training_summary: ScalableTrainingResult
    system_profile: SystemResourceSnapshot
    generated_at: str = Field(default_factory=_utc_now_str)
