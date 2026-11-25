"""
CANCERWISE AI — Large-Scale Data Processor
Implements high-throughput Polars lazy execution, Pandas chunk-based streaming,
PyArrow zero-copy tables, Snappy-compressed Parquet storage, and memory-aware type downcasting.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from backend.schemas.large_scale import (
    ProcessingFramework,
    MemoryOptimizationResult,
    DataProcessingBenchmarkItem,
)
from data_mining.large_scale.performance_monitor import PerformanceMonitor

logger = logging.getLogger("cancerwise.large_scale.processor")


class LargeScaleDataProcessor:
    """
    Optimized data processing engine for out-of-core and large-scale biomedical tabular data.
    """

    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        self.monitor = monitor or PerformanceMonitor()

    def optimize_memory(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> Tuple[pd.DataFrame, MemoryOptimizationResult]:
        """
        Downcasts floating-point and integer columns to minimal sufficient precision
        (e.g., float64 -> float32, int64 -> int32/int16), measuring RAM reduction.
        """
        orig_bytes = df.memory_usage(deep=True).sum()
        orig_mb = orig_bytes / (1024.0 * 1024.0)

        df_opt = df.copy()
        downcasted: Dict[str, str] = {}

        for col in df_opt.columns:
            dtype = df_opt[col].dtype
            if pd.api.types.is_float_dtype(dtype):
                df_opt[col] = pd.to_numeric(df_opt[col], downcast="float")
                if str(df_opt[col].dtype) != str(dtype):
                    downcasted[col] = str(df_opt[col].dtype)
            elif pd.api.types.is_integer_dtype(dtype):
                df_opt[col] = pd.to_numeric(df_opt[col], downcast="integer")
                if str(df_opt[col].dtype) != str(dtype):
                    downcasted[col] = str(df_opt[col].dtype)

        opt_bytes = df_opt.memory_usage(deep=True).sum()
        opt_mb = opt_bytes / (1024.0 * 1024.0)
        saved_mb = max(0.0, orig_mb - opt_mb)
        saved_pct = (saved_mb / orig_mb) * 100.0 if orig_mb > 0 else 0.0

        result = MemoryOptimizationResult(
            dataset_name=dataset_name,
            original_memory_mb=round(orig_mb, 4),
            optimized_memory_mb=round(opt_mb, 4),
            memory_reduced_mb=round(saved_mb, 4),
            memory_savings_percent=round(saved_pct, 2),
            downcasted_columns=downcasted,
        )

        logger.info(
            "Memory optimization for '%s': %.2fMB -> %.2fMB (%.1f%% saved, %d cols downcasted)",
            dataset_name, orig_mb, opt_mb, saved_pct, len(downcasted),
        )
        return df_opt, result

    def stream_csv_chunks(
        self,
        csv_path: Union[str, Path],
        chunksize: int = 500,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Streams CSV data chunk-by-chunk to enable out-of-core memory-bounded ingestion.
        """
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            yield chunk

    def to_polars_lazy(self, df_or_parquet: Union[pd.DataFrame, str, Path]) -> pl.LazyFrame:
        """
        Converts DataFrame or Parquet file path into a Polars LazyFrame for deferred query optimization.
        """
        if isinstance(df_or_parquet, (str, Path)):
            return pl.scan_parquet(str(df_or_parquet))
        return pl.from_pandas(df_or_parquet).lazy()

    def process_with_polars(
        self,
        df: pd.DataFrame,
        lazy: bool = True,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Executes parallel column standardizations and aggregations via Polars.
        """
        start = time.perf_counter()
        pl_df = pl.from_pandas(df)

        numeric_cols = [col for col, dtype in zip(pl_df.columns, pl_df.dtypes) if dtype.is_numeric()]

        if lazy:
            lazy_query = (
                pl_df.lazy()
                .select([
                    pl.col(c) for c in pl_df.columns
                ])
                .with_columns([
                    ((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-8)).alias(f"{c}_norm")
                    for c in numeric_cols[:5]
                ])
            )
            processed_pl = lazy_query.collect()
        else:
            processed_pl = pl_df.with_columns([
                ((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-8)).alias(f"{c}_norm")
                for c in numeric_cols[:5]
            ])

        elapsed = time.perf_counter() - start
        return processed_pl.to_pandas(), elapsed

    def benchmark_frameworks(
        self,
        df: pd.DataFrame,
        frameworks: Optional[List[ProcessingFramework]] = None,
    ) -> List[DataProcessingBenchmarkItem]:
        """
        Empirically benchmarks execution speed, memory footprint, and throughput across:
        Pandas Standard, Pandas Chunked, Polars Eager, Polars Lazy, and PyArrow.
        """
        if frameworks is None:
            frameworks = [
                ProcessingFramework.PANDAS_STANDARD,
                ProcessingFramework.PANDAS_CHUNKED,
                ProcessingFramework.POLARS_EAGER,
                ProcessingFramework.POLARS_LAZY,
                ProcessingFramework.PYARROW,
            ]

        n_rows, n_cols = df.shape
        results: List[DataProcessingBenchmarkItem] = []

        for fw in frameworks:
            start_rss = self.monitor.process.memory_info().rss / (1024.0 * 1024.0)
            start_time = time.perf_counter()

            if fw == ProcessingFramework.PANDAS_STANDARD:
                num_df = df.select_dtypes(include=[np.number])
                _ = (num_df - num_df.mean()) / (num_df.std() + 1e-8)

            elif fw == ProcessingFramework.PANDAS_CHUNKED:
                chunk_size = max(len(df) // 4, 10)
                chunks_out = []
                for i in range(0, len(df), chunk_size):
                    c = df.iloc[i : i + chunk_size].select_dtypes(include=[np.number])
                    chunks_out.append((c - c.mean()) / (c.std() + 1e-8))
                _ = pd.concat(chunks_out)

            elif fw == ProcessingFramework.POLARS_EAGER:
                pl_df = pl.from_pandas(df)
                num_cols = [c for c, d in zip(pl_df.columns, pl_df.dtypes) if d.is_numeric()]
                _ = pl_df.select([((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-8)) for c in num_cols[:5]])

            elif fw == ProcessingFramework.POLARS_LAZY:
                pl_lazy = pl.from_pandas(df).lazy()
                num_cols = [c for c, d in zip(df.columns, df.dtypes) if pd.api.types.is_numeric_dtype(d)]
                _ = pl_lazy.select([((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-8)) for c in num_cols[:5]]).collect()

            elif fw == ProcessingFramework.PYARROW:
                arrow_table = pa.Table.from_pandas(df)
                # Compute schema and zero-copy slices
                _ = arrow_table.slice(0, len(df))

            elapsed = max(time.perf_counter() - start_time, 1e-5)
            end_rss = self.monitor.process.memory_info().rss / (1024.0 * 1024.0)
            peak_mem = max(end_rss - start_rss, 0.5)
            throughput = n_rows / elapsed

            results.append(
                DataProcessingBenchmarkItem(
                    framework=fw,
                    dataset_rows=n_rows,
                    dataset_columns=n_cols,
                    execution_time_seconds=round(elapsed, 4),
                    throughput_records_per_sec=round(throughput, 1),
                    peak_memory_mb=round(peak_mem, 2),
                    memory_efficiency_score=round(throughput / peak_mem, 2),
                )
            )

        # Sort by throughput descending
        results.sort(key=lambda r: r.throughput_records_per_sec, reverse=True)
        return results
