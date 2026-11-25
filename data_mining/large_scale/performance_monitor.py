"""
CANCERWISE AI — Performance Monitor & Telemetry Engine
Monitors host CPU, physical RAM, GPU VRAM, and process-level memory footprints.
Provides high-precision contextual timing, throughput profiling, and hardware status snapshots.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Dict, Generator, Optional, Tuple
import psutil
import torch

from backend.schemas.large_scale import SystemResourceSnapshot

logger = logging.getLogger("cancerwise.large_scale.monitor")


class PerformanceMonitor:
    """
    Real-time telemetry and resource usage tracker.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_system_snapshot(self) -> SystemResourceSnapshot:
        """
        Captures instantaneous hardware and process resource state.
        """
        vm = psutil.virtual_memory()
        mem_info = self.process.memory_info()

        # Check GPU availability
        gpu_avail = torch.cuda.is_available()
        gpu_name = None
        gpu_alloc_mb = None

        if gpu_avail:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024.0 * 1024.0), 2)
            except Exception:
                gpu_avail = False

        return SystemResourceSnapshot(
            cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
            cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            ram_total_mb=round(vm.total / (1024.0 * 1024.0), 2),
            ram_used_mb=round(vm.used / (1024.0 * 1024.0), 2),
            ram_available_mb=round(vm.available / (1024.0 * 1024.0), 2),
            ram_percent=float(vm.percent),
            process_rss_mb=round(mem_info.rss / (1024.0 * 1024.0), 2),
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            gpu_memory_allocated_mb=gpu_alloc_mb,
        )

    @contextlib.contextmanager
    def track_resources(self, operation_name: str = "operation") -> Generator[Dict[str, Any], None, None]:
        """
        Context manager measuring duration, start/end RSS, peak memory delta, and throughput.
        """
        start_time = time.perf_counter()
        start_rss = self.process.memory_info().rss / (1024.0 * 1024.0)
        perf_data: Dict[str, Any] = {"operation": operation_name}

        yield perf_data

        elapsed = time.perf_counter() - start_time
        end_rss = self.process.memory_info().rss / (1024.0 * 1024.0)
        mem_delta = max(0.0, end_rss - start_rss)

        perf_data["elapsed_seconds"] = round(elapsed, 4)
        perf_data["start_rss_mb"] = round(start_rss, 2)
        perf_data["end_rss_mb"] = round(end_rss, 2)
        perf_data["memory_delta_mb"] = round(mem_delta, 2)

        if "num_records" in perf_data and elapsed > 0:
            throughput = perf_data["num_records"] / elapsed
            perf_data["throughput_records_per_sec"] = round(throughput, 1)

        logger.info(
            "Performance: '%s' completed in %.4fs | Memory: %.2fMB -> %.2fMB (Delta: %.2fMB)",
            operation_name, elapsed, start_rss, end_rss, mem_delta,
        )
