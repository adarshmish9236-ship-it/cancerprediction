"""
CANCERWISE AI — Asynchronous Job Management Service
Coordinates non-blocking background execution for long-running model training,
large-scale dataset processing, and publication report compilation.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import enum
import logging
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from backend.core.exceptions import JobNotFoundError

logger = logging.getLogger("cancerwise.services.job")


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobInfo(BaseModel):
    """Pydantic representation of a background job's lifecycle and telemetry."""

    job_id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Normalized progress from 0.0 to 1.0")
    progress_percent: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage 0-100%")
    message: str = "Job queued"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    logs: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    failure_details: Optional[Dict[str, Any]] = None


class JobContext:
    """Thread-safe context passed to worker functions to record progress and logs."""

    def __init__(self, job: JobInfo, lock: threading.Lock, cancel_event: threading.Event):
        self._job = job
        self._lock = lock
        self._cancel_event = cancel_event

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.is_cancelled:
            raise InterruptedError(f"Job {self._job.job_id} was cancelled by user.")

    def log(self, text: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {text}"
        with self._lock:
            self._job.logs.append(entry)
        logger.info("[Job %s] %s", self._job.job_id, text)

    def set_progress(self, progress: float, message: Optional[str] = None) -> None:
        progress = max(0.0, min(1.0, float(progress)))
        with self._lock:
            self._job.progress = round(progress, 4)
            self._job.progress_percent = round(progress * 100.0, 2)
            if message:
                self._job.message = message
        if message:
            self.log(f"Progress: {self._job.progress_percent:.1f}% — {message}")


class JobManager:
    """
    Singleton manager tracking async worker jobs.
    Uses a dedicated thread pool to keep the FastAPI event loop responsive.
    """

    _instance: Optional["JobManager"] = None
    _init_lock = threading.Lock()

    def __new__(cls, max_workers: int = 4) -> "JobManager":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_workers: int = 4) -> None:
        if getattr(self, "_initialized", False):
            return
        self.max_workers = max_workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cw_job_worker",
        )
        self._jobs: Dict[str, JobInfo] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        self._initialized = True
        logger.info("JobManager initialized with %d worker threads.", max_workers)

    def submit(
        self,
        job_type: str,
        target_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> JobInfo:
        """Submits a callable for background execution and returns initial JobInfo."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = JobInfo(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            message="Job accepted and queued for worker pool execution.",
        )
        cancel_event = threading.Event()

        with self._lock:
            self._jobs[job_id] = job
            self._cancel_events[job_id] = cancel_event

        future = self._executor.submit(
            self._execute_wrapper,
            job_id,
            target_fn,
            cancel_event,
            *args,
            **kwargs,
        )

        with self._lock:
            self._futures[job_id] = future

        logger.info("Job submitted: %s (Type: %s)", job_id, job_type)
        return job

    def _execute_wrapper(
        self,
        job_id: str,
        target_fn: Callable[..., Any],
        cancel_event: threading.Event,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Internal execution wrapper executing on worker thread."""
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.message = "Execution started"

        start_time = time.perf_counter()
        ctx = JobContext(job, self._lock, cancel_event)
        ctx.log(f"Starting execution of {job.job_type}")

        try:
            # Check if callable accepts JobContext as first argument or keyword 'ctx'
            import inspect
            sig = inspect.signature(target_fn)
            if "ctx" in sig.parameters or len(sig.parameters) > len(args):
                result = target_fn(ctx, *args, **kwargs)
            else:
                result = target_fn(*args, **kwargs)

            # Check if cancellation was triggered
            if cancel_event.is_set():
                with self._lock:
                    job.status = JobStatus.CANCELLED
                    job.message = "Job execution cancelled by user request."
            else:
                with self._lock:
                    job.status = JobStatus.COMPLETED
                    job.progress = 1.0
                    job.progress_percent = 100.0
                    job.message = "Execution completed successfully."
                    # Serialize result safely if dict or pydantic
                    if hasattr(result, "model_dump"):
                        job.result = result.model_dump()
                    elif isinstance(result, dict):
                        job.result = result
                    elif result is not None:
                        job.result = {"output": str(result)}

            ctx.log(f"Finished {job.job_type} successfully.")

        except InterruptedError as ie:
            with self._lock:
                job.status = JobStatus.CANCELLED
                job.message = str(ie)
            ctx.log(f"Execution cancelled: {ie}")

        except Exception as e:
            tb = traceback.format_exc()
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.message = f"Execution failed: {str(e)}"
                job.failure_details = {
                    "exception_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": tb,
                }
            ctx.log(f"Execution failed with exception: {str(e)}")
            logger.error("Job %s failed: %s\n%s", job_id, e, tb)

        finally:
            elapsed = round(time.perf_counter() - start_time, 3)
            with self._lock:
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.elapsed_seconds = elapsed

    def get_job(self, job_id: str) -> JobInfo:
        """Retrieves JobInfo by identifier or raises JobNotFoundError."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        return job

    def list_jobs(
        self,
        limit: int = 50,
        status_filter: Optional[JobStatus] = None,
    ) -> List[JobInfo]:
        """Lists active and past background jobs."""
        with self._lock:
            jobs = list(self._jobs.values())

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        # Sort newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> JobInfo:
        """Cancels a pending or running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            cancel_event = self._cancel_events.get(job_id)
            future = self._futures.get(job_id)

        if not job:
            raise JobNotFoundError(job_id)

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return job

        if cancel_event:
            cancel_event.set()

        if future and not future.running():
            future.cancel()

        with self._lock:
            job.status = JobStatus.CANCELLED
            job.message = "Job cancellation requested by client."
            job.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info("Cancellation requested for job %s", job_id)
        return job


_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
