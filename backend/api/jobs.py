"""
CANCERWISE AI — Background Job Management API Endpoints
Provides routes for monitoring, querying, and cancelling asynchronous background tasks.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query, status

from backend.core.logging_setup import get_logger
from backend.services.job_service import get_job_manager, JobInfo, JobStatus

logger = get_logger("api.jobs")
router = APIRouter(prefix="/jobs", tags=["Background Job Management"])

_job_manager = get_job_manager()


@router.get("", response_model=List[JobInfo], summary="List Background Jobs")
def list_jobs(
    limit: int = Query(50, ge=1, le=200, description="Maximum jobs to return"),
    status: Optional[JobStatus] = Query(None, description="Filter by status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED"),
):
    """Lists submitted background execution jobs sorted by creation time."""
    if hasattr(limit, "default"):
        limit = limit.default
    if hasattr(status, "default"):
        status = status.default
    return _job_manager.list_jobs(limit=limit, status_filter=status)



@router.get("/{job_id}", response_model=JobInfo, summary="Get Job Status & Telemetry")
def get_job(job_id: str):
    """
    Retrieves detailed execution status, progress percentage, execution logs,
    completion timestamp, and failure details for a background job.
    """
    return _job_manager.get_job(job_id)


@router.post("/{job_id}/cancel", response_model=JobInfo, summary="Cancel Background Job")
def cancel_job(job_id: str):
    """Requests cancellation of an active or pending background execution job."""
    return _job_manager.cancel_job(job_id)

