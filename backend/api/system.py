"""
CANCERWISE AI — System Telemetry & Status API Endpoints
Provides system status, service availability, and hardware resource telemetry.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter

from backend.core.logging_setup import get_logger
from backend.schemas.large_scale import SystemResourceSnapshot
from backend.services.system_service import SystemService

logger = get_logger("api.system")
router = APIRouter(prefix="/system", tags=["System Telemetry & Status"])

_system_service = SystemService()


@router.get("/status", summary="Get System Status")
def get_system_status() -> Dict[str, Any]:
    """Captures operational status of all platform engines and data registry counts."""
    return _system_service.get_system_status()


@router.get("/resources", response_model=SystemResourceSnapshot, summary="Get Host Resource Telemetry")
def get_system_resources():
    """Captures hardware telemetry: CPU load, core count, physical RAM, process memory, and GPU VRAM."""
    return _system_service.get_system_resources()
