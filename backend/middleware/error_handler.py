"""
CANCERWISE AI — Centralized Error Handling Middleware
Translates internal exceptions into standardized, clinically-safe HTTP JSON responses.
"""

from __future__ import annotations

from datetime import datetime, timezone
import traceback
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.core.exceptions import CancerwiseException
from backend.core.logging_setup import get_logger

logger = get_logger("middleware.error_handler")


def create_error_payload(
    code: str,
    message: str,
    status_code: int,
    path: str,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Formats standardized error payload."""
    return {
        "error": {
            "code": code,
            "message": message,
            "status_code": status_code,
            "path": path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Registers centralized exception handlers on the FastAPI application."""

    @app.exception_handler(CancerwiseException)
    async def cancerwise_exception_handler(request: Request, exc: CancerwiseException) -> JSONResponse:
        logger.warning(
            "CancerwiseException [%s] on %s: %s (status %d)",
            exc.code,
            request.url.path,
            exc.message,
            exc.status_code,
        )
        payload = create_error_payload(
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            path=request.url.path,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed on %s: %s", request.url.path, exc.errors())
        details = {"validation_errors": exc.errors()}
        payload = create_error_payload(
            code="REQUEST_VALIDATION_ERROR",
            message="Request body or query parameters failed schema validation.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            path=request.url.path,
            details=details,
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        code = "HTTP_ERROR"
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            code = "NOT_FOUND"
        elif exc.status_code == status.HTTP_401_UNAUTHORIZED:
            code = "UNAUTHORIZED"
        elif exc.status_code == status.HTTP_403_FORBIDDEN:
            code = "FORBIDDEN"
        elif exc.status_code == status.HTTP_400_BAD_REQUEST:
            code = "BAD_REQUEST"

        detail_msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        details = exc.detail if isinstance(exc.detail, dict) else {}
        payload = create_error_payload(
            code=code,
            message=detail_msg,
            status_code=exc.status_code,
            path=request.url.path,
            details=details,
        )
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception on %s: %s\n%s", request.url.path, exc, traceback.format_exc())
        payload = create_error_payload(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred during processing.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            path=request.url.path,
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload)
