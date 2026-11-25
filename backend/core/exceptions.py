"""
CANCERWISE AI — Core Exceptions
Centralized domain exception hierarchy for the CANCERWISE AI backend.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CancerwiseException(Exception):
    """Base exception for all CANCERWISE AI domain errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class DatasetNotFoundError(CancerwiseException):
    """Raised when a requested dataset is not found in the registry or storage."""

    def __init__(self, dataset_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Dataset '{dataset_id}' was not found in the registry.",
            code="DATASET_NOT_FOUND",
            status_code=404,
            details=details or {"dataset_id": dataset_id},
        )


class ModelNotFoundError(CancerwiseException):
    """Raised when a requested model artifact or manifest is not found."""

    def __init__(self, model_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Model artifact '{model_id}' was not found.",
            code="MODEL_NOT_FOUND",
            status_code=404,
            details=details or {"model_id": model_id},
        )


class JobNotFoundError(CancerwiseException):
    """Raised when a requested background job identifier is not found."""

    def __init__(self, job_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Background job '{job_id}' does not exist.",
            code="JOB_NOT_FOUND",
            status_code=404,
            details=details or {"job_id": job_id},
        )


class FileValidationError(CancerwiseException):
    """Raised when an uploaded file fails security or format validation."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="FILE_VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class FileSizeExceededError(CancerwiseException):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        size_mb = round(size_bytes / (1024 * 1024), 2)
        max_mb = round(max_bytes / (1024 * 1024), 2)
        super().__init__(
            message=f"Uploaded file size ({size_mb} MB) exceeds maximum allowed size ({max_mb} MB).",
            code="FILE_SIZE_EXCEEDED",
            status_code=413,
            details={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )


class IncompatibleDataError(CancerwiseException):
    """Raised when dataset schemas or feature shapes are incompatible."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="INCOMPATIBLE_DATA_ERROR",
            status_code=422,
            details=details,
        )


class UnapprovedModelError(CancerwiseException):
    """Raised when attempting inference on an unapproved or deprecated model."""

    def __init__(self, model_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Model '{model_id}' has not been approved for clinical inference.",
            code="UNAPPROVED_MODEL_ERROR",
            status_code=403,
            details=details or {"model_id": model_id},
        )


class FeatureMismatchError(CancerwiseException):
    """Raised when inference features do not match trained model expectations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="FEATURE_MISMATCH_ERROR",
            status_code=400,
            details=details,
        )


class AuthenticationError(CancerwiseException):
    """Raised when API authentication fails."""

    def __init__(self, message: str = "Invalid or missing authentication credentials.") -> None:
        super().__init__(
            message=message,
            code="UNAUTHENTICATED",
            status_code=401,
        )


class AuthorizationError(CancerwiseException):
    """Raised when an authenticated user lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions to perform this operation.") -> None:
        super().__init__(
            message=message,
            code="FORBIDDEN",
            status_code=403,
        )


class ProcessingError(CancerwiseException):
    """Raised when an internal machine learning or data processing error occurs."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="PROCESSING_ERROR",
            status_code=500,
            details=details,
        )
