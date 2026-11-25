"""
CANCERWISE AI — Request Size Limits & Safe File Paths Middleware
Guards against path traversal attacks, unapproved file extensions, and memory exhaustion.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Set

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from backend.core.exceptions import FileSizeExceededError, FileValidationError

# Maximum upload limit: 100 MB default
MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024

# Allowed biomedical dataset extensions
ALLOWED_DATASET_EXTENSIONS: Set[str] = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json"}


def sanitize_filename(filename: str) -> str:
    """
    Strips directory paths, illegal characters, and enforces safe biomedical filenames.
    Prevents path traversal attacks (e.g. '../../etc/passwd').
    """
    # Extract only the base name
    clean_name = Path(filename).name

    # Remove dangerous characters
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name)

    # Prevent hidden files
    while clean_name.startswith("."):
        clean_name = clean_name[1:]

    if not clean_name:
        clean_name = "uploaded_dataset.csv"

    return clean_name


def validate_file_extension(filename: str) -> str:
    """Verifies that the file possesses an approved medical dataset extension."""
    clean_name = sanitize_filename(filename)
    suffix = Path(clean_name).suffix.lower()
    if suffix not in ALLOWED_DATASET_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file format '{suffix}'. Approved formats: {sorted(list(ALLOWED_DATASET_EXTENSIONS))}"
        )
    return clean_name


def validate_file_content_size(content_bytes: bytes, max_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> None:
    """Validates that loaded bytes do not exceed allowed upload limits."""
    if len(content_bytes) == 0:
        raise FileValidationError("Uploaded file is empty (0 bytes).")
    if len(content_bytes) > max_bytes:
        raise FileSizeExceededError(size_bytes=len(content_bytes), max_bytes=max_bytes)


class RequestSizeLimitMiddleware:
    """
    Pure ASGI middleware checking Content-Length header against max upload limits.
    Rejects oversized requests early before streaming large payloads into memory.
    """

    def __init__(self, app, max_upload_size: int = MAX_UPLOAD_SIZE_BYTES):
        self.app = app
        self.max_upload_size = max_upload_size

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length:
                try:
                    length = int(content_length)
                    if length > self.max_upload_size:
                        size_mb = round(length / (1024 * 1024), 2)
                        max_mb = round(self.max_upload_size / (1024 * 1024), 2)
                        from fastapi.responses import JSONResponse
                        from backend.middleware.error_handler import create_error_payload
                        payload = create_error_payload(
                            code="FILE_SIZE_EXCEEDED",
                            message=f"Request size ({size_mb} MB) exceeds maximum allowed limit ({max_mb} MB).",
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            path=scope.get("path", ""),
                            details={"content_length": length, "max_allowed": self.max_upload_size},
                        )
                        response = JSONResponse(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content=payload,
                        )
                        await response(scope, receive, send)
                        return
                except ValueError:
                    pass

        await self.app(scope, receive, send)
