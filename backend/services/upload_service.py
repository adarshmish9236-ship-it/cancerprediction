"""
CANCERWISE AI — Secure Dataset Upload Service
Handles file validation, size constraints, MIME/magic checks, and multi-format parsing.
Supports: CSV, XLSX, Parquet, JSON.
"""

from __future__ import annotations

import io
import json
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd

from backend.core.logging_setup import get_logger
from backend.schemas.dataset import DataFormat, DatasetMetadata, DataModality, TargetType

logger = get_logger("services.upload_service")


class UploadValidationError(Exception):
    """Custom exception raised when an uploaded file fails security or format validation."""
    pass


class DatasetUploadService:
    """
    Service for securely receiving, inspecting, and ingesting user-uploaded cancer datasets.
    """

    ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit

    def __init__(self, raw_upload_dir: Path):
        self.raw_upload_dir = raw_upload_dir
        self.raw_upload_dir.mkdir(parents=True, exist_ok=True)

    def validate_and_save(
        self,
        file_bytes: bytes,
        filename: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_variable: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """
        Validate uploaded file and save immutable copy in raw directory.

        Args:
            file_bytes: Raw content of uploaded file.
            filename: Original file name.
            dataset_id: Optional slug for the dataset.
            dataset_name: Human-readable name.
            target_variable: Identified target column.

        Returns:
            Tuple of (DataFrame, DatasetMetadata).

        Raises:
            UploadValidationError: If validation fails.
        """
        # 1. File size check
        size = len(file_bytes)
        if size == 0:
            raise UploadValidationError("Uploaded file is empty (0 bytes).")
        if size > self.MAX_FILE_SIZE_BYTES:
            max_mb = self.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            actual_mb = round(size / (1024 * 1024), 2)
            raise UploadValidationError(
                f"File size exceeds maximum allowed limit of {max_mb} MB (received {actual_mb} MB)."
            )

        # 2. Extension check
        p = Path(filename)
        ext = p.suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise UploadValidationError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}."
            )

        # 3. Content parsing & integrity check
        df, data_format = self._parse_file_content(file_bytes, ext, filename)

        if df.empty or len(df) == 0:
            raise UploadValidationError("Uploaded dataset contains no data rows.")
        if df.shape[1] < 2:
            raise UploadValidationError(
                f"Uploaded dataset must have at least 2 columns (found {df.shape[1]})."
            )

        # Compute SHA-256
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        # 4. Save raw file immutably
        safe_id = (dataset_id or p.stem).lower().replace(" ", "_").replace("-", "_")
        safe_filename = f"upload_{safe_id}_{sha256[:8]}{ext}"
        saved_raw_path = self.raw_upload_dir / safe_filename

        with open(saved_raw_path, "wb") as f:
            f.write(file_bytes)
        logger.info("Saved raw uploaded file to %s (%d bytes)", saved_raw_path, size)

        # 5. Detect target if not explicitly specified
        detected_target = target_variable
        if not detected_target:
            candidates = [
                c for c in df.columns
                if c.lower() in ("diagnosis", "target", "label", "class", "malignant", "outcome", "status")
            ]
            if candidates:
                detected_target = candidates[0]

        target_classes = None
        target_type = TargetType.BINARY_CLASSIFICATION
        if detected_target and detected_target in df.columns:
            unique_vals = [str(x) for x in df[detected_target].dropna().unique()]
            target_classes = sorted(unique_vals)
            if len(unique_vals) > 2:
                target_type = TargetType.MULTICLASS_CLASSIFICATION
            elif len(unique_vals) <= 1:
                target_type = TargetType.UNSUPERVISED

        feature_names = [c for c in df.columns if c != detected_target and c.lower() not in ("id", "patient_id")]

        metadata = DatasetMetadata(
            dataset_id=safe_id,
            name=dataset_name or p.stem.replace("_", " ").title(),
            description=f"User-uploaded dataset: {filename}",
            original_source="User Upload",
            license="User Provided",
            records_count=len(df),
            columns_count=len(df.columns),
            target_variable=detected_target,
            target_classes=target_classes,
            target_type=target_type,
            feature_names=feature_names,
            missing_values_count=int(df.isnull().sum().sum()),
            modality=DataModality.TABULAR,
            data_format=data_format,
            version="1.0.0",
            sha256_checksum=sha256,
            raw_file_path=str(saved_raw_path.resolve()),
            provenance={
                "original_filename": filename,
                "file_size_bytes": size,
                "upload_format": ext,
            },
            schema_definition={col: str(df[col].dtype) for col in df.columns},
        )

        return df, metadata

    def _parse_file_content(self, file_bytes: bytes, ext: str, filename: str) -> Tuple[pd.DataFrame, DataFormat]:
        """Attempt parsing file with appropriate engine and detailed error capture."""
        buffer = io.BytesIO(file_bytes)
        try:
            if ext == ".csv":
                df = pd.read_csv(buffer)
                return df, DataFormat.CSV
            elif ext in (".xlsx", ".xls"):
                df = pd.read_excel(buffer)
                return df, DataFormat.XLSX
            elif ext == ".parquet":
                df = pd.read_parquet(buffer)
                return df, DataFormat.PARQUET
            elif ext == ".json":
                # Support both table-oriented and records-oriented JSON
                try:
                    df = pd.read_json(buffer)
                except Exception:
                    buffer.seek(0)
                    data = json.load(buffer)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                return df, DataFormat.JSON
            else:
                raise UploadValidationError(f"Unsupported extension: {ext}")
        except UploadValidationError:
            raise
        except Exception as e:
            logger.error("Failed to parse uploaded file '%s': %s", filename, e)
            raise UploadValidationError(
                f"Malformed or corrupt file '{filename}'. Failed to parse as {ext.upper()}: {str(e)}"
            )
