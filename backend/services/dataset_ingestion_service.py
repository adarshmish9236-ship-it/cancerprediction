"""
CANCERWISE AI — Dataset Ingestion Service
Handles secure uploads, multi-modal connector execution, and data provenance registration.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.core.exceptions import FileValidationError, FileSizeExceededError
from backend.core.logging_setup import get_logger
from backend.core.config import get_config
from backend.middleware.request_limits import (
    sanitize_filename,
    validate_file_extension,
    validate_file_content_size,
)
from backend.schemas.dataset import DatasetMetadata
from backend.services.dataset_engine import DatasetRegistry
from backend.services.upload_service import DatasetUploadService
from backend.services.connectors.uci_wdbc_connector import UCIWisconsinDiagnosticConnector
from backend.services.connectors.histopathology_connector import HistopathologyImageConnector
from backend.services.connectors.tcga_genomic_connector import TCGAGenomicConnector

logger = get_logger("services.dataset_ingestion")


class DatasetIngestionService:
    """
    Dedicated service handling external data acquisition and authenticated user uploads.
    """

    def __init__(self, data_root: Optional[Path] = None):
        config = get_config()
        self.data_root = data_root or Path(config.paths.data.raw).parent
        self.registry = DatasetRegistry(self.data_root)
        self.upload_service = DatasetUploadService(self.registry.raw_dir)
        self.connectors = {
            "uci_wdbc": UCIWisconsinDiagnosticConnector(),
            "histopathology_breakhis": HistopathologyImageConnector(),
            "tcga_brca_genomic": TCGAGenomicConnector(),
        }

    def ingest_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_variable: Optional[str] = None,
    ) -> DatasetMetadata:
        """
        Validates uploaded file size, sanitized filename, parses content, and registers metadata.
        """
        # 1. Enforce size and extension limits
        validate_file_content_size(file_bytes)
        safe_filename = validate_file_extension(filename)

        # 2. Delegate to upload validation and persistence
        try:
            df, metadata = self.upload_service.validate_and_save(
                file_bytes=file_bytes,
                filename=safe_filename,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                target_variable=target_variable,
            )
            # 3. Register in dataset registry
            registered = self.registry.register_dataset(df=df, metadata=metadata, save_processed=True)
            logger.info("Uploaded dataset registered successfully: %s (%d rows)", registered.dataset_id, len(df))
            return registered
        except Exception as e:
            if isinstance(e, (FileValidationError, FileSizeExceededError)):
                raise
            logger.error("Failed to ingest uploaded dataset '%s': %s", safe_filename, e)
            raise FileValidationError(f"Could not parse or ingest uploaded dataset: {str(e)}")

    def ingest_wdbc(self, force: bool = False) -> DatasetMetadata:
        """Downloads, standardizes, and registers the primary UCI WDBC cohort."""
        connector = self.connectors["uci_wdbc"]
        raw_path = connector.fetch_or_download(self.registry.raw_dir, force=force)
        df, extra_info = connector.load_and_parse(raw_path)
        metadata = connector.create_metadata(
            df=df,
            raw_path=raw_path,
            processed_path=self.registry.processed_dir / "uci_wdbc.parquet",
            extra_info=extra_info,
        )
        return self.registry.register_dataset(df=df, metadata=metadata, save_processed=True)

    def ingest_connector(self, connector_key: str, force: bool = False) -> DatasetMetadata:
        """Ingests multi-modal cancer dataset through registered connector."""
        if connector_key not in self.connectors:
            raise KeyError(f"Unknown connector '{connector_key}'. Available: {list(self.connectors.keys())}")
        connector = self.connectors[connector_key]
        raw_path = connector.fetch_or_download(self.registry.raw_dir, force=force)
        df, extra_info = connector.load_and_parse(raw_path)
        metadata = connector.create_metadata(df, raw_path=raw_path, extra_info=extra_info)
        return self.registry.register_dataset(df, metadata, save_processed=True)

    def list_sources(self) -> List[Dict[str, Any]]:
        """Returns registered biomedical data sources."""
        return self.registry.source_mgr.list_sources()
