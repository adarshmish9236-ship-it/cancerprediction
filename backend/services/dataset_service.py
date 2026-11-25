"""
CANCERWISE AI — Unified Dataset Ingestion Service
Orchestrates downloading, connector execution, storage routing, and registry updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from backend.core.config import get_config
from backend.core.logging_setup import get_logger
from backend.schemas.dataset import (
    DatasetMetadata, DatasetPreview, DatasetStatisticsSummary,
    DatasetValidationReport, SchemaCompatibilityReport
)
from backend.services.dataset_engine import DatasetRegistry
from backend.services.upload_service import DatasetUploadService
from backend.services.compatibility_checker import SchemaCompatibilityChecker
from backend.services.connectors.uci_wdbc_connector import UCIWisconsinDiagnosticConnector
from backend.services.connectors.histopathology_connector import HistopathologyImageConnector
from backend.services.connectors.tcga_genomic_connector import TCGAGenomicConnector

logger = get_logger("services.dataset_service")


class DatasetService:
    """
    Facade service coordinating all dataset ingestion, registration, and inspection.
    """

    def __init__(self, data_root: Optional[Path] = None):
        config = get_config()
        self.data_root = data_root or Path(config.paths.data.raw).parent
        self.registry = DatasetRegistry(self.data_root)
        self.upload_service = DatasetUploadService(self.registry.raw_dir)
        self.compatibility_checker = SchemaCompatibilityChecker()

        # Registered connectors
        self.connectors = {
            "uci_wdbc": UCIWisconsinDiagnosticConnector(),
            "histopathology_breakhis": HistopathologyImageConnector(),
            "tcga_brca_genomic": TCGAGenomicConnector(),
        }

    def ingest_wdbc(self, force: bool = False) -> DatasetMetadata:
        """
        Ingest the primary Wisconsin Diagnostic Breast Cancer dataset from UCI.
        """
        connector = self.connectors["uci_wdbc"]
        logger.info("Starting ingestion of primary dataset: %s", connector.name)

        # 1. Download/fetch raw
        raw_path = connector.fetch_or_download(self.registry.raw_dir, force=force)

        # 2. Parse and validate
        df, extra_info = connector.load_and_parse(raw_path)

        # 3. Create metadata
        metadata = connector.create_metadata(
            df=df,
            raw_path=raw_path,
            processed_path=self.registry.processed_dir / "uci_wdbc.parquet",
            extra_info=extra_info,
        )

        # 4. Register
        registered_meta = self.registry.register_dataset(df=df, metadata=metadata, save_processed=True)
        logger.info("UCI WDBC dataset successfully ingested and registered (%d rows, %d cols).", len(df), len(df.columns))
        return registered_meta

    def ingest_connector(self, connector_key: str, force: bool = False) -> DatasetMetadata:
        """Ingest any registered connector (histopathology, tcga, etc.)."""
        if connector_key not in self.connectors:
            raise KeyError(f"Unknown connector '{connector_key}'. Available: {list(self.connectors.keys())}")

        connector = self.connectors[connector_key]
        raw_path = connector.fetch_or_download(self.registry.raw_dir, force=force)
        df, extra_info = connector.load_and_parse(raw_path)
        meta = connector.create_metadata(df, raw_path=raw_path, extra_info=extra_info)
        return self.registry.register_dataset(df, meta, save_processed=True)

    def upload_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_variable: Optional[str] = None,
    ) -> DatasetMetadata:
        """Process user upload, validate, and register."""
        df, metadata = self.upload_service.validate_and_save(
            file_bytes=file_bytes,
            filename=filename,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            target_variable=target_variable,
        )
        return self.registry.register_dataset(df=df, metadata=metadata, save_processed=True)

    def list_datasets(self) -> List[DatasetMetadata]:
        return self.registry.list_datasets()

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        return self.registry.get_dataset(dataset_id)

    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Alias for get_dataset."""
        return self.registry.get_dataset(dataset_id)

    def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        return self.registry.get_metadata(dataset_id)

    def get_preview(self, dataset_id: str, n_rows: int = 5) -> Optional[DatasetPreview]:
        return self.registry.get_preview(dataset_id, n_rows=n_rows)

    def get_statistics(self, dataset_id: str) -> Optional[DatasetStatisticsSummary]:
        return self.registry.get_statistics(dataset_id)

    def validate_dataset(self, dataset_id: str) -> Optional[DatasetValidationReport]:
        return self.registry.validate_dataset(dataset_id)

    def check_compatibility(self, dataset_a_id: str, dataset_b_id: str) -> SchemaCompatibilityReport:
        meta_a = self.registry.get_metadata(dataset_a_id)
        meta_b = self.registry.get_metadata(dataset_b_id)
        if not meta_a:
            raise ValueError(f"Dataset '{dataset_a_id}' is not registered.")
        if not meta_b:
            raise ValueError(f"Dataset '{dataset_b_id}' is not registered.")

        df_a = self.registry.get_dataset(dataset_a_id)
        df_b = self.registry.get_dataset(dataset_b_id)
        return self.compatibility_checker.check_compatibility(meta_a, meta_b, df_a, df_b)

    def get_sources(self) -> List[Dict[str, Any]]:
        return self.registry.source_mgr.list_sources()

    def get_data_dictionary(self, name: str = "wdbc") -> Dict[str, Any]:
        """Retrieve attribute definitions and clinical target semantics."""
        connector = self.connectors.get("uci_wdbc")
        cols = connector.COLUMN_NAMES if connector else (
            ["id", "diagnosis"]
            + [f"{f}_{suffix}" for suffix in ["mean", "se", "worst"] for f in [
                "radius", "texture", "perimeter", "area", "smoothness",
                "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
            ]]
        )
        attributes = []
        for c in cols:
            attributes.append({
                "name": c,
                "type": "categorical" if c == "diagnosis" else ("identifier" if c == "id" else "continuous"),
                "description": f"Measurement for {c}",
            })
        return {
            "dataset": name,
            "attributes": attributes,
            "target_semantics": {
                "M": "Malignant",
                "B": "Benign",
                "Malignant": "Confirmed malignant cytological biopsy",
                "Benign": "Confirmed benign cytological biopsy",
            },
        }
