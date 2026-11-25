"""
CANCERWISE AI — Dataset Management Service
Orchestrates registry lookups, dataset deletion, structural profiling, and previews.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.core.exceptions import DatasetNotFoundError
from backend.core.logging_setup import get_logger
from backend.schemas.dataset import (
    DatasetMetadata,
    DatasetPreview,
    DatasetStatisticsSummary,
    DatasetValidationReport,
)
from backend.services.dataset_engine import DatasetRegistry
from backend.core.config import get_config

logger = get_logger("services.dataset_management")


class DatasetManagementService:
    """
    Dedicated service for dataset lifecycle management, discovery, deletion, and profiling.
    """

    def __init__(self, data_root: Optional[Path] = None):
        config = get_config()
        self.data_root = data_root or Path(config.paths.data.raw).parent
        self.registry = DatasetRegistry(self.data_root)

    def list_datasets(self) -> List[DatasetMetadata]:
        """Lists all registered oncology datasets."""
        return self.registry.list_datasets()

    def get_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Retrieves metadata for a specific dataset or raises DatasetNotFoundError."""
        meta = self.registry.get_metadata(dataset_id)
        if not meta:
            raise DatasetNotFoundError(dataset_id)
        return meta

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """Loads dataset into a Pandas DataFrame or raises DatasetNotFoundError."""
        df = self.registry.get_dataset(dataset_id)
        if df is None:
            raise DatasetNotFoundError(dataset_id)
        return df

    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Deletes raw, processed, and metadata artifacts for a given dataset ID.
        """
        meta = self.registry.get_metadata(dataset_id)
        if not meta:
            raise DatasetNotFoundError(dataset_id)

        deleted_files: List[str] = []

        # Delete raw file
        if meta.raw_file_path:
            raw_p = Path(meta.raw_file_path)
            if raw_p.exists():
                try:
                    raw_p.unlink()
                    deleted_files.append(str(raw_p))
                except Exception as e:
                    logger.warning("Could not delete raw file %s: %s", raw_p, e)

        # Delete processed file
        if meta.processed_file_path:
            proc_p = Path(meta.processed_file_path)
            if proc_p.exists():
                try:
                    proc_p.unlink()
                    deleted_files.append(str(proc_p))
                except Exception as e:
                    logger.warning("Could not delete processed file %s: %s", proc_p, e)

        # Delete metadata file
        for suffix in [f"{dataset_id}.json", f"{dataset_id}_metadata.json"]:
            meta_p = self.registry.metadata_dir / suffix
            if meta_p.exists():
                try:
                    meta_p.unlink()
                    deleted_files.append(str(meta_p))
                except Exception as e:
                    logger.warning("Could not delete metadata file %s: %s", meta_p, e)
        if hasattr(self.registry, "_metadata_cache") and dataset_id in self.registry._metadata_cache:
            del self.registry._metadata_cache[dataset_id]


        logger.info("Successfully deleted dataset '%s' (%d files removed)", dataset_id, len(deleted_files))
        return {
            "status": "SUCCESS",
            "message": f"Dataset '{dataset_id}' successfully deleted.",
            "deleted_files": deleted_files,
        }

    def get_preview(self, dataset_id: str, n_rows: int = 5) -> DatasetPreview:
        """Returns head, tail, dimensions, and types for dataset."""
        preview = self.registry.get_preview(dataset_id, n_rows=n_rows)
        if not preview:
            raise DatasetNotFoundError(dataset_id)
        return preview

    def get_statistics(self, dataset_id: str) -> DatasetStatisticsSummary:
        """Returns descriptive statistical metrics."""
        stats = self.registry.get_statistics(dataset_id)
        if not stats:
            raise DatasetNotFoundError(dataset_id)
        return stats

    def get_profile(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generates a comprehensive dataset profile combining dimensions, dtypes,
        missing value percentages, duplicate counts, memory usage, and column distributions.
        """
        df = self.get_dataframe(dataset_id)
        meta = self.get_metadata(dataset_id)

        col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
        missing_pct = {col: round(cnt / len(df) * 100.0, 2) for col, cnt in missing_counts.items()}
        duplicate_count = int(df.duplicated().sum())
        memory_bytes = int(df.memory_usage(deep=True).sum())

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        num_summary = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            num_summary = {
                col: {
                    stat: round(float(val), 4) if pd.notnull(val) else None
                    for stat, val in stats.items()
                }
                for col, stats in desc.items()
            }

        target_distribution = {}
        if meta.target_variable and meta.target_variable in df.columns:
            target_distribution = {
                str(k): int(v) for k, v in df[meta.target_variable].value_counts().items()
            }

        return {
            "dataset_id": dataset_id,
            "dataset_name": meta.name,
            "modality": meta.modality.value if hasattr(meta.modality, "value") else str(meta.modality),
            "total_samples": len(df),
            "total_features": len(df.columns),
            "target_variable": meta.target_variable,
            "target_distribution": target_distribution,
            "duplicate_samples": duplicate_count,
            "memory_usage_mb": round(memory_bytes / (1024 * 1024), 3),
            "column_types": col_types,
            "missing_values": missing_counts,
            "missing_percentages": missing_pct,
            "numeric_feature_summary": num_summary,
            "created_at": getattr(meta, "download_timestamp", getattr(meta, "created_at", None)),
            "provenance": meta.provenance,
        }

