"""
CANCERWISE AI — Dataset Management Engine
Contains core management components:
- DatasetRegistry
- DatasetLoader
- DatasetValidator
- DatasetMetadataManager
- DatasetVersionManager
- DatasetSourceManager
- DatasetStatistics
"""

from __future__ import annotations

import json
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.core.logging_setup import get_logger
from backend.schemas.dataset import (
    DatasetMetadata, DatasetValidationReport, DatasetStatisticsSummary,
    DatasetColumnInfo, DatasetPreview, DataModality, DataFormat, TargetType
)

logger = get_logger("services.dataset_engine")


# ===========================================================================
# 1. DatasetStatistics
# ===========================================================================
class DatasetStatistics:
    """Computes comprehensive dataset exploration metrics and descriptive statistics."""

    @staticmethod
    def compute(df: pd.DataFrame, target_col: Optional[str] = None) -> DatasetStatisticsSummary:
        total_missing = int(df.isnull().sum().sum())
        total_cells = df.shape[0] * df.shape[1] if df.shape[1] > 0 else 1
        missing_pct = round((total_missing / total_cells) * 100, 2)
        memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3)
        dup_count = int(df.duplicated().sum())

        target_dist: Optional[Dict[str, int]] = None
        target_pcts: Optional[Dict[str, float]] = None

        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts().to_dict()
            target_dist = {str(k): int(v) for k, v in counts.items()}
            total_targets = len(df[target_col].dropna())
            if total_targets > 0:
                target_pcts = {
                    str(k): round((v / total_targets) * 100, 2)
                    for k, v in counts.items()
                }

        cols_info: List[DatasetColumnInfo] = []
        num_count = 0
        cat_count = 0

        for col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            if is_num:
                num_count += 1
            else:
                cat_count += 1

            null_c = int(df[col].isnull().sum())
            null_p = round((null_c / len(df)) * 100, 2) if len(df) > 0 else 0.0
            unique_c = int(df[col].nunique())

            is_id = col.lower() in ("id", "patient_id", "sample_id", "case_id")

            cols_info.append(
                DatasetColumnInfo(
                    name=col,
                    dtype=str(df[col].dtype),
                    null_count=null_c,
                    null_percentage=null_p,
                    unique_count=unique_c,
                    is_target=(col == target_col),
                    is_identifier=is_id,
                )
            )

        return DatasetStatisticsSummary(
            records_count=int(df.shape[0]),
            columns_count=int(df.shape[1]),
            memory_usage_mb=memory_mb,
            duplicate_rows_count=dup_count,
            total_missing_values=total_missing,
            missing_percentage=missing_pct,
            target_distribution=target_dist,
            target_percentages=target_pcts,
            numeric_feature_count=num_count,
            categorical_feature_count=cat_count,
            columns=cols_info,
        )


# ===========================================================================
# 2. DatasetValidator
# ===========================================================================
class DatasetValidator:
    """Validates schema integrity, data types, value boundaries, and target variables."""

    @staticmethod
    def validate(
        df: pd.DataFrame,
        metadata: Optional[DatasetMetadata] = None,
        expected_target: Optional[str] = None,
        max_missing_ratio: float = 0.5,
    ) -> DatasetValidationReport:
        dataset_id = metadata.dataset_id if metadata else "unregistered"
        errors: List[str] = []
        warnings: List[str] = []
        checks_total = 0
        checks_passed = 0

        # Check 1: Non-empty DataFrame
        checks_total += 1
        if df.empty or len(df) == 0:
            errors.append("Dataset is completely empty.")
        else:
            checks_passed += 1

        # Check 2: Minimum records threshold
        checks_total += 1
        if len(df) < 5:
            warnings.append(f"Very small dataset sample size: {len(df)} records.")
        else:
            checks_passed += 1

        # Check 3: Minimum features threshold
        checks_total += 1
        if df.shape[1] < 2:
            errors.append(f"Insufficient number of columns: {df.shape[1]}. Need at least 2.")
        else:
            checks_passed += 1

        # Check 4: Target variable presence
        target = expected_target or (metadata.target_variable if metadata else None)
        checks_total += 1
        if target:
            if target not in df.columns:
                errors.append(f"Required target variable '{target}' not found in columns.")
            else:
                checks_passed += 1
                # Check 5: Target has at least 2 distinct classes
                checks_total += 1
                n_classes = df[target].nunique(dropna=True)
                if n_classes < 2:
                    errors.append(f"Target '{target}' has only {n_classes} distinct class. Must have at least 2.")
                else:
                    checks_passed += 1
        else:
            checks_passed += 1

        # Check 6: Missing values ratio
        checks_total += 1
        high_missing_cols = []
        for col in df.columns:
            ratio = df[col].isnull().mean()
            if ratio > max_missing_ratio:
                high_missing_cols.append(f"{col} ({round(ratio*100, 1)}%)")
        if high_missing_cols:
            warnings.append(f"Columns with >{int(max_missing_ratio*100)}% missing values: {', '.join(high_missing_cols)}")
        checks_passed += 1

        # Check 7: Duplicate records
        checks_total += 1
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate row(s) in dataset.")
        checks_passed += 1

        # Check 8: Column naming hygiene
        checks_total += 1
        whitespace_cols = [c for c in df.columns if str(c).strip() != str(c) or " " in str(c)]
        if whitespace_cols:
            warnings.append(f"Columns with whitespace in name: {whitespace_cols[:5]}")
        checks_passed += 1

        # Check 9: File integrity & Checksum (if metadata provided)
        checksum_ok = True
        if metadata and metadata.raw_file_path:
            checks_total += 1
            raw_p = Path(metadata.raw_file_path)
            if raw_p.exists():
                sha256 = hashlib.sha256()
                with open(raw_p, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                computed = sha256.hexdigest()
                if computed == metadata.sha256_checksum:
                    checks_passed += 1
                else:
                    checksum_ok = False
                    errors.append("SHA-256 checksum mismatch: Raw file may have been altered or corrupted.")
            else:
                warnings.append(f"Raw source file not accessible at: {metadata.raw_file_path}")
                checks_passed += 1

        is_valid = (len(errors) == 0)

        return DatasetValidationReport(
            dataset_id=dataset_id,
            is_valid=is_valid,
            total_checks=checks_total,
            passed_checks=checks_passed,
            failed_checks=len(errors),
            errors=errors,
            warnings=warnings,
            integrity_verified=is_valid,
            checksum_matched=checksum_ok,
            summary={
                "shape": list(df.shape),
                "columns": list(df.columns),
                "duplicate_rows": int(dup_count),
                "total_nulls": int(df.isnull().sum().sum()),
            },
        )


# ===========================================================================
# 3. DatasetMetadataManager
# ===========================================================================
class DatasetMetadataManager:
    """Manages metadata serialization, schema definitions, and dictionary retrieval."""

    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: DatasetMetadata) -> Path:
        out_file = self.metadata_dir / f"{metadata.dataset_id}_metadata.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        logger.info("Saved metadata for '%s' to %s", metadata.dataset_id, out_file)
        return out_file

    def load_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        meta_file = self.metadata_dir / f"{dataset_id}_metadata.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DatasetMetadata(**data)

    def list_all_metadata(self) -> List[DatasetMetadata]:
        records: List[DatasetMetadata] = []
        for p in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    records.append(DatasetMetadata(**json.load(f)))
            except Exception as e:
                logger.warning("Could not read metadata file %s: %s", p, e)
        return records


# ===========================================================================
# 4. DatasetVersionManager
# ===========================================================================
class DatasetVersionManager:
    """Tracks dataset version snapshots, history, and prevents accidental overwrites."""

    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.version_dir / "version_history.json"
        self._history = self._load_history()

    def _load_history(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_history(self) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def register_version(
        self,
        dataset_id: str,
        version: str,
        checksum: str,
        records_count: int,
        columns_count: int,
        changelog: str = "Initial ingestion",
    ) -> Dict[str, Any]:
        entry = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "records_count": records_count,
            "columns_count": columns_count,
            "changelog": changelog,
        }
        if dataset_id not in self._history:
            self._history[dataset_id] = []
        self._history[dataset_id].append(entry)
        self._save_history()
        logger.info("Registered version %s for dataset '%s'", version, dataset_id)
        return entry

    def get_version_history(self, dataset_id: str) -> List[Dict[str, Any]]:
        return self._history.get(dataset_id, [])


# ===========================================================================
# 5. DatasetSourceManager
# ===========================================================================
class DatasetSourceManager:
    """Maintains register of verified biomedical data sources and repositories."""

    DEFAULT_SOURCES = [
        {
            "source_id": "uci_ml_repository",
            "name": "UCI Machine Learning Repository",
            "url": "https://archive.ics.uci.edu",
            "license_type": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "supported_modalities": ["TABULAR"],
            "cancer_cohorts": ["Wisconsin Diagnostic Breast Cancer", "Cervical Cancer Risk", "Haberman's Survival"],
            "trust_level": "High (Peer-Reviewed Academic)",
        },
        {
            "source_id": "tcga_gdc",
            "name": "The Cancer Genome Atlas (NCI GDC)",
            "url": "https://portal.gdc.cancer.gov",
            "license_type": "Open Access NIH Genomic Data Sharing Policy",
            "supported_modalities": ["GENOMIC", "TABULAR", "IMAGE"],
            "cancer_cohorts": ["TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD"],
            "trust_level": "Clinical/Genomic Gold Standard",
        },
        {
            "source_id": "breakhis_database",
            "name": "Breast Cancer Histopathological Database (BreakHis)",
            "url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
            "license_type": "Academic Non-Commercial Research License",
            "supported_modalities": ["IMAGE"],
            "cancer_cohorts": ["Microscopic Biopsy Slides (40X - 400X)"],
            "trust_level": "Peer-Reviewed Pathological Benchmark",
        },
    ]

    def __init__(self, metadata_dir: Path):
        self.sources_file = metadata_dir / "data_sources.json"
        self._sources = self._load_or_initialize()

    def _load_or_initialize(self) -> List[Dict[str, Any]]:
        if self.sources_file.exists():
            try:
                with open(self.sources_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        with open(self.sources_file, "w", encoding="utf-8") as f:
            json.dump(self.DEFAULT_SOURCES, f, indent=2)
        return list(self.DEFAULT_SOURCES)

    def list_sources(self) -> List[Dict[str, Any]]:
        return self._sources

    def register_source(self, source_info: Dict[str, Any]) -> None:
        self._sources.append(source_info)
        with open(self.sources_file, "w", encoding="utf-8") as f:
            json.dump(self._sources, f, indent=2)


# ===========================================================================
# 6. DatasetLoader
# ===========================================================================
class DatasetLoader:
    """Loads datasets into pandas or polars from raw/processed storage."""

    @staticmethod
    def load(file_path: Union[str, Path], format_hint: Optional[DataFormat] = None) -> pd.DataFrame:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found at: {p}")

        suffix = p.suffix.lower()
        if suffix == ".csv" or format_hint == DataFormat.CSV:
            return pd.read_csv(p)
        elif suffix == ".parquet" or format_hint == DataFormat.PARQUET:
            return pd.read_parquet(p)
        elif suffix in (".xlsx", ".xls") or format_hint == DataFormat.XLSX:
            return pd.read_excel(p)
        elif suffix == ".json" or format_hint == DataFormat.JSON:
            return pd.read_json(p)
        else:
            # Fallback attempt as CSV
            return pd.read_csv(p)


# ===========================================================================
# 7. DatasetRegistry
# ===========================================================================
class DatasetRegistry:
    """
    Central hub for managing all datasets in CANCERWISE AI.
    Handles indexing, lookup, caching, and coordinated ingestion.
    """

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.raw_dir = data_root / "raw"
        self.interim_dir = data_root / "interim"
        self.processed_dir = data_root / "processed"
        self.external_dir = data_root / "external"
        self.metadata_dir = data_root / "metadata"

        for d in (self.raw_dir, self.interim_dir, self.processed_dir, self.external_dir, self.metadata_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.metadata_mgr = DatasetMetadataManager(self.metadata_dir)
        self.version_mgr = DatasetVersionManager(self.metadata_dir / "versions")
        self.source_mgr = DatasetSourceManager(self.metadata_dir)
        self.validator = DatasetValidator()
        self.loader = DatasetLoader()

        self._cache: Dict[str, pd.DataFrame] = {}

    def register_dataset(
        self,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        save_processed: bool = True,
    ) -> DatasetMetadata:
        """Register a new or updated dataset into the platform."""
        if save_processed:
            proc_csv = self.processed_dir / f"{metadata.dataset_id}.csv"
            proc_parquet = self.processed_dir / f"{metadata.dataset_id}.parquet"
            df.to_csv(proc_csv, index=False)
            df.to_parquet(proc_parquet, index=False)
            metadata.processed_file_path = str(proc_parquet.resolve())

        # Update metadata counts
        metadata.records_count = len(df)
        metadata.columns_count = len(df.columns)
        metadata.missing_values_count = int(df.isnull().sum().sum())
        metadata.feature_names = [c for c in df.columns if c != metadata.target_variable and c.lower() not in ("id", "patient_id")]

        self.metadata_mgr.save_metadata(metadata)
        self.version_mgr.register_version(
            dataset_id=metadata.dataset_id,
            version=metadata.version,
            checksum=metadata.sha256_checksum,
            records_count=metadata.records_count,
            columns_count=metadata.columns_count,
        )

        self._cache[metadata.dataset_id] = df.copy()
        logger.info("Dataset '%s' successfully registered in DatasetRegistry.", metadata.dataset_id)
        return metadata

    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Fetch DataFrame for registered dataset (cached or loaded from processed/raw)."""
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        meta = self.metadata_mgr.load_metadata(dataset_id)
        if not meta:
            return None

        path_to_load = meta.processed_file_path or meta.raw_file_path
        if not path_to_load or not Path(path_to_load).exists():
            return None

        df = self.loader.load(path_to_load)
        self._cache[dataset_id] = df
        return df

    def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        return self.metadata_mgr.load_metadata(dataset_id)

    def list_datasets(self) -> List[DatasetMetadata]:
        return self.metadata_mgr.list_all_metadata()

    def get_preview(self, dataset_id: str, n_rows: int = 5) -> Optional[DatasetPreview]:
        df = self.get_dataset(dataset_id)
        meta = self.get_metadata(dataset_id)
        if df is None or meta is None:
            return None

        # Clean JSON-serializable representations
        head_records = df.head(n_rows).replace({np.nan: None}).to_dict(orient="records")
        tail_records = df.tail(n_rows).replace({np.nan: None}).to_dict(orient="records")
        dtypes_dict = {col: str(df[col].dtype) for col in df.columns}

        target_dist = None
        if meta.target_variable and meta.target_variable in df.columns:
            target_dist = {str(k): int(v) for k, v in df[meta.target_variable].value_counts().items()}

        return DatasetPreview(
            dataset_id=dataset_id,
            name=meta.name,
            shape=list(df.shape),
            columns=list(df.columns),
            dtypes=dtypes_dict,
            head=head_records,
            tail=tail_records,
            target_distribution=target_dist,
        )

    def get_statistics(self, dataset_id: str) -> Optional[DatasetStatisticsSummary]:
        df = self.get_dataset(dataset_id)
        meta = self.get_metadata(dataset_id)
        if df is None:
            return None
        target = meta.target_variable if meta else None
        return DatasetStatistics.compute(df, target_col=target)

    def validate_dataset(self, dataset_id: str) -> Optional[DatasetValidationReport]:
        df = self.get_dataset(dataset_id)
        meta = self.get_metadata(dataset_id)
        if df is None:
            return None
        return self.validator.validate(df, metadata=meta)
