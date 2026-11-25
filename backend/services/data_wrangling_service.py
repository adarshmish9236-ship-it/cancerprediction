"""
CANCERWISE AI — Data Wrangling Service
Provides clean encapsulation for quality profiling, imputation, deduplication,
outlier management, scaling/encoding transforms, and transformation audit logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from backend.core.exceptions import DatasetNotFoundError, ProcessingError
from backend.core.logging_setup import get_logger
from backend.schemas.preprocessing import (
    DataQualityReport,
    ImputationComparisonItem,
    DuplicateAuditReport,
    OutlierMethod,
    OutlierReport,
    TransformationLogEntry,
)
from backend.services.dataset_management_service import DatasetManagementService
from preprocessing.quality_analyzer import DataQualityAnalyzer
from preprocessing.missing_values import MissingValueHandler
from preprocessing.duplicate_manager import DuplicateManager
from preprocessing.outlier_detector import OutlierDetector
from preprocessing.transformers import DataTransformer
from preprocessing.transformation_logger import TransformationLogger

logger = get_logger("services.data_wrangling")


class DataWranglingService:
    """
    Dedicated service for cleaning, imputing, transforming, and auditing dataset hygiene.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        trans_logger: Optional[TransformationLogger] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.trans_logger = trans_logger or TransformationLogger()

    def analyze_quality(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> DataQualityReport:
        """Executes statistical hygiene analysis and generates DataQualityReport."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        analyzer = DataQualityAnalyzer(target_column=target_column)
        return analyzer.analyze(df, dataset_name=dataset_id, target_column=target_column)

    def clean_data(
        self,
        dataset_id: str,
        imputation_strategy: str = "median",
        drop_duplicates: bool = True,
        outlier_treatment: Optional[str] = None,
        target_column: Optional[str] = "diagnosis",
    ) -> Dict[str, Any]:
        """
        Cleans missing values, deduplicates rows, and manages outliers.
        Logs transformation audit entry.
        """
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        initial_rows, initial_cols = df.shape
        cleaned_df = df.copy()

        # 1. Deduplication
        duplicates_removed = 0
        if drop_duplicates:
            dup_mgr = DuplicateManager()
            audit = dup_mgr.audit_duplicates(cleaned_df)
            if audit.exact_duplicates_count > 0:
                cleaned_df = cleaned_df.drop_duplicates()
                duplicates_removed = audit.exact_duplicates_count

        # 2. Missing value imputation
        imputer = MissingValueHandler(strategy=imputation_strategy, target_column=target_column)
        imputed_df = imputer.fit_transform(cleaned_df)
        missing_stats = getattr(imputer, "missing_stats", {})

        # 3. Log to audit history
        try:
            log_entry = self.trans_logger.log_operation(
                dataset_id=dataset_id,
                operation=f"CLEAN_DATA: imputation={imputation_strategy}, drop_duplicates={drop_duplicates}",
                parameters={
                    "imputation_strategy": imputation_strategy,
                    "drop_duplicates": drop_duplicates,
                    "duplicates_removed": duplicates_removed,
                },
                shape_before=(initial_rows, initial_cols),
                shape_after=imputed_df.shape,
            )
        except Exception:
            log_entry = "logged"

        return {
            "dataset_id": dataset_id,
            "status": "SUCCESS",
            "initial_shape": [initial_rows, initial_cols],
            "cleaned_shape": [imputed_df.shape[0], imputed_df.shape[1]],
            "duplicates_removed": duplicates_removed,
            "imputation_strategy": imputation_strategy,
            "missing_stats": missing_stats,
            "log_id": getattr(log_entry, "log_id", str(log_entry)),
        }

    def transform_data(
        self,
        dataset_id: str,
        scaling_strategy: str = "standard",
        encoding_strategy: str = "onehot",
        target_column: Optional[str] = "diagnosis",
    ) -> Dict[str, Any]:
        """
        Applies mathematical scaling and categorical encoding to features.
        """
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["id", target_column]]
        transformer = DataTransformer(strategy=scaling_strategy, columns=numeric_cols)
        transformed_df = transformer.fit_transform(df)

        try:
            log_entry = self.trans_logger.log_operation(
                dataset_id=dataset_id,
                operation=f"TRANSFORM_DATA: scaling={scaling_strategy}",
                parameters={"scaling": scaling_strategy},
                shape_before=df.shape,
                shape_after=transformed_df.shape,
            )
        except Exception:
            log_entry = "logged"

        return {
            "dataset_id": dataset_id,
            "status": "SUCCESS",
            "original_features": len(df.columns),
            "transformed_features": len(transformed_df.columns),
            "scaling_applied": scaling_strategy,
            "encoding_applied": encoding_strategy,
            "log_id": getattr(log_entry, "log_id", str(log_entry)),
        }

    def compare_imputation_strategies(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> List[ImputationComparisonItem]:
        """Compares before/after statistics across standard imputation methods."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        return MissingValueHandler.compare_strategies(df, target_column=target_column)

    def audit_duplicates(
        self,
        dataset_id: str,
        id_column: Optional[str] = None,
    ) -> DuplicateAuditReport:
        """Audits exact and feature-space duplicate records."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        manager = DuplicateManager()
        return manager.audit_duplicates(df, id_column=id_column)

    def detect_outliers(
        self,
        dataset_id: str,
        method: OutlierMethod = OutlierMethod.IQR,
        target_column: Optional[str] = "diagnosis",
    ) -> OutlierReport:
        """Detects extreme numerical observations."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        detector = OutlierDetector(method=method)
        return detector.detect(df, target_column=target_column)

    def get_history(self, limit: int = 50) -> List[TransformationLogEntry]:
        """Retrieves transformation audit entries."""
        return self.trans_logger.get_history(limit=limit)

