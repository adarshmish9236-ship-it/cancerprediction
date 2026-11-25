"""
CANCERWISE AI — Data Validation Service
Provides clinical schema validation, type constraints, Pandera contract verification,
and multimodal compatibility assessment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd

from backend.core.exceptions import DatasetNotFoundError, IncompatibleDataError
from backend.core.logging_setup import get_logger
from backend.schemas.dataset import SchemaCompatibilityReport
from backend.schemas.preprocessing import SchemaValidationReport
from backend.services.dataset_management_service import DatasetManagementService
from backend.services.compatibility_checker import SchemaCompatibilityChecker
from preprocessing.data_validator import DataValidationEngine

logger = get_logger("services.data_validation")


class DataValidationService:
    """
    Dedicated service enforcing data integrity contracts, schema checks, and modality compatibility.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        checker: Optional[SchemaCompatibilityChecker] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.checker = checker or SchemaCompatibilityChecker()

    def validate_schema(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> SchemaValidationReport:
        """Validates dataset columns against Pandera schema expectations."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        validator = DataValidationEngine(target_column=target_column)
        return validator.validate(df, dataset_name=dataset_id)

    def check_compatibility(
        self,
        dataset_a_id: str,
        dataset_b_id: str,
    ) -> SchemaCompatibilityReport:
        """Evaluates schema, modality, units, and label compatibility between two datasets."""
        meta_a = self.dataset_mgmt.get_metadata(dataset_a_id)
        meta_b = self.dataset_mgmt.get_metadata(dataset_b_id)
        df_a = self.dataset_mgmt.get_dataframe(dataset_a_id)
        df_b = self.dataset_mgmt.get_dataframe(dataset_b_id)
        return self.checker.check_compatibility(meta_a, meta_b, df_a, df_b)
