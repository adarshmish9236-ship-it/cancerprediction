"""
CANCERWISE AI — Feature Engineering Service
Coordinates cytological morphometrics synthesis, ratio expansion, polynomial terms,
logarithmic transformations, and derived feature provenance registration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from backend.core.exceptions import DatasetNotFoundError
from backend.core.logging_setup import get_logger
from backend.schemas.feature_engineering import (
    DerivedFeatureDefinition,
    FeatureEngineeringResult,
)
from backend.services.dataset_management_service import DatasetManagementService
from feature_engineering.engineering_engine import FeatureEngineeringEngine

logger = get_logger("services.feature_engineering")


class FeatureEngineeringService:
    """
    Dedicated service for biomedical and statistical feature creation.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        engine: Optional[FeatureEngineeringEngine] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.engine = engine or FeatureEngineeringEngine()

    def engineer_features(
        self,
        dataset_id: str,
        apply_domain_features: bool = True,
        apply_ratios: bool = True,
        apply_polynomials: bool = False,
        apply_log_transforms: bool = False,
        apply_aggregations: bool = True,
    ) -> FeatureEngineeringResult:
        """Applies feature synthesis and returns engineering metadata."""
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        _, result = self.engine.engineer_features(
            df,
            dataset_name=dataset_id,
            apply_domain_features=apply_domain_features,
            apply_ratios=apply_ratios,
            apply_polynomials=apply_polynomials,
            apply_log_transforms=apply_log_transforms,
            apply_aggregations=apply_aggregations,
        )
        return result

    def get_registered_definitions(self) -> List[DerivedFeatureDefinition]:
        """Returns definitions of cytological and morphological features."""
        return self.engine.get_registered_definitions()
