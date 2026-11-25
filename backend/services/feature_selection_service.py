"""
CANCERWISE AI — Feature Selection Service
Coordinates filter, wrapper, embedded feature selection, zero-leakage cross-validated comparisons,
consensus ranking, and dimensionality reduction (PCA, KPCA, SVD, UMAP).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from backend.core.exceptions import DatasetNotFoundError, IncompatibleDataError
from backend.core.logging_setup import get_logger
from backend.schemas.feature_engineering import (
    SelectionComparisonItem,
    DimensionalityReductionResult,
    DimensionalityMethod,
)
from backend.services.dataset_management_service import DatasetManagementService
from feature_engineering.selection_engine import FeatureSelectionEngine
from feature_engineering.reduction_engine import DimensionalityReductionEngine

logger = get_logger("services.feature_selection")


class FeatureSelectionService:
    """
    Dedicated service for feature selection, ranking, and dimensionality reduction.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        selection_engine: Optional[FeatureSelectionEngine] = None,
        reduction_engine: Optional[DimensionalityReductionEngine] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.selection_engine = selection_engine or FeatureSelectionEngine()
        self.reduction_engine = reduction_engine or DimensionalityReductionEngine()

    def _split_features_target(
        self,
        dataset_id: str,
        target_column: Optional[str] = "diagnosis",
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.dataset_mgmt.get_dataframe(dataset_id)
        y = None
        if target_column and target_column in df.columns:
            y_raw = df[target_column]
            # Map standard oncology labels if categorical
            if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
                unique_vals = list(y_raw.unique())
                if "M" in unique_vals and "B" in unique_vals:
                    mapping = {"M": 1, "B": 0}
                elif "malignant" in unique_vals or "benign" in unique_vals:
                    mapping = {"malignant": 1, "benign": 0}
                else:
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                y = y_raw.map(mapping)
            else:
                y = y_raw

        X = df.drop(columns=[target_column]) if (target_column and target_column in df.columns) else df
        # Select numeric only for feature selection
        X = X.select_dtypes(include=[np.number]).fillna(0.0)
        return X, y

    def select_features(
        self,
        dataset_id: str,
        method: str = "mutual_info",
        k: int = 10,
        target_column: str = "diagnosis",
    ) -> Dict[str, Any]:
        """Runs filter, wrapper, or embedded selection algorithm."""
        X, y = self._split_features_target(dataset_id, target_column)
        if y is None:
            raise IncompatibleDataError(f"Target column '{target_column}' is required for feature selection.")

        selected_features, rankings = self.selection_engine.select_features(X, y, method=method, k=k)
        return {
            "dataset_id": dataset_id,
            "method": method,
            "target_column": target_column,
            "original_feature_count": X.shape[1],
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "rankings": rankings,
        }

    def compare_selection_strategies(
        self,
        dataset_id: str,
        k: int = 10,
        cv_folds: int = 5,
        target_column: str = "diagnosis",
    ) -> List[SelectionComparisonItem]:
        """Runs multi-method cross-validated comparison with strict data leakage prevention."""
        X, y = self._split_features_target(dataset_id, target_column)
        if y is None:
            raise IncompatibleDataError(f"Target column '{target_column}' not found.")
        return self.selection_engine.compare_selection_strategies(X, y, k=k, cv_folds=cv_folds)

    def reduce_dimensions(
        self,
        dataset_id: str,
        method: str = "pca",
        n_components: int = 2,
        target_column: Optional[str] = "diagnosis",
    ) -> DimensionalityReductionResult:
        """Applies PCA, KPCA, SVD, or UMAP dimensionality reduction."""
        X, y = self._split_features_target(dataset_id, target_column)
        try:
            enum_method = DimensionalityMethod(method.lower())
        except ValueError:
            enum_method = DimensionalityMethod.PCA

        res = self.reduction_engine.reduce(
            X,
            method=enum_method,
            n_components=n_components,
            y=y,
        )

        if isinstance(res, tuple):
            return res[1]
        return res

