"""
CANCERWISE AI — Missing Value Handler
Provides mean, median, mode, constant, KNN, and iterative (MICE) imputation,
along with missingness indicator generation and strategy comparison tools.

CRITICAL RULE: Diagnostic target labels must NEVER be imputed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator

# Enable experimental IterativeImputer in scikit-learn
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    HAS_ITERATIVE = True
except ImportError:
    HAS_ITERATIVE = False

from backend.schemas.preprocessing import (
    ImputationStrategy,
    ImputationComparisonItem,
)

logger = logging.getLogger("cancerwise.preprocessing.missing")


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Automated and configurable missing value handler.
    Strictly forbids imputation of target variables.
    """

    def __init__(
        self,
        strategy: Union[ImputationStrategy, str] = ImputationStrategy.MEDIAN,
        columns: Optional[List[str]] = None,
        fill_value: Optional[Union[float, int, str]] = None,
        knn_neighbors: int = 5,
        max_iter: int = 10,
        add_indicator: bool = False,
        target_column: Optional[str] = None,
    ) -> None:
        self.strategy = ImputationStrategy(strategy) if isinstance(strategy, str) else strategy
        self.columns = columns
        self.fill_value = fill_value
        self.knn_neighbors = knn_neighbors
        self.max_iter = max_iter
        self.add_indicator = add_indicator
        self.target_column = target_column

        self._imputer: Optional[Any] = None
        self._indicator: Optional[MissingIndicator] = None
        self._fitted_columns: List[str] = []
        self._indicator_feature_names: List[str] = []
        self._statistics: Dict[str, Any] = {}

    def _validate_target(self, df: pd.DataFrame) -> None:
        """Ensure target column is not subjected to imputation."""
        if self.target_column and self.target_column in df.columns:
            if self.columns and self.target_column in self.columns:
                raise ValueError(
                    f"Target column '{self.target_column}' cannot be imputed. "
                    "Diagnostic labels must never be synthesized or statistically filled."
                )
            # If columns is None, target will be automatically excluded
            if df[self.target_column].isna().any():
                logger.warning(
                    "Target column '%s' contains %d missing labels. "
                    "These records must be excluded from supervised training or handled via semi-supervised learning.",
                    self.target_column, int(df[self.target_column].isna().sum()),
                )

    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueHandler":
        self._validate_target(X)

        # Determine target columns to impute
        if self.columns:
            # Exclude target column if explicitly listed
            self._fitted_columns = [c for c in self.columns if c in X.columns and c != self.target_column]
        else:
            if self.strategy in (ImputationStrategy.KNN, ImputationStrategy.ITERATIVE):
                # KNN and MICE require all numeric features for neighbor distances and regression
                self._fitted_columns = [
                    c for c in X.select_dtypes(include=np.number).columns
                    if c != self.target_column
                ]
            else:
                # Default to all feature columns with missing values (excluding target)
                self._fitted_columns = [
                    c for c in X.columns
                    if c != self.target_column and X[c].isna().any()
                ]
                if not self._fitted_columns:
                    # If no column has missing values, default to all non-target columns
                    self._fitted_columns = [c for c in X.columns if c != self.target_column]

        if not self._fitted_columns:
            logger.info("No feature columns identified for imputation.")
            return self

        subset = X[self._fitted_columns]

        # Build appropriate scikit-learn imputer
        if self.strategy == ImputationStrategy.MEAN:
            self._imputer = SimpleImputer(strategy="mean")
            self._imputer.fit(subset)
            self._statistics = dict(zip(self._fitted_columns, self._imputer.statistics_))
        elif self.strategy == ImputationStrategy.MEDIAN:
            self._imputer = SimpleImputer(strategy="median")
            self._imputer.fit(subset)
            self._statistics = dict(zip(self._fitted_columns, self._imputer.statistics_))
        elif self.strategy == ImputationStrategy.MODE:
            self._imputer = SimpleImputer(strategy="most_frequent")
            self._imputer.fit(subset)
            self._statistics = dict(zip(self._fitted_columns, self._imputer.statistics_))
        elif self.strategy == ImputationStrategy.CONSTANT:
            val = self.fill_value if self.fill_value is not None else 0.0
            self._imputer = SimpleImputer(strategy="constant", fill_value=val)
            self._imputer.fit(subset)
            self._statistics = {c: val for c in self._fitted_columns}
        elif self.strategy == ImputationStrategy.KNN:
            self._imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self._imputer.fit(subset)
            self._statistics = {"knn_neighbors": self.knn_neighbors}
        elif self.strategy == ImputationStrategy.ITERATIVE:
            if not HAS_ITERATIVE:
                logger.warning("IterativeImputer not available, falling back to median imputation.")
                self._imputer = SimpleImputer(strategy="median")
                self._imputer.fit(subset)
            else:
                self._imputer = IterativeImputer(max_iter=self.max_iter, random_state=42)
                self._imputer.fit(subset)
            self._statistics = {"max_iter": self.max_iter}

        # Optional missingness indicators
        if self.add_indicator:
            self._indicator = MissingIndicator(features="all")
            self._indicator.fit(subset)
            self._indicator_feature_names = [f"{c}_isna" for c in self._fitted_columns]

        logger.info(
            "MissingValueHandler fitted with strategy '%s' on %d columns.",
            self.strategy.value, len(self._fitted_columns),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._imputer is None:
            return X.copy()

        df_out = X.copy()
        subset = df_out[self._fitted_columns]

        imputed_array = self._imputer.transform(subset)
        df_out[self._fitted_columns] = imputed_array

        if self._indicator is not None and self._indicator_feature_names:
            indicator_array = self._indicator.transform(subset)
            for idx, col_name in enumerate(self._indicator_feature_names):
                df_out[col_name] = indicator_array[:, idx].astype(int)

        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @classmethod
    def compare_strategies(
        cls,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        strategies: Optional[List[ImputationStrategy]] = None,
        target_column: Optional[str] = None,
    ) -> List[ImputationComparisonItem]:
        """
        Compare the statistical impact (mean, std, median) of multiple imputation strategies.
        """
        if strategies is None:
            strategies = [
                ImputationStrategy.MEAN,
                ImputationStrategy.MEDIAN,
                ImputationStrategy.KNN,
            ]

        target_cols = columns or [c for c in df.select_dtypes(include=np.number).columns if c != target_column and df[c].isna().any()]
        if not target_cols:
            return []

        comparison_results: List[ImputationComparisonItem] = []

        for strat in strategies:
            try:
                handler = cls(strategy=strat, columns=target_cols, target_column=target_column)
                df_imputed = handler.fit_transform(df)

                for col in target_cols:
                    orig_s = df[col]
                    imp_s = df_imputed[col]

                    comparison_results.append(
                        ImputationComparisonItem(
                            strategy=strat.value,
                            column=col,
                            mean_before=round(float(orig_s.mean()), 4) if not orig_s.dropna().empty else None,
                            mean_after=round(float(imp_s.mean()), 4),
                            std_before=round(float(orig_s.std()), 4) if len(orig_s.dropna()) > 1 else None,
                            std_after=round(float(imp_s.std()), 4) if len(imp_s) > 1 else None,
                            median_before=round(float(orig_s.median()), 4) if not orig_s.dropna().empty else None,
                            median_after=round(float(imp_s.median()), 4),
                            missing_before=int(orig_s.isna().sum()),
                            missing_after=int(imp_s.isna().sum()),
                        )
                    )
            except Exception as e:
                logger.error("Failed to compare strategy %s: %s", strat.value, e)

        return comparison_results
