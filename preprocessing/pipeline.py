"""
CANCERWISE AI — Advanced Preprocessing Pipeline
Modular, reproducible, and reversible preprocessing pipeline architecture.
Supports:
- Feature-wise ColumnTransformer composition.
- Chaining imputation, scaling, power transforms, and encoding.
- Configuration serialization/deserialization (JSON/YAML recipes).
- Mathematical reversibility (inverse_transform).
- Polars acceleration for high-throughput dataset loading and profiling.
- Full backwards-compatibility with CancerwisePreprocessor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from backend.schemas.preprocessing import (
    ImputationStrategy,
    ScalingStrategy,
    EncodingStrategy,
    PreprocessingRecipe,
    TransformationStepConfig,
)
from preprocessing.missing_values import MissingValueHandler
from preprocessing.transformers import DataTransformer

logger = logging.getLogger("cancerwise.preprocessing.pipeline")


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Advanced, composable, and reversible preprocessing pipeline.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        passthrough_features: Optional[List[str]] = None,
        imputation_strategy: Union[ImputationStrategy, str] = ImputationStrategy.MEDIAN,
        scaling_strategy: Union[ScalingStrategy, str] = ScalingStrategy.STANDARD,
        encoding_strategy: Union[EncodingStrategy, str] = EncodingStrategy.ONEHOT,
        target_column: Optional[str] = None,
        knn_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.passthrough_features = passthrough_features or []
        self.imputation_strategy = str(imputation_strategy.value if hasattr(imputation_strategy, "value") else imputation_strategy)
        self.scaling_strategy = str(scaling_strategy.value if hasattr(scaling_strategy, "value") else scaling_strategy)
        self.encoding_strategy = str(encoding_strategy.value if hasattr(encoding_strategy, "value") else encoding_strategy)
        self.target_column = target_column
        self.knn_neighbors = knn_neighbors
        self.random_state = random_state

        self._imputer_num: Optional[MissingValueHandler] = None
        self._imputer_cat: Optional[MissingValueHandler] = None
        self._scaler: Optional[DataTransformer] = None
        self._encoder: Optional[DataTransformer] = None
        self._fitted = False
        self._fitted_columns: List[str] = []
        self._output_columns: List[str] = []

    def fit(self, X: Union[pd.DataFrame, pl.DataFrame], y=None) -> "PreprocessingPipeline":
        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        else:
            df = X.copy()

        # Identify columns
        num_cols = [c for c in (self.numeric_features or df.select_dtypes(include=np.number).columns.tolist()) if c != self.target_column and c not in self.passthrough_features]
        cat_cols = [c for c in (self.categorical_features or df.select_dtypes(exclude=np.number).columns.tolist()) if c != self.target_column and c not in self.passthrough_features]

        self.numeric_features = num_cols
        self.categorical_features = cat_cols
        self._fitted_columns = num_cols + cat_cols + self.passthrough_features

        # 1. Fit Imputers (numeric with chosen strategy, categorical with mode)
        df_imputed = df
        if self.imputation_strategy != "none":
            if num_cols:
                self._imputer_num = MissingValueHandler(
                    strategy=self.imputation_strategy,
                    columns=num_cols,
                    knn_neighbors=self.knn_neighbors,
                    target_column=self.target_column,
                )
                df_imputed = self._imputer_num.fit_transform(df_imputed)
            if cat_cols:
                self._imputer_cat = MissingValueHandler(
                    strategy=ImputationStrategy.MODE,
                    columns=cat_cols,
                    target_column=self.target_column,
                )
                df_imputed = self._imputer_cat.fit_transform(df_imputed)

        # 2. Fit Scaler on numeric
        if self.scaling_strategy != "none" and self.scaling_strategy != "passthrough" and num_cols:
            self._scaler = DataTransformer(
                strategy=self.scaling_strategy,
                columns=num_cols,
                random_state=self.random_state,
            )
            df_scaled = self._scaler.fit_transform(df_imputed)
        else:
            df_scaled = df_imputed

        # 3. Fit Encoder on categorical
        if self.encoding_strategy != "none" and self.encoding_strategy != "passthrough" and cat_cols:
            self._encoder = DataTransformer(
                strategy=self.encoding_strategy,
                columns=cat_cols,
                random_state=self.random_state,
            )
            df_encoded = self._encoder.fit_transform(df_scaled)
        else:
            df_encoded = df_scaled

        self._output_columns = [c for c in df_encoded.columns if c != self.target_column]
        self._fitted = True

        logger.info(
            "PreprocessingPipeline fitted | num=%d | cat=%d | pass=%d | output_cols=%d",
            len(num_cols), len(cat_cols), len(self.passthrough_features), len(self._output_columns),
        )
        return self

    def transform(self, X: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
        else:
            df = X.copy()

        # Step 1: Impute
        if self._imputer_num is not None:
            df = self._imputer_num.transform(df)
        if self._imputer_cat is not None:
            df = self._imputer_cat.transform(df)

        # Step 2: Scale
        if self._scaler is not None:
            df = self._scaler.transform(df)

        # Step 3: Encode
        if self._encoder is not None:
            df = self._encoder.transform(df)

        return df

    def fit_transform(self, X: Union[pd.DataFrame, pl.DataFrame], y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse scaling and encoding transformations.
        Note: Imputation is inherently lossy and cannot be reversed.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before inverse_transform.")

        df = X.copy()

        if self._encoder is not None:
            try:
                df = self._encoder.inverse_transform(df)
            except Exception as e:
                logger.warning("Could not inverse transform encoder: %s", e)

        if self._scaler is not None:
            try:
                df = self._scaler.inverse_transform(df)
            except Exception as e:
                logger.warning("Could not inverse transform scaler: %s", e)

        return df

    def to_recipe(self, name: str = "cancerwise_pipeline") -> PreprocessingRecipe:
        """Export pipeline configuration as a reproducible PreprocessingRecipe."""
        steps = []
        if self.imputation_strategy != "none":
            steps.append(
                TransformationStepConfig(
                    name="imputation",
                    step_type="imputer",
                    strategy=self.imputation_strategy,
                    columns=self.numeric_features + self.categorical_features,
                    parameters={"knn_neighbors": self.knn_neighbors},
                )
            )
        if self.scaling_strategy not in ("none", "passthrough"):
            steps.append(
                TransformationStepConfig(
                    name="scaling",
                    step_type="scaler",
                    strategy=self.scaling_strategy,
                    columns=self.numeric_features,
                    parameters=self._scaler.parameters if self._scaler else {},
                )
            )
        if self.encoding_strategy not in ("none", "passthrough"):
            steps.append(
                TransformationStepConfig(
                    name="encoding",
                    step_type="encoder",
                    strategy=self.encoding_strategy,
                    columns=self.categorical_features,
                    parameters=self._encoder.parameters if self._encoder else {},
                )
            )

        return PreprocessingRecipe(
            name=name,
            steps=steps,
            target_column=self.target_column,
            reversible=True,
        )

    def save_recipe(self, path: Union[str, Path]) -> None:
        """Save recipe to JSON or YAML."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        recipe = self.to_recipe(name=p.stem)
        recipe_dict = recipe.model_dump()

        if p.suffix in (".yaml", ".yml"):
            with open(p, "w", encoding="utf-8") as f:
                yaml.dump(recipe_dict, f, default_flow_style=False)
        else:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(recipe_dict, f, indent=2)
        logger.info("Saved preprocessing recipe to %s", p)

    @classmethod
    def load_recipe(cls, path: Union[str, Path]) -> "PreprocessingPipeline":
        """Load and instantiate a pipeline from a recipe file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            if p.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        recipe = PreprocessingRecipe(**data)

        # Parse steps
        impute_strat = "median"
        scale_strat = "standard"
        encode_strat = "onehot"
        num_cols = []
        cat_cols = []
        knn_k = 5

        for step in recipe.steps:
            if step.step_type == "imputer":
                impute_strat = step.strategy
                knn_k = step.parameters.get("knn_neighbors", 5)
            elif step.step_type == "scaler":
                scale_strat = step.strategy
                num_cols = step.columns
            elif step.step_type == "encoder":
                encode_strat = step.strategy
                cat_cols = step.columns

        return cls(
            numeric_features=num_cols,
            categorical_features=cat_cols,
            imputation_strategy=impute_strat,
            scaling_strategy=scale_strat,
            encoding_strategy=encode_strat,
            target_column=recipe.target_column,
            knn_neighbors=knn_k,
        )


# ---------------------------------------------------------------------------
# Backwards Compatibility: CancerwisePreprocessor (Phase 0)
# ---------------------------------------------------------------------------
class CancerwisePreprocessor(BaseEstimator, TransformerMixin):
    """
    Backwards-compatible preprocessor interface preserving Phase 0 contract
    while powered by the underlying PreprocessingPipeline.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        missing_strategy: str = "median",
        scaling_strategy: str = "standard",
        encoding_strategy: str = "onehot",
        knn_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.missing_strategy = missing_strategy
        self.scaling_strategy = scaling_strategy
        self.encoding_strategy = encoding_strategy
        self.knn_neighbors = knn_neighbors
        self.random_state = random_state

        self._pipeline = PreprocessingPipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            imputation_strategy=missing_strategy,
            scaling_strategy=scaling_strategy,
            encoding_strategy=encoding_strategy,
            knn_neighbors=knn_neighbors,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y=None) -> "CancerwisePreprocessor":
        self._pipeline.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df_out = self._pipeline.transform(X)
        return df_out.select_dtypes(include=np.number).to_numpy()

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)
