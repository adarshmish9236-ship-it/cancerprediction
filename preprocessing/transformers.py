"""
CANCERWISE AI — Data Transformers
Mathematical transformation and feature scaling engine.
Supports StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer
(Yeo-Johnson & Box-Cox), QuantileTransformer, Log1p, OneHotEncoder, and OrdinalEncoder.

Prevents invalid operations (e.g. non-positive values in Box-Cox/log) and preserves
fitted parameters for reversible transforms and reproducible inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)

from backend.schemas.preprocessing import ScalingStrategy, EncodingStrategy

logger = logging.getLogger("cancerwise.preprocessing.transformers")


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Safely computes log(x + offset) with offset ensuring strictly positive values."""

    def __init__(self, offset: float = 1.0) -> None:
        self.offset = offset
        self.min_val_: Optional[float] = None
        self.computed_offset_: float = offset

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        min_x = np.nanmin(X_arr)
        self.min_val_ = float(min_x)
        # If minimum value <= -self.offset, shift offset so values are strictly > 0
        if min_x <= -self.offset:
            self.computed_offset_ = abs(min_x) + 1.0
            logger.info("Auto-adjusted log offset to %.4f to prevent non-positive log operands.", self.computed_offset_)
        else:
            self.computed_offset_ = self.offset
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        shifted = X_arr + self.computed_offset_
        if np.any(shifted <= 0):
            shifted = np.clip(shifted, a_min=1e-8, a_max=None)
        return np.log(shifted)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.exp(X_arr) - self.computed_offset_


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    Unified transformer for feature scaling and encoding.
    Preserves fitted parameters and supports inverse transformations.
    """

    def __init__(
        self,
        strategy: Union[ScalingStrategy, EncodingStrategy, str] = ScalingStrategy.STANDARD,
        columns: Optional[List[str]] = None,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        norm: str = "l2",
        n_quantiles: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.strategy = str(strategy.value if hasattr(strategy, "value") else strategy)
        self.columns = columns
        self.feature_range = feature_range
        self.norm = norm
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        self._fitted_transformer: Optional[Any] = None
        self._fitted_columns: List[str] = []
        self._output_columns: List[str] = []
        self._parameters: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "DataTransformer":
        self._fitted_columns = self.columns or X.columns.tolist()
        subset = X[self._fitted_columns]

        strat = self.strategy.lower()

        if strat == ScalingStrategy.STANDARD.value:
            self._fitted_transformer = StandardScaler()
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {
                "mean": [float(m) for m in self._fitted_transformer.mean_],
                "scale": [float(s) for s in self._fitted_transformer.scale_],
            }

        elif strat == ScalingStrategy.MINMAX.value:
            self._fitted_transformer = MinMaxScaler(feature_range=self.feature_range)
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {
                "data_min": [float(m) for m in self._fitted_transformer.data_min_],
                "data_max": [float(m) for m in self._fitted_transformer.data_max_],
            }

        elif strat == ScalingStrategy.ROBUST.value:
            self._fitted_transformer = RobustScaler()
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {
                "center": [float(c) for c in self._fitted_transformer.center_],
                "scale": [float(s) for s in self._fitted_transformer.scale_],
            }

        elif strat == ScalingStrategy.NORMALIZER.value:
            self._fitted_transformer = Normalizer(norm=self.norm)
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {"norm": self.norm}

        elif strat == ScalingStrategy.POWER_YEO_JOHNSON.value:
            self._fitted_transformer = PowerTransformer(method="yeo-johnson")
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {
                "lambdas": [float(l) for l in self._fitted_transformer.lambdas_],
            }

        elif strat == ScalingStrategy.POWER_BOX_COX.value:
            # Box-Cox requires strictly positive values
            min_val = np.nanmin(subset.to_numpy())
            if min_val <= 0:
                logger.warning(
                    "Box-Cox requires strictly positive values (min=%.4f). Auto-falling back to Yeo-Johnson.",
                    min_val,
                )
                self._fitted_transformer = PowerTransformer(method="yeo-johnson")
            else:
                self._fitted_transformer = PowerTransformer(method="box-cox")
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {
                "lambdas": [float(l) for l in self._fitted_transformer.lambdas_],
            }

        elif strat == ScalingStrategy.QUANTILE_UNIFORM.value:
            n_q = min(self.n_quantiles, max(10, len(subset)))
            self._fitted_transformer = QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=n_q,
                random_state=self.random_state,
            )
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {"n_quantiles": n_q, "output_distribution": "uniform"}

        elif strat == ScalingStrategy.QUANTILE_NORMAL.value:
            n_q = min(self.n_quantiles, max(10, len(subset)))
            self._fitted_transformer = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_q,
                random_state=self.random_state,
            )
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {"n_quantiles": n_q, "output_distribution": "normal"}

        elif strat == ScalingStrategy.LOG1P.value:
            self._fitted_transformer = Log1pTransformer()
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {"computed_offset": self._fitted_transformer.computed_offset_}

        elif strat == EncodingStrategy.ONEHOT.value:
            self._fitted_transformer = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            self._fitted_transformer.fit(subset)
            encoded_names = self._fitted_transformer.get_feature_names_out(self._fitted_columns).tolist()
            self._output_columns = encoded_names
            self._parameters = {
                "categories": [list(cats) for cats in self._fitted_transformer.categories_],
                "feature_names_out": encoded_names,
            }

        elif strat == EncodingStrategy.ORDINAL.value:
            self._fitted_transformer = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self._fitted_transformer.fit(subset)
            self._output_columns = list(self._fitted_columns)
            self._parameters = {
                "categories": [list(cats) for cats in self._fitted_transformer.categories_],
            }

        elif strat == ScalingStrategy.PASSTHROUGH.value:
            self._fitted_transformer = FunctionTransformer(validate=False)
            self._output_columns = list(self._fitted_columns)

        else:
            raise ValueError(f"Unsupported transformation strategy: '{self.strategy}'.")

        logger.info(
            "DataTransformer fitted | strategy=%s | input_cols=%d | output_cols=%d",
            self.strategy, len(self._fitted_columns), len(self._output_columns),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fitted_transformer is None:
            raise RuntimeError("DataTransformer must be fitted before calling transform.")

        df_out = X.copy()
        subset = df_out[self._fitted_columns]

        transformed = self._fitted_transformer.transform(subset)

        if self.strategy == EncodingStrategy.ONEHOT.value:
            # Drop original categorical columns and append one-hot columns
            df_out = df_out.drop(columns=self._fitted_columns)
            df_encoded = pd.DataFrame(transformed, columns=self._output_columns, index=X.index)
            df_out = pd.concat([df_out, df_encoded], axis=1)
        else:
            df_out[self._fitted_columns] = transformed

        return df_out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse transformations where mathematically possible.
        """
        if self._fitted_transformer is None:
            raise RuntimeError("DataTransformer must be fitted before calling inverse_transform.")

        if not hasattr(self._fitted_transformer, "inverse_transform"):
            raise NotImplementedError(f"Strategy '{self.strategy}' does not support exact inverse transformation.")

        df_out = X.copy()

        if self.strategy == EncodingStrategy.ONEHOT.value:
            encoded_subset = df_out[self._output_columns]
            recovered = self._fitted_transformer.inverse_transform(encoded_subset)
            df_out = df_out.drop(columns=self._output_columns)
            for idx, col in enumerate(self._fitted_columns):
                df_out[col] = recovered[:, idx]
        else:
            subset = df_out[self._fitted_columns]
            recovered = self._fitted_transformer.inverse_transform(subset)
            df_out[self._fitted_columns] = recovered

        return df_out

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns
