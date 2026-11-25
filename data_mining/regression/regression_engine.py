"""
CANCERWISE AI — Continuous Regression Engine
Implements Linear Regression, Ridge, Lasso, ElasticNet, Random Forest Regressor,
Gradient Boosting Regressor, and Support Vector Regression (SVR).

GOVERNANCE DIRECTIVE:
Activated strictly when genuine continuous oncology targets (e.g. tumor cellularity,
Ki-67 mitotic proliferation index) exist. Binary cancer labels are NEVER converted to continuous scores.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.schemas.advanced_learning import (
    RegressionMethod,
    RegressionExperimentResult,
)

logger = logging.getLogger("cancerwise.regression.engine")


class RegressionEngine:
    """
    Evaluates continuous predictive modeling for oncology biomarkers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_regression_experiment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Union[RegressionMethod, str] = RegressionMethod.RIDGE,
        target_name: str = "tumor_cellularity_percent",
        dataset_name: str = "Synthetic Tumor Cellularity Benchmark",
        test_size: float = 0.20,
    ) -> RegressionExperimentResult:
        """
        Fits continuous regression model and evaluates MSE, RMSE, MAE, R², and adjusted R².
        """
        if isinstance(method, str):
            method = RegressionMethod(method)

        start_time = time.perf_counter()

        # Clinical governance check: verify genuine continuous target
        unique_count = len(np.unique(y))
        if unique_count <= 5:
            raise ValueError(
                f"Continuous regression engine requires a genuine continuous target, but found only {unique_count} distinct values. "
                "Per clinical governance, discrete binary or multiclass labels must not be coerced into continuous targets."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if method == RegressionMethod.LINEAR_REGRESSION:
            model = LinearRegression()
        elif method == RegressionMethod.RIDGE:
            model = Ridge(alpha=1.0, random_state=self.random_state)
        elif method == RegressionMethod.LASSO:
            model = Lasso(alpha=0.1, random_state=self.random_state)
        elif method == RegressionMethod.ELASTIC_NET:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        elif method == RegressionMethod.RANDOM_FOREST_REGRESSOR:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        elif method == RegressionMethod.GRADIENT_BOOSTING_REGRESSOR:
            model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        elif method == RegressionMethod.SVR:
            model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        else:
            raise ValueError(f"Unknown regression method: {method}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        # Adjusted R²
        n = len(y_test)
        p = X_train.shape[1]
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)) if (n - p - 1) > 0 else r2

        elapsed = round(time.perf_counter() - start_time, 3)

        return RegressionExperimentResult(
            method=method,
            dataset_name=dataset_name,
            target_name=target_name,
            mse=round(mse, 4),
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            r2_score=round(r2, 4),
            adjusted_r2=round(adj_r2, 4),
            execution_time_seconds=elapsed,
        )
