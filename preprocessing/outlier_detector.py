"""
CANCERWISE AI — Outlier Detector
Multivariate and univariate anomaly detection using IQR, Z-score, Modified Z-score,
Isolation Forest, Local Outlier Factor (LOF), and Robust Covariance (EllipticEnvelope).

CRITICAL RULE: Extreme values in oncological biomarkers and nuclear cytology morphometrics
may indicate aggressive malignant phenotypes or rare biological variants rather than measurement errors.
Do not automatically delete biological extremes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from backend.schemas.preprocessing import OutlierMethod, OutlierReport

logger = logging.getLogger("cancerwise.preprocessing.outliers")


class OutlierDetector:
    """
    Comprehensive anomaly and outlier detector for biomedical cancer data.
    Provides diagnostic bounding, robust statistics, and non-destructive remediation.
    """

    def __init__(
        self,
        method: Union[OutlierMethod, str] = OutlierMethod.IQR,
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        modified_z_threshold: float = 3.5,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        random_state: int = 42,
    ) -> None:
        self.method = OutlierMethod(method) if isinstance(method, str) else method
        self.iqr_multiplier = iqr_multiplier
        self.z_threshold = z_threshold
        self.modified_z_threshold = modified_z_threshold
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self._bounds: Dict[str, Tuple[float, float]] = {}
        self._model: Optional[Any] = None

    def detect(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> OutlierReport:
        """
        Execute outlier detection across specified numeric columns.
        """
        # Exclude target and non-numeric columns
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_column and target_column in num_cols:
            num_cols.remove(target_column)

        eval_cols = [c for c in (columns or num_cols) if c in num_cols]
        n_rows = len(df)

        if not eval_cols or n_rows == 0:
            return OutlierReport(
                method=self.method,
                total_outliers=0,
                outlier_percentage=0.0,
                outlier_indices=[],
                column_outlier_counts={},
                threshold_params={},
            )

        subset = df[eval_cols].copy()
        # Impute temporarily with median if missing values exist for multivariate algorithms
        subset_filled = subset.fillna(subset.median())

        column_counts: Dict[str, int] = {c: 0 for c in eval_cols}
        outlier_indices_set: set[int] = set()
        threshold_params: Dict[str, Any] = {}

        if self.method == OutlierMethod.IQR:
            threshold_params["multiplier"] = self.iqr_multiplier
            for col in eval_cols:
                s = subset[col].dropna()
                q25 = s.quantile(0.25)
                q75 = s.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - self.iqr_multiplier * iqr
                upper = q75 + self.iqr_multiplier * iqr
                self._bounds[col] = (float(lower), float(upper))

                is_outlier = (subset[col] < lower) | (subset[col] > upper)
                col_indices = subset.index[is_outlier].tolist()
                column_counts[col] = len(col_indices)
                outlier_indices_set.update(col_indices)

        elif self.method == OutlierMethod.Z_SCORE:
            threshold_params["threshold"] = self.z_threshold
            for col in eval_cols:
                s = subset[col].dropna()
                mean = s.mean()
                std = s.std()
                if std > 0:
                    z_scores = np.abs((subset[col] - mean) / std)
                    is_outlier = z_scores > self.z_threshold
                    col_indices = subset.index[is_outlier].tolist()
                    column_counts[col] = len(col_indices)
                    outlier_indices_set.update(col_indices)
                    self._bounds[col] = (float(mean - self.z_threshold * std), float(mean + self.z_threshold * std))

        elif self.method == OutlierMethod.MODIFIED_Z_SCORE:
            threshold_params["threshold"] = self.modified_z_threshold
            # Boris Iglewicz and David Hoaglin (1993): M_i = 0.6745 * (x_i - median) / MAD
            for col in eval_cols:
                s = subset[col].dropna()
                median = s.median()
                mad = np.median(np.abs(s - median))
                if mad > 0:
                    mod_z = 0.6745 * np.abs(subset[col] - median) / mad
                    is_outlier = mod_z > self.modified_z_threshold
                    col_indices = subset.index[is_outlier].tolist()
                    column_counts[col] = len(col_indices)
                    outlier_indices_set.update(col_indices)

        elif self.method == OutlierMethod.ISOLATION_FOREST:
            threshold_params["contamination"] = self.contamination
            clf = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
            preds = clf.fit_predict(subset_filled)
            outlier_mask = preds == -1
            col_indices = subset.index[outlier_mask].tolist()
            outlier_indices_set.update(col_indices)
            self._model = clf
            # Record multivariate count
            threshold_params["model"] = "IsolationForest"

        elif self.method == OutlierMethod.LOCAL_OUTLIER_FACTOR:
            threshold_params["contamination"] = self.contamination
            threshold_params["n_neighbors"] = self.n_neighbors
            lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, max(2, n_rows - 1)),
                contamination=self.contamination,
                n_jobs=-1,
            )
            preds = lof.fit_predict(subset_filled)
            outlier_mask = preds == -1
            col_indices = subset.index[outlier_mask].tolist()
            outlier_indices_set.update(col_indices)
            threshold_params["model"] = "LocalOutlierFactor"

        elif self.method == OutlierMethod.ROBUST_COVARIANCE:
            threshold_params["contamination"] = self.contamination
            ee = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
            )
            ee.fit(subset_filled)
            preds = ee.predict(subset_filled)
            outlier_mask = preds == -1
            col_indices = subset.index[outlier_mask].tolist()
            outlier_indices_set.update(col_indices)
            threshold_params["model"] = "EllipticEnvelope"

        total_outliers = len(outlier_indices_set)
        pct = round((total_outliers / n_rows) * 100, 2) if n_rows > 0 else 0.0

        logger.info(
            "OutlierDetector (%s) identified %d outliers (%.2f%%) across %d features.",
            self.method.value, total_outliers, pct, len(eval_cols),
        )

        return OutlierReport(
            method=self.method,
            total_outliers=total_outliers,
            outlier_percentage=pct,
            outlier_indices=sorted(list(outlier_indices_set)),
            column_outlier_counts={k: v for k, v in column_counts.items() if v > 0},
            threshold_params=threshold_params,
        )

    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Winsorize/clip numeric features to specified quantile bounds.
        Preserves sample count while limiting leverage of extreme values.
        """
        df_out = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_column and target_column in num_cols:
            num_cols.remove(target_column)

        target_cols = [c for c in (columns or num_cols) if c in num_cols]

        for col in target_cols:
            lower = df_out[col].quantile(lower_percentile)
            upper = df_out[col].quantile(upper_percentile)
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)

        logger.info("Winsorized %d columns to percentiles [%.2f, %.2f].", len(target_cols), lower_percentile, upper_percentile)
        return df_out

    def remove_outliers(
        self,
        df: pd.DataFrame,
        outlier_indices: List[int],
        confirm_biological_override: bool = False,
    ) -> pd.DataFrame:
        """
        Remove outlier records.

        Requires confirm_biological_override=True to prevent unintended erasure of genuine
        malignant phenotypes or aggressive oncological biological extremes.
        """
        if not confirm_biological_override:
            raise PermissionError(
                "Refusing to delete outliers without clinical confirmation. "
                "Extreme values in cytology or genomic biomarkers may represent aggressive cancer phenotypes. "
                "Set confirm_biological_override=True if removal is clinically warranted."
            )

        df_out = df.drop(index=outlier_indices, errors="ignore")
        logger.warning(
            "Purged %d records marked as outliers. (Remaining: %d)",
            len(outlier_indices), len(df_out),
        )
        return df_out
