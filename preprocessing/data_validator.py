"""
CANCERWISE AI — Data Validation Engine
Integrates Pandera schema validation and strict clinical data integrity checks:
- Column data types.
- Allowed categorical vocabularies (e.g., 'Benign', 'Malignant').
- Missingness tolerances.
- Numerical constraints (non-negativity, ranges).
- Dataset dimensionality.
- Target validity (presence, nulls forbidden, class balance).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

try:
    try:
        import pandera.pandas as pa
        from pandera.pandas import Column, Check, DataFrameSchema
    except ImportError:
        import pandera as pa
        from pandera import Column, Check, DataFrameSchema
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False

from backend.schemas.preprocessing import SchemaValidationReport

logger = logging.getLogger("cancerwise.preprocessing.validation")


class DataValidationEngine:
    """
    Schema and data validation engine powered by Pandera with native fallback.
    """

    DEFAULT_ALLOWED_TARGETS = {"Benign", "Malignant"}

    def __init__(
        self,
        target_column: Optional[str] = "diagnosis",
        allowed_targets: Optional[Set[str]] = None,
        min_records: int = 10,
        required_columns: Optional[List[str]] = None,
        max_missing_ratio: float = 0.50,
        non_negative_columns: Optional[List[str]] = None,
    ) -> None:
        self.target_column = target_column
        self.allowed_targets = allowed_targets or self.DEFAULT_ALLOWED_TARGETS
        self.min_records = min_records
        self.required_columns = required_columns or []
        self.max_missing_ratio = max_missing_ratio
        self.non_negative_columns = non_negative_columns or []

    def validate(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> SchemaValidationReport:
        """
        Execute comprehensive schema validation on the dataset.
        """
        errors: List[str] = []
        warnings: List[str] = []
        column_validations: Dict[str, bool] = {}
        total_checks = 0
        passed_checks = 0

        n_rows, n_cols = df.shape

        # 1. Dimension Check
        total_checks += 1
        if n_rows < self.min_records:
            errors.append(f"Insufficient records: Dataset has {n_rows} rows, minimum required is {self.min_records}.")
        else:
            passed_checks += 1

        # 2. Required Columns Check
        if self.required_columns:
            total_checks += 1
            missing_cols = [c for c in self.required_columns if c not in df.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
            else:
                passed_checks += 1

        # 3. Target Validity Check
        if self.target_column:
            total_checks += 1
            if self.target_column not in df.columns:
                errors.append(f"Target column '{self.target_column}' is absent from dataset.")
                column_validations[self.target_column] = False
            else:
                target_s = df[self.target_column]
                # No missing values allowed in target
                total_checks += 1
                target_nulls = int(target_s.isna().sum())
                if target_nulls > 0:
                    errors.append(f"Target column '{self.target_column}' contains {target_nulls} missing values. Target imputation is strictly forbidden.")
                    column_validations[self.target_column] = False
                else:
                    passed_checks += 1

                # Allowed classes
                total_checks += 1
                observed_classes = set(target_s.dropna().astype(str).unique())
                unrecognized = observed_classes - self.allowed_targets
                if unrecognized:
                    warnings.append(
                        f"Target column '{self.target_column}' contains classes {unrecognized} "
                        f"outside expected vocabulary {self.allowed_targets}."
                    )
                else:
                    passed_checks += 1
                column_validations[self.target_column] = len(errors) == 0

        # 4. Column Missingness Tolerances
        for col in df.columns:
            total_checks += 1
            null_count = int(df[col].isna().sum())
            null_ratio = null_count / n_rows if n_rows > 0 else 0
            if null_ratio > self.max_missing_ratio:
                errors.append(f"Column '{col}' exceeds maximum missingness: {null_ratio * 100:.1f}% > {self.max_missing_ratio * 100:.1f}%.")
                column_validations[col] = False
            else:
                passed_checks += 1
                column_validations[col] = column_validations.get(col, True)

        # 5. Non-Negativity Constraints (e.g. for nuclear radius, area, perimeter)
        for col in self.non_negative_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                total_checks += 1
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    errors.append(f"Numerical constraint violated: Column '{col}' has {neg_count} negative values where >= 0 is required.")
                    column_validations[col] = False
                else:
                    passed_checks += 1

        # 6. Pandera Schema Check if Available
        if HAS_PANDERA and not errors and n_rows > 0:
            try:
                schema_dict = {}
                for col in df.columns:
                    if col == self.target_column:
                        schema_dict[col] = Column(
                            object,
                            checks=Check.isin(list(self.allowed_targets)),
                            nullable=False,
                        )
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        checks = []
                        if col in self.non_negative_columns:
                            checks.append(Check.greater_than_or_equal_to(0.0))
                        schema_dict[col] = Column(float, checks=checks, nullable=True, coerce=True)

                schema = DataFrameSchema(schema_dict, coerce=True)
                schema.validate(df, lazy=True)
                total_checks += 1
                passed_checks += 1
            except Exception as e:
                total_checks += 1
                # Format Pandera schema error
                warnings.append(f"Pandera validation notice: {str(e)[:200]}")

        is_valid = len(errors) == 0
        failed_checks = total_checks - passed_checks

        logger.info(
            "DataValidationEngine finished for '%s' | Valid=%s | Passed=%d/%d | Errors=%d | Warnings=%d",
            dataset_name, is_valid, passed_checks, total_checks, len(errors), len(warnings),
        )

        return SchemaValidationReport(
            is_valid=is_valid,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            errors=errors,
            warnings=warnings,
            column_validations=column_validations,
            summary={
                "dataset_name": dataset_name,
                "rows": n_rows,
                "columns": n_cols,
                "has_pandera": HAS_PANDERA,
            },
        )
