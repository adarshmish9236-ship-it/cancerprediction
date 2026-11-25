"""
CANCERWISE AI — Data Quality Analyzer
Automated statistical and structural data hygiene profiling engine.
Detects missingness, duplicates, invalid values, constant columns, near-zero variance,
outliers, high cardinality, inconsistent types, and potential data leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from backend.schemas.preprocessing import (
    ColumnQualityIssue,
    DataQualityReport,
)

logger = logging.getLogger("cancerwise.preprocessing.quality")


class DataQualityAnalyzer:
    """
    Comprehensive statistical and structural data hygiene profiler.

    NOTE: The computed quality score reflects data integrity, completeness, and consistency
    for machine learning pipelines. It is explicitly NOT a clinical quality assessment or
    a measure of biological validity.
    """

    DEFAULT_RULES = {
        "missing_critical_threshold": 0.30,  # >30% missing is critical
        "missing_warning_threshold": 0.05,   # >5% missing is warning
        "near_zero_variance_threshold": 1e-4, # variance below this
        "near_zero_freq_ratio": 19.0,        # 95/5 ratio
        "high_cardinality_threshold": 50,     # >50 unique values for categorical
        "high_cardinality_ratio": 0.50,      # >50% unique
        "leakage_correlation_threshold": 0.99, # correlation with target > 0.99
        "outlier_iqr_multiplier": 1.5,
    }

    def __init__(
        self,
        rules: Optional[Dict[str, Any]] = None,
        id_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> None:
        self.rules = {**self.DEFAULT_RULES, **(rules or {})}
        self.id_columns = id_columns or ["id", "patient_id", "sample_id", "case_id"]
        self.target_column = target_column

    def analyze(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        target_column: Optional[str] = None,
    ) -> DataQualityReport:
        """
        Execute comprehensive data quality evaluation on a DataFrame.
        """
        target_col = target_column or self.target_column
        issues: List[ColumnQualityIssue] = []
        n_rows, n_cols = df.shape

        if n_rows == 0:
            return DataQualityReport(
                dataset_name=dataset_name,
                total_records=0,
                total_columns=n_cols,
                overall_quality_score=0.0,
                score_interpretation="Critical: Empty dataset.",
                issues=[
                    ColumnQualityIssue(
                        column="ALL",
                        issue_type="empty_dataset",
                        severity="critical",
                        details="The dataset contains zero records.",
                    )
                ],
            )

        # 1. Missing Values Analysis
        missing_summary: Dict[str, Any] = {}
        total_cells = n_rows * n_cols
        total_missing = int(df.isna().sum().sum())
        missing_summary["total_missing_cells"] = total_missing
        missing_summary["total_missing_percentage"] = round((total_missing / total_cells) * 100, 2)
        missing_by_col = df.isna().sum().to_dict()
        missing_summary["by_column"] = {k: int(v) for k, v in missing_by_col.items() if v > 0}

        for col, count in missing_by_col.items():
            if count > 0:
                pct = count / n_rows
                severity = "critical" if pct >= self.rules["missing_critical_threshold"] else (
                    "warning" if pct >= self.rules["missing_warning_threshold"] else "info"
                )
                issues.append(
                    ColumnQualityIssue(
                        column=col,
                        issue_type="missing_values",
                        severity=severity,
                        details=f"{count} missing values ({pct * 100:.2f}% of column).",
                        affected_count=count,
                        affected_percentage=round(pct * 100, 2),
                    )
                )

        # 2. Duplicate Rows
        duplicate_rows_count = int(df.duplicated().sum())
        if duplicate_rows_count > 0:
            dup_pct = duplicate_rows_count / n_rows
            issues.append(
                ColumnQualityIssue(
                    column="ROW_LEVEL",
                    issue_type="duplicate_rows",
                    severity="warning" if dup_pct < 0.1 else "critical",
                    details=f"{duplicate_rows_count} exact duplicate rows ({dup_pct * 100:.2f}%).",
                    affected_count=duplicate_rows_count,
                    affected_percentage=round(dup_pct * 100, 2),
                )
            )

        # 3. Duplicate Identifiers
        duplicate_identifiers_count = 0
        detected_id_col = None
        for candidate_id in self.id_columns:
            if candidate_id in df.columns:
                detected_id_col = candidate_id
                dup_id_count = int(df[candidate_id].duplicated().sum())
                if dup_id_count > 0:
                    duplicate_identifiers_count = dup_id_count
                    issues.append(
                        ColumnQualityIssue(
                            column=candidate_id,
                            issue_type="duplicate_identifiers",
                            severity="critical",
                            details=(
                                f"Identifier column '{candidate_id}' has {dup_id_count} non-unique values. "
                                "May indicate duplicated patient samples or longitudinal records."
                            ),
                            affected_count=dup_id_count,
                            affected_percentage=round((dup_id_count / n_rows) * 100, 2),
                        )
                    )
                break

        # 4. Inconsistent Data Types & Invalid Values
        inconsistent_dtype_columns: List[str] = []
        for col in df.columns:
            if col == detected_id_col or col == target_col:
                continue
            # Check if an object column is secretly numeric with bad tokens (e.g. "?", "N/A", "null")
            if df[col].dtype == object or str(df[col].dtype) == "string":
                converted = pd.to_numeric(df[col], errors="coerce")
                non_null_original = df[col].dropna()
                valid_converted = converted.dropna()
                if len(valid_converted) > 0 and len(valid_converted) >= 0.7 * len(non_null_original) and len(valid_converted) < len(non_null_original):
                    inconsistent_dtype_columns.append(col)
                    diff = len(non_null_original) - len(valid_converted)
                    issues.append(
                        ColumnQualityIssue(
                            column=col,
                            issue_type="inconsistent_data_type",
                            severity="critical",
                            details=(
                                f"Column '{col}' is stored as object but {len(valid_converted)}/{len(non_null_original)} "
                                f"values are numeric. {diff} invalid non-numeric tokens detected."
                            ),
                            affected_count=diff,
                            affected_percentage=round((diff / n_rows) * 100, 2),
                        )
                    )

        # 5. Constant & Near-Zero Variance Columns
        constant_columns: List[str] = []
        near_zero_variance_columns: List[str] = []

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            unique_vals = series.nunique()
            if unique_vals <= 1:
                constant_columns.append(col)
                issues.append(
                    ColumnQualityIssue(
                        column=col,
                        issue_type="constant_column",
                        severity="warning",
                        details=f"Column '{col}' contains a single constant value. Provides zero analytical variance.",
                        affected_count=n_rows,
                        affected_percentage=100.0,
                    )
                )
            elif pd.api.types.is_numeric_dtype(series):
                var = float(series.var()) if len(series) > 1 else 0.0
                if var < self.rules["near_zero_variance_threshold"]:
                    near_zero_variance_columns.append(col)
                    issues.append(
                        ColumnQualityIssue(
                            column=col,
                            issue_type="near_zero_variance",
                            severity="info",
                            details=f"Variance is {var:.6e}, below threshold {self.rules['near_zero_variance_threshold']}.",
                            affected_count=n_rows,
                            affected_percentage=100.0,
                        )
                    )
            else:
                # Frequency ratio check for categorical
                top_counts = series.value_counts(ascending=False).values
                if len(top_counts) >= 2 and (top_counts[0] / max(top_counts[1], 1)) > self.rules["near_zero_freq_ratio"]:
                    near_zero_variance_columns.append(col)
                    issues.append(
                        ColumnQualityIssue(
                            column=col,
                            issue_type="near_zero_variance",
                            severity="info",
                            details=f"Most frequent value dominates {top_counts[0]}/{len(series)} records (frequency ratio > {self.rules['near_zero_freq_ratio']}).",
                            affected_count=int(top_counts[0]),
                            affected_percentage=round((top_counts[0] / n_rows) * 100, 2),
                        )
                    )

        # 6. Outlier Summary (IQR method per numeric column)
        outlier_summary: Dict[str, int] = {}
        for col in df.select_dtypes(include=np.number).columns:
            if col == detected_id_col:
                continue
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - self.rules["outlier_iqr_multiplier"] * iqr
                upper = q75 + self.rules["outlier_iqr_multiplier"] * iqr
                outliers_count = int(((s < lower) | (s > upper)).sum())
                if outliers_count > 0:
                    outlier_summary[col] = outliers_count
                    if outliers_count / n_rows > 0.05:  # >5% outliers
                        issues.append(
                            ColumnQualityIssue(
                                column=col,
                                issue_type="high_outlier_count",
                                severity="info",
                                details=f"{outliers_count} IQR outliers ({outliers_count / n_rows * 100:.2f}%). Note: biological extremes may be valid.",
                                affected_count=outliers_count,
                                affected_percentage=round((outliers_count / n_rows) * 100, 2),
                            )
                        )

        # 7. High-Cardinality Categorical Columns
        high_cardinality_columns: List[str] = []
        for col in df.select_dtypes(exclude=np.number).columns:
            if col == detected_id_col or col == target_col:
                continue
            unique_count = df[col].nunique()
            unique_ratio = unique_count / n_rows
            if unique_count > self.rules["high_cardinality_threshold"] or unique_ratio > self.rules["high_cardinality_ratio"]:
                high_cardinality_columns.append(col)
                issues.append(
                    ColumnQualityIssue(
                        column=col,
                        issue_type="high_cardinality",
                        severity="warning",
                        details=f"Column has {unique_count} distinct categories ({unique_ratio * 100:.1f}% unique).",
                        affected_count=unique_count,
                        affected_percentage=round(unique_ratio * 100, 2),
                    )
                )

        # 8. Potential Data Leakage Detection
        potential_leakage_columns: List[str] = []
        if target_col and target_col in df.columns:
            y = df[target_col]
            # Convert target to numeric if binary
            if not pd.api.types.is_numeric_dtype(y):
                y_num = pd.factorize(y)[0]
            else:
                y_num = y.values

            for col in df.columns:
                if col == target_col or col == detected_id_col:
                    continue
                # Check column identity / exact copy
                if df[col].equals(df[target_col]):
                    potential_leakage_columns.append(col)
                    issues.append(
                        ColumnQualityIssue(
                            column=col,
                            issue_type="data_leakage_duplicate_target",
                            severity="critical",
                            details=f"Column '{col}' is identical to target '{target_col}'. Severe data leakage.",
                            affected_count=n_rows,
                            affected_percentage=100.0,
                        )
                    )
                # Check correlation
                elif pd.api.types.is_numeric_dtype(df[col]):
                    valid_mask = (~df[col].isna()) & (~pd.isna(y_num))
                    if valid_mask.sum() > 20:
                        corr = np.corrcoef(df[col][valid_mask], y_num[valid_mask])[0, 1]
                        if abs(corr) >= self.rules["leakage_correlation_threshold"]:
                            potential_leakage_columns.append(col)
                            issues.append(
                                ColumnQualityIssue(
                                    column=col,
                                    issue_type="data_leakage_high_correlation",
                                    severity="critical",
                                    details=f"Correlation with target is {corr:.4f} (>= {self.rules['leakage_correlation_threshold']}). High risk of leakage.",
                                    affected_count=n_rows,
                                    affected_percentage=100.0,
                                )
                            )

        # 9. Compute Data Quality Score (0 to 100)
        # Weighted deduction system
        score = 100.0

        # Completeness penalty: up to -30
        missing_fraction = total_missing / total_cells if total_cells > 0 else 0
        score -= min(30.0, missing_fraction * 100.0)

        # Uniqueness penalty: up to -15
        dup_row_frac = duplicate_rows_count / n_rows
        score -= min(15.0, dup_row_frac * 50.0)

        # Identifier integrity: -15 if duplicate IDs
        if duplicate_identifiers_count > 0:
            score -= 15.0

        # Inconsistent dtypes: -10 per column (max -20)
        score -= min(20.0, len(inconsistent_dtype_columns) * 10.0)

        # Constant columns: -5 per column (max -15)
        score -= min(15.0, len(constant_columns) * 5.0)

        # Data leakage: -25 if detected
        if len(potential_leakage_columns) > 0:
            score -= 25.0

        overall_score = round(max(0.0, min(100.0, score)), 1)

        # Qualitative interpretation
        if overall_score >= 90:
            interpretation = "Excellent: High data integrity, ready for pipeline modeling."
        elif overall_score >= 75:
            interpretation = "Good: Minor issues detected (outliers, minor missingness), manageable with standard preprocessing."
        elif overall_score >= 50:
            interpretation = "Moderate: Noticeable missingness, duplicates, or constant columns requiring wrangling."
        else:
            interpretation = "Poor: Severe data quality issues (high missingness, duplicate IDs, or potential leakage) must be resolved."

        logger.info(
            "DataQualityAnalyzer complete for '%s' | Score=%.1f (%s) | Issues=%d",
            dataset_name, overall_score, interpretation, len(issues),
        )

        return DataQualityReport(
            dataset_name=dataset_name,
            total_records=n_rows,
            total_columns=n_cols,
            overall_quality_score=overall_score,
            score_interpretation=interpretation,
            issues=issues,
            missing_summary=missing_summary,
            duplicate_rows_count=duplicate_rows_count,
            duplicate_identifiers_count=duplicate_identifiers_count,
            constant_columns=constant_columns,
            near_zero_variance_columns=near_zero_variance_columns,
            outlier_summary=outlier_summary,
            high_cardinality_columns=high_cardinality_columns,
            inconsistent_dtype_columns=inconsistent_dtype_columns,
            potential_leakage_columns=potential_leakage_columns,
            rules_applied=self.rules,
        )
