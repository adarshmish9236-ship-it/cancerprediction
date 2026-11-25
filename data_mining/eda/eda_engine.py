"""
CANCERWISE AI — Exploratory Data Analysis Engine
Master coordinator for dataset overview, univariate/bivariate profiling,
multivariate correlation and dimensionality reduction, hypothesis testing,
and cancer-specific morphometric analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.schemas.eda import (
    DatasetOverview,
    UnivariateStats,
    BivariateAnalysisResult,
    MultivariateStats,
    StatisticalTestResult,
    WDBCSpecificFindings,
)
from data_mining.eda.statistical_tests import StatisticalMiningEngine

logger = logging.getLogger("cancerwise.data_mining.eda.engine")


class EDAEngine:
    """
    Core exploratory data analysis and statistical mining orchestrator.
    """

    def __init__(self, stats_engine: Optional[StatisticalMiningEngine] = None) -> None:
        self.stats_engine = stats_engine or StatisticalMiningEngine()

    # -----------------------------------------------------------------------
    # 1. Dataset Overview
    # -----------------------------------------------------------------------
    def get_overview(
        self,
        df: pd.DataFrame,
        dataset_id: str = "dataset",
        dataset_name: str = "Cancer Dataset",
        target_column: Optional[str] = "diagnosis",
    ) -> DatasetOverview:
        """Generate high-level dataset metrics, dimensions, target balance, and memory usage."""
        n_rows, n_cols = df.shape
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        total_cells = n_rows * n_cols
        missing_cells = int(df.isna().sum().sum())
        missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        dup_rows = int(df.duplicated().sum())
        dup_pct = round((dup_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

        target_dist: Dict[str, int] = {}
        target_props: Dict[str, float] = {}

        if target_column and target_column in df.columns:
            counts = df[target_column].value_counts()
            target_dist = {str(k): int(v) for k, v in counts.items()}
            target_props = {str(k): round(float(v / n_rows), 4) for k, v in counts.items()}

        mem_bytes = int(df.memory_usage(deep=True).sum())
        mem_mb = round(mem_bytes / (1024 * 1024), 2)

        summary_text = (
            f"Dataset '{dataset_name}' contains {n_rows:,} instances across {n_cols} attributes "
            f"({len(num_cols)} numerical, {len(cat_cols)} categorical). "
            f"Missing data: {missing_pct}%. Duplicate records: {dup_pct}%. "
            f"Memory footprint: {mem_mb} MB."
        )

        return DatasetOverview(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            total_records=n_rows,
            total_features=n_cols,
            numerical_features_count=len(num_cols),
            categorical_features_count=len(cat_cols),
            numerical_features=num_cols,
            categorical_features=cat_cols,
            missing_value_cells=missing_cells,
            missing_value_percentage=missing_pct,
            duplicate_records_count=dup_rows,
            duplicate_records_percentage=dup_pct,
            target_column=target_column,
            target_distribution=target_dist,
            target_proportions=target_props,
            memory_utilization_bytes=mem_bytes,
            memory_utilization_mb=mem_mb,
            summary_text=summary_text,
        )

    # -----------------------------------------------------------------------
    # 2. Univariate Profile
    # -----------------------------------------------------------------------
    def get_univariate_profile(
        self,
        df: pd.DataFrame,
        feature_name: str,
    ) -> UnivariateStats:
        """Compute comprehensive distribution metrics and normality tests for a feature."""
        if feature_name not in df.columns:
            raise KeyError(f"Feature '{feature_name}' not found in dataset.")

        series = df[feature_name]
        n_rows = len(df)
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
        distinct_count = int(series.nunique())

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            mean_val = round(float(clean.mean()), 4) if not clean.empty else None
            std_val = round(float(clean.std()), 4) if len(clean) > 1 else None
            median_val = round(float(clean.median()), 4) if not clean.empty else None
            q25 = round(float(clean.quantile(0.25)), 4) if not clean.empty else None
            q75 = round(float(clean.quantile(0.75)), 4) if not clean.empty else None
            iqr = round(q75 - q25, 4) if (q25 is not None and q75 is not None) else None
            min_val = round(float(clean.min()), 4) if not clean.empty else None
            max_val = round(float(clean.max()), 4) if not clean.empty else None
            skew_val = round(float(clean.skew()), 4) if len(clean) > 2 else None
            kurt_val = round(float(clean.kurtosis()), 4) if len(clean) > 3 else None

            is_norm, norm_stat, norm_p, test_name = self.stats_engine.check_normality(clean)

            return UnivariateStats(
                feature_name=feature_name,
                dtype=str(series.dtype),
                count=int(clean.count()),
                missing_count=missing_count,
                missing_percentage=missing_pct,
                distinct_count=distinct_count,
                mean=mean_val,
                std=std_val,
                median=median_val,
                q25=q25,
                q75=q75,
                iqr=iqr,
                min_val=min_val,
                max_val=max_val,
                skewness=skew_val,
                kurtosis=kurt_val,
                normality_test_name=test_name,
                normality_statistic=norm_stat,
                normality_p_value=norm_p,
                is_normally_distributed=is_norm,
            )
        else:
            # Categorical feature
            top_cats = {str(k): int(v) for k, v in series.value_counts().head(10).items()}
            return UnivariateStats(
                feature_name=feature_name,
                dtype=str(series.dtype),
                count=int(series.count()),
                missing_count=missing_count,
                missing_percentage=missing_pct,
                distinct_count=distinct_count,
                top_categories=top_cats,
            )

    # -----------------------------------------------------------------------
    # 3. Bivariate Profile
    # -----------------------------------------------------------------------
    def get_bivariate_profile(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        target_column: Optional[str] = None,
    ) -> BivariateAnalysisResult:
        """Compute correlations and group statistics for two features."""
        corrs = self.stats_engine.calculate_correlations(df[x_col], df[y_col])
        p_r, p_p = corrs["pearson"]
        s_r, s_p = corrs["spearman"]
        k_tau, k_p = corrs["kendall"]

        # Relationship strength
        abs_r = abs(p_r)
        if abs_r >= 0.8:
            strength = "Very Strong"
        elif abs_r >= 0.6:
            strength = "Strong"
        elif abs_r >= 0.4:
            strength = "Moderate"
        elif abs_r >= 0.2:
            strength = "Weak"
        else:
            strength = "Negligible"

        # Group statistics if target column present
        group_stats: Dict[str, Dict[str, float]] = {}
        if target_column and target_column in df.columns:
            for grp, subset in df.groupby(target_column):
                group_stats[str(grp)] = {
                    f"{x_col}_mean": round(float(subset[x_col].mean()), 4),
                    f"{y_col}_mean": round(float(subset[y_col].mean()), 4),
                }

        # Subsample for scatter preview
        sample_df = df[[x_col, y_col]].dropna().head(100)
        sample_records = sample_df.to_dict(orient="records")

        return BivariateAnalysisResult(
            feature_x=x_col,
            feature_y=y_col,
            pearson_r=p_r,
            pearson_p=p_p,
            spearman_rho=s_r,
            spearman_p=s_p,
            kendall_tau=k_tau,
            kendall_p=k_p,
            relationship_strength=strength,
            scatter_sample=sample_records,
            group_statistics=group_stats,
        )

    # -----------------------------------------------------------------------
    # 4. Multivariate Profile
    # -----------------------------------------------------------------------
    def get_multivariate_profile(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        target_column: Optional[str] = "diagnosis",
    ) -> MultivariateStats:
        """Compute correlation matrix, top correlated pairs, and PCA explained variance."""
        cols = numeric_columns or [c for c in df.select_dtypes(include=np.number).columns if c != "id" and c != target_column]
        cols = cols[:30]  # Limit to 30 features

        subset = df[cols].dropna()

        corr_matrix_df = subset.corr(method="pearson").round(4)
        corr_dict = corr_matrix_df.to_dict()

        # Find top correlated pairs (excluding diagonal)
        pairs = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr_matrix_df.loc[c1, c2]
                pairs.append({"feature_1": c1, "feature_2": c2, "correlation": round(float(val), 4)})
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset)
        pca = PCA(n_components=min(10, len(cols)), random_state=42)
        pca.fit(X_scaled)
        exp_vars = [round(float(v), 4) for v in pca.explained_variance_ratio_]

        return MultivariateStats(
            features_included=cols,
            correlation_matrix=corr_dict,
            pca_explained_variance_ratio=exp_vars,
            pca_total_variance_explained=round(float(sum(exp_vars)), 4),
            top_correlated_pairs=pairs[:15],
        )

    # -----------------------------------------------------------------------
    # 5. Full Hypothesis Testing Pipeline
    # -----------------------------------------------------------------------
    def run_statistical_tests(
        self,
        df: pd.DataFrame,
        target_column: str = "diagnosis",
        alpha: float = 0.05,
    ) -> List[StatisticalTestResult]:
        """
        Execute hypothesis tests (Mann-Whitney U) across all numeric features
        against binary target, applying Benjamini-Hochberg FDR multiple testing correction.
        """
        if target_column not in df.columns:
            return []

        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "id" and c != target_column]
        raw_results = []

        for col in num_cols:
            res = self.stats_engine.compare_two_groups(
                series=df[col],
                groups=df[target_column],
                feature_name=col,
                target_name=target_column,
            )
            raw_results.append(res)

        # Apply Benjamini-Hochberg FDR correction
        corrected_results = self.stats_engine.apply_multiple_testing_correction(raw_results, method="fdr_bh")
        return corrected_results

    # -----------------------------------------------------------------------
    # 6. Cancer Specific Analysis (WDBC)
    # -----------------------------------------------------------------------
    def analyze_cancer_wdbc(
        self,
        df: pd.DataFrame,
        target_column: str = "diagnosis",
    ) -> WDBCSpecificFindings:
        """Run specialized cytology analysis for Wisconsin Diagnostic Breast Cancer."""
        return self.stats_engine.analyze_wdbc_cytology(df, target_column=target_column)
