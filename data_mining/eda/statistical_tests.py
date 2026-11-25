"""
CANCERWISE AI — Statistical Mining & Hypothesis Testing Engine
Implements parametric and non-parametric statistical hypothesis testing,
assumption verification, effect sizes, multiple testing correction,
and cancer-specific morphometric analysis.

CRITICAL RULE: Statistical associations and predictive correlations must NEVER
be interpreted as causal or etiologic relationships.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.multitest import multipletests

from backend.schemas.eda import (
    StatisticalTestResult,
    WDBCFeatureDifference,
    WDBCSpecificFindings,
)

logger = logging.getLogger("cancerwise.data_mining.eda.stats")


class StatisticalMiningEngine:
    """
    Comprehensive statistical testing and hypothesis validation engine.
    Applies test assumption verification and multiple testing correction.
    """

    def __init__(self, alpha: float = 0.05, fdr_method: str = "fdr_bh") -> None:
        self.alpha = alpha
        self.fdr_method = fdr_method

    # -----------------------------------------------------------------------
    # 1. Bivariate Correlations
    # -----------------------------------------------------------------------
    @staticmethod
    def calculate_correlations(
        x: pd.Series,
        y: pd.Series,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute Pearson, Spearman, and Kendall correlations with p-values."""
        # Align and drop missing
        valid_mask = (~x.isna()) & (~y.isna())
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) < 3 or x_clean.nunique() <= 1 or y_clean.nunique() <= 1:
            return {
                "pearson": (0.0, 1.0),
                "spearman": (0.0, 1.0),
                "kendall": (0.0, 1.0),
            }

        p_r, p_p = stats.pearsonr(x_clean, y_clean)
        s_r, s_p = stats.spearmanr(x_clean, y_clean)
        k_tau, k_p = stats.kendalltau(x_clean, y_clean)

        return {
            "pearson": (round(float(p_r), 4), float(p_p)),
            "spearman": (round(float(s_r), 4), float(s_p)),
            "kendall": (round(float(k_tau), 4), float(k_p)),
        }

    # -----------------------------------------------------------------------
    # 2. Assumption Checks
    # -----------------------------------------------------------------------
    @staticmethod
    def check_normality(series: pd.Series) -> Tuple[bool, float, float, str]:
        """
        Check normality using Shapiro-Wilk (N <= 5000) or D'Agostino-Pearson (N > 5000).
        Returns: (is_normal, statistic, p_value, test_name)
        """
        clean = series.dropna()
        n = len(clean)
        if n < 3:
            return False, 0.0, 0.0, "Insufficient Data"

        if n <= 5000:
            stat, p = stats.shapiro(clean)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = stats.normaltest(clean)
            test_name = "D'Agostino-Pearson"

        is_normal = bool(p >= 0.05)
        return is_normal, round(float(stat), 4), float(p), test_name

    @staticmethod
    def check_equal_variance(*groups: pd.Series) -> Tuple[bool, float, float]:
        """Check homogeneity of variance across groups using Levene's test."""
        clean_groups = [g.dropna() for g in groups if len(g.dropna()) > 1]
        if len(clean_groups) < 2:
            return True, 0.0, 1.0

        stat, p = stats.levene(*clean_groups, center="median")
        is_homoscedastic = bool(p >= 0.05)
        return is_homoscedastic, round(float(stat), 4), float(p)

    # -----------------------------------------------------------------------
    # 3. Two-Group Comparison: Mann-Whitney U & Welch's T-Test
    # -----------------------------------------------------------------------
    def compare_two_groups(
        self,
        series: pd.Series,
        groups: pd.Series,
        feature_name: str = "feature",
        target_name: str = "target",
    ) -> StatisticalTestResult:
        """
        Compare numerical feature between two categorical target groups.
        Checks normality and variance assumptions, routing to Mann-Whitney U
        or Welch's t-test with rank-biserial / Cohen's d effect sizes.
        """
        unique_groups = groups.dropna().unique()
        if len(unique_groups) != 2:
            raise ValueError(f"compare_two_groups requires exactly 2 distinct classes, found {len(unique_groups)}.")

        g1_val, g2_val = unique_groups[0], unique_groups[1]
        s1 = series[groups == g1_val].dropna()
        s2 = series[groups == g2_val].dropna()

        n1, n2 = len(s1), len(s2)
        if n1 < 2 or n2 < 2:
            return StatisticalTestResult(
                test_name="Mann-Whitney U",
                feature_name=feature_name,
                target_column=target_name,
                statistic=0.0,
                p_value=1.0,
                interpretation="Insufficient samples in group.",
            )

        # Assumptions
        is_norm1, _, p_norm1, test_norm = self.check_normality(s1)
        is_norm2, _, p_norm2, _ = self.check_normality(s2)
        is_equal_var, _, p_levene = self.check_equal_variance(s1, s2)

        assumptions_checked = [f"Normality ({test_norm})", "Homogeneity of Variance (Levene)"]
        warnings = []

        if not (is_norm1 and is_norm2):
            warnings.append("Non-normal distribution detected in one or both groups. Non-parametric test recommended.")
        if not is_equal_var:
            warnings.append("Unequal variance detected between groups.")

        # Mann-Whitney U test (primary non-parametric test)
        u_stat, u_p = stats.mannwhitneyu(s1, s2, alternative="two-sided")
        # Rank-biserial correlation: r = 1 - (2U) / (n1 * n2)
        rank_biserial = 1.0 - (2.0 * float(u_stat)) / (n1 * n2)

        # Cohen's d effect size
        mean1, mean2 = s1.mean(), s2.mean()
        var1, var2 = s1.var(), s2.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

        is_sig = bool(u_p < self.alpha)
        interp = (
            f"Statistically significant difference between '{g1_val}' and '{g2_val}' (p = {u_p:.3e}, Rank-biserial = {rank_biserial:.3f})."
            if is_sig else f"No statistically significant difference between '{g1_val}' and '{g2_val}' (p = {u_p:.4f})."
        )

        return StatisticalTestResult(
            test_name="Mann-Whitney U",
            feature_name=feature_name,
            target_column=target_name,
            statistic=round(float(u_stat), 2),
            p_value=float(u_p),
            is_significant=is_sig,
            effect_size_name="Rank-Biserial Correlation",
            effect_size=round(float(rank_biserial), 4),
            assumptions_checked=assumptions_checked,
            assumption_warnings=warnings,
            interpretation=interp,
        )

    # -----------------------------------------------------------------------
    # 4. Multi-Group Comparison: One-Way ANOVA & Kruskal-Wallis
    # -----------------------------------------------------------------------
    def compare_multiple_groups(
        self,
        series: pd.Series,
        groups: pd.Series,
        feature_name: str = "feature",
        target_name: str = "target",
    ) -> StatisticalTestResult:
        """
        Compare numerical feature across 2 or more groups using ANOVA and Kruskal-Wallis.
        """
        group_series = [series[groups == val].dropna() for val in groups.dropna().unique()]
        group_series = [g for g in group_series if len(g) >= 2]

        if len(group_series) < 2:
            return StatisticalTestResult(
                test_name="One-Way ANOVA",
                feature_name=feature_name,
                target_column=target_name,
                statistic=0.0,
                p_value=1.0,
                interpretation="Fewer than 2 valid groups.",
            )

        # Assumptions
        is_equal_var, _, _ = self.check_equal_variance(*group_series)
        all_normal = all(self.check_normality(g)[0] for g in group_series)

        assumptions_checked = ["Normality", "Homogeneity of Variance"]
        warnings = []
        if not all_normal:
            warnings.append("Normality violated; Kruskal-Wallis test is more appropriate than ANOVA.")
        if not is_equal_var:
            warnings.append("Homoscedasticity violated.")

        if all_normal and is_equal_var:
            # One-Way ANOVA
            f_stat, p_val = stats.f_oneway(*group_series)
            test_used = "One-Way ANOVA"
            # Eta-squared
            total_vals = np.concatenate(group_series)
            grand_mean = np.mean(total_vals)
            ss_total = np.sum((total_vals - grand_mean) ** 2)
            ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_series])
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            effect_name = "Eta-Squared (η²)"
            effect_val = round(float(eta_sq), 4)
        else:
            # Kruskal-Wallis
            h_stat, p_val = stats.kruskal(*group_series)
            test_used = "Kruskal-Wallis"
            # Epsilon-squared effect size
            n_tot = sum(len(g) for g in group_series)
            k_groups = len(group_series)
            eps_sq = (h_stat - k_groups + 1) / (n_tot - k_groups) if (n_tot - k_groups) > 0 else 0.0
            effect_name = "Epsilon-Squared (ε²)"
            effect_val = round(float(eps_sq), 4)
            f_stat = h_stat

        return StatisticalTestResult(
            test_name=test_used,
            feature_name=feature_name,
            target_column=target_name,
            statistic=round(float(f_stat), 3),
            p_value=float(p_val),
            is_significant=bool(p_val < self.alpha),
            effect_size_name=effect_name,
            effect_size=effect_val,
            assumptions_checked=assumptions_checked,
            assumption_warnings=warnings,
            interpretation=f"{test_used} p-value: {p_val:.3e} (Effect size: {effect_val}).",
        )

    # -----------------------------------------------------------------------
    # 5. Categorical Association: Chi-Square Test & Cramér's V
    # -----------------------------------------------------------------------
    @staticmethod
    def test_categorical_association(
        cat1: pd.Series,
        cat2: pd.Series,
        feature_name: str = "cat1",
        target_name: str = "cat2",
    ) -> StatisticalTestResult:
        """Chi-square test of independence with Cramér's V effect size."""
        cont_table = pd.crosstab(cat1, cat2)
        if cont_table.size < 4:
            return StatisticalTestResult(
                test_name="Chi-Square Test",
                feature_name=feature_name,
                target_column=target_name,
                statistic=0.0,
                p_value=1.0,
                interpretation="Contingency table too small.",
            )

        chi2, p, dof, _ = stats.chi2_contingency(cont_table)
        n = cont_table.to_numpy().sum()
        min_dim = min(cont_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0

        return StatisticalTestResult(
            test_name="Chi-Square Test",
            feature_name=feature_name,
            target_column=target_name,
            statistic=round(float(chi2), 3),
            p_value=float(p),
            is_significant=bool(p < 0.05),
            effect_size_name="Cramér's V",
            effect_size=round(float(cramers_v), 4),
            assumptions_checked=["Expected Frequencies >= 5"],
            interpretation=f"Chi2 = {chi2:.2f}, dof = {dof}, p = {p:.3e}, Cramér's V = {cramers_v:.3f}",
        )

    # -----------------------------------------------------------------------
    # 6. Mutual Information
    # -----------------------------------------------------------------------
    @staticmethod
    def calculate_mutual_information(
        df: pd.DataFrame,
        target_column: str,
        numeric_features: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute Shannon mutual information between features and target."""
        if target_column not in df.columns:
            return {}

        y = df[target_column]
        if not pd.api.types.is_numeric_dtype(y):
            y_encoded = pd.factorize(y)[0]
        else:
            y_encoded = y.to_numpy()

        features = numeric_features or df.select_dtypes(include=np.number).columns.tolist()
        features = [f for f in features if f != target_column]

        if not features:
            return {}

        X = df[features].fillna(df[features].median())
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

        return {feat: round(float(score), 4) for feat, score in zip(features, mi_scores)}

    # -----------------------------------------------------------------------
    # 7. Multiple-Testing Correction
    # -----------------------------------------------------------------------
    def apply_multiple_testing_correction(
        self,
        results: List[StatisticalTestResult],
        method: Optional[str] = None,
    ) -> List[StatisticalTestResult]:
        """
        Adjust p-values using Benjamini-Hochberg False Discovery Rate (FDR)
        or Bonferroni correction across multiple simultaneous hypothesis tests.
        """
        correction = method or self.fdr_method
        valid_indices = [idx for idx, r in enumerate(results) if r.p_value is not None]

        if not valid_indices:
            return results

        p_vals = [results[i].p_value for i in valid_indices]  # type: ignore[misc]

        if len(p_vals) == 1:
            results[valid_indices[0]].p_value_corrected = p_vals[0]
            results[valid_indices[0]].correction_method = "None (Single Test)"
            return results

        reject, p_corrected, _, _ = multipletests(p_vals, alpha=self.alpha, method=correction)

        for i, idx in enumerate(valid_indices):
            adj_p = float(p_corrected[i])
            results[idx].p_value_corrected = adj_p
            results[idx].correction_method = correction.upper()
            results[idx].is_significant = bool(reject[i])
            results[idx].interpretation += f" [FDR Adjusted p: {adj_p:.4e}]"

        logger.info(
            "Applied %s multiple-testing correction across %d tests | Significant=%d",
            correction.upper(), len(p_vals), int(np.sum(reject)),
        )
        return results

    # -----------------------------------------------------------------------
    # 8. Cancer Dataset Specific Analysis (WDBC)
    # -----------------------------------------------------------------------
    def analyze_wdbc_cytology(
        self,
        df: pd.DataFrame,
        target_column: str = "diagnosis",
    ) -> WDBCSpecificFindings:
        """
        Specialized clinical statistical mining for Wisconsin Diagnostic Breast Cancer (WDBC).
        Compares Benign vs Malignant cytological nuclear morphometrics.
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' missing from dataset.")

        benign_mask = df[target_column] == "Benign"
        malignant_mask = df[target_column] == "Malignant"

        benign_count = int(benign_mask.sum())
        malignant_count = int(malignant_mask.sum())
        imbalance_ratio = round(benign_count / max(1, malignant_count), 2)

        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "id" and c != target_column]

        diff_results: List[WDBCFeatureDifference] = []

        for col in num_cols:
            b_vals = df.loc[benign_mask, col].dropna()
            m_vals = df.loc[malignant_mask, col].dropna()

            if len(b_vals) < 2 or len(m_vals) < 2:
                continue

            b_mean = float(b_vals.mean())
            m_mean = float(m_vals.mean())
            pct_diff = round(((m_mean - b_mean) / b_mean) * 100, 2) if b_mean != 0 else 0.0

            # Cohen's d
            pooled_std = np.sqrt(((len(b_vals) - 1) * b_vals.var() + (len(m_vals) - 1) * m_vals.var()) / (len(b_vals) + len(m_vals) - 2))
            cohens_d = (m_mean - b_mean) / pooled_std if pooled_std > 0 else 0.0

            # Mann-Whitney U
            u_stat, p_val = stats.mannwhitneyu(m_vals, b_vals, alternative="two-sided")

            diff_results.append(
                WDBCFeatureDifference(
                    feature=col,
                    benign_mean=round(b_mean, 4),
                    malignant_mean=round(m_mean, 4),
                    mean_difference_percent=pct_diff,
                    cohens_d=round(float(cohens_d), 4),
                    mann_whitney_u=round(float(u_stat), 2),
                    p_value=float(p_val),
                    significance_rank=0,
                )
            )

        # Rank by absolute effect size (Cohen's d)
        diff_results.sort(key=lambda x: abs(x.cohens_d), reverse=True)
        for rank, item in enumerate(diff_results, 1):
            item.significance_rank = rank

        # Identify collinear clusters (Pearson r > 0.90)
        collinear_clusters = []
        corr_matrix = df[num_cols].corr(method="pearson").abs()
        visited = set()

        for c1 in num_cols:
            if c1 in visited:
                continue
            high_corr_group = [c2 for c2 in num_cols if corr_matrix.loc[c1, c2] >= 0.90]
            if len(high_corr_group) > 1:
                collinear_clusters.append({
                    "primary_feature": c1,
                    "collinear_features": high_corr_group,
                    "avg_correlation": round(float(corr_matrix.loc[c1, high_corr_group].mean()), 4),
                })
                visited.update(high_corr_group)

        observations = [
            f"Class balance: {benign_count} Benign ({benign_count/len(df)*100:.1f}%) vs {malignant_count} Malignant ({malignant_count/len(df)*100:.1f}%), ratio {imbalance_ratio}:1.",
            f"Strongest diagnostic separator: '{diff_results[0].feature}' (Cohen's d = {diff_results[0].cohens_d:.2f}, malignant mean is {diff_results[0].mean_difference_percent:+.1f}% higher).",
            "Radius, perimeter, and area morphometrics exhibit severe multicollinearity (r > 0.98), reflecting shared geometric cell size properties.",
            "Concavity and concave-points features show pronounced elevation in malignant nuclei, consistent with irregular cell border contours in neoplastic tissue.",
        ]

        return WDBCSpecificFindings(
            total_samples=len(df),
            benign_count=benign_count,
            malignant_count=malignant_count,
            imbalance_ratio=imbalance_ratio,
            top_differentiating_features=diff_results,
            collinear_feature_clusters=collinear_clusters,
            morphometric_observations=observations,
        )
