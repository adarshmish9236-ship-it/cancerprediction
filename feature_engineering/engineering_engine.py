"""
CANCERWISE AI — Feature Engineering Engine
Generates polynomial, interaction, ratio, statistical aggregation, logarithmic,
and domain-informed nuclear cytology features while maintaining a reproducible derived feature registry.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from backend.schemas.feature_engineering import (
    DerivedFeatureDefinition,
    FeatureEngineeringResult,
)

logger = logging.getLogger("cancerwise.feature_engineering.engine")


class FeatureEngineeringEngine:
    """
    Automated feature generation and cytological domain-informed feature creation.
    Tracks mathematical derivation and biomedical rationale for all generated features.
    """

    def __init__(self) -> None:
        self.derived_registry: Dict[str, DerivedFeatureDefinition] = {}

    def clear_registry(self) -> None:
        """Clear the derived feature registry."""
        self.derived_registry = {}

    def create_cytology_domain_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[DerivedFeatureDefinition]]:
        """
        Create domain-informed nuclear morphology features for cancer cytology:
        - Sphericity approximation: sqrt(pi * area) / perimeter
        - Circularity index: 4 * pi * area / (perimeter^2)
        - Nuclear irregularity index: concavity * fractal_dimension
        - Border roughness: perimeter / radius
        - Worst-to-Mean dynamic expansion ratios
        """
        df_out = df.copy()
        new_defs: List[DerivedFeatureDefinition] = []

        # 1. Circularity Index: 4 * pi * area / (perimeter^2)
        if "area_mean" in df.columns and "perimeter_mean" in df.columns:
            feat_name = "circularity_index_mean"
            df_out[feat_name] = (4.0 * np.pi * df_out["area_mean"]) / (df_out["perimeter_mean"] ** 2 + 1e-8)
            d = DerivedFeatureDefinition(
                feature_name=feat_name,
                base_features=["area_mean", "perimeter_mean"],
                transformation_type="domain",
                formula="4 * pi * area_mean / (perimeter_mean^2)",
                domain_rationale="Measures circular deviation; malignant nuclei exhibit lower circularity due to pleomorphic membrane indentations.",
            )
            self.derived_registry[feat_name] = d
            new_defs.append(d)

        # 2. Sphericity Approximation: sqrt(pi * area) / perimeter
        if "area_mean" in df.columns and "perimeter_mean" in df.columns:
            feat_name = "sphericity_mean"
            df_out[feat_name] = np.sqrt(np.pi * np.maximum(df_out["area_mean"], 0)) / (df_out["perimeter_mean"] + 1e-8)
            d = DerivedFeatureDefinition(
                feature_name=feat_name,
                base_features=["area_mean", "perimeter_mean"],
                transformation_type="domain",
                formula="sqrt(pi * area_mean) / perimeter_mean",
                domain_rationale="Quantifies boundary regular shape; rounded benign nuclei score close to 0.5, irregular malignant nuclei drop significantly.",
            )
            self.derived_registry[feat_name] = d
            new_defs.append(d)

        # 3. Nuclear Irregularity Index: concavity_mean * fractal_dimension_mean
        if "concavity_mean" in df.columns and "fractal_dimension_mean" in df.columns:
            feat_name = "nuclear_irregularity_mean"
            df_out[feat_name] = df_out["concavity_mean"] * df_out["fractal_dimension_mean"]
            d = DerivedFeatureDefinition(
                feature_name=feat_name,
                base_features=["concavity_mean", "fractal_dimension_mean"],
                transformation_type="domain",
                formula="concavity_mean * fractal_dimension_mean",
                domain_rationale="Couples structural contour indentation with boundary fractal complexity; strongly elevated in high-grade carcinoma.",
            )
            self.derived_registry[feat_name] = d
            new_defs.append(d)

        # 4. Nuclear Border Roughness: perimeter_mean / radius_mean
        if "perimeter_mean" in df.columns and "radius_mean" in df.columns:
            feat_name = "border_roughness_mean"
            df_out[feat_name] = df_out["perimeter_mean"] / (df_out["radius_mean"] + 1e-8)
            d = DerivedFeatureDefinition(
                feature_name=feat_name,
                base_features=["perimeter_mean", "radius_mean"],
                transformation_type="domain",
                formula="perimeter_mean / radius_mean",
                domain_rationale="Ratio exceeding 2*pi (~6.28) indicates ruffled nuclear membrane folding.",
            )
            self.derived_registry[feat_name] = d
            new_defs.append(d)

        # 5. Dynamic Expansion Ratios (Worst / Mean)
        paired_morphometrics = ["radius", "area", "perimeter", "concavity", "concave_points"]
        for p in paired_morphometrics:
            worst_col = f"{p}_worst"
            mean_col = f"{p}_mean"
            if worst_col in df.columns and mean_col in df.columns:
                feat_name = f"{p}_expansion_ratio"
                df_out[feat_name] = df_out[worst_col] / (df_out[mean_col] + 1e-8)
                d = DerivedFeatureDefinition(
                    feature_name=feat_name,
                    base_features=[worst_col, mean_col],
                    transformation_type="ratio",
                    formula=f"{worst_col} / {mean_col}",
                    domain_rationale=f"Assesses intra-tumor morphological heterogeneity; malignant aspirates have higher worst-to-mean {p} variance.",
                )
                self.derived_registry[feat_name] = d
                new_defs.append(d)

        logger.info("Engineered %d cytology domain features.", len(new_defs))
        return df_out, new_defs

    def create_ratio_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[DerivedFeatureDefinition]]:
        """
        Creates perimeter-to-area and dynamic expansion ratios.
        """
        df_out = df.copy()
        new_defs: List[DerivedFeatureDefinition] = []

        # Perimeter to Area ratio
        for suffix in ["mean", "worst", "se"]:
            p_col = f"perimeter_{suffix}"
            a_col = f"area_{suffix}"
            if p_col in df.columns and a_col in df.columns:
                feat_name = f"perimeter_to_area_{suffix}"
                df_out[feat_name] = df_out[p_col] / (df_out[a_col] + 1e-8)
                d = DerivedFeatureDefinition(
                    feature_name=feat_name,
                    base_features=[p_col, a_col],
                    transformation_type="ratio",
                    formula=f"{p_col} / {a_col}",
                    domain_rationale=f"Boundary perimeter relative to interior cellular area ({suffix}).",
                )
                self.derived_registry[feat_name] = d
                new_defs.append(d)

        # Worst to mean ratio for all matching pairs
        worst_cols = [c for c in df.columns if c.endswith("_worst")]
        for w_col in worst_cols:
            base_stem = w_col[:-6]
            m_col = f"{base_stem}_mean"
            if m_col in df.columns:
                feat_name = f"ratio_worst_to_mean_{w_col}"
                df_out[feat_name] = df_out[w_col] / (df_out[m_col] + 1e-8)
                d = DerivedFeatureDefinition(
                    feature_name=feat_name,
                    base_features=[w_col, m_col],
                    transformation_type="ratio",
                    formula=f"{w_col} / {m_col}",
                    domain_rationale=f"Ratio of worst to mean {base_stem}.",
                )
                self.derived_registry[feat_name] = d
                new_defs.append(d)

        logger.info("Engineered %d ratio features.", len(new_defs))
        return df_out, new_defs

    def create_polynomial_and_interaction_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        degree: int = 2,
        interaction_only: bool = True,
    ) -> Tuple[pd.DataFrame, List[DerivedFeatureDefinition]]:
        """
        Generate interaction products and polynomial features on selected numerical attributes.
        """
        df_out = df.copy()
        new_defs: List[DerivedFeatureDefinition] = []

        if features is None:
            features = [c for c in ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "concavity_mean"] if c in df.columns]

        valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if len(valid_features) < 2:
            return df_out, new_defs

        subset = df[valid_features].fillna(df[valid_features].median())

        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_arr = poly.fit_transform(subset)
        poly_names = poly.get_feature_names_out(valid_features)

        for idx, name in enumerate(poly_names):
            if name not in valid_features:
                clean_name = name.replace(" ", "_x_")
                df_out[clean_name] = poly_arr[:, idx]
                d = DerivedFeatureDefinition(
                    feature_name=clean_name,
                    base_features=name.split(" "),
                    transformation_type="interaction" if " " in name else "polynomial",
                    formula=name.replace(" ", " * "),
                    domain_rationale="Non-linear feature interaction capturing synergistic morphometric deviations.",
                )
                self.derived_registry[clean_name] = d
                new_defs.append(d)

        logger.info("Engineered %d polynomial/interaction features.", len(new_defs))
        return df_out, new_defs

    def create_statistical_aggregations(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[DerivedFeatureDefinition]]:
        """
        Extract row-wise statistical summaries (mean, std, min, max) across cytology categories
        (e.g., across all _mean features, across all _worst features).
        """
        df_out = df.copy()
        new_defs: List[DerivedFeatureDefinition] = []

        for category in ["mean", "se", "worst"]:
            cols = [c for c in df.select_dtypes(include=np.number).columns if c.endswith(f"_{category}") and c != "id"]
            if len(cols) >= 3:
                # Row-wise mean
                name_mean = f"{category}_morphology_composite_mean"
                df_out[name_mean] = df[cols].mean(axis=1)
                d1 = DerivedFeatureDefinition(
                    feature_name=name_mean,
                    base_features=cols,
                    transformation_type="statistical",
                    formula=f"mean({cols})",
                    domain_rationale=f"Global morphometric magnitude across all {category} cellular dimensions.",
                )
                self.derived_registry[name_mean] = d1
                new_defs.append(d1)

                # Row-wise max
                name_max = f"{category}_morphology_composite_max"
                df_out[name_max] = df[cols].max(axis=1)
                d2 = DerivedFeatureDefinition(
                    feature_name=name_max,
                    base_features=cols,
                    transformation_type="statistical",
                    formula=f"max({cols})",
                    domain_rationale=f"Maximum morphometric extent across all {category} cellular dimensions.",
                )
                self.derived_registry[name_max] = d2
                new_defs.append(d2)

                # Row-wise standard deviation (intra-cellular heterogeneity)
                name_std = f"{category}_morphology_composite_std"
                df_out[name_std] = df[cols].std(axis=1)
                d3 = DerivedFeatureDefinition(
                    feature_name=name_std,
                    base_features=cols,
                    transformation_type="statistical",
                    formula=f"std({cols})",
                    domain_rationale=f"Cellular morphometric dispersion across all {category} measurements.",
                )
                self.derived_registry[name_std] = d3
                new_defs.append(d3)

        logger.info("Engineered %d statistical aggregation features.", len(new_defs))
        return df_out, new_defs

    def create_logarithmic_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[DerivedFeatureDefinition]]:
        """
        Apply log1p transformation to highly right-skewed morphometric features (area, perimeter).
        """
        df_out = df.copy()
        new_defs: List[DerivedFeatureDefinition] = []

        target_cols = features or [c for c in df.select_dtypes(include=np.number).columns if "area" in c or "perimeter" in c]

        for col in target_cols:
            if col in df.columns:
                name_log = f"log1p_{col}"
                min_val = df[col].min()
                offset = abs(min_val) + 1.0 if min_val < 0 else 1.0
                df_out[name_log] = np.log1p(np.maximum(df[col] + (offset - 1.0), 0))
                d = DerivedFeatureDefinition(
                    feature_name=name_log,
                    base_features=[col],
                    transformation_type="log",
                    formula=f"log1p({col})",
                    domain_rationale="Log-transformed scaling to stabilize exponential variance in cellular area/perimeter.",
                )
                self.derived_registry[name_log] = d
                new_defs.append(d)

        logger.info("Engineered %d logarithmic features.", len(new_defs))
        return df_out, new_defs

    def engineer_features(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        apply_domain_features: bool = True,
        apply_ratios: bool = True,
        apply_polynomials: bool = False,
        apply_log_transforms: bool = False,
        apply_aggregations: bool = True,
        target_column: Optional[str] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """
        Execute configured feature engineering transformations and compile registry.
        """
        start_time = time.time()
        orig_count = df.shape[1]
        df_current = df.copy()
        all_defs: List[DerivedFeatureDefinition] = []

        if apply_domain_features:
            df_current, defs = self.create_cytology_domain_features(df_current)
            all_defs.extend(defs)

        if apply_ratios:
            df_current, defs = self.create_ratio_features(df_current)
            all_defs.extend(defs)

        if apply_aggregations:
            df_current, defs = self.create_statistical_aggregations(df_current)
            all_defs.extend(defs)

        if apply_log_transforms:
            df_current, defs = self.create_logarithmic_features(df_current)
            all_defs.extend(defs)

        if apply_polynomials:
            df_current, defs = self.create_polynomial_and_interaction_features(df_current, degree=2, interaction_only=True)
            all_defs.extend(defs)

        elapsed = round(time.time() - start_time, 3)

        result = FeatureEngineeringResult(
            dataset_name=dataset_name,
            original_feature_count=orig_count,
            derived_feature_count=len(all_defs),
            total_feature_count=df_current.shape[1],
            derived_definitions=all_defs,
            execution_time_seconds=elapsed,
        )

        logger.info(
            "FeatureEngineeringEngine complete | Orig: %d | Derived: %d | Total: %d | Elapsed: %.3fs",
            orig_count, len(all_defs), df_current.shape[1], elapsed,
        )
        return df_current, result

    def run_full_engineering_pipeline(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        include_domain: bool = True,
        include_interactions: bool = True,
        include_aggregations: bool = True,
        include_log: bool = True,
    ) -> Tuple[pd.DataFrame, FeatureEngineeringResult]:
        """Backward-compatible alias for full pipeline."""
        return self.engineer_features(
            df=df,
            dataset_name=dataset_name,
            apply_domain_features=include_domain,
            apply_ratios=include_domain,
            apply_polynomials=include_interactions,
            apply_log_transforms=include_log,
            apply_aggregations=include_aggregations,
        )
