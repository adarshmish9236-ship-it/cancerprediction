"""
CANCERWISE AI — Feature Ranking & Explainability Service
Implements Permutation Importance, SHAP-based feature importance, model-specific
native importance, and multi-method Consensus Ranking (Borda Count).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from backend.schemas.feature_engineering import (
    FeatureRankingItem,
    FeatureImportanceSummary,
)

logger = logging.getLogger("cancerwise.ranking")


class FeatureRankingService:
    """
    Computes explainable feature importance rankings using permutation, SHAP,
    model-native metrics, and consensus Borda counts across biological markers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def compute_permutation_importance(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        scoring: str = "roc_auc",
    ) -> List[FeatureRankingItem]:
        """
        Computes permutation importance on validation dataset.
        Permutes feature values and assesses decline in model scoring.
        """
        try:
            r = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=-1,
            )
            importances_mean = r.importances_mean
        except Exception as e:
            logger.warning(f"Permutation importance with scoring '{scoring}' failed ({e}), falling back to accuracy.")
            r = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                scoring="accuracy",
            )
            importances_mean = r.importances_mean

        features = X.columns.tolist()
        sorted_indices = np.argsort(importances_mean)[::-1]

        rankings: List[FeatureRankingItem] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            rankings.append(
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=rank,
                    score=round(float(importances_mean[idx]), 6),
                    method="permutation_importance",
                    is_selected=bool(importances_mean[idx] > 0),
                )
            )
        return rankings

    def compute_shap_importance(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        max_samples: int = 200,
    ) -> List[FeatureRankingItem]:
        """
        Calculates mean absolute SHAP values using shap.TreeExplainer or shap.Explainer.
        """
        features = X.columns.tolist()
        X_sample = X.sample(min(len(X), max_samples), random_state=self.random_state) if len(X) > max_samples else X

        try:
            import shap
            if hasattr(model, "tree_") or hasattr(model, "estimators_") or "xgb" in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample).values

            # Normalize shap_values dimensionality
            if isinstance(shap_values, list):
                # Binary classification list of arrays [class0, class1]
                vals = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0])
            elif hasattr(shap_values, "values"):
                vals = np.abs(shap_values.values)
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    vals = np.abs(shap_values[:, :, 1])
                else:
                    vals = np.abs(shap_values)
            else:
                vals = np.abs(np.array(shap_values))

            mean_abs_shap = np.mean(vals, axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = np.mean(mean_abs_shap, axis=-1)

            sorted_indices = np.argsort(mean_abs_shap)[::-1]
            rankings: List[FeatureRankingItem] = []
            for rank, idx in enumerate(sorted_indices, start=1):
                rankings.append(
                    FeatureRankingItem(
                        feature_name=features[idx],
                        rank=rank,
                        score=round(float(mean_abs_shap[idx]), 6),
                        method="shap_importance",
                        is_selected=bool(mean_abs_shap[idx] > 0),
                    )
                )
            return rankings

        except Exception as e:
            logger.warning(f"SHAP importance calculation fallback due to: {e}")
            # Fallback to model feature_importances_ or coefficients
            return self.compute_native_importance(model, features)

    def compute_native_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str],
    ) -> List[FeatureRankingItem]:
        """
        Extracts model-specific feature importances or coefficient magnitudes.
        """
        if hasattr(model, "feature_importances_"):
            scores = model.feature_importances_
            method = "native_tree_importance"
        elif hasattr(model, "coef_"):
            scores = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
            method = "native_linear_coefficient"
        else:
            scores = np.ones(len(feature_names))
            method = "uniform_fallback"

        sorted_indices = np.argsort(scores)[::-1]
        rankings: List[FeatureRankingItem] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            rankings.append(
                FeatureRankingItem(
                    feature_name=feature_names[idx],
                    rank=rank,
                    score=round(float(scores[idx]), 6),
                    method=method,
                    is_selected=bool(scores[idx] > 0),
                )
            )
        return rankings

    def compute_consensus_ranking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> FeatureImportanceSummary:
        """
        Computes consensus rankings using Borda Count across 4 complementary methodologies:
        1. Pearson correlation magnitude (linear bivariate filter)
        2. Mutual Information (non-linear filter)
        3. Random Forest Gini Importance (non-linear embedded)
        4. Permutation Importance (post-hoc model inspection)
        """
        numeric_df = X.select_dtypes(include=[np.number]).dropna(axis=1)
        features = numeric_df.columns.tolist()
        n_features = len(features)
        y_arr = np.asarray(y)

        # Ensure y_numeric is a numeric array for correlation computation
        if y_arr.dtype.kind not in 'iufb':
            unique_vals = np.unique(y_arr)
            val_map = {val: idx for idx, val in enumerate(unique_vals)}
            y_numeric = np.array([val_map[v] for v in y_arr], dtype=float)
        else:
            y_numeric = y_arr.astype(float)

        # 1. Pearson Correlation
        pearson_scores = []
        for col in features:
            corr = np.corrcoef(numeric_df[col], y_numeric)[0, 1]
            pearson_scores.append(abs(corr) if not np.isnan(corr) else 0.0)
        p_ranks = {features[idx]: rank for rank, idx in enumerate(np.argsort(pearson_scores)[::-1], start=1)}

        # 2. Mutual Information
        mi = mutual_info_classif(numeric_df, y_numeric.astype(int), random_state=self.random_state)
        mi_ranks = {features[idx]: rank for rank, idx in enumerate(np.argsort(mi)[::-1], start=1)}

        # 3. Random Forest Gini Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(numeric_df, y_arr)
        rf_scores = rf.feature_importances_
        rf_ranks = {features[idx]: rank for rank, idx in enumerate(np.argsort(rf_scores)[::-1], start=1)}

        # 4. Permutation Importance
        perm_items = self.compute_permutation_importance(rf, numeric_df, y_arr)
        perm_ranks = {item.feature_name: item.rank for item in perm_items}

        # Borda Count Aggregation
        # Each method assigns points: (N - rank + 1)
        consensus_items = []
        for col in features:
            r_p = p_ranks.get(col, n_features)
            r_mi = mi_ranks.get(col, n_features)
            r_rf = rf_ranks.get(col, n_features)
            r_perm = perm_ranks.get(col, n_features)

            borda_score = (
                (n_features - r_p + 1)
                + (n_features - r_mi + 1)
                + (n_features - r_rf + 1)
                + (n_features - r_perm + 1)
            )
            mean_rank = round((r_p + r_mi + r_rf + r_perm) / 4.0, 2)
            rank_variance = round(float(np.var([r_p, r_mi, r_rf, r_perm])), 2)

            consensus_items.append({
                "feature_name": col,
                "borda_score": borda_score,
                "mean_rank": mean_rank,
                "rank_variance": rank_variance,
                "pearson_rank": r_p,
                "mutual_info_rank": r_mi,
                "random_forest_rank": r_rf,
                "permutation_rank": r_perm,
            })

        consensus_items.sort(key=lambda x: x["borda_score"], reverse=True)
        for consensus_rank, item in enumerate(consensus_items, start=1):
            item["consensus_rank"] = consensus_rank

        ranking_items = [
            FeatureRankingItem(
                feature_name=item["feature_name"],
                rank=item["consensus_rank"],
                score=float(item["borda_score"]),
                method="consensus_borda",
                is_selected=True,
            )
            for item in consensus_items
        ]

        return FeatureImportanceSummary(
            model_type="Ensemble_Consensus",
            importance_type="Borda_Count_Consensus",
            rankings=ranking_items,
            consensus_rankings=consensus_items,
        )
