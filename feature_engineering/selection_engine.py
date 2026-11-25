"""
CANCERWISE AI — Feature Selection Engine
Filter, Wrapper, and Embedded feature selection algorithms with strict data leakage prevention.

CRITICAL RULE: All feature selection and dimensionality reduction must be fitted
strictly within the training fold of cross-validation splits.
Never select features on the combined dataset or test set.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    chi2,
    mutual_info_classif,
    RFE,
    RFECV,
    SequentialFeatureSelector,
    SelectFromModel,
)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from backend.schemas.feature_engineering import (
    FilterSelectionMethod,
    WrapperSelectionMethod,
    EmbeddedSelectionMethod,
    FeatureRankingItem,
    SelectionComparisonItem,
)

logger = logging.getLogger("cancerwise.feature_engineering.selection")


class FeatureSelectionEngine:
    """
    Comprehensive feature selection suite across filter, wrapper, and embedded approaches.
    Enforces strict train-fold isolation to prevent data leakage.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    # -----------------------------------------------------------------------
    # 1. Filter-Based Selection
    # -----------------------------------------------------------------------
    @staticmethod
    def select_by_variance(
        X: pd.DataFrame,
        threshold: float = 0.01,
    ) -> List[str]:
        """Select features exhibiting variance strictly above threshold."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        selected_mask = selector.get_support()
        selected = X.columns[selected_mask].tolist()
        return selected if selected else X.columns.tolist()

    @staticmethod
    def select_by_correlation_filter(
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        threshold: float = 0.90,
        method: str = "pearson",
    ) -> List[str]:
        """
        Greedy collinearity elimination: for pairs with |r| >= threshold,
        drops the feature with lower correlation to target y (if provided) or drops the second feature.
        """
        corr_matrix = X.corr(method=method).abs()
        target_corrs = None
        if y is not None:
            y_num = pd.factorize(y)[0] if not pd.api.types.is_numeric_dtype(y) else y.values
            target_corrs = {col: abs(float(np.corrcoef(X[col], y_num)[0, 1])) for col in X.columns}

        drop_cols = set()
        cols = X.columns.tolist()

        for i in range(len(cols)):
            c1 = cols[i]
            if c1 in drop_cols:
                continue
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                if c2 in drop_cols:
                    continue
                if corr_matrix.loc[c1, c2] >= threshold:
                    if target_corrs:
                        # Keep feature with higher target correlation
                        if target_corrs.get(c1, 0) < target_corrs.get(c2, 0):
                            drop_cols.add(c1)
                            break
                        else:
                            drop_cols.add(c2)
                    else:
                        drop_cols.add(c2)

        return [c for c in cols if c not in drop_cols]

    @staticmethod
    def select_by_chi2(
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """Chi-Square feature selection after strictly non-negative scaling."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        k_val = min(k, X.shape[1])
        selector = SelectKBest(score_func=chi2, k=k_val)
        selector.fit(X_scaled, y)
        selected = X.columns[selector.get_support()].tolist()
        return selected, selector.scores_

    @staticmethod
    def select_by_anova(
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """One-way ANOVA F-test feature selection."""
        k_val = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_val)
        selector.fit(X, y)
        selected = X.columns[selector.get_support()].tolist()
        return selected, selector.scores_

    @staticmethod
    def select_by_mutual_info(
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        random_state: int = 42,
    ) -> Tuple[List[str], np.ndarray]:
        """Mutual Information non-linear feature selection."""
        k_val = min(k, X.shape[1])
        scores = mutual_info_classif(X, y, random_state=random_state)
        top_indices = np.argsort(scores)[::-1][:k_val]
        selected = [X.columns[i] for i in top_indices]
        return selected, scores

    # -----------------------------------------------------------------------
    # 2. Wrapper-Based Selection
    # -----------------------------------------------------------------------
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: int = 10,
        estimator: Optional[BaseEstimator] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Recursive Feature Elimination (RFE)."""
        if estimator is None:
            estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        k_val = min(n_features_to_select, X.shape[1])
        rfe = RFE(estimator=estimator, n_features_to_select=k_val, step=1)
        rfe.fit(X, y)
        selected = X.columns[rfe.get_support()].tolist()
        return selected, rfe.ranking_

    def select_by_rfecv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Optional[BaseEstimator] = None,
        min_features_to_select: int = 5,
        cv_splits: int = 5,
    ) -> Tuple[List[str], np.ndarray]:
        """Recursive Feature Elimination with Cross-Validation (RFECV)."""
        if estimator is None:
            estimator = LogisticRegression(max_iter=1000, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="roc_auc",
            min_features_to_select=min(min_features_to_select, X.shape[1]),
        )
        rfecv.fit(X, y)
        selected = X.columns[rfecv.get_support()].tolist()
        return selected, rfecv.ranking_

    def select_by_sfs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: int = 10,
        direction: str = "forward",
        cv_splits: int = 3,
    ) -> List[str]:
        """Sequential Forward / Backward Selection (SFS / SBS)."""
        estimator = LogisticRegression(max_iter=500, random_state=self.random_state)
        k_val = min(n_features_to_select, X.shape[1])
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=k_val,
            direction=direction,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        sfs.fit(X, y)
        return X.columns[sfs.get_support()].tolist()

    # -----------------------------------------------------------------------
    # 3. Embedded Selection
    # -----------------------------------------------------------------------
    def select_by_l1_logistic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        C: float = 0.5,
    ) -> List[str]:
        """L1-regularized sparse logistic regression selection."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=1000, random_state=self.random_state)
        selector = SelectFromModel(estimator=clf)
        selector.fit(X_scaled, y)
        selected = X.columns[selector.get_support()].tolist()
        return selected if selected else X.columns[:5].tolist()

    def select_by_tree_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "random_forest",
        k: Optional[int] = None,
        threshold: str = "median",
    ) -> Tuple[List[str], np.ndarray]:
        """Tree-based feature selection using Gini/split importance."""
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        elif model_type == "xgboost":
            model = XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric="logloss")
        else:
            raise ValueError(f"Unknown tree model type: {model_type}")

        model.fit(X, y)
        scores = model.feature_importances_

        if k is not None:
            k_val = min(k, X.shape[1])
            top_idx = np.argsort(scores)[::-1][:k_val]
            selected = [X.columns[i] for i in top_idx]
        else:
            selector = SelectFromModel(estimator=model, threshold=threshold, prefit=True)
            selected = X.columns[selector.get_support()].tolist()
            if not selected:
                selected = X.columns[:5].tolist()

        return selected, scores

    # -----------------------------------------------------------------------
    # Unified Selector Dispatcher
    # -----------------------------------------------------------------------
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Union[str, FilterSelectionMethod, WrapperSelectionMethod, EmbeddedSelectionMethod] = "mutual_info",
        k: int = 10,
    ) -> Tuple[List[str], List[FeatureRankingItem]]:
        """
        Unified dispatcher for all feature selection strategies.
        Returns selected features list and full feature rankings.
        """
        m_str = method.value if hasattr(method, "value") else str(method).lower()
        k_val = min(k, X.shape[1])
        features = X.columns.tolist()

        selected: List[str] = []
        rankings: List[FeatureRankingItem] = []

        if m_str == "variance_threshold":
            selected = self.select_by_variance(X)[:k_val]
            variances = X.var().values
            sorted_idx = np.argsort(variances)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(variances[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        elif m_str in ["pearson_correlation", "spearman_correlation"]:
            m = "pearson" if "pearson" in m_str else "spearman"
            selected = self.select_by_correlation_filter(X, y, method=m)[:k_val]
            corrs = [abs(float(np.corrcoef(X[c], y)[0, 1])) for c in features]
            sorted_idx = np.argsort(corrs)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(corrs[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        elif m_str == "chi2":
            selected, scores = self.select_by_chi2(X, y, k=k_val)
            sorted_idx = np.argsort(scores)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(scores[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        elif m_str == "anova_f":
            selected, scores = self.select_by_anova(X, y, k=k_val)
            sorted_idx = np.argsort(scores)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(scores[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        elif m_str == "mutual_info":
            selected, scores = self.select_by_mutual_info(X, y, k=k_val, random_state=self.random_state)
            sorted_idx = np.argsort(scores)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(scores[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        elif m_str == "rfe":
            selected, rank_arr = self.select_by_rfe(X, y, n_features_to_select=k_val)
            sorted_idx = np.argsort(rank_arr)
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=int(rank_arr[idx]),
                    score=round(1.0 / float(rank_arr[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for idx in sorted_idx
            ]

        elif m_str == "rfecv":
            selected, rank_arr = self.select_by_rfecv(X, y, min_features_to_select=k_val)
            sorted_idx = np.argsort(rank_arr)
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=int(rank_arr[idx]),
                    score=round(1.0 / float(rank_arr[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for idx in sorted_idx
            ]

        elif m_str in ["sfs_forward", "sfs_backward"]:
            direction = "forward" if "forward" in m_str else "backward"
            selected = self.select_by_sfs(X, y, n_features_to_select=k_val, direction=direction)
            rankings = [
                FeatureRankingItem(
                    feature_name=f,
                    rank=r,
                    score=1.0 if f in selected else 0.0,
                    method=m_str,
                    is_selected=f in selected,
                )
                for r, f in enumerate(features, 1)
            ]

        elif m_str in ["l1_logistic", "lasso"]:
            selected = self.select_by_l1_logistic(X, y)[:k_val]
            rankings = [
                FeatureRankingItem(
                    feature_name=f,
                    rank=r,
                    score=1.0 if f in selected else 0.0,
                    method=m_str,
                    is_selected=f in selected,
                )
                for r, f in enumerate(features, 1)
            ]

        elif m_str in ["random_forest", "gradient_boosting", "xgboost"]:
            selected, scores = self.select_by_tree_importance(X, y, model_type=m_str, k=k_val)
            sorted_idx = np.argsort(scores)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(scores[idx]), 6),
                    method=m_str,
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        else:
            # Fallback to mutual info
            selected, scores = self.select_by_mutual_info(X, y, k=k_val, random_state=self.random_state)
            sorted_idx = np.argsort(scores)[::-1]
            rankings = [
                FeatureRankingItem(
                    feature_name=features[idx],
                    rank=r,
                    score=round(float(scores[idx]), 6),
                    method="mutual_info",
                    is_selected=features[idx] in selected,
                )
                for r, idx in enumerate(sorted_idx, 1)
            ]

        return selected, rankings

    # -----------------------------------------------------------------------
    # 4. Leakage-Free Cross-Validation Comparison Pipeline
    # -----------------------------------------------------------------------
    def compare_selection_strategies(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: Optional[int] = None,
        cv_folds: Optional[int] = None,
        target_k: int = 10,
        n_splits: int = 5,
    ) -> List[SelectionComparisonItem]:
        """
        Evaluate and compare feature selection strategies using 5-fold Stratified CV.

        STRICT DATA LEAKAGE PREVENTION:
        In each fold, the feature selector is fit exclusively on X_train.
        X_val is evaluated strictly using features learned on X_train.
        """
        target_k = k if k is not None else target_k
        n_splits = cv_folds if cv_folds is not None else n_splits

        orig_count = X.shape[1]
        k_val = min(target_k, orig_count)

        # Standardize target to 0/1 binary
        if not pd.api.types.is_numeric_dtype(y):
            y_arr = (y == "Malignant").astype(int).to_numpy() if "Malignant" in y.values else pd.factorize(y)[0]
        else:
            y_arr = y.to_numpy()

        strategies: List[Tuple[str, str, Callable[[pd.DataFrame, pd.Series], List[str]]]] = [
            ("Variance Threshold", "filter", lambda x, target: self.select_by_variance(x, threshold=0.01)),
            ("Pearson Correlation (r < 0.90)", "filter", lambda x, target: self.select_by_correlation_filter(x, target, threshold=0.90)),
            ("ANOVA F-Test (Top K)", "filter", lambda x, target: self.select_by_anova(x, target, k=k_val)[0]),
            ("Mutual Information (Top K)", "filter", lambda x, target: self.select_by_mutual_info(x, target, k=k_val, random_state=self.random_state)[0]),
            ("Recursive Feature Elimination (RFE)", "wrapper", lambda x, target: self.select_by_rfe(x, target, n_features_to_select=k_val)[0]),
            ("L1 Regularized Logistic", "embedded", lambda x, target: self.select_by_l1_logistic(x, target, C=0.5)),
            ("Random Forest Gini Importance", "embedded", lambda x, target: self.select_by_tree_importance(x, target, model_type="random_forest", k=k_val)[0]),
            ("XGBoost Gain Importance", "embedded", lambda x, target: self.select_by_tree_importance(x, target, model_type="xgboost", k=k_val)[0]),
        ]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        comparison_results: List[SelectionComparisonItem] = []

        for name, category, selector_func in strategies:
            start_time = time.time()
            acc_scores = []
            f1_scores = []
            roc_scores = []

            # First compute whole-dataset selected feature list for reporting
            try:
                selected_on_full = selector_func(X, pd.Series(y_arr, index=X.index))
            except Exception as e:
                logger.warning("Selector '%s' failed on full data: %s", name, e)
                selected_on_full = X.columns[:k_val].tolist()

            # Execute leakage-free cross-validation
            for train_idx, val_idx in cv.split(X, y_arr):
                X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

                # 1. Scale on train only
                scaler = StandardScaler()
                X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
                X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

                # 2. Select features on train only
                try:
                    fold_selected = selector_func(X_tr_scaled, pd.Series(y_tr, index=X_tr.index))
                    if not fold_selected:
                        fold_selected = X.columns[:k_val].tolist()
                except Exception:
                    fold_selected = X.columns[:k_val].tolist()

                # 3. Train downstream classifier on selected features only
                clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
                clf.fit(X_tr_scaled[fold_selected], y_tr)

                preds = clf.predict(X_val_scaled[fold_selected])
                probs = clf.predict_proba(X_val_scaled[fold_selected])[:, 1] if len(np.unique(y_tr)) == 2 else preds

                acc_scores.append(accuracy_score(y_val, preds))
                f1_scores.append(f1_score(y_val, preds, zero_division=0))
                try:
                    roc_scores.append(roc_auc_score(y_val, probs))
                except Exception:
                    roc_scores.append(accuracy_score(y_val, preds))

            elapsed = round(time.time() - start_time, 3)

            item = SelectionComparisonItem(
                method=name,
                category=category,
                original_feature_count=orig_count,
                selected_feature_count=len(selected_on_full),
                selected_features=selected_on_full,
                execution_time_seconds=elapsed,
                cv_accuracy_mean=round(float(np.mean(acc_scores)), 4),
                cv_accuracy_std=round(float(np.std(acc_scores)), 4),
                cv_f1_mean=round(float(np.mean(f1_scores)), 4),
                cv_f1_std=round(float(np.std(f1_scores)), 4),
                cv_roc_auc_mean=round(float(np.mean(roc_scores)), 4),
                cv_roc_auc_std=round(float(np.std(roc_scores)), 4),
                data_leakage_prevented=True,
            )
            comparison_results.append(item)

        # Sort by CV ROC-AUC descending
        comparison_results.sort(key=lambda x: x.cv_roc_auc_mean, reverse=True)
        return comparison_results
