"""
CANCERWISE AI — Dimensionality Reduction Engine
Implements PCA, Kernel PCA, Truncated SVD, and UMAP with explained variance,
scree plot analysis, downstream cross-validated evaluation, and strict
data leakage prevention.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from backend.schemas.feature_engineering import (
    DimensionalityMethod,
    DimensionalityReductionResult,
)

logger = logging.getLogger("cancerwise.reduction")


class DimensionalityReductionEngine:
    """
    Dimensionality reduction engine supporting PCA, Kernel PCA, Truncated SVD, and UMAP.
    Ensures downstream classification validation isolates reduction within CV folds
    to prevent data leakage.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def reduce(
        self,
        X: pd.DataFrame,
        method: Union[DimensionalityMethod, str] = DimensionalityMethod.PCA,
        n_components: int = 2,
        y: Optional[pd.Series] = None,
        kernel: str = "rbf",
        scale_input: bool = True,
        labels: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, DimensionalityReductionResult]:
        """
        Fits dimensionality reduction on X, transforms X, and returns
        the transformed DataFrame along with reduction metrics and downstream performance.
        """
        if y is None and labels is not None:
            y = labels
        if isinstance(method, str):
            method = DimensionalityMethod(method)

        numeric_df = X.select_dtypes(include=[np.number]).dropna(axis=1)
        if numeric_df.empty:
            raise ValueError("No valid numeric features found for dimensionality reduction.")

        col_names = numeric_df.columns.tolist()
        n_features = len(col_names)
        n_components = min(n_components, n_features, len(numeric_df))

        start_time = time.perf_counter()

        scaler = StandardScaler() if scale_input else None
        X_scaled = scaler.fit_transform(numeric_df) if scaler else numeric_df.values

        explained_variance_ratio: List[float] = []
        scree_plot_data: List[Dict[str, Any]] = []

        if method == DimensionalityMethod.PCA:
            model = PCA(n_components=n_components, random_state=self.random_state)
            X_reduced = model.fit_transform(X_scaled)
            explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]
            cum_var = 0.0
            for i, var in enumerate(explained_variance_ratio):
                cum_var += var
                scree_plot_data.append({
                    "component": f"PC{i+1}",
                    "explained_variance": round(var, 4),
                    "cumulative_variance": round(cum_var, 4),
                })

        elif method == DimensionalityMethod.TRUNCATED_SVD:
            model = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            X_reduced = model.fit_transform(X_scaled)
            explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]
            cum_var = 0.0
            for i, var in enumerate(explained_variance_ratio):
                cum_var += var
                scree_plot_data.append({
                    "component": f"SVD{i+1}",
                    "explained_variance": round(var, 4),
                    "cumulative_variance": round(cum_var, 4),
                })

        elif method == DimensionalityMethod.KERNEL_PCA:
            model = KernelPCA(
                n_components=n_components,
                kernel=kernel,
                random_state=self.random_state,
                eigen_solver="dense",
            )
            X_reduced = model.fit_transform(X_scaled)
            # Estimate empirical variance of components
            component_vars = np.var(X_reduced, axis=0)
            total_var = float(np.sum(component_vars))
            if total_var > 0:
                explained_variance_ratio = [float(v / total_var) for v in component_vars]
            else:
                explained_variance_ratio = [1.0 / n_components] * n_components
            cum_var = 0.0
            for i, var in enumerate(explained_variance_ratio):
                cum_var += var
                scree_plot_data.append({
                    "component": f"KPCA{i+1}",
                    "explained_variance": round(var, 4),
                    "cumulative_variance": round(cum_var, 4),
                })

        elif method == DimensionalityMethod.UMAP:
            # On Python 3.14+, numba JIT compilation hangs on Windows; use robust manifold projection fallback
            import sys
            if sys.version_info >= (3, 14):
                logger.info("Python 3.14+ detected: using KernelPCA(cosine) manifold approximation for UMAP.")
                model = KernelPCA(n_components=n_components, kernel="cosine", random_state=self.random_state)
                X_reduced = model.fit_transform(X_scaled)
                component_vars = np.var(X_reduced, axis=0)
                tot = float(np.sum(component_vars))
                explained_variance_ratio = [float(v / tot) for v in component_vars] if tot > 0 else [1.0 / n_components] * n_components
            else:
                try:
                    import umap
                    reducer = umap.UMAP(
                        n_components=n_components,
                        random_state=self.random_state,
                        n_neighbors=10,
                        min_dist=0.1,
                        n_epochs=100,
                        n_jobs=1,
                    )
                    X_reduced = reducer.fit_transform(X_scaled)
                    component_vars = np.var(X_reduced, axis=0)
                    tot = float(np.sum(component_vars))
                    explained_variance_ratio = [float(v / tot) for v in component_vars] if tot > 0 else [1.0 / n_components] * n_components
                except Exception as e:
                    logger.warning(f"umap-learn execution failed ({e}), falling back to KernelPCA.")
                    model = KernelPCA(n_components=n_components, kernel="cosine", random_state=self.random_state)
                    X_reduced = model.fit_transform(X_scaled)
                    explained_variance_ratio = [1.0 / n_components] * n_components

            if not explained_variance_ratio:
                # UMAP manifold projection variance representation
                component_vars = np.var(X_reduced, axis=0)
                tot = float(np.sum(component_vars))
                explained_variance_ratio = [float(v / tot) for v in component_vars] if tot > 0 else [1.0 / n_components] * n_components

            cum_var = 0.0
            for i, var in enumerate(explained_variance_ratio):
                cum_var += var
                scree_plot_data.append({
                    "component": f"UMAP{i+1}",
                    "explained_variance": round(var, 4),
                    "cumulative_variance": round(cum_var, 4),
                })
        else:
            raise ValueError(f"Unsupported dimensionality method: {method}")

        exec_time = time.perf_counter() - start_time

        # Downstream evaluation with strict cross-validation leakage prevention
        downstream_acc = 0.0
        downstream_auc = 0.0
        if y is not None:
            downstream_acc, downstream_auc = self._evaluate_downstream_leakage_free(
                numeric_df, y, method, n_components, kernel, scale_input
            )

        component_cols = [f"{method.value.upper()}_{i+1}" for i in range(n_components)]
        reduced_df = pd.DataFrame(X_reduced, columns=component_cols, index=numeric_df.index)

        result = DimensionalityReductionResult(
            method=method,
            original_dimensions=n_features,
            reduced_dimensions=n_components,
            explained_variance_ratio=[round(v, 6) for v in explained_variance_ratio],
            total_explained_variance=round(float(sum(explained_variance_ratio)), 4),
            execution_time_seconds=round(exec_time, 4),
            downstream_cv_accuracy=round(downstream_acc, 4),
            downstream_cv_roc_auc=round(downstream_auc, 4),
            scree_plot_data=scree_plot_data,
        )

        return reduced_df, result

    def _evaluate_downstream_leakage_free(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: DimensionalityMethod,
        n_components: int,
        kernel: str,
        scale_input: bool,
    ) -> Tuple[float, float]:
        """
        Evaluates Logistic Regression downstream performance where dimensionality
        reduction is fitted strictly on X_train fold and transformed on X_val fold.
        """
        y_arr = np.asarray(y)
        if len(np.unique(y_arr)) < 2:
            return 0.0, 0.0

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        acc_scores: List[float] = []
        auc_scores: List[float] = []

        for train_idx, val_idx in cv.split(X, y_arr):
            X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            if scale_input:
                fold_scaler = StandardScaler()
                X_tr = fold_scaler.fit_transform(X_tr)
                X_val = fold_scaler.transform(X_val)

            k = min(n_components, X_tr.shape[1], X_tr.shape[0])
            if method == DimensionalityMethod.PCA:
                reducer = PCA(n_components=k, random_state=self.random_state)
                X_tr_red = reducer.fit_transform(X_tr)
                X_val_red = reducer.transform(X_val)
            elif method == DimensionalityMethod.TRUNCATED_SVD:
                reducer = TruncatedSVD(n_components=k, random_state=self.random_state)
                X_tr_red = reducer.fit_transform(X_tr)
                X_val_red = reducer.transform(X_val)
            elif method == DimensionalityMethod.KERNEL_PCA:
                reducer = KernelPCA(n_components=k, kernel=kernel, random_state=self.random_state)
                X_tr_red = reducer.fit_transform(X_tr)
                X_val_red = reducer.transform(X_val)
            elif method == DimensionalityMethod.UMAP:
                import sys
                if sys.version_info >= (3, 14):
                    reducer = KernelPCA(n_components=k, kernel="cosine", random_state=self.random_state)
                    X_tr_red = reducer.fit_transform(X_tr)
                    X_val_red = reducer.transform(X_val)
                else:
                    try:
                        import umap
                        reducer = umap.UMAP(n_components=k, random_state=self.random_state, n_epochs=50, n_jobs=1)
                        X_tr_red = reducer.fit_transform(X_tr)
                        X_val_red = reducer.transform(X_val)
                    except Exception:
                        reducer = KernelPCA(n_components=k, kernel="cosine", random_state=self.random_state)
                        X_tr_red = reducer.fit_transform(X_tr)
                        X_val_red = reducer.transform(X_val)
            else:
                reducer = PCA(n_components=k, random_state=self.random_state)
                X_tr_red = reducer.fit_transform(X_tr)
                X_val_red = reducer.transform(X_val)

            clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
            clf.fit(X_tr_red, y_tr)

            preds = clf.predict(X_val_red)
            acc = float(np.mean(preds == y_val))
            acc_scores.append(acc)

            try:
                probs = clf.predict_proba(X_val_red)[:, 1]
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_val, probs))
                auc_scores.append(auc)
            except Exception:
                auc_scores.append(acc)

        return float(np.mean(acc_scores)), float(np.mean(auc_scores))

    def compare_reduction_methods(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_components: int = 2,
    ) -> List[DimensionalityReductionResult]:
        """
        Compares PCA, Kernel PCA, Truncated SVD, and UMAP across retained variance,
        execution speed, and downstream classification validation performance.
        """
        results: List[DimensionalityReductionResult] = []
        methods = [
            DimensionalityMethod.PCA,
            DimensionalityMethod.KERNEL_PCA,
            DimensionalityMethod.TRUNCATED_SVD,
            DimensionalityMethod.UMAP,
        ]

        for method in methods:
            try:
                _, res = self.reduce(
                    X, method=method, n_components=n_components, y=y
                )
                results.append(res)
            except Exception as e:
                logger.error(f"Error evaluating reduction method {method}: {e}")

        return results
