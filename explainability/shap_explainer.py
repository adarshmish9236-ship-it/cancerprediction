"""
CANCERWISE AI — SHAP Explainer Engine
Computes global Shapley feature attributions, individual local prediction explanations,
dependence relationships, and partial dependence / ICE curves.
Selects the appropriate SHAP explainer (TreeExplainer, LinearExplainer, Explainer) per architecture.

GOVERNANCE DIRECTIVE:
SHAP values represent additive mathematical contributions to model predictions.
They MUST NOT be interpreted as causal oncogenic mechanisms or clinical proof.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence

from backend.schemas.explainability import (
    SHAPFeatureContribution,
    LocalSHAPExplanation,
    GlobalSHAPImportanceItem,
    SHAPExplanationSummary,
    PartialDependenceResult,
)

logger = logging.getLogger("cancerwise.explainability.shap")


class SHAPExplainer:
    """
    Unified SHAP interpretability service supporting tree, linear, and neural estimators.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _select_explainer(
        self,
        model: Any,
        X_sample: np.ndarray,
    ) -> shap.Explainer:
        """Dynamically selects optimal SHAP explainer based on model class."""
        model_str = str(type(model)).lower()
        if hasattr(model, "tree_") or hasattr(model, "estimators_") or "xgb" in model_str or "lgbm" in model_str or "catboost" in model_str:
            return shap.TreeExplainer(model)
        elif hasattr(model, "coef_") and not hasattr(model, "kernel"):
            # Linear models
            return shap.LinearExplainer(model, X_sample)
        else:
            # Fallback to general Explainer or KernelExplainer
            try:
                return shap.Explainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, X_sample)
            except Exception:
                return shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, X_sample[:30])

    def explain_global(
        self,
        model: Any,
        X: Optional[pd.DataFrame] = None,
        model_id: str = "model",
        max_background_samples: int = 150,
        max_samples: Optional[int] = None,
        X_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> SHAPExplanationSummary:
        """
        Computes global Shapley feature importance (mean absolute SHAP) across samples.
        """
        df = X if X is not None else X_data
        if df is None:
            raise ValueError("DataFrame must be provided via 'X' or 'X_data'.")
        features = df.columns.tolist()
        if max_samples is not None:
            max_background_samples = max_samples
        n_samples = min(len(df), max_background_samples)
        X_sub = df.sample(n_samples, random_state=self.random_state) if len(df) > n_samples else df

        explainer = self._select_explainer(model, X_sub.values)
        explainer_type = type(explainer).__name__

        shap_values = explainer.shap_values(X_sub.values) if hasattr(explainer, "shap_values") else explainer(X_sub.values).values

        # Normalize dimensional structure
        if isinstance(shap_values, list):
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                vals = shap_values[:, :, 1]
            else:
                vals = shap_values
        else:
            vals = np.array(shap_values)

        mean_abs = np.mean(np.abs(vals), axis=0)
        sorted_indices = np.argsort(mean_abs)[::-1]

        global_items: List[GlobalSHAPImportanceItem] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            feat_vals = X_sub[features[idx]].values
            corr = np.corrcoef(feat_vals, vals[:, idx])[0, 1] if np.std(vals[:, idx]) > 0 else 0.0
            global_items.append(
                GlobalSHAPImportanceItem(
                    feature_name=features[idx],
                    rank=rank,
                    mean_abs_shap=round(float(mean_abs[idx]), 6),
                    correlation_with_risk=round(float(corr) if not np.isnan(corr) else 0.0, 4),
                )
            )

        return SHAPExplanationSummary(
            model_id=model_id,
            explainer_type=explainer_type,
            num_samples_analyzed=len(X_sub),
            global_importances=global_items,
        )

    def explain_local_sample(
        self,
        model: Any,
        X_sample: pd.Series,
        X_background: pd.DataFrame,
        sample_index: int = 0,
    ) -> LocalSHAPExplanation:
        """
        Decomposes an individual prediction into additive feature Shapley contributions.
        """
        features = X_background.columns.tolist()
        bg_sub = X_background.sample(min(len(X_background), 100), random_state=self.random_state)
        explainer = self._select_explainer(model, bg_sub.values)

        x_vec = X_sample.values.reshape(1, -1)
        shap_vals = explainer.shap_values(x_vec) if hasattr(explainer, "shap_values") else explainer(x_vec).values

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif hasattr(shap_vals, "values"):
            sv = shap_vals.values[0, :, 1] if shap_vals.values.ndim == 3 else shap_vals.values[0]
        elif isinstance(shap_vals, np.ndarray):
            sv = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]
        else:
            sv = np.array(shap_vals)[0]

        # Base value (expected value)
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            base_val = float(ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else ev)
        else:
            base_val = 0.5

        x_df = pd.DataFrame([X_sample.values], columns=features)
        # Prediction probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_df)[0]
            prob_mal = float(probs[1]) if len(probs) > 1 else float(probs[0])
            pred_class = int(prob_mal >= 0.5)
        else:
            preds = model.predict(x_df)[0]
            pred_class = int(preds)
            prob_mal = 1.0 if pred_class == 1 else 0.0

        label = "Malignant" if pred_class == 1 else "Benign"

        contributions: List[SHAPFeatureContribution] = []
        for i, feat in enumerate(features):
            val = float(sv[i])
            direction = "Increases Risk (Malignant)" if val > 0 else "Decreases Risk (Benign)"
            contributions.append(
                SHAPFeatureContribution(
                    feature_name=feat,
                    feature_value=round(float(X_sample[feat]), 4),
                    shap_value=round(val, 6),
                    contribution_direction=direction,
                )
            )

        # Sort contributions by absolute impact descending
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return LocalSHAPExplanation(
            sample_index=sample_index,
            base_value=round(base_val, 4),
            predicted_probability=round(prob_mal, 4),
            predicted_class=pred_class,
            predicted_label=label,
            contributions=contributions,
        )

    def compute_partial_dependence_curves(
        self,
        model: Any,
        X: Optional[pd.DataFrame] = None,
        feature_name: str = "mean radius",
        grid_resolution: int = 30,
        X_data: Optional[pd.DataFrame] = None,
    ) -> PartialDependenceResult:
        """
        Calculates Partial Dependence (PDP) and Individual Conditional Expectation (ICE).
        """
        df = X if X is not None else X_data
        if df is None:
            raise ValueError("DataFrame must be provided via 'X' or 'X_data'.")
        if feature_name not in df.columns:
            raise ValueError(f"Feature '{feature_name}' not in DataFrame columns.")

        X_sub = df.sample(min(len(df), 100), random_state=self.random_state)
        pdp_res = partial_dependence(
            model,
            X_sub,
            features=[feature_name],
            grid_resolution=grid_resolution,
            kind="both",
        )

        grid_values = [float(v) for v in pdp_res["grid_values"][0]]
        avg_raw = pdp_res["average"]
        if hasattr(avg_raw, "ndim") and avg_raw.ndim == 2:
            avg_pdp = [float(v) for v in avg_raw[0]]
        elif hasattr(avg_raw, "ndim") and avg_raw.ndim == 1:
            avg_pdp = [float(v) for v in avg_raw]
        else:
            avg_pdp = [float(v) for v in avg_raw[0]]

        ind_raw = pdp_res.get("individual", None)
        ice_curves: List[List[float]] = []
        if ind_raw is not None:
            if hasattr(ind_raw, "ndim") and ind_raw.ndim == 3:
                ice_curves = [[float(v) for v in curve] for curve in ind_raw[0][:20]]
            elif hasattr(ind_raw, "ndim") and ind_raw.ndim == 2:
                ice_curves = [[float(v) for v in curve] for curve in ind_raw[:20]]
            else:
                ice_curves = [[float(v) for v in curve] for curve in ind_raw[0][:20]]

        return PartialDependenceResult(
            feature_name=feature_name,
            grid_values=grid_values,
            average_partial_dependence=avg_pdp,
            ice_curves=ice_curves,
        )

    # Aliases for compatibility
    compute_partial_dependence = compute_partial_dependence_curves

    def explain_instance(
        self,
        model: Any,
        instance: Union[pd.DataFrame, pd.Series],
        X_background: Optional[pd.DataFrame] = None,
        sample_index: int = 0,
        top_k: Optional[int] = None,
        model_id: str = "model",
        **kwargs,
    ) -> LocalSHAPExplanation:
        """
        Alias for explain_local_sample with top_k truncation and DataFrame/model_id compatibility.
        """
        if isinstance(instance, pd.DataFrame):
            if X_background is None:
                X_background = instance
            sample_series = instance.iloc[sample_index]
        else:
            sample_series = instance
            if X_background is None:
                X_background = pd.DataFrame([sample_series.values], columns=sample_series.index)

        res = self.explain_local_sample(
            model=model,
            X_sample=sample_series,
            X_background=X_background,
            sample_index=sample_index,
        )
        if top_k and top_k < len(res.contributions):
            res.contributions = res.contributions[:top_k]
        return res
