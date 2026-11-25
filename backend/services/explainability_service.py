"""
CANCERWISE AI — Explainability Service
Coordinates SHAP global/local feature importance, LIME surrogate rules,
and Partial Dependence / Individual Conditional Expectation curves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from backend.core.exceptions import DatasetNotFoundError, ModelNotFoundError
from backend.core.logging_setup import get_logger
from backend.schemas.explainability import (
    SHAPExplanationSummary,
    LocalSHAPExplanation,
    LIMEExplanationResult,
    PartialDependenceResult,
)
from backend.services.dataset_management_service import DatasetManagementService
from data_mining.classification.serializer import ModelSerializer
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer

logger = get_logger("services.explainability")


class ExplainabilityService:
    """
    Dedicated service for Explainable AI (XAI) calculations and model interpretation.
    """

    def __init__(
        self,
        dataset_mgmt: Optional[DatasetManagementService] = None,
        serializer: Optional[ModelSerializer] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
        lime_explainer: Optional[LIMEExplainer] = None,
    ):
        self.dataset_mgmt = dataset_mgmt or DatasetManagementService()
        self.serializer = serializer or ModelSerializer()
        self.shap_explainer = shap_explainer or SHAPExplainer()
        self.lime_explainer = lime_explainer or LIMEExplainer()
        self._model_cache: Dict[str, Any] = {}

    def _get_dataset(self, dataset_id: str = "uci_wdbc") -> Tuple[pd.DataFrame, pd.Series]:
        """Loads dataset and prepares features X and target y."""
        try:
            df = self.dataset_mgmt.get_dataframe(dataset_id)
        except DatasetNotFoundError:
            # Fallback to loading any available or ingesting
            try:
                datasets = self.dataset_mgmt.list_datasets()
                if datasets:
                    df = self.dataset_mgmt.get_dataframe(datasets[0].dataset_id)
                else:
                    raise DatasetNotFoundError(dataset_id)
            except Exception:
                raise DatasetNotFoundError(dataset_id)

        target_col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
        y_raw = df[target_col]
        mapping = {"M": 1, "B": 0, "malignant": 1, "benign": 0, "1": 1, "0": 0, 1: 1, 0: 0}
        y = pd.Series(
            [mapping.get(str(v).strip().lower(), mapping.get(v, 0)) for v in y_raw],
            name=target_col,
            index=df.index,
            dtype=int,
        )
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col and c != "id"]
        X = df[feature_cols].fillna(df[feature_cols].median())
        return X, y

    def get_or_load_model_and_data(
        self, model_id: str = "baseline_rf"
    ) -> Tuple[Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Retrieves model pipeline from registry or creates baseline model on WDBC."""
        X, y = self._get_dataset("uci_wdbc")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        if model_id in self._model_cache:
            return self._model_cache[model_id], X_train, y_train, X_test, y_test

        try:
            bundle = self.serializer.load_model(model_id)
            model = bundle.get("pipeline", bundle.get("model"))
            if model is not None:
                self._model_cache[model_id] = model
                return model, X_train, y_train, X_test, y_test
        except Exception:
            pass

        # Baseline fallback
        if "logistic" in model_id.lower() or "linear" in model_id.lower():
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)

        model.fit(X_train, y_train)
        self._model_cache[model_id] = model
        return model, X_train, y_train, X_test, y_test

    def explain_global_shap(
        self,
        model_id: str,
        max_samples: int = 80,
    ) -> SHAPExplanationSummary:
        """Calculates global mean absolute Shapley values."""
        model, X_train, y_train, X_test, y_test = self.get_or_load_model_and_data(model_id)
        return self.shap_explainer.explain_global(
            model=model,
            X_data=X_test,
            max_samples=max_samples,
            model_id=model_id,
        )

    def explain_local_shap(
        self,
        model_id: str,
        sample_index: int = 0,
        num_features: int = 8,
    ) -> LocalSHAPExplanation:
        """Calculates additive local SHAP waterfall contributions for a sample."""
        model, X_train, y_train, X_test, y_test = self.get_or_load_model_and_data(model_id)
        idx = max(0, min(sample_index, len(X_test) - 1))
        instance = X_test.iloc[idx]
        return self.shap_explainer.explain_instance(
            model=model,
            instance=instance,
            X_background=X_train.iloc[:60],
            sample_index=idx,
            top_k=num_features,
        )

    def explain_local_lime(
        self,
        model_id: str,
        sample_index: int = 0,
        num_features: int = 8,
    ) -> LIMEExplanationResult:
        """Fits interpretable sparse linear surrogate rules for an individual observation."""
        model, X_train, y_train, X_test, y_test = self.get_or_load_model_and_data(model_id)
        idx = max(0, min(sample_index, len(X_test) - 1))
        instance = X_test.iloc[idx]
        return self.lime_explainer.explain_instance(
            model=model,
            instance=instance,
            X_train=X_train,
            num_features=num_features,
            sample_index=idx,
            model_id=model_id,
        )

    def compute_pdp(
        self,
        model_id: str,
        feature_name: str = "mean radius",
        grid_resolution: int = 25,
    ) -> PartialDependenceResult:
        """Calculates Partial Dependence (PDP) and Individual Conditional Expectation (ICE) curves."""
        model, X_train, y_train, X_test, y_test = self.get_or_load_model_and_data(model_id)
        feat = feature_name if feature_name in X_test.columns else X_test.columns[0]
        return self.shap_explainer.compute_partial_dependence(
            model=model,
            X_data=X_test,
            feature_name=feat,
            grid_resolution=grid_resolution,
        )

    def get_model_explainability(self, model_id: str) -> Dict[str, Any]:
        """Provides an explainability summary for a model."""
        summary = self.explain_global_shap(model_id, max_samples=40)
        return {
            "model_id": model_id,
            "explanation_available": True,
            "methods_supported": ["shap_global", "shap_local", "lime_local", "pdp_ice"],
            "top_features": [
                {
                    "rank": item.rank,
                    "feature": item.feature_name,
                    "importance": item.mean_abs_shap,
                    "direction": getattr(item, "risk_correlation_direction", "positive"),
                }

                for item in summary.global_importances[:5]
            ],
            "governance_notice": getattr(summary, "governance_notice", "Clinical decision support only."),

        }

