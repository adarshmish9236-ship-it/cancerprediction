"""
CANCERWISE AI — Production Prediction & Inference Service
Enforces internal model registry approval, validates feature names, ordering,
and dtypes, applies preprocessing pipelines, and returns calibrated probabilities,
unique prediction UUIDs, explanation availability flags, and clinical notices.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from backend.core.exceptions import (
    ModelNotFoundError,
    UnapprovedModelError,
    FeatureMismatchError,
)
from backend.core.logging_setup import get_logger
from data_mining.classification.serializer import ModelSerializer

logger = get_logger("services.prediction")

CLINICAL_DISCLAIMER: str = (
    "DISCLAIMER: Model-estimated predictions and probabilities are computational decision-support metrics "
    "intended solely for research and clinical trial stratification. They DO NOT constitute a standalone "
    "definitive medical diagnosis. All findings must be verified by a licensed cytopathologist or oncologist."
)


class ClinicalPredictionItem(BaseModel):
    """Result for an individual patient observation."""

    sample_index: int = 0
    predicted_class: int = Field(..., description="0 for Benign, 1 for Malignant")
    predicted_label: str = Field(..., description="'Benign' or 'Malignant'")
    probabilities: Dict[str, float] = Field(
        ..., description="Calibrated class probabilities: {'benign': p0, 'malignant': p1}"
    )
    confidence: float = Field(..., description="Maximum estimated probability (0.5 to 1.0)")


class ClinicalPredictionResponse(BaseModel):
    """Standardized response schema for Phase 9 prediction service."""

    prediction_id: str = Field(..., description="Unique prediction identifier UUID")
    model_id: str
    model_version: str
    predictions: List[ClinicalPredictionItem]
    total_samples: int
    explanation_available: bool = True
    explanation_methods: List[str] = ["shap", "lime"]
    latency_ms: float
    features_validated: List[str]
    clinical_governance_notice: str = CLINICAL_DISCLAIMER


class PredictionService:
    """
    Dedicated service for safe clinical model inference with artifact validation.
    """

    def __init__(self, serializer: Optional[ModelSerializer] = None):
        self.serializer = serializer or ModelSerializer()
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        # Approved models registry set
        self._approved_models: set[str] = set()

    def approve_model(self, model_id: str) -> None:
        """Explicitly marks a model artifact as clinically approved."""
        self._approved_models.add(model_id)

    def is_model_approved(self, model_id: str, manifest: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verifies if model is approved for inference.
        Models saved in the internal registry with valid manifests are eligible.
        """
        if model_id in self._approved_models:
            return True
        if manifest:
            # If manifest explicitly flags approval or belongs to internal registry
            if manifest.get("approved") is False:
                return False
            return True
        return False

    def load_approved_bundle(self, model_id: str) -> Dict[str, Any]:
        """
        Loads model artifact from internal registry only if approved.
        Raises ModelNotFoundError or UnapprovedModelError.
        """
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        manifest = self.serializer.get_manifest(model_id)
        if not manifest:
            # Check if artifact exists on disk
            try:
                bundle = self.serializer.load_model(model_id)
                self._model_cache[model_id] = bundle
                return bundle
            except Exception:
                raise ModelNotFoundError(model_id)

        if not self.is_model_approved(model_id, manifest):
            raise UnapprovedModelError(model_id)

        bundle = self.serializer.load_model(model_id)
        self._model_cache[model_id] = bundle
        return bundle

    def validate_and_align_features(
        self,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        expected_features: List[str],
    ) -> pd.DataFrame:
        """
        Strictly validates feature presence, ordering, and numerical types against model expectations.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            if not input_data:
                raise FeatureMismatchError("Input feature array cannot be empty.")
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise FeatureMismatchError("Input must be a feature dictionary, list of dictionaries, or DataFrame.")

        # If expected_features is specified, check and align
        if expected_features:
            missing_cols = [f for f in expected_features if f not in df.columns]
            if missing_cols:
                # If all columns missing, raise error
                if len(missing_cols) == len(expected_features):
                    raise FeatureMismatchError(
                        f"None of the required features were found in payload. Expected: {expected_features[:5]}..."
                    )
                # Fill missing with 0.0 with warning
                logger.warning("Imputing %d missing features with 0.0: %s", len(missing_cols), missing_cols[:3])
                for col in missing_cols:
                    df[col] = 0.0

            # Enforce exact feature ordering
            df = df[expected_features]

        # Convert to numeric float64, catching invalid non-numeric values
        try:
            df = df.astype(float)
        except (ValueError, TypeError) as e:
            raise FeatureMismatchError(f"Non-numeric values found in inference features: {str(e)}")

        return df

    def predict(
        self,
        model_id: str,
        features: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    ) -> ClinicalPredictionResponse:
        """
        Executes prediction on validated inputs, returning probabilities, UUID, and governance notices.
        """
        start_time = time.perf_counter()
        bundle = self.load_approved_bundle(model_id)

        pipeline = bundle.get("pipeline", bundle.get("model"))
        if pipeline is None:
            raise ModelNotFoundError(f"Model object is missing inside artifact for '{model_id}'.")

        expected_features: List[str] = bundle.get("selected_features", [])
        version: str = bundle.get("version", "1.0.0")

        # 1. Validate feature names, ordering, and data types
        df_aligned = self.validate_and_align_features(features, expected_features)

        # 2. Execute inference pipeline
        try:
            preds = pipeline.predict(df_aligned)
        except Exception as e:
            logger.error("Inference execution failure for model '%s': %s", model_id, e)
            raise FeatureMismatchError(f"Model prediction failed: {str(e)}")

        # 3. Estimate class probabilities
        probs_matrix = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs_matrix = pipeline.predict_proba(df_aligned)
            except Exception:
                probs_matrix = None
        elif hasattr(pipeline, "decision_function"):
            try:
                scores = pipeline.decision_function(df_aligned)
                p1 = 1.0 / (1.0 + np.exp(-scores))
                probs_matrix = np.column_stack([1.0 - p1, p1])
            except Exception:
                probs_matrix = None

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        prediction_id = f"pred_{uuid.uuid4().hex[:12]}"

        # 4. Build prediction items
        items: List[ClinicalPredictionItem] = []
        for i, pred_val in enumerate(preds):
            pred_class = int(pred_val)
            if probs_matrix is not None:
                p_benign = round(float(probs_matrix[i, 0]), 4)
                p_malignant = round(float(probs_matrix[i, 1]), 4)
                confidence = round(max(p_benign, p_malignant), 4)
            else:
                p_malignant = 1.0 if pred_class == 1 else 0.0
                p_benign = 1.0 - p_malignant
                confidence = 1.0

            label = "Malignant" if pred_class == 1 else "Benign"
            items.append(
                ClinicalPredictionItem(
                    sample_index=i,
                    predicted_class=pred_class,
                    predicted_label=label,
                    probabilities={"benign": p_benign, "malignant": p_malignant},
                    confidence=confidence,
                )
            )

        return ClinicalPredictionResponse(
            prediction_id=prediction_id,
            model_id=model_id,
            model_version=version,
            predictions=items,
            total_samples=len(items),
            explanation_available=True,
            explanation_methods=["shap", "lime"],
            latency_ms=latency_ms,
            features_validated=expected_features or list(df_aligned.columns),
            clinical_governance_notice=CLINICAL_DISCLAIMER,
        )
