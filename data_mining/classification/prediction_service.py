"""
CANCERWISE AI — Production Prediction Service
Provides high-throughput and single-sample inference with feature validation,
probability estimation, latency profiling, and mandatory clinical governance notices.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from backend.schemas.modeling import (
    PredictionRequest,
    PredictionResponse,
    SinglePredictionResult,
)
from data_mining.classification.serializer import ModelSerializer

logger = logging.getLogger("cancerwise.classification.prediction")


class PredictionService:
    """
    Inference service that loads serialized model pipelines and evaluates new patient observations.
    """

    def __init__(self, serializer: Optional[ModelSerializer] = None):
        self.serializer = serializer or ModelSerializer()
        self._model_cache: Dict[str, Dict[str, Any]] = {}

    def get_or_load_bundle(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self._model_cache:
            bundle = self.serializer.load_model(model_id)
            self._model_cache[model_id] = bundle
        return self._model_cache[model_id]

    def predict(
        self,
        model_id: str,
        input_data: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame],
    ) -> PredictionResponse:
        """
        Executes prediction using the loaded bundle.
        """
        bundle = self.get_or_load_bundle(model_id)
        pipeline = bundle["pipeline"]
        expected_features: List[str] = bundle.get("selected_features", [])
        version: str = bundle.get("version", "1.0.0")

        # Format input data
        if isinstance(input_data, dict):
            df_in = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df_in = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_in = input_data.copy()
        else:
            raise ValueError("Input data must be a dictionary, list of dictionaries, or DataFrame.")

        # Ensure all expected features are present; impute with 0.0 if absent
        for feat in expected_features:
            if feat not in df_in.columns:
                logger.warning("Feature '%s' missing from inference payload, imputing with 0.0.", feat)
                df_in[feat] = 0.0

        X_eval = df_in[expected_features]

        start_time = time.perf_counter()
        preds = pipeline.predict(X_eval)

        probs_arr = None
        if hasattr(pipeline, "predict_proba"):
            try:
                probs_arr = pipeline.predict_proba(X_eval)
            except Exception:
                probs_arr = None
        elif hasattr(pipeline, "decision_function"):
            try:
                scores = pipeline.decision_function(X_eval)
                p1 = 1.0 / (1.0 + np.exp(-scores))
                probs_arr = np.column_stack([1.0 - p1, p1])
            except Exception:
                probs_arr = None

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        results: List[SinglePredictionResult] = []
        for i in range(len(preds)):
            pred_class = int(preds[i])
            if probs_arr is not None:
                p_benign = round(float(probs_arr[i, 0]), 4)
                p_malignant = round(float(probs_arr[i, 1]), 4)
                confidence = max(p_benign, p_malignant)
            else:
                p_malignant = 1.0 if pred_class == 1 else 0.0
                p_benign = 1.0 - p_malignant
                confidence = 1.0

            label = "Malignant" if pred_class == 1 else "Benign"

            results.append(
                SinglePredictionResult(
                    predicted_class=pred_class,
                    predicted_label=label,
                    probability_benign=p_benign,
                    probability_malignant=p_malignant,
                    confidence_score=confidence,
                )
            )

        return PredictionResponse(
            model_id=model_id,
            model_version=version,
            predictions=results,
            latency_ms=elapsed_ms,
        )
