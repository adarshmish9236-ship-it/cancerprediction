"""
CANCERWISE AI — LIME Explainer Engine
Local Interpretable Model-agnostic Explanations (LIME) for individual patient predictions.
Explains local decision boundaries using sparse linear surrogate models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from lime import lime_tabular

from backend.schemas.explainability import (
    LIMEFeatureRule,
    LIMEExplanationResult,
)

logger = logging.getLogger("cancerwise.explainability.lime")


class LIMEExplainer:
    """
    Computes local sparse surrogate model explanations using LIME.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def explain_instance(
        self,
        model: Any,
        instance: pd.Series,
        X_train: pd.DataFrame,
        num_features: int = 8,
        sample_index: int = 0,
        model_id: str = "model",
    ) -> LIMEExplanationResult:
        """
        Fits local surrogate linear model around the patient instance.
        """
        features = X_train.columns.tolist()
        class_names = ["Benign", "Malignant"]

        # Predict function wrapper
        if hasattr(model, "predict_proba"):
            predict_fn = model.predict_proba
        else:
            def predict_fn(arr):
                preds = model.predict(arr)
                p1 = np.asarray(preds, dtype=float)
                return np.column_stack([1.0 - p1, p1])

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.to_numpy(dtype=float),
            feature_names=features,
            class_names=class_names,
            mode="classification",
            random_state=self.random_state,
        )

        inst_vec = instance.to_numpy(dtype=float)
        exp = explainer.explain_instance(
            data_row=inst_vec,
            predict_fn=predict_fn,
            num_features=min(num_features, len(features)),
            labels=(1,),  # Explain Malignant class
        )

        probs = predict_fn(inst_vec.reshape(1, -1))[0]
        prob_benign = float(probs[0])
        prob_malignant = float(probs[1])
        pred_class = int(prob_malignant >= 0.5)
        pred_label = "Malignant" if pred_class == 1 else "Benign"

        intercept = float(exp.intercept[1]) if hasattr(exp, "intercept") and 1 in exp.intercept else 0.5
        score = float(exp.score) if hasattr(exp, "score") else 0.90

        rules: List[LIMEFeatureRule] = []
        for rule_str, weight in exp.as_list(label=1):
            direction = "Supports Malignant" if weight > 0 else "Supports Benign"
            # Extract feature name from rule
            matched_feat = next((f for f in features if f in rule_str), "feature")
            rules.append(
                LIMEFeatureRule(
                    rule=rule_str,
                    feature_name=matched_feat,
                    weight=round(float(weight), 6),
                    direction=direction,
                )
            )

        return LIMEExplanationResult(
            model_id=model_id,
            sample_index=sample_index,
            predicted_class=pred_class,
            predicted_label=pred_label,
            probability_benign=round(prob_benign, 4),
            probability_malignant=round(prob_malignant, 4),
            local_intercept=round(intercept, 4),
            rules=rules,
            explanation_fidelity_score=round(score, 4),
        )
