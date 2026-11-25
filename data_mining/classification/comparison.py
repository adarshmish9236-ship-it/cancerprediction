"""
CANCERWISE AI — Model Comparison Engine
Multi-model benchmarking suite evaluating predictive discrimination, generalization gap,
clinical sensitivity (Recall), specificity, probabilistic calibration (Brier score),
computational efficiency, and memory footprints.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backend.schemas.modeling import (
    AlgorithmType,
    ModelComparisonItem,
    ModelComparisonSummary,
    ModelTrainingConfig,
    ModelTrainingResult,
)
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.registry import ModelRegistry

logger = logging.getLogger("cancerwise.classification.comparison")


class ModelComparisonEngine:
    """
    Ranks and evaluates competing machine learning architectures on identical dataset splits.
    """

    def __init__(self, trainer: Optional[ModelTrainer] = None):
        self.trainer = trainer or ModelTrainer()
        self.registry = self.trainer.registry

    def benchmark_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_id: str,
        algorithms: Optional[List[AlgorithmType]] = None,
        ranking_metric: str = "test_roc_auc",
        selected_features: Optional[List[str]] = None,
    ) -> ModelComparisonSummary:
        """
        Trains and compares multiple classification models under identical evaluation splits.
        """
        if algorithms is None:
            algorithms = [
                AlgorithmType.LOGISTIC_REGRESSION,
                AlgorithmType.RANDOM_FOREST,
                AlgorithmType.GRADIENT_BOOSTING,
                AlgorithmType.RBF_SVM,
                AlgorithmType.KNN,
                AlgorithmType.GAUSSIAN_NB,
                AlgorithmType.XGBOOST,
                AlgorithmType.MLP_SKLEARN,
            ]

        items: List[ModelComparisonItem] = []

        for alg in algorithms:
            logger.info("Benchmarking algorithm '%s'...", alg.value)
            config = ModelTrainingConfig(
                algorithm=alg,
                dataset_id=dataset_id,
                selected_features=selected_features,
                test_size=0.20,
                random_state=42,
            )
            try:
                pipeline, res = self.trainer.train(X, y, config)
                meta = self.registry.get_algorithm_metadata(alg)

                # Determine model size
                size_kb = 0.0
                if os.path.exists(res.artifact_path):
                    size_kb = round(os.path.getsize(res.artifact_path) / 1024.0, 2)

                # Generalization gap: CV mean accuracy/roc minus test accuracy/roc
                gen_gap = round(abs(res.cv_roc_auc_mean - res.test_metrics.roc_auc), 4)

                item = ModelComparisonItem(
                    model_id=res.model_id,
                    model_name=meta.display_name,
                    algorithm=alg,
                    category=meta.category,
                    test_accuracy=res.test_metrics.accuracy,
                    test_precision=res.test_metrics.precision,
                    test_recall_sensitivity=res.test_metrics.recall_sensitivity,
                    test_specificity=res.test_metrics.specificity,
                    test_f1_score=res.test_metrics.f1_score,
                    test_balanced_accuracy=res.test_metrics.balanced_accuracy,
                    test_roc_auc=res.test_metrics.roc_auc,
                    test_pr_auc=res.test_metrics.pr_auc,
                    test_mcc=res.test_metrics.matthews_corr_coef,
                    test_brier_score=res.test_metrics.brier_score,
                    training_time_seconds=res.training_time_seconds,
                    inference_latency_ms=res.inference_latency_ms,
                    model_size_kb=size_kb,
                    generalization_gap=gen_gap,
                )
                items.append(item)

            except Exception as e:
                logger.error("Failed benchmarking algorithm '%s': %s", alg.value, e)

        if not items:
            raise RuntimeError("No models successfully completed benchmarking.")

        # Sort based on selected ranking metric (descending for score, ascending for brier/gap/latency)
        reverse_sort = True
        if ranking_metric in ["test_brier_score", "generalization_gap", "inference_latency_ms", "training_time_seconds"]:
            reverse_sort = False

        items.sort(key=lambda m: getattr(m, ranking_metric, 0.0), reverse=reverse_sort)
        best_model = items[0]

        rationale = (
            f"Model '{best_model.model_name}' ranks #1 based on {ranking_metric} ({getattr(best_model, ranking_metric)}). "
            f"In biomedical diagnostics, high sensitivity (Recall={best_model.test_recall_sensitivity:.4f}) and "
            f"ROC-AUC ({best_model.test_roc_auc:.4f}) are critical to prevent fatal false-negative misclassifications, "
            f"while maintaining low calibration error (Brier Score={best_model.test_brier_score:.4f})."
        )

        return ModelComparisonSummary(
            ranking_metric=ranking_metric,
            ranked_models=items,
            top_recommended_model=best_model.model_name,
            clinical_rationale=rationale,
        )
