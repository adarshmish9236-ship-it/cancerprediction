"""
CANCERWISE AI — Unified Model Trainer
Orchestrates stratified train/test splitting, preprocessing pipeline creation,
in-fold class resampling (SMOTE / Oversampling / Undersampling), cross-validation,
hyperparameter tuning, untouched holdout evaluation, and artifact serialization.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from backend.schemas.modeling import (
    AlgorithmType,
    ResamplingStrategy,
    TuningStrategy,
    ModelTrainingConfig,
    ModelTrainingResult,
    ModelEvaluationMetrics,
)
from data_mining.classification.registry import ModelRegistry
from data_mining.classification.evaluator import ModelEvaluator
from data_mining.classification.tuning import HyperparameterOptimizer
from data_mining.classification.serializer import ModelSerializer

logger = logging.getLogger("cancerwise.classification.trainer")


class ModelTrainer:
    """
    End-to-end model training engine enforcing clinical machine learning best practices.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        evaluator: Optional[ModelEvaluator] = None,
        serializer: Optional[ModelSerializer] = None,
        random_state: int = 42,
    ):
        self.registry = registry or ModelRegistry()
        self.evaluator = evaluator or ModelEvaluator()
        self.serializer = serializer or ModelSerializer()
        self.tuner = HyperparameterOptimizer(random_state=random_state)
        self.random_state = random_state

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelTrainingConfig,
    ) -> Tuple[Any, ModelTrainingResult]:
        """
        Executes end-to-end supervised training workflow.
        """
        # 1. Feature selection / filtering
        if config.selected_features:
            valid_feats = [f for f in config.selected_features if f in X.columns]
            if not valid_feats:
                raise ValueError("None of the specified selected_features exist in X.")
            X_data = X[valid_feats].copy()
        else:
            X_data = X.select_dtypes(include=[np.number]).dropna(axis=1).copy()
            valid_feats = X_data.columns.tolist()

        # 2. Target standardization to 0/1 binary
        if not pd.api.types.is_numeric_dtype(y):
            mapping = {"M": 1, "B": 0, "malignant": 1, "benign": 0}
            unique_vals = list(y.unique())
            if all(v in mapping for v in unique_vals):
                y_arr = y.map(mapping).to_numpy()
            else:
                y_arr = pd.factorize(y)[0]
        else:
            y_arr = y.to_numpy().astype(int)

        # 3. Stratified Train / Test Split — test set kept completely untouched
        X_train, X_test, y_train, y_test = train_test_split(
            X_data,
            y_arr,
            test_size=config.test_size,
            stratify=y_arr,
            random_state=config.random_state,
        )

        logger.info(
            "Stratified split | Train: %d (%d pos) | Test: %d (%d pos)",
            len(X_train), int(np.sum(y_train)), len(X_test), int(np.sum(y_test)),
        )

        start_train_time = time.perf_counter()

        # 4. Estimator Initialization
        estimator = self.registry.create_estimator(
            config.algorithm,
            hyperparameters=config.hyperparameters,
            random_state=config.random_state,
        )

        # 5. Resampling Strategy Formulation
        resampler = None
        if config.resampling == ResamplingStrategy.SMOTE:
            resampler = SMOTE(random_state=config.random_state)
        elif config.resampling == ResamplingStrategy.RANDOM_OVERSAMPLING:
            resampler = RandomOverSampler(random_state=config.random_state)
        elif config.resampling == ResamplingStrategy.RANDOM_UNDERSAMPLING:
            resampler = RandomUnderSampler(random_state=config.random_state)
        elif config.resampling == ResamplingStrategy.CLASS_WEIGHTS:
            if hasattr(estimator, "class_weight"):
                estimator.set_params(class_weight="balanced")

        # 6. Pipeline Assembly
        scaler = StandardScaler()
        if resampler is not None:
            pipeline = ImblearnPipeline([
                ("scaler", scaler),
                ("resampler", resampler),
                ("classifier", estimator),
            ])
        else:
            pipeline = SklearnPipeline([
                ("scaler", scaler),
                ("classifier", estimator),
            ])

        # 7. Cross-Validation on Training Fold Only
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
            cv_roc_mean = float(np.mean(cv_scores))
            cv_roc_std = float(np.std(cv_scores))
        except Exception as e:
            logger.warning("CV scoring failed (%s), defaulting to accuracy scoring.", e)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
            cv_roc_mean = float(np.mean(cv_scores))
            cv_roc_std = float(np.std(cv_scores))

        # 8. Hyperparameter Optimization (if requested)
        tuned_params: Dict[str, Any] = {}
        if config.tuning_strategy != TuningStrategy.NONE:
            meta = self.registry.get_algorithm_metadata(config.algorithm)
            # Prefix search grid for pipeline classifier
            clf_grid = {f"classifier__{k}": v for k, v in meta.search_grid.items()}
            pipeline, best_p, best_s = self.tuner.optimize(
                pipeline,
                param_grid=clf_grid,
                X=X_train,
                y=y_train,
                strategy=config.tuning_strategy,
                n_iter=config.tuning_trials,
                cv_folds=config.cv_folds,
            )
            tuned_params = {k.replace("classifier__", ""): v for k, v in best_p.items()}
        else:
            pipeline.fit(X_train, y_train)

        total_train_time = time.perf_counter() - start_train_time

        # 9. Untouched Test Set Evaluation
        infer_start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        infer_elapsed = time.perf_counter() - infer_start
        latency_ms_per_sample = round((infer_elapsed / len(X_test)) * 1000.0, 3)

        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)
            except Exception:
                y_prob = None
        elif hasattr(pipeline, "decision_function"):
            try:
                df_scores = pipeline.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-df_scores))
            except Exception:
                y_prob = None

        test_metrics = self.evaluator.evaluate(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            compute_ci=True,
            random_state=config.random_state,
        )

        # 10. Serialization
        uid_slug = uuid.uuid4().hex[:8]
        model_id = f"{config.dataset_id}_{config.algorithm.value}_{uid_slug}"
        meta = self.registry.get_algorithm_metadata(config.algorithm)

        final_hyperparams = {**meta.default_params, **config.hyperparameters, **tuned_params}

        artifact_path = self.serializer.save_model(
            model_id=model_id,
            pipeline=pipeline,
            selected_features=valid_feats,
            algorithm=config.algorithm.value,
            hyperparameters=final_hyperparams,
            test_metrics=test_metrics.model_dump(),
            training_config=config.model_dump(),
            version="1.0.0",
        )

        result = ModelTrainingResult(
            model_id=model_id,
            model_name=f"{meta.display_name} ({model_id})",
            algorithm=config.algorithm,
            dataset_id=config.dataset_id,
            target_column=config.target_column,
            version="1.0.0",
            selected_features=valid_feats,
            resampling_applied=config.resampling,
            hyperparameters=final_hyperparams,
            training_time_seconds=round(total_train_time, 4),
            inference_latency_ms=latency_ms_per_sample,
            artifact_path=str(artifact_path),
            cv_roc_auc_mean=round(cv_roc_mean, 4),
            cv_roc_auc_std=round(cv_roc_std, 4),
            test_metrics=test_metrics,
        )

        return pipeline, result
