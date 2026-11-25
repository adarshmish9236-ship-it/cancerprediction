"""
CANCERWISE AI — Phase 5 Comprehensive Test Suite
Tests PyTorch Classifier, Model Evaluator (12 metrics + 95% Bootstrap CIs),
Model Registry, Resampling with SMOTE, Model Trainer, Serializer,
Multi-Model Comparison Engine, Prediction Service, and REST Endpoints.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from backend.schemas.modeling import (
    AlgorithmType,
    ResamplingStrategy,
    TuningStrategy,
    ModelTrainingConfig,
    ModelTrainingResult,
    ModelEvaluationMetrics,
)
from data_mining.classification.base import PyTorchMLPClassifier
from data_mining.classification.evaluator import ModelEvaluator
from data_mining.classification.registry import ModelRegistry
from data_mining.classification.tuning import HyperparameterOptimizer
from data_mining.classification.serializer import ModelSerializer
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.comparison import ModelComparisonEngine
from data_mining.classification.prediction_service import PredictionService
from backend.api.models import (
    list_supported_algorithms,
    train_model,
    list_saved_models,
    predict,
)
from backend.services.dataset_service import DatasetService


@pytest.fixture
def synthetic_classification_data():
    """Generates synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.65, 0.35],
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df["diagnosis"] = y
    return df


# ---------------------------------------------------------------------------
# 1. PyTorch MLP Classifier Tests
# ---------------------------------------------------------------------------
class TestPyTorchMLPClassifier:
    def test_pytorch_mlp_fit_predict(self, synthetic_classification_data):
        X = synthetic_classification_data.drop(columns=["diagnosis"])
        y = synthetic_classification_data["diagnosis"]

        clf = PyTorchMLPClassifier(epochs=10, batch_size=32, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

        probs = clf.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Model Evaluator Tests
# ---------------------------------------------------------------------------
class TestModelEvaluator:
    def test_all_twelve_metrics(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.85, 0.15, 0.4, 0.2, 0.3, 0.95, 0.8, 0.1, 0.9, 0.2, 0.88, 0.15, 0.45, 0.7, 0.3, 0.2, 0.9, 0.1, 0.85])

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred, y_prob, compute_ci=True, n_bootstraps=20)

        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall_sensitivity <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert 0.0 <= metrics.balanced_accuracy <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert 0.0 <= metrics.pr_auc <= 1.0
        assert -1.0 <= metrics.matthews_corr_coef <= 1.0
        assert metrics.brier_score >= 0.0
        assert len(metrics.confusion_matrix) == 2
        assert "accuracy" in metrics.confidence_intervals
        assert "roc_auc" in metrics.confidence_intervals


# ---------------------------------------------------------------------------
# 3. Model Registry Tests
# ---------------------------------------------------------------------------
class TestModelRegistry:
    def test_algorithms_availability(self):
        registry = ModelRegistry()
        algs = registry.list_available_algorithms()
        assert len(algs) >= 14

        categories = {a["category"] for a in algs}
        assert "Linear" in categories
        assert "Tree" in categories
        assert "Kernel" in categories
        assert "Instance" in categories
        assert "Probabilistic" in categories
        assert "Boosting" in categories
        assert "Neural Network" in categories

    def test_estimator_instantiation(self):
        registry = ModelRegistry()
        for alg in [AlgorithmType.LOGISTIC_REGRESSION, AlgorithmType.RANDOM_FOREST, AlgorithmType.XGBOOST, AlgorithmType.MLP_PYTORCH]:
            est = registry.create_estimator(alg)
            assert hasattr(est, "fit")
            assert hasattr(est, "predict")


# ---------------------------------------------------------------------------
# 4. Model Trainer & In-Fold Resampling Tests
# ---------------------------------------------------------------------------
class TestModelTrainer:
    def test_training_workflow_with_smote(self, synthetic_classification_data):
        X = synthetic_classification_data.drop(columns=["diagnosis"])
        y = synthetic_classification_data["diagnosis"]

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="synthetic_test",
            target_column="diagnosis",
            resampling=ResamplingStrategy.SMOTE,
            test_size=0.20,
            cv_folds=3,
        )

        pipeline, result = trainer.train(X, y, config)
        assert isinstance(result, ModelTrainingResult)
        assert result.resampling_applied == ResamplingStrategy.SMOTE
        assert result.test_metrics.roc_auc > 0.60
        assert result.cv_roc_auc_mean > 0.60
        assert result.training_time_seconds > 0.0
        assert result.inference_latency_ms >= 0.0
        assert result.artifact_path != ""

    def test_hyperparameter_tuning(self, synthetic_classification_data):
        X = synthetic_classification_data.drop(columns=["diagnosis"])
        y = synthetic_classification_data["diagnosis"]

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.RANDOM_FOREST,
            dataset_id="synthetic_test",
            target_column="diagnosis",
            tuning_strategy=TuningStrategy.RANDOMIZED_SEARCH,
            tuning_trials=3,
            cv_folds=3,
        )

        pipeline, result = trainer.train(X, y, config)
        assert result.test_metrics.accuracy > 0.65


# ---------------------------------------------------------------------------
# 5. Multi-Model Comparison Engine Tests
# ---------------------------------------------------------------------------
class TestModelComparisonEngine:
    def test_benchmark_multiple_algorithms(self, synthetic_classification_data):
        X = synthetic_classification_data.drop(columns=["diagnosis"])
        y = synthetic_classification_data["diagnosis"]

        engine = ModelComparisonEngine()
        algs = [
            AlgorithmType.LOGISTIC_REGRESSION,
            AlgorithmType.RANDOM_FOREST,
            AlgorithmType.XGBOOST,
            AlgorithmType.KNN,
            AlgorithmType.GAUSSIAN_NB,
            AlgorithmType.MLP_SKLEARN,
        ]

        summary = engine.benchmark_models(
            X, y,
            dataset_id="synthetic_test",
            algorithms=algs,
            ranking_metric="test_roc_auc",
        )

        assert len(summary.ranked_models) == 6
        assert summary.ranked_models[0].test_roc_auc >= summary.ranked_models[-1].test_roc_auc
        assert summary.top_recommended_model != ""


# ---------------------------------------------------------------------------
# 6. Serializer & Prediction Service Tests
# ---------------------------------------------------------------------------
class TestPredictionService:
    def test_serialization_and_inference(self, synthetic_classification_data):
        X = synthetic_classification_data.drop(columns=["diagnosis"])
        y = synthetic_classification_data["diagnosis"]

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.RANDOM_FOREST,
            dataset_id="synthetic_pred_test",
            target_column="diagnosis",
        )
        _, result = trainer.train(X, y, config)

        service = PredictionService()
        sample_input = X.iloc[0].to_dict()

        pred_resp = service.predict(result.model_id, sample_input)
        assert len(pred_resp.predictions) == 1
        p0 = pred_resp.predictions[0]
        assert p0.predicted_class in [0, 1]
        assert p0.predicted_label in ["Benign", "Malignant"]
        assert 0.0 <= p0.probability_benign <= 1.0
        assert 0.0 <= p0.probability_malignant <= 1.0
        assert "clinical_disclaimer" in pred_resp.model_dump()


# ---------------------------------------------------------------------------
# 7. Models REST API Tests
# ---------------------------------------------------------------------------
class TestModelsAPI:
    def test_api_routes(self):
        # 1. List algorithms
        algs = list_supported_algorithms()
        assert len(algs) >= 10

        # 2. Train on UCI WDBC
        dataset_id = "uci_wdbc"
        ds = DatasetService()
        if ds.load_dataset(dataset_id) is not None:
            config = ModelTrainingConfig(
                algorithm=AlgorithmType.LOGISTIC_REGRESSION,
                dataset_id=dataset_id,
                target_column="diagnosis",
                cv_folds=3,
            )
            res = train_model(dataset_id=dataset_id, config=config)
            assert res.test_metrics.roc_auc > 0.90

            # 3. List saved models
            saved = list_saved_models()
            assert any(m["model_id"] == res.model_id for m in saved)
