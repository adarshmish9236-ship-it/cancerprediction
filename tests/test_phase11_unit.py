"""
CANCERWISE AI — Phase 11 Unit Test Suite
Validates individual functionality across all 13 core algorithmic and analytical components:
1. Dataset Loader
2. Dataset Validator
3. Missing-Value Handler
4. Outlier Detector
5. Data Transformer
6. Feature Selector
7. Classification Algorithms
8. Semi-Supervised Engine
9. Multitask Engine
10. Model Evaluator
11. SHAP Explainer
12. LIME Explainer
13. Prediction Service
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Dataset Loader & Management
from backend.services.dataset_management_service import DatasetManagementService
from backend.services.dataset_ingestion_service import DatasetIngestionService

# 2. Preprocessing Components
from preprocessing import (
    DataQualityAnalyzer,
    MissingValueHandler,
    OutlierDetector,
    DataTransformer,
    DataValidationEngine,
)
from backend.schemas.preprocessing import OutlierMethod

# 3. Feature Selection & Engineering
from feature_engineering.selection_engine import FeatureSelectionEngine

# 4. Classification & Modeling
from backend.schemas.modeling import AlgorithmType, ModelTrainingConfig
from data_mining.classification.registry import ModelRegistry
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.prediction_service import PredictionService

# 5. Advanced Learning Engines
from data_mining.semi_supervised.semi_supervised_engine import SemiSupervisedEngine
from data_mining.multitask.multitask_engine import MultitaskEngine

# 6. Evaluation & Explainability
from evaluation.model_validator import ClinicalModelValidator
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer


@pytest.fixture(scope="module")
def wdbc_data():
    """Loads Wisconsin Diagnostic Breast Cancer dataset."""
    mgmt = DatasetManagementService()
    try:
        df = mgmt.get_dataframe("uci_wdbc")
    except Exception:
        ingest = DatasetIngestionService()
        ingest.ingest_wdbc()
        df = mgmt.get_dataframe("uci_wdbc")
    return df


@pytest.fixture(scope="module")
def prepared_X_y(wdbc_data):
    """Splits into numerical features X and binary target y."""
    y_raw = wdbc_data["diagnosis"]
    y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
    X = wdbc_data.drop(columns=["diagnosis", "id"], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0.0)
    return X, y


# ===========================================================================
# 1. Dataset Loader Tests
# ===========================================================================
class TestDatasetLoader:
    def test_load_existing_dataset(self, wdbc_data):
        assert isinstance(wdbc_data, pd.DataFrame)
        assert len(wdbc_data) == 569
        assert "diagnosis" in wdbc_data.columns
        assert wdbc_data.shape[1] >= 31

    def test_dataset_metadata_integrity(self):
        mgmt = DatasetManagementService()
        meta = mgmt.get_metadata("uci_wdbc")
        assert meta.dataset_id == "uci_wdbc"
        assert meta.records_count == 569
        assert meta.target_variable == "diagnosis"


# ===========================================================================
# 2. Dataset Validator Tests
# ===========================================================================
class TestDatasetValidator:
    def test_data_validation_clean_data(self, wdbc_data):
        validator = DataValidationEngine(target_column="diagnosis")
        val_result = validator.validate(wdbc_data)
        assert val_result.is_valid is True
        assert len(val_result.errors) == 0

    def test_data_validation_missing_target(self, wdbc_data):
        validator = DataValidationEngine(target_column="diagnosis")
        df_no_target = wdbc_data.drop(columns=["diagnosis"])
        val_result = validator.validate(df_no_target)
        assert val_result.is_valid is False
        assert any("target" in str(err).lower() for err in val_result.errors)


# ===========================================================================
# 3. Missing-Value Handler Tests
# ===========================================================================
class TestMissingValueHandler:
    def test_missing_value_imputation(self, prepared_X_y):
        X, _ = prepared_X_y
        X_corrupt = X.copy()
        mask = np.random.RandomState(42).rand(*X_corrupt.shape) < 0.05
        X_corrupt[mask] = np.nan
        assert X_corrupt.isna().sum().sum() > 0

        imputer = MissingValueHandler(strategy="median")
        X_imputed = imputer.fit_transform(X_corrupt)
        assert X_imputed.isna().sum().sum() == 0
        assert X_imputed.shape == X.shape


# ===========================================================================
# 4. Outlier Detector Tests
# ===========================================================================
class TestOutlierDetector:
    def test_outlier_detection_iqr(self, wdbc_data):
        detector = OutlierDetector(method=OutlierMethod.IQR)
        report = detector.detect(wdbc_data, target_column="diagnosis")
        assert report.total_outliers >= 0
        assert report.method == OutlierMethod.IQR


# ===========================================================================
# 5. Data Transformer Tests
# ===========================================================================
class TestDataTransformer:
    def test_standard_scaling_transformation(self, prepared_X_y):
        X, _ = prepared_X_y
        transformer = DataTransformer(strategy="standard")
        X_trans = transformer.fit_transform(X)
        assert X_trans.shape == X.shape
        means = np.nanmean(X_trans.values, axis=0)
        assert np.all(np.abs(means) < 1e-2)


# ===========================================================================
# 6. Feature Selector Tests
# ===========================================================================
class TestFeatureSelector:
    def test_feature_selection_methods(self, prepared_X_y):
        X, y = prepared_X_y
        selector = FeatureSelectionEngine()
        selected, rankings = selector.select_features(X, y, method="mutual_info", k=10)
        assert len(selected) == 10
        assert len(rankings) == X.shape[1]
        assert all(f in X.columns for f in selected)


# ===========================================================================
# 7. Classification Algorithms Tests
# ===========================================================================
class TestClassificationAlgorithms:
    def test_model_registry_instantiations(self):
        reg = ModelRegistry()
        algs = reg.list_available_algorithms()
        assert len(algs) >= 10

    def test_logistic_regression_fit_predict(self, prepared_X_y):
        X, y = prepared_X_y
        reg = ModelRegistry()
        clf = reg.create_estimator(AlgorithmType.LOGISTIC_REGRESSION)
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        assert len(preds) == len(y)
        assert probs.shape == (len(y), 2)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_random_forest_fit_predict(self, prepared_X_y):
        X, y = prepared_X_y
        reg = ModelRegistry()
        rf = reg.create_estimator(AlgorithmType.RANDOM_FOREST)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == len(y)
        acc = np.mean(preds == y.values)
        assert acc > 0.90


# ===========================================================================
# 8. Semi-Supervised Engine Tests
# ===========================================================================
class TestSemiSupervisedEngine:
    def test_label_spreading(self, prepared_X_y):
        X, y = prepared_X_y
        engine = SemiSupervisedEngine(random_state=42)
        result = engine.run_experiment(
            X, y,
            method="label_spreading",
            proportions=[0.3],
            test_size=0.2,
        )
        assert len(result.proportion_evaluations) == 1
        assert result.proportion_evaluations[0].test_accuracy > 0.80


# ===========================================================================
# 9. Multitask Engine Tests
# ===========================================================================
class TestMultitaskEngine:
    def test_multitask_training(self, prepared_X_y):
        X, y = prepared_X_y
        engine = MultitaskEngine()
        y_recurrence = (X["area_mean"] > X["area_mean"].median()).astype(int).values
        res = engine.train_multitask(
            X,
            targets={"primary": y.values, "recurrence": y_recurrence},
            task_types={"primary": "binary", "recurrence": "binary"},
            epochs=5,
            batch_size=32,
        )
        assert len(res.tasks) == 2
        assert any(t.task_name == "primary" for t in res.tasks)
        assert res.combined_test_loss < 2.0


# ===========================================================================
# 10. Model Evaluator Tests
# ===========================================================================
class TestModelEvaluator:
    def test_clinical_model_validator(self, prepared_X_y):
        X, y = prepared_X_y
        reg = ModelRegistry()
        clf = reg.create_estimator(AlgorithmType.LOGISTIC_REGRESSION)
        clf.fit(X, y)
        probs = clf.predict_proba(X)[:, 1]
        preds = clf.predict(X)

        validator = ClinicalModelValidator(random_state=42)
        metrics = validator.evaluate_clinical_metrics(y.values, preds, probs)
        assert metrics.sensitivity > 0.85
        assert metrics.specificity > 0.85
        assert metrics.roc_auc > 0.90
        assert 0.0 <= metrics.brier_score <= 0.25


# ===========================================================================
# 11. SHAP Explainer Tests
# ===========================================================================
class TestSHAPExplainer:
    def test_shap_explanation(self, prepared_X_y):
        X, y = prepared_X_y
        reg = ModelRegistry()
        rf = reg.create_estimator(AlgorithmType.RANDOM_FOREST)
        rf.fit(X, y)

        explainer = SHAPExplainer(random_state=42)
        summary = explainer.explain_global(rf, X, max_samples=30, model_id="test_rf")
        assert len(summary.global_importances) == X.shape[1]
        assert summary.global_importances[0].mean_abs_shap >= summary.global_importances[-1].mean_abs_shap

        instance_exp = explainer.explain_instance(rf, X, sample_index=0, model_id="test_rf")
        assert instance_exp.waterfall_features is not None


# ===========================================================================
# 12. LIME Explainer Tests
# ===========================================================================
class TestLIMEExplainer:
    def test_lime_explanation(self, prepared_X_y):
        X, y = prepared_X_y
        reg = ModelRegistry()
        rf = reg.create_estimator(AlgorithmType.RANDOM_FOREST)
        rf.fit(X, y)

        lime = LIMEExplainer(random_state=42)
        res = lime.explain_instance(
            model=rf,
            instance=X.iloc[0],
            X_train=X,
            num_features=5,
            sample_index=0,
            model_id="test_rf",
        )
        assert len(res.rules) == 5
        assert res.predicted_label in ["Benign", "Malignant"]


# ===========================================================================
# 13. Prediction Service Tests
# ===========================================================================
class TestPredictionService:
    def test_prediction_service_inference(self, prepared_X_y):
        X, y = prepared_X_y
        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.RANDOM_FOREST,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        pipeline, result = trainer.train(X, y, config)

        pred_svc = PredictionService()
        sample_dict = X.iloc[0].to_dict()
        pred_res = pred_svc.predict(result.model_id, sample_dict)
        assert len(pred_res.predictions) == 1
        p0 = pred_res.predictions[0]
        assert p0.predicted_label in ["Benign", "Malignant"]
        assert 0.0 <= p0.probability_malignant <= 1.0
