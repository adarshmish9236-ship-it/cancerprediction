"""
CANCERWISE AI — Phase 11 Integration Test Suite
Verifies the 11 integrated pipeline connections:
1. Dataset upload to dataset registry
2. Dataset registry to data profiler
3. Data profiler to preprocessing
4. Preprocessing to EDA
5. EDA to feature engineering
6. Feature engineering to training
7. Training to evaluation
8. Evaluation to model registry
9. Model registry to prediction
10. Prediction to explainability
11. Backend to frontend (APIClient integration)
"""

from __future__ import annotations

import io
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.services.dataset_management_service import DatasetManagementService
from backend.services.dataset_ingestion_service import DatasetIngestionService
from preprocessing.quality_analyzer import DataQualityAnalyzer
from preprocessing.missing_values import MissingValueHandler
from preprocessing.transformers import DataTransformer
from data_mining.eda.eda_engine import EDAEngine
from feature_engineering.engineering_engine import FeatureEngineeringEngine
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.serializer import ModelSerializer
from data_mining.classification.prediction_service import PredictionService
from evaluation.model_validator import ClinicalModelValidator
from explainability.shap_explainer import SHAPExplainer
from frontend.utils.api_client import APIClient
from backend.schemas.modeling import AlgorithmType, ModelTrainingConfig


@pytest.fixture(scope="module")
def primary_dataset():
    """Ensures uci_wdbc is available."""
    mgmt = DatasetManagementService()
    try:
        df = mgmt.get_dataframe("uci_wdbc")
    except Exception:
        ingest = DatasetIngestionService()
        ingest.ingest_wdbc()
        df = mgmt.get_dataframe("uci_wdbc")
    return df


class TestIntegrationPipeline:
    # -----------------------------------------------------------------------
    # Hop 1: Dataset Upload to Dataset Registry
    # -----------------------------------------------------------------------
    def test_hop1_upload_to_registry(self):
        mgmt = DatasetManagementService()
        ingest = DatasetIngestionService()

        csv_content = (
            "id,radius_mean,texture_mean,diagnosis\n"
            "9901,14.5,19.2,M\n"
            "9902,12.1,17.4,B\n"
            "9903,15.2,20.1,M\n"
            "9904,11.8,16.5,B\n"
        )
        meta = ingest.ingest_uploaded_file(
            file_bytes=csv_content.encode("utf-8"),
            filename="hop1_test_cohort.csv",
            dataset_id="hop1_test_cohort",
            dataset_name="Hop 1 Verification Cohort",
            target_variable="diagnosis",
        )
        assert meta.dataset_id == "hop1_test_cohort"
        assert meta.records_count == 4

        # Verify dataset exists in registry
        registered_df = mgmt.get_dataframe("hop1_test_cohort")
        assert len(registered_df) == 4
        assert "diagnosis" in registered_df.columns

        # Clean up
        mgmt.delete_dataset("hop1_test_cohort")

    # -----------------------------------------------------------------------
    # Hop 2: Dataset Registry to Data Profiler
    # -----------------------------------------------------------------------
    def test_hop2_registry_to_profiler(self, primary_dataset):
        mgmt = DatasetManagementService()
        df = mgmt.get_dataframe("uci_wdbc")
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df, dataset_name="uci_wdbc", target_column="diagnosis")

        assert report.dataset_name == "uci_wdbc"
        assert report.total_records == 569
        assert report.total_columns >= 31
        assert 0.0 <= report.overall_quality_score <= 100.0

    # -----------------------------------------------------------------------
    # Hop 3: Data Profiler to Preprocessing
    # -----------------------------------------------------------------------
    def test_hop3_profiler_to_preprocessing(self, primary_dataset):
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(primary_dataset, dataset_name="uci_wdbc", target_column="diagnosis")
        assert report.overall_quality_score >= 90.0

        imputer = MissingValueHandler(strategy="median")
        cleaned_df = imputer.fit_transform(primary_dataset.select_dtypes(include=[np.number]))
        assert cleaned_df.isna().sum().sum() == 0

    # -----------------------------------------------------------------------
    # Hop 4: Preprocessing to EDA
    # -----------------------------------------------------------------------
    def test_hop4_preprocessing_to_eda(self, primary_dataset):
        transformer = DataTransformer(strategy="standard")
        num_cols = primary_dataset.select_dtypes(include=[np.number]).columns
        trans_df = transformer.fit_transform(primary_dataset[num_cols])
        trans_df["diagnosis"] = primary_dataset["diagnosis"]

        eda_engine = EDAEngine()
        overview = eda_engine.get_overview(trans_df, target_column="diagnosis")
        assert overview.total_records == 569
        assert "Malignant" in overview.target_distribution or "M" in overview.target_distribution

    # -----------------------------------------------------------------------
    # Hop 5: EDA to Feature Engineering
    # -----------------------------------------------------------------------
    def test_hop5_eda_to_feature_engineering(self, primary_dataset):
        eda_engine = EDAEngine()
        bi_res = eda_engine.get_bivariate_profile(primary_dataset, "radius_mean", "area_mean")
        assert bi_res.pearson_correlation > 0.90

        fe_engine = FeatureEngineeringEngine()
        fe_df, metadata = fe_engine.engineer_features(
            primary_dataset,
            apply_domain_features=True,
            apply_ratios=True,
        )
        assert metadata.derived_feature_count > 0

    # -----------------------------------------------------------------------
    # Hop 6: Feature Engineering to Training
    # -----------------------------------------------------------------------
    def test_hop6_feature_engineering_to_training(self, primary_dataset):
        fe_engine = FeatureEngineeringEngine()
        fe_df, _ = fe_engine.engineer_features(primary_dataset, apply_ratios=True)
        
        y_raw = fe_df["diagnosis"]
        y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = fe_df.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=3,
        )
        pipeline, result = trainer.train(X, y, config)
        assert result.test_metrics.roc_auc > 0.90
        assert pipeline is not None

    # -----------------------------------------------------------------------
    # Hop 7: Training to Evaluation
    # -----------------------------------------------------------------------
    def test_hop7_training_to_evaluation(self, primary_dataset):
        y_raw = primary_dataset["diagnosis"]
        y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = primary_dataset.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.RANDOM_FOREST,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        pipeline, train_result = trainer.train(X, y, config)

        validator = ClinicalModelValidator(random_state=42)
        probs = pipeline.predict_proba(X)[:, 1]
        preds = pipeline.predict(X)
        metrics = validator.evaluate_clinical_metrics(y.values, preds, probs)
        assert metrics.sensitivity > 0.90
        assert metrics.roc_auc > 0.95

    # -----------------------------------------------------------------------
    # Hop 8: Evaluation to Model Registry
    # -----------------------------------------------------------------------
    def test_hop8_evaluation_to_model_registry(self, primary_dataset):
        y_raw = primary_dataset["diagnosis"]
        y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = primary_dataset.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        pipeline, train_res = trainer.train(X, y, config)

        serializer = ModelSerializer()
        artifact_path = serializer.save_model(
            model_id=train_res.model_id,
            pipeline=pipeline,
            selected_features=X.columns.tolist(),
            algorithm="logistic_regression",
            hyperparameters={},
            test_metrics={"test_roc_auc": train_res.test_metrics.roc_auc},
            training_config={},
        )
        assert Path(artifact_path).exists()
        loaded = serializer.load_model(train_res.model_id)
        assert loaded is not None

    # -----------------------------------------------------------------------
    # Hop 9: Model Registry to Prediction
    # -----------------------------------------------------------------------
    def test_hop9_registry_to_prediction(self, primary_dataset):
        y_raw = primary_dataset["diagnosis"]
        y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = primary_dataset.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        _, train_res = trainer.train(X, y, config)

        pred_svc = PredictionService()
        sample = X.iloc[0].to_dict()
        pred_res = pred_svc.predict(train_res.model_id, sample)
        assert len(pred_res.predictions) == 1
        assert pred_res.predictions[0].predicted_label in ["Benign", "Malignant"]

    # -----------------------------------------------------------------------
    # Hop 10: Prediction to Explainability
    # -----------------------------------------------------------------------
    def test_hop10_prediction_to_explainability(self, primary_dataset):
        y_raw = primary_dataset["diagnosis"]
        y = y_raw.map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = primary_dataset.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.RANDOM_FOREST,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        pipeline, train_res = trainer.train(X, y, config)

        explainer = SHAPExplainer(random_state=42)
        summary = explainer.explain_global(pipeline, X, max_samples=30, model_id=train_res.model_id)
        assert len(summary.global_importances) > 0
        local_exp = explainer.explain_instance(pipeline, X, sample_index=0, model_id=train_res.model_id)
        assert local_exp.waterfall_features is not None

    # -----------------------------------------------------------------------
    # Hop 11: Backend to Frontend (APIClient Integration)
    # -----------------------------------------------------------------------
    def test_hop11_backend_to_frontend(self):
        client = APIClient()
        datasets = client.list_datasets()
        assert len(datasets) >= 1
        assert any(d.get("dataset_id") == "uci_wdbc" for d in datasets)

        stats = client.get_overview_stats()
        assert stats["total_datasets"] >= 1
        assert stats["total_models"] >= 1
