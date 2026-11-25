"""
CANCERWISE AI — Phase 9 Comprehensive Test Suite
Validates complete FastAPI backend, 17 service integrations, background job execution,
Pydantic schemas, centralized error handling, and all versioned /api/v1 endpoints.
"""

from __future__ import annotations

import io
import time
import asyncio
import pytest
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from fastapi import UploadFile

from backend.main import health_check, api_health_check, root
from backend.services.job_service import get_job_manager, JobStatus
from backend.services.dataset_management_service import DatasetManagementService
from backend.services.dataset_ingestion_service import DatasetIngestionService
from backend.services.supervised_learning_service import SupervisedLearningService
from backend.services.prediction_service import PredictionService
from backend.core.exceptions import (
    DatasetNotFoundError,
    ModelNotFoundError,
    JobNotFoundError,
    FileValidationError,
    FeatureMismatchError,
)

# Route controllers
from backend.api.system import get_system_status, get_system_resources
from backend.api.datasets import (
    list_datasets,
    get_dataset_metadata,
    get_dataset_profile,
    upload_dataset,
    delete_dataset,
)
from backend.api.wrangling import (
    analyze_dataset_quality,
    clean_dataset,
    transform_dataset,
    get_transformation_history,
)
from backend.api.eda import (
    analyze_eda,
    get_eda_visualizations,
    get_eda_report,
)
from backend.api.features import (
    engineer_features,
    select_features,
    reduce_dimensions,
)
from backend.api.models import (
    list_models,
    train_model,
    get_model_details,
    evaluate_model,
    predict_model,
)
from backend.api.advanced import (
    train_semi_supervised,
    train_multitask,
    train_multiclass,
    train_one_class,
    train_regression,
    train_kernel,
)
from backend.api.explainability import (
    calculate_shap,
    calculate_lime,
    get_model_explainability,
)
from backend.api.jobs import (
    list_jobs,
    get_job,
    cancel_job,
)

# Request schemas
from backend.schemas.phase9_schemas import (
    WranglingAnalyzeRequest,
    WranglingCleanRequest,
    WranglingTransformRequest,
    EDAAnalyzeRequest,
    FeatureEngineerRequest,
    FeatureSelectRequest,
    FeatureReduceRequest,
    ModelTrainRequest,
    ClinicalPredictionRequest,
    AdvancedSemiSupervisedRequest,
    AdvancedMultitaskRequest,
    AdvancedMulticlassRequest,
    AdvancedOneClassRequest,
    AdvancedRegressionRequest,
    AdvancedKernelRequest,
    ExplainabilitySHAPRequest,
    ExplainabilityLIMERequest,
)
from backend.schemas.modeling import AlgorithmType, ModelTrainingConfig, ResamplingStrategy
from backend.schemas.advanced_learning import (
    SemiSupervisedMethod,
    MulticlassStrategy,
    OneClassMethod,
    RegressionMethod,
)


@pytest.fixture(scope="module")
def prepared_dataset():
    """Ensures primary dataset is available for integration testing."""
    mgmt = DatasetManagementService()
    try:
        mgmt.get_dataframe("uci_wdbc")
    except Exception:
        ingest = DatasetIngestionService()
        ingest.ingest_wdbc()
    return "uci_wdbc"


@pytest.fixture(scope="module")
def trained_model_id(prepared_dataset):
    """Trains a baseline model for prediction, evaluation, and explanation tests."""
    sup = SupervisedLearningService()
    cfg = ModelTrainingConfig(
        dataset_id=prepared_dataset,
        algorithm=AlgorithmType.LOGISTIC_REGRESSION,
        target_column="diagnosis",
        cv_folds=2,
    )
    res = sup.train_model(cfg, async_job=False)
    return res.model_id


# ===========================================================================
# 1. System & Telemetry APIs
# ===========================================================================
class TestSystemEndpoints:
    def test_root_endpoint(self):
        res = root()
        assert res["name"] == "CANCERWISE AI"
        assert "documentation" in res
        assert "endpoints" in res

    def test_health_check_endpoints(self):
        r1 = health_check()
        assert r1.status_code == 200
        r2 = api_health_check()
        assert r2.status_code == 200



    def test_system_status(self):
        status = get_system_status()
        assert status["platform_name"] == "CANCERWISE AI"
        assert status["overall_status"] == "ONLINE"
        assert "subsystems" in status
        assert status["subsystems"]["dataset_engine"] == "operational"
        assert status["subsystems"]["model_registry"] == "operational"

    def test_system_resources(self):
        res = get_system_resources()
        assert res.cpu_cores_logical >= 1
        assert res.ram_total_mb > 1000.0
        assert res.process_memory_rss_mb > 0.0


# ===========================================================================
# 2. Dataset APIs
# ===========================================================================
class TestDatasetEndpoints:
    def test_list_datasets(self, prepared_dataset):
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 1
        assert any(d.dataset_id == prepared_dataset for d in datasets)

    def test_get_dataset_metadata(self, prepared_dataset):
        meta = get_dataset_metadata(prepared_dataset)
        assert meta.dataset_id == prepared_dataset
        assert meta.records_count == 569
        assert meta.columns_count == 32

    def test_get_dataset_metadata_not_found(self):
        with pytest.raises(DatasetNotFoundError):
            get_dataset_metadata("non_existent_dataset_999")

    def test_get_dataset_profile(self, prepared_dataset):
        profile = get_dataset_profile(prepared_dataset)
        assert profile["dataset_id"] == prepared_dataset
        assert profile["total_samples"] == 569
        assert profile["total_features"] == 32
        assert "column_types" in profile
        assert "missing_percentages" in profile
        assert "numeric_feature_summary" in profile

    def test_upload_and_delete_dataset(self):
        csv_bytes = b"sample_id,marker_1,marker_2,diagnosis\nS01,1.5,3.2,M\nS02,2.1,4.5,B\nS03,3.0,5.1,M\n"
        upload_file = UploadFile(
            filename="temp_phase9_test.csv",
            file=io.BytesIO(csv_bytes),
        )
        meta = upload_dataset(
            file=upload_file,
            dataset_id="temp_phase9_test",
            dataset_name="Temp Phase 9 Test Cohort",
            target_variable="diagnosis",
        )
        assert meta.dataset_id == "temp_phase9_test"
        assert meta.records_count == 3



        # Delete uploaded dataset
        del_res = delete_dataset("temp_phase9_test")
        assert del_res["status"] == "SUCCESS"

        with pytest.raises(DatasetNotFoundError):
            get_dataset_metadata("temp_phase9_test")


# ===========================================================================
# 3. Data Wrangling APIs
# ===========================================================================
class TestWranglingEndpoints:
    def test_wrangling_analyze(self, prepared_dataset):
        req = WranglingAnalyzeRequest(dataset_id=prepared_dataset, target_column="diagnosis")
        report = analyze_dataset_quality(req)
        assert report.dataset_name == prepared_dataset
        assert 0.0 <= report.overall_cleanliness_score <= 100.0
        assert report.missingness.total_missing_cells == 0

    def test_wrangling_clean(self, prepared_dataset):
        req = WranglingCleanRequest(
            dataset_id=prepared_dataset,
            imputation_strategy="median",
            drop_duplicates=True,
            target_column="diagnosis",
        )
        res = clean_dataset(req)
        assert res["status"] == "SUCCESS"
        assert res["cleaned_shape"][0] > 0
        assert "log_id" in res

    def test_wrangling_transform(self, prepared_dataset):
        req = WranglingTransformRequest(
            dataset_id=prepared_dataset,
            scaling_strategy="standard",
            encoding_strategy="onehot",
            target_column="diagnosis",
        )
        res = transform_dataset(req)
        assert res["status"] == "SUCCESS"
        assert res["scaling_applied"] == "standard"
        assert res["transformed_features"] > 0

    def test_wrangling_history(self):
        history = get_transformation_history(limit=10)
        assert isinstance(history, list)


# ===========================================================================
# 4. Exploratory Data Analysis (EDA) APIs
# ===========================================================================
class TestEDAEndpoints:
    def test_eda_analyze(self, prepared_dataset):
        req = EDAAnalyzeRequest(dataset_id=prepared_dataset, target_column="diagnosis")
        res = analyze_eda(req)
        assert res["dataset_id"] == prepared_dataset
        assert "overview" in res
        assert "multivariate" in res
        assert len(res["top_statistical_tests"]) > 0

    def test_eda_visualizations(self, prepared_dataset):
        res = get_eda_visualizations(
            dataset_id=prepared_dataset,
            chart_type="correlation_heatmap",
        )
        assert res["chart_type"] == "correlation_heatmap"
        assert "figure" in res

    def test_eda_report(self, prepared_dataset):
        res = get_eda_report(dataset_id=prepared_dataset)
        assert res["status"] == "SUCCESS"
        assert Path(res["report_path"]).exists()
        assert res["report_size_kb"] > 0.0


# ===========================================================================
# 5. Feature Engineering & Selection APIs
# ===========================================================================
class TestFeatureEndpoints:
    def test_features_engineer(self, prepared_dataset):
        req = FeatureEngineerRequest(
            dataset_id=prepared_dataset,
            apply_domain_features=True,
            apply_ratios=True,
            apply_polynomials=False,
            apply_log_transforms=False,
            apply_aggregations=True,
        )
        res = engineer_features(req)
        assert res.status == "SUCCESS"
        assert res.engineered_feature_count > 0
        assert len(res.new_features_created) > 0

    def test_features_select(self, prepared_dataset):
        req = FeatureSelectRequest(
            dataset_id=prepared_dataset,
            method="mutual_info",
            k=8,
            target_column="diagnosis",
        )
        res = select_features(req)
        assert res["selected_feature_count"] == 8
        assert len(res["selected_features"]) == 8
        assert "rankings" in res

    def test_features_reduce(self, prepared_dataset):
        req = FeatureReduceRequest(
            dataset_id=prepared_dataset,
            method="pca",
            n_components=2,
            target_column="diagnosis",
        )
        res = reduce_dimensions(req)
        method_str = res.method.value if hasattr(res.method, "value") else str(res.method)
        assert method_str == "pca"
        assert res.n_components == 2
        assert len(res.explained_variance_ratio) == 2



# ===========================================================================
# 6. Machine Learning & Prediction APIs
# ===========================================================================
class TestMachineLearningEndpoints:
    def test_list_models(self):
        models = list_models()
        assert isinstance(models, list)

    def test_train_model_synchronous(self, prepared_dataset):
        req = ModelTrainRequest(
            dataset_id=prepared_dataset,
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            target_column="diagnosis",
            resampling=ResamplingStrategy.NONE,
            cv_folds=2,
            async_job=False,
        )
        res = train_model(req)
        assert res.status == "SUCCESS"
        assert res.validation_metrics.roc_auc >= 0.5
        assert res.model_id.startswith("model_") or "_" in res.model_id


    def test_train_model_async_job(self, prepared_dataset):
        req = ModelTrainRequest(
            dataset_id=prepared_dataset,
            algorithm=AlgorithmType.RANDOM_FOREST,
            target_column="diagnosis",
            cv_folds=2,
            async_job=True,
        )
        job = train_model(req)
        assert job.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED]

        jm = get_job_manager()
        # Wait briefly for completion
        for _ in range(30):
            current = jm.get_job(job.job_id)
            if current.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            time.sleep(0.3)

        assert current.status == JobStatus.COMPLETED
        assert current.progress == 1.0

    def test_get_model_details(self, trained_model_id):
        manifest = get_model_details(trained_model_id)
        assert manifest["model_id"] == trained_model_id
        assert "selected_features" in manifest
        assert "algorithm" in manifest

    def test_evaluate_model(self, trained_model_id):
        metrics = evaluate_model(trained_model_id, compute_bootstrap=False)
        assert 0.0 <= metrics.sensitivity_recall <= 1.0
        assert 0.0 <= metrics.specificity <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert len(metrics.prevalence_sensitivity_analysis) > 0

    def test_predict_model_validation_and_features(self, trained_model_id):
        manifest = get_model_details(trained_model_id)
        selected_features = manifest["selected_features"]

        # Valid feature payload
        sample_obs = {f: 1.0 for f in selected_features}
        req = ClinicalPredictionRequest(features=sample_obs)
        resp = predict_model(trained_model_id, req)

        assert resp.prediction_id.startswith("pred_")
        assert resp.model_id == trained_model_id
        assert resp.total_samples == 1
        item = resp.predictions[0]
        assert item.predicted_class in [0, 1]
        assert item.predicted_label in ["Benign", "Malignant"]
        assert "benign" in item.probabilities
        assert "malignant" in item.probabilities
        assert 0.5 <= item.confidence <= 1.0
        assert resp.explanation_available is True
        assert "shap" in resp.explanation_methods
        assert "DISCLAIMER" in resp.clinical_governance_notice

    def test_predict_model_rejects_missing_model(self):
        with pytest.raises(ModelNotFoundError):
            req = ClinicalPredictionRequest(features={"mean radius": 14.0})
            predict_model("non_existent_model_888", req)

    def test_predict_model_rejects_incompatible_features(self, trained_model_id):
        with pytest.raises(FeatureMismatchError):
            req = ClinicalPredictionRequest(features=[])
            predict_model(trained_model_id, req)


# ===========================================================================
# 7. Advanced Learning APIs
# ===========================================================================
class TestAdvancedLearningEndpoints:
    def test_advanced_semi_supervised(self):
        req = AdvancedSemiSupervisedRequest(method=SemiSupervisedMethod.LABEL_SPREADING, async_job=False)
        res = train_semi_supervised(req)
        assert res.method == "label_spreading"
        assert len(res.labeled_partition_results) >= 1

    def test_advanced_multitask(self):
        req = AdvancedMultitaskRequest(epochs=15, learning_rate=0.01, async_job=False)
        res = train_multitask(req)
        assert res.classification_accuracy >= 0.0
        assert res.regression_mse >= 0.0

    def test_advanced_multiclass(self):
        req = AdvancedMulticlassRequest(strategy=MulticlassStrategy.MULTINOMIAL_LOGISTIC, async_job=False)
        res = train_multiclass(req)
        assert res.macro_f1 >= 0.0
        assert len(res.confusion_matrix) >= 2

    def test_advanced_one_class(self):
        req = AdvancedOneClassRequest(method=OneClassMethod.ISOLATION_FOREST, contamination=0.05, async_job=False)
        res = train_one_class(req)
        assert res.benign_inlier_rate >= 0.0
        assert res.malignant_detection_rate >= 0.0

    def test_advanced_regression(self):
        req = AdvancedRegressionRequest(method=RegressionMethod.RIDGE, async_job=False)
        res = train_regression(req)
        assert res.mse >= 0.0
        assert "r2_score" in res.model_dump()

    def test_advanced_kernel(self):
        req = AdvancedKernelRequest()
        res = train_kernel(req)
        assert len(res.kernel_performances) >= 4
        assert res.best_kernel in ["linear", "poly", "rbf", "sigmoid"]


# ===========================================================================
# 8. Explainability APIs
# ===========================================================================
class TestExplainabilityEndpoints:
    def test_explainability_shap_global(self, trained_model_id):
        req = ExplainabilitySHAPRequest(model_id=trained_model_id, type="global", max_samples=25)
        res = calculate_shap(req)
        assert len(res.global_importances) > 0
        assert res.model_id == trained_model_id

    def test_explainability_shap_local(self, trained_model_id):
        req = ExplainabilitySHAPRequest(model_id=trained_model_id, type="local", sample_index=0, num_features=5)
        res = calculate_shap(req)
        assert len(res.contributions) <= 5
        assert 0.0 <= res.predicted_probability <= 1.0

    def test_explainability_lime(self, trained_model_id):
        req = ExplainabilityLIMERequest(model_id=trained_model_id, sample_index=0, num_features=4)
        res = calculate_lime(req)
        assert len(res.rules) == 4
        assert res.predicted_label in ["Benign", "Malignant"]

    def test_get_model_explainability(self, trained_model_id):
        res = get_model_explainability(trained_model_id)
        assert res["model_id"] == trained_model_id
        assert res["explanation_available"] is True
        assert len(res["top_features"]) > 0


# ===========================================================================
# 9. Background Job Management APIs
# ===========================================================================
class TestJobManagementEndpoints:
    def test_job_submission_and_query(self):
        jm = get_job_manager()
        job = jm.submit("test_compute_task", lambda ctx: {"result": "ok"})
        assert job.job_id.startswith("job_")

        # Query job
        fetched = get_job(job.job_id)
        assert fetched.job_id == job.job_id

        # List jobs
        jobs = list_jobs(limit=10)
        assert any(j.job_id == job.job_id for j in jobs)

    def test_job_cancellation(self):
        jm = get_job_manager()

        def _long_step(ctx):
            for _ in range(20):
                ctx.check_cancelled()
                time.sleep(0.1)

        job = jm.submit("cancellable_job", _long_step)
        cancelled = cancel_job(job.job_id)
        assert cancelled.status == JobStatus.CANCELLED

    def test_job_not_found(self):
        with pytest.raises(JobNotFoundError):
            get_job("non_existent_job_xyz_999")
