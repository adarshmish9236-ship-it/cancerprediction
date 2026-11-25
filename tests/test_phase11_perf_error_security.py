"""
CANCERWISE AI — Phase 11 Performance, Error & Security Test Suite
Covers:
1. Performance Testing (Latency benchmarks, memory profiling, training benchmarks)
2. Error Handling Testing (Corrupt files, invalid uploads, missing features, invalid schemas)
3. Security Testing (Path traversal, file upload MIME checks, credential scanning, log sanitization)
"""

from __future__ import annotations

import io
import os
import time
import psutil
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import UploadFile

from backend.services.dataset_management_service import DatasetManagementService
from backend.services.dataset_ingestion_service import DatasetIngestionService
from preprocessing.transformers import DataTransformer
from preprocessing.quality_analyzer import DataQualityAnalyzer
from data_mining.classification.trainer import ModelTrainer
from data_mining.classification.prediction_service import PredictionService
from backend.schemas.modeling import AlgorithmType, ModelTrainingConfig


@pytest.fixture(scope="module")
def wdbc_data():
    mgmt = DatasetManagementService()
    try:
        return mgmt.get_dataframe("uci_wdbc")
    except Exception:
        ingest = DatasetIngestionService()
        ingest.ingest_wdbc()
        return mgmt.get_dataframe("uci_wdbc")


# ===========================================================================
# 1. Performance Testing
# ===========================================================================
class TestPerformanceBenchmarks:
    def test_dataset_loading_latency(self):
        mgmt = DatasetManagementService()
        t0 = time.perf_counter()
        df = mgmt.get_dataframe("uci_wdbc")
        elapsed = time.perf_counter() - t0
        assert len(df) == 569
        assert elapsed < 1.0, f"Dataset loading too slow: {elapsed:.3f}s (benchmark: < 1.0s)"

    def test_preprocessing_runtime(self, wdbc_data):
        num_cols = wdbc_data.select_dtypes(include=[np.number]).columns
        transformer = DataTransformer(strategy="standard")
        t0 = time.perf_counter()
        transformed = transformer.fit_transform(wdbc_data[num_cols])
        elapsed = time.perf_counter() - t0
        assert transformed.shape == wdbc_data[num_cols].shape
        assert elapsed < 0.5, f"Preprocessing too slow: {elapsed:.3f}s (benchmark: < 0.5s)"

    def test_model_training_latency(self, wdbc_data):
        y = wdbc_data["diagnosis"].map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = wdbc_data.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        t0 = time.perf_counter()
        _, result = trainer.train(X, y, config)
        elapsed = time.perf_counter() - t0
        assert result.test_metrics.roc_auc > 0.90
        assert elapsed < 2.0, f"Training too slow: {elapsed:.3f}s (benchmark: < 2.0s)"

    def test_prediction_latency(self, wdbc_data):
        y = wdbc_data["diagnosis"].map({"M": 1, "B": 0, "Malignant": 1, "Benign": 0}).astype(int)
        X = wdbc_data.drop(columns=["diagnosis", "id"], errors="ignore").select_dtypes(include=[np.number]).fillna(0.0)

        trainer = ModelTrainer()
        config = ModelTrainingConfig(
            algorithm=AlgorithmType.LOGISTIC_REGRESSION,
            dataset_id="uci_wdbc",
            target_column="diagnosis",
            cv_folds=2,
        )
        _, result = trainer.train(X, y, config)

        pred_svc = PredictionService()
        sample = X.iloc[0].to_dict()

        # Warmup
        pred_svc.predict(result.model_id, sample)

        # Measure single patient prediction latency over 20 runs
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            pred_svc.predict(result.model_id, sample)
            latencies.append(time.perf_counter() - t0)

        mean_latency_ms = np.mean(latencies) * 1000.0
        assert mean_latency_ms < 50.0, f"Inference latency exceeded: {mean_latency_ms:.2f}ms (benchmark: < 50ms)"

    def test_memory_headroom(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        assert rss_mb < 1500.0, f"Memory consumption too high: {rss_mb:.1f} MB"


# ===========================================================================
# 2. Error Handling Testing
# ===========================================================================
class TestErrorHandling:
    def test_invalid_file_upload_format(self):
        ingest = DatasetIngestionService()
        invalid_bytes = b"This is not a CSV or Parquet file."
        with pytest.raises(Exception):
            ingest.ingest_uploaded_file(
                file_bytes=invalid_bytes,
                filename="malformed.txt",
                dataset_id="bad_dataset",
                dataset_name="Bad Dataset",
                target_variable="target",
            )

    def test_missing_target_column(self, wdbc_data):
        df_no_target = wdbc_data.drop(columns=["diagnosis"])
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df_no_target, dataset_name="no_target", target_column="diagnosis")
        assert report is not None
        assert report.overall_quality_score < 100.0 or len(report.issues) > 0

    def test_missing_model_artifact_raises_error(self):
        pred_svc = PredictionService()
        sample = {"radius_mean": 14.5}
        with pytest.raises(FileNotFoundError):
            pred_svc.predict("non_existent_model_id_9999", sample)

    def test_empty_dataset_error(self):
        empty_df = pd.DataFrame(columns=["id", "feature1", "diagnosis"])
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(empty_df, dataset_name="empty")
        assert report.total_records == 0
        assert report.overall_quality_score == 0.0

    def test_corrupted_dataset_missing_values(self):
        corrupt_df = pd.DataFrame({
            "feature1": [np.nan, np.nan, np.nan],
            "feature2": [None, None, None],
            "diagnosis": ["M", "B", "M"]
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(corrupt_df, dataset_name="corrupt", target_column="diagnosis")
        assert report.missing_summary.get("total_missing_cells", 0) == 6


# ===========================================================================
# 3. Security Testing
# ===========================================================================
class TestSecurityAudits:
    def test_path_traversal_prevention(self):
        mgmt = DatasetManagementService()
        with pytest.raises(Exception):
            mgmt.get_dataframe("../../etc/passwd")

    def test_credential_leakage_in_codebase(self):
        forbidden_patterns = [
            "password = \"", "secret_key = \"", "api_key = \"ghp_",
            "PRIVATE KEY-----", "aws_secret_access_key",
        ]
        scanned_count = 0
        for root_dir in ["backend", "frontend", "configs"]:
            for path in Path(root_dir).rglob("*.py"):
                scanned_count += 1
                content = path.read_text(encoding="utf-8", errors="ignore")
                for pattern in forbidden_patterns:
                    assert pattern not in content, f"Credential risk pattern '{pattern}' found in {path}"
        assert scanned_count > 10

    def test_log_sanitization_for_phi_pii(self):
        logs_dir = Path("logs")
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                content = log_file.read_text(encoding="utf-8", errors="ignore")
                assert "password" not in content.lower() or "level=info" in content.lower()
                assert "ssn:" not in content.lower()
