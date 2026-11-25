"""
CANCERWISE AI — Phase 1 Test Suite: Data Collection & Ingestion Engine
Tests:
- UCI Wisconsin Diagnostic Breast Cancer (WDBC) ingestion & column standardization
- Diagnostic labels ('Benign', 'Malignant') vs cancer stage distinction
- DatasetRegistry, DatasetLoader, DatasetValidator, DatasetMetadataManager, DatasetVersionManager, DatasetStatistics
- Modality-specific connectors (Histopathology IMAGE, TCGA GENOMIC)
- Multi-format upload support (CSV, XLSX, Parquet, JSON) with secure validation
- Malformed file rejection with clear error handling
- Schema compatibility checker rules (modality, feature overlap, label definition)
- API endpoint integration
"""

from __future__ import annotations

import io
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from backend.main import app
from backend.schemas.dataset import DataModality, DataFormat, TargetType
from backend.services.dataset_service import DatasetService
from backend.services.connectors.uci_wdbc_connector import UCIWisconsinDiagnosticConnector
from backend.services.connectors.histopathology_connector import HistopathologyImageConnector
from backend.services.connectors.tcga_genomic_connector import TCGAGenomicConnector
from backend.services.upload_service import UploadValidationError
from backend.services.compatibility_checker import SchemaCompatibilityChecker
from backend.api.datasets import (
    ingest_uci_wdbc,
    list_datasets,
    get_dataset_preview,
    get_dataset_statistics,
    validate_dataset,
    get_data_dictionary,
    list_data_sources,
)


# ===========================================================================
# 1. UCI WDBC Connector Tests
# ===========================================================================
class TestUCIWDBCConnector:

    def test_wdbc_connector_instantiation(self):
        connector = UCIWisconsinDiagnosticConnector()
        assert connector.modality == DataModality.TABULAR
        assert connector.data_format == DataFormat.CSV
        assert len(connector.COLUMN_NAMES) == 32

    def test_wdbc_fetch_and_parse(self, tmp_path: Path):
        connector = UCIWisconsinDiagnosticConnector()
        raw_path = connector.fetch_or_download(tmp_path)
        assert raw_path.exists()
        assert raw_path.stat().st_size > 0

        df, extra_info = connector.load_and_parse(raw_path)
        assert len(df) == 569
        assert len(df.columns) == 32
        assert "id" in df.columns
        assert "diagnosis" in df.columns

        # Verify diagnostic label standardization: ONLY Benign and Malignant
        unique_targets = set(df["diagnosis"].unique())
        assert unique_targets == {"Benign", "Malignant"}
        assert "M" not in unique_targets
        assert "B" not in unique_targets

        # Confirm all 30 features are numeric float
        feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]
        assert len(feature_cols) == 30
        for col in feature_cols:
            assert pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col])

    def test_wdbc_metadata_generation(self, tmp_path: Path):
        connector = UCIWisconsinDiagnosticConnector()
        raw_path = connector.fetch_or_download(tmp_path)
        df, extra_info = connector.load_and_parse(raw_path)
        meta = connector.create_metadata(df, raw_path=raw_path, extra_info=extra_info)

        assert meta.dataset_id == "uci_wdbc"
        assert meta.records_count == 569
        assert meta.columns_count == 32
        assert meta.target_variable == "diagnosis"
        assert meta.target_classes == ["Benign", "Malignant"]
        assert len(meta.feature_names) == 30
        assert meta.sha256_checksum is not None
        assert len(meta.sha256_checksum) == 64


# ===========================================================================
# 2. Modality Connectors (Histopathology & TCGA Genomic)
# ===========================================================================
class TestMultiModalConnectors:

    def test_histopathology_image_connector(self, tmp_path: Path):
        connector = HistopathologyImageConnector()
        assert connector.modality == DataModality.IMAGE
        assert connector.data_format == DataFormat.IMAGE_DIRECTORY

        raw_path = connector.fetch_or_download(tmp_path)
        assert raw_path.exists()
        df, extra_info = connector.load_and_parse(raw_path)
        assert len(df) > 0
        assert "magnification" in df.columns
        assert "diagnosis" in df.columns

        meta = connector.create_metadata(df, raw_path)
        assert meta.modality == DataModality.IMAGE

    def test_tcga_genomic_connector(self, tmp_path: Path):
        connector = TCGAGenomicConnector()
        assert connector.modality == DataModality.GENOMIC

        raw_path = connector.fetch_or_download(tmp_path)
        assert raw_path.exists()
        df, extra_info = connector.load_and_parse(raw_path)
        assert len(df) > 0
        assert "pam50_subtype" in df.columns
        assert "BRCA1" in df.columns
        assert "TP53" in df.columns

        meta = connector.create_metadata(df, raw_path)
        assert meta.modality == DataModality.GENOMIC
        assert meta.target_type == TargetType.MULTICLASS_CLASSIFICATION


# ===========================================================================
# 3. Dataset Management Engine Tests
# ===========================================================================
class TestDatasetEngine:

    def test_dataset_service_ingest_and_registry(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        meta = service.ingest_wdbc()
        assert meta.dataset_id == "uci_wdbc"

        # Check retrieval from registry
        retrieved_df = service.get_dataset("uci_wdbc")
        assert retrieved_df is not None
        assert len(retrieved_df) == 569

        retrieved_meta = service.get_metadata("uci_wdbc")
        assert retrieved_meta is not None
        assert retrieved_meta.records_count == 569

    def test_dataset_statistics(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        service.ingest_wdbc()
        stats = service.get_statistics("uci_wdbc")

        assert stats is not None
        assert stats.records_count == 569
        assert stats.columns_count == 32
        assert stats.total_missing_values == 0
        assert stats.target_distribution is not None
        assert "Benign" in stats.target_distribution
        assert "Malignant" in stats.target_distribution
        assert stats.target_distribution["Benign"] == 357
        assert stats.target_distribution["Malignant"] == 212

    def test_dataset_validator(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        service.ingest_wdbc()
        val_report = service.validate_dataset("uci_wdbc")

        assert val_report is not None
        assert val_report.is_valid is True
        assert val_report.failed_checks == 0
        assert val_report.checksum_matched is True


# ===========================================================================
# 4. Multi-format Upload & Secure File Validation Tests
# ===========================================================================
class TestUploadService:

    def test_upload_csv(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        csv_data = b"patient_id,diagnosis,marker_a,marker_b\nP01,Benign,1.2,3.4\nP02,Malignant,5.6,7.8\nP03,Benign,2.1,4.3\n"
        meta = service.upload_dataset(
            file_bytes=csv_data,
            filename="cohort_study.csv",
            dataset_id="test_cohort",
            target_variable="diagnosis",
        )
        assert meta.dataset_id == "test_cohort"
        assert meta.records_count == 3
        assert meta.columns_count == 4
        assert meta.target_variable == "diagnosis"
        assert meta.data_format == DataFormat.CSV

    def test_upload_parquet(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        df_orig = pd.DataFrame({
            "sample_id": [101, 102, 103],
            "expression": [12.4, 9.8, 15.2],
            "subtype": ["LumA", "LumB", "Basal"],
        })
        buffer = io.BytesIO()
        df_orig.to_parquet(buffer, index=False)
        meta = service.upload_dataset(
            file_bytes=buffer.getvalue(),
            filename="expression.parquet",
            dataset_id="test_parquet",
        )
        assert meta.data_format == DataFormat.PARQUET
        assert meta.records_count == 3

    def test_upload_json(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        json_data = json.dumps([
            {"id": "A1", "score": 4.5, "label": "B"},
            {"id": "A2", "score": 8.9, "label": "M"},
        ]).encode("utf-8")
        meta = service.upload_dataset(
            file_bytes=json_data,
            filename="records.json",
            dataset_id="test_json",
        )
        assert meta.data_format == DataFormat.JSON
        assert meta.records_count == 2

    def test_upload_rejects_empty_file(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        with pytest.raises(UploadValidationError, match="empty"):
            service.upload_dataset(file_bytes=b"", filename="empty.csv")

    def test_upload_rejects_unsupported_format(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        with pytest.raises(UploadValidationError, match="Unsupported file format"):
            service.upload_dataset(file_bytes=b"some content", filename="bad.exe")

    def test_upload_rejects_malformed_csv(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        with pytest.raises(UploadValidationError):
            service.upload_dataset(
                file_bytes=b"col1\n1\n", # Only 1 column (minimum 2 required)
                filename="one_col.csv",
            )


# ===========================================================================
# 5. Schema Compatibility Checker Tests (Section 7 Integration Rules)
# ===========================================================================
class TestCompatibilityChecker:

    def test_incompatible_modalities(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        meta_tab = service.ingest_wdbc()
        meta_img = service.ingest_connector("histopathology_breakhis")

        report = service.check_compatibility(meta_tab.dataset_id, meta_img.dataset_id)
        assert report.compatible is False
        assert report.modality_match is False
        assert any("Modality mismatch" in r for r in report.reasons)

    def test_disjoint_features_rejected(self, tmp_path: Path):
        service = DatasetService(data_root=tmp_path)
        service.ingest_wdbc()

        csv_unrelated = b"blood_pressure,cholesterol,cardiac_risk\n120,200,Low\n140,250,High\n"
        service.upload_dataset(file_bytes=csv_unrelated, filename="cardiac.csv", dataset_id="cardiac_study")

        report = service.check_compatibility("uci_wdbc", "cardiac_study")
        assert report.compatible is False
        assert any("No shared feature columns" in r for r in report.reasons)


# ===========================================================================
# 6. REST API Endpoints Tests
# ===========================================================================
class TestDatasetAPI:

    def test_api_ingest_wdbc(self):
        metadata = ingest_uci_wdbc(force=False)
        assert metadata.dataset_id == "uci_wdbc"
        assert metadata.records_count == 569
        assert metadata.columns_count == 32
        assert metadata.target_variable == "diagnosis"

    def test_api_list_datasets(self):
        items = list_datasets()
        assert isinstance(items, list)
        assert any(d.dataset_id == "uci_wdbc" for d in items)

    def test_api_dataset_preview(self):
        preview = get_dataset_preview("uci_wdbc", n_rows=5)
        assert preview.dataset_id == "uci_wdbc"
        assert len(preview.head) == 5
        assert "diagnosis" in preview.columns

    def test_api_dataset_statistics(self):
        stats = get_dataset_statistics("uci_wdbc")
        assert stats.records_count == 569
        assert stats.target_distribution["Benign"] == 357
        assert stats.target_distribution["Malignant"] == 212

    def test_api_dataset_validation(self):
        val = validate_dataset("uci_wdbc")
        assert val.is_valid is True
        assert val.checksum_matched is True
        assert val.total_checks >= 5
        assert val.passed_checks == val.total_checks

    def test_api_data_dictionary(self):
        dictionary = get_data_dictionary("wdbc")
        assert len(dictionary["attributes"]) == 32
        assert "Benign" in dictionary["target_semantics"]
        assert "Malignant" in dictionary["target_semantics"]

    def test_api_sources(self):
        sources = list_data_sources()
        assert len(sources) >= 1
        assert any(s["name"] == "UCI Machine Learning Repository" for s in sources)
