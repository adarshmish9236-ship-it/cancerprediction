"""
CANCERWISE AI — Dataset Management & Ingestion API Endpoints
Provides versioned routes for dataset upload, discovery, inspection, deletion, and profiling.
All routes delegate to dedicated service classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status, Depends

from backend.core.logging_setup import get_logger
from backend.schemas.dataset import (
    DatasetMetadata,
    DatasetPreview,
    DatasetStatisticsSummary,
    DatasetValidationReport,
    SchemaCompatibilityReport,
)
from backend.services.dataset_management_service import DatasetManagementService
from backend.services.dataset_ingestion_service import DatasetIngestionService
from backend.services.data_validation_service import DataValidationService

logger = get_logger("api.datasets")
router = APIRouter(prefix="/datasets", tags=["Dataset Management & Ingestion"])

_mgmt_service = DatasetManagementService()
_ingest_service = DatasetIngestionService()
_val_service = DataValidationService()


# ---------------------------------------------------------------------------
# Section 2 Dataset APIs
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=DatasetMetadata, status_code=status.HTTP_201_CREATED, summary="Upload Dataset")
def upload_dataset(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None),
    dataset_name: Optional[str] = Form(None),
    target_variable: Optional[str] = Form(None),
):
    """
    Upload a custom cancer dataset (CSV, XLSX, Parquet, or JSON).
    Performs secure file validation, size limit checks, safe filename sanitization, and registration.
    """
    content = file.file.read()
    filename = file.filename or "uploaded_dataset.csv"
    metadata = _ingest_service.ingest_uploaded_file(
        file_bytes=content,
        filename=filename,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        target_variable=target_variable,
    )
    return metadata



@router.get("", response_model=List[DatasetMetadata], summary="List All Datasets")
def list_datasets():
    """List all registered datasets with metadata summaries."""
    return _mgmt_service.list_datasets()


@router.get("/{dataset_id}", response_model=DatasetMetadata, summary="Get Dataset Metadata")
def get_dataset_metadata(dataset_id: str):
    """Retrieve full metadata for a specific dataset."""
    return _mgmt_service.get_metadata(dataset_id)


@router.delete("/{dataset_id}", summary="Delete Dataset")
def delete_dataset(dataset_id: str):
    """Deletes dataset raw, processed, and metadata artifacts from platform."""
    return _mgmt_service.delete_dataset(dataset_id)


@router.get("/{dataset_id}/profile", summary="Get Dataset Profile")
def get_dataset_profile(dataset_id: str):
    """
    Comprehensive structural and quality profile combining dimensions, types,
    missingness percentages, duplicates, memory footprint, and column distributions.
    """
    return _mgmt_service.get_profile(dataset_id)


# ---------------------------------------------------------------------------
# Ingestion & Validation Utility Endpoints
# ---------------------------------------------------------------------------
@router.post("/ingest/wdbc", response_model=DatasetMetadata, status_code=status.HTTP_201_CREATED, summary="Ingest UCI WDBC Cohort")
def ingest_uci_wdbc(force: bool = False):
    """Ingest primary Wisconsin Diagnostic Breast Cancer cohort from UCI."""
    return _ingest_service.ingest_wdbc(force=force)


@router.post("/ingest/{connector_key}", response_model=DatasetMetadata, status_code=status.HTTP_201_CREATED, summary="Ingest via Connector")
def ingest_connector(connector_key: str, force: bool = False):
    """Ingest dataset via registered multi-modal connector."""
    try:
        return _ingest_service.ingest_connector(connector_key=connector_key, force=force)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sources", summary="List Data Sources")
def list_data_sources():
    """List trusted biomedical data sources registered in the platform."""
    return _ingest_service.list_sources()


@router.get("/{dataset_id}/preview", response_model=DatasetPreview, summary="Preview Dataset")
def get_dataset_preview(dataset_id: str, n_rows: int = 5):
    """Preview dataset head, tail, dimensions, and dtypes."""
    return _mgmt_service.get_preview(dataset_id, n_rows=n_rows)


@router.get("/{dataset_id}/statistics", response_model=DatasetStatisticsSummary, summary="Get Dataset Statistics")
def get_dataset_statistics(dataset_id: str):
    """Descriptive statistics, null counts, duplicate counts, and class distributions."""
    return _mgmt_service.get_statistics(dataset_id)


@router.get("/{dataset_id}/validate", response_model=DatasetValidationReport, summary="Validate Dataset Integrity")
def validate_dataset(dataset_id: str):
    """Run data integrity, schema consistency, and checksum verification."""
    from backend.services.dataset_service import DatasetService
    srv = DatasetService()
    report = srv.validate_dataset(dataset_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return report


@router.get("/dictionary/{name}", summary="Get Data Dictionary")
def get_data_dictionary(name: str = "wdbc"):
    """Retrieve attribute definitions and clinical target semantics."""
    from backend.services.dataset_service import DatasetService
    srv = DatasetService()
    return srv.get_data_dictionary(name)


@router.post("/check-compatibility", response_model=SchemaCompatibilityReport, summary="Check Dataset Schema Compatibility")
def check_schema_compatibility(dataset_a_id: str, dataset_b_id: str):
    """Evaluate schema, modality, units, and label compatibility between two datasets."""
    return _val_service.check_compatibility(dataset_a_id, dataset_b_id)
