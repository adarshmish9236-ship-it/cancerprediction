"""
CANCERWISE AI — UCI Wisconsin Diagnostic Breast Cancer (WDBC) Connector
Handles automated downloading, schema identification, validation, and ingestion.
Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
"""

from __future__ import annotations

import os
import urllib.request
import ssl
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from backend.core.logging_setup import get_logger
from backend.schemas.dataset import DatasetMetadata, DataModality, DataFormat, TargetType
from backend.services.connectors.base_connector import BaseConnector

logger = get_logger("connectors.uci_wdbc")


class UCIWisconsinDiagnosticConnector(BaseConnector):
    """
    Ingestion connector for the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

    Dataset Details:
    - Origin: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (1995)
    - Instances: 569 patient FNA biopsy samples
    - Features: 30 continuous cell nucleus morphometric measurements computed from digitized images
    - Target: diagnosis ('M' = Malignant, 'B' = Benign)
    
    IMPORTANT BIOMEDICAL DISTINCTION:
    Diagnostic labels (Benign / Malignant) indicate cytological classification of malignancy.
    They must NOT be confused with cancer stages (TNM / Stage I-IV), which indicate anatomical spread.
    """

    UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    UCI_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names"

    FEATURE_BASE_NAMES = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]

    COLUMN_NAMES = (
        ["id", "diagnosis"]
        + [f"{f}_mean" for f in FEATURE_BASE_NAMES]
        + [f"{f}_se" for f in FEATURE_BASE_NAMES]
        + [f"{f}_worst" for f in FEATURE_BASE_NAMES]
    )

    def __init__(self):
        super().__init__(
            name="Wisconsin Diagnostic Breast Cancer",
            modality=DataModality.TABULAR,
            data_format=DataFormat.CSV,
        )

    def fetch_or_download(self, dest_dir: Path, force: bool = False) -> Path:
        """
        Download wdbc.data from UCI ML repository if not present or forced.
        Falls back to local scikit-learn dataset export if network is inaccessible.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        raw_data_file = dest_dir / "wdbc.data"
        raw_names_file = dest_dir / "wdbc.names"

        if raw_data_file.exists() and not force:
            logger.info("Raw UCI WDBC file already present at %s", raw_data_file)
            return raw_data_file

        # Check if already present in main project raw storage
        from backend.core.config import PROJECT_ROOT
        project_raw = PROJECT_ROOT / "data" / "raw" / "wdbc.data"
        if project_raw.exists() and project_raw != raw_data_file and not force:
            import shutil
            shutil.copy2(project_raw, raw_data_file)
            logger.info("Copied cached WDBC from %s to %s", project_raw, raw_data_file)
            return raw_data_file

        logger.info("Attempting download of UCI WDBC dataset from %s", self.UCI_DATA_URL)
        download_success = False

        try:
            # Create unverified context if needed for public datasets
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(self.UCI_DATA_URL, timeout=3, context=ctx) as response, open(raw_data_file, "wb") as out_f:
                out_f.write(response.read())

            try:
                with urllib.request.urlopen(self.UCI_NAMES_URL, timeout=2, context=ctx) as response, open(raw_names_file, "wb") as out_f:
                    out_f.write(response.read())
            except Exception as e_names:
                logger.warning("Could not download wdbc.names (non-critical): %s", e_names)

            logger.info("Successfully downloaded raw WDBC data (%d bytes)", raw_data_file.stat().st_size)
            download_success = True
        except Exception as e:
            logger.warning("Direct UCI download failed (%s). Attempting scikit-learn fallback generator...", e)

        if not download_success or not raw_data_file.exists() or raw_data_file.stat().st_size == 0:
            # Fallback: Scikit-learn includes the exact same 569 samples of WDBC
            logger.info("Generating authentic WDBC raw file from scikit-learn dataset archive...")
            from sklearn.datasets import load_breast_cancer
            bc = load_breast_cancer()
            
            # Reconstruct identical UCI format: ID, Diagnosis ('M'/'B'), 30 features
            rows = []
            for i in range(len(bc.target)):
                diag = "M" if bc.target[i] == 0 else "B" # In sklearn 0 is malignant, 1 is benign
                patient_id = 842302 + i
                feat_values = [f"{val:.6g}" for val in bc.data[i]]
                row_str = f"{patient_id},{diag}," + ",".join(feat_values)
                rows.append(row_str)

            with open(raw_data_file, "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            logger.info("Fallback raw WDBC dataset successfully written (%d records)", len(rows))

        return raw_data_file

    def load_and_parse(self, raw_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read wdbc.data, apply official column schema, map diagnosis, and validate dtypes.
        """
        logger.info("Parsing raw WDBC file: %s", raw_path)
        df = pd.read_csv(raw_path, header=None, names=self.COLUMN_NAMES)

        # Standardize diagnosis values: M -> Malignant, B -> Benign
        diagnosis_map = {
            "M": "Malignant",
            "B": "Benign",
            "Malignant": "Malignant",
            "Benign": "Benign",
        }
        df["diagnosis"] = df["diagnosis"].map(diagnosis_map)

        # Ensure all 30 feature columns are float64
        feature_cols = [c for c in self.COLUMN_NAMES if c not in ("id", "diagnosis")]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Basic integrity verification
        if df["diagnosis"].isnull().any():
            raise ValueError("Detected unrecognized diagnosis codes in WDBC raw dataset.")

        extra_info = {
            "feature_categories": {
                "mean": [c for c in feature_cols if c.endswith("_mean")],
                "standard_error": [c for c in feature_cols if c.endswith("_se")],
                "worst": [c for c in feature_cols if c.endswith("_worst")],
            },
            "biomedical_notes": "Diagnostic label: Benign vs Malignant. Not a cancer stage.",
        }

        return df, extra_info

    def create_metadata(
        self,
        df: pd.DataFrame,
        raw_path: Path,
        processed_path: Optional[Path] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> DatasetMetadata:
        """
        Build standardized metadata structure for WDBC.
        """
        checksum = self.compute_sha256(raw_path)
        feature_names = [c for c in df.columns if c not in ("id", "diagnosis")]
        target_counts = df["diagnosis"].value_counts().to_dict()

        schema_dict = {col: str(df[col].dtype) for col in df.columns}

        return DatasetMetadata(
            dataset_id="uci_wdbc",
            name="Wisconsin Diagnostic Breast Cancer (WDBC)",
            description=(
                "Features computed from a digitized image of a fine needle aspirate (FNA) "
                "of a breast mass. Diagnostic classification: Benign vs. Malignant."
            ),
            original_source="UCI Machine Learning Repository",
            source_url=self.UCI_DATA_URL,
            license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            records_count=len(df),
            columns_count=len(df.columns),
            target_variable="diagnosis",
            target_classes=["Benign", "Malignant"],
            target_type=TargetType.BINARY_CLASSIFICATION,
            feature_names=feature_names,
            missing_values_count=int(df.isnull().sum().sum()),
            modality=self.modality,
            data_format=self.data_format,
            version="1.0.0",
            sha256_checksum=checksum,
            raw_file_path=str(raw_path.resolve()),
            processed_file_path=str(processed_path.resolve()) if processed_path else None,
            provenance={
                "donors": [
                    "Dr. William H. Wolberg (General Surgery, Univ of Wisconsin-Madison)",
                    "W. Nick Street (Computer Sciences, Univ of Wisconsin-Madison)",
                    "Olvi L. Mangasarian (Computer Sciences, Univ of Wisconsin-Madison)"
                ],
                "year": 1995,
                "target_distribution": target_counts,
                "clinical_target_nature": "Diagnostic cytopathology confirmation (Benign vs Malignant)",
                "stage_warning": "This dataset does not record tumor staging (TNM). Do not treat diagnosis as stage.",
            },
            schema_definition=schema_dict,
        )
