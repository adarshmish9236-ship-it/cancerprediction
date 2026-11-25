"""
CANCERWISE AI — Breast Cancer Histopathology Image Connector
Extensible connector for microscopic tissue biopsy image collections (e.g., BreakHis).
Modality: IMAGE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from backend.core.logging_setup import get_logger
from backend.schemas.dataset import DatasetMetadata, DataModality, DataFormat, TargetType
from backend.services.connectors.base_connector import BaseConnector

logger = get_logger("connectors.histopathology")


class HistopathologyImageConnector(BaseConnector):
    """
    Connector for breast cancer histopathological whole-slide and patch image datasets.
    Supports BreakHis-style structures:
    - Magnification levels: 40X, 100X, 200X, 400X
    - Clinical classes: Benign (adenosis, fibroadenoma, tubular adenoma, phyllodes tumor)
                        Malignant (ductal carcinoma, lobular carcinoma, mucinous carcinoma, papillary carcinoma)
    """

    def __init__(self):
        super().__init__(
            name="Breast Cancer Histopathology Connector",
            modality=DataModality.IMAGE,
            data_format=DataFormat.IMAGE_DIRECTORY,
        )

    def fetch_or_download(self, dest_dir: Path, force: bool = False) -> Path:
        """
        Creates/verifies histopathology directory structure and sample manifest.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = dest_dir / "histopathology_manifest.json"

        if manifest_file.exists() and not force:
            return manifest_file

        # Create structured sample manifest template for histopathology
        manifest = {
            "connector": "HistopathologyImageConnector",
            "modality": "IMAGE",
            "supported_magnifications": ["40X", "100X", "200X", "400X"],
            "supported_subtypes": {
                "Benign": ["Adenosis", "Fibroadenoma", "TubularAdenoma", "PhyllodesTumor"],
                "Malignant": ["DuctalCarcinoma", "LobularCarcinoma", "MucinousCarcinoma", "PapillaryCarcinoma"]
            },
            "image_specifications": {
                "color_space": "RGB",
                "staining": "Hematoxylin & Eosin (H&E)",
                "formats": ["PNG", "TIFF", "JPG"]
            },
            "samples": [
                {
                    "slide_id": "SOB_B_A-14-22549AB",
                    "magnification": "40X",
                    "diagnosis": "Benign",
                    "subtype": "Adenosis",
                    "patient_id": "14-22549",
                    "pathology_notes": "Preserved glandular architecture without malignant cytological atypia"
                },
                {
                    "slide_id": "SOB_M_DC-14-20636",
                    "magnification": "100X",
                    "diagnosis": "Malignant",
                    "subtype": "DuctalCarcinoma",
                    "patient_id": "14-20636",
                    "pathology_notes": "Infiltrating ductal sheets with nuclear pleomorphism and mitotic activity"
                }
            ]
        }

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Initialized histopathology connector manifest at %s", manifest_file)
        return manifest_file

    def load_and_parse(self, raw_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse histopathology manifest into tabular metadata representation."""
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data.get("samples", []))
        extra_info = {
            "magnifications": data.get("supported_magnifications", []),
            "subtypes": data.get("supported_subtypes", {}),
            "staining": data.get("image_specifications", {}).get("staining", "H&E"),
        }
        return df, extra_info

    def create_metadata(
        self,
        df: pd.DataFrame,
        raw_path: Path,
        processed_path: Optional[Path] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> DatasetMetadata:
        checksum = self.compute_sha256(raw_path)
        return DatasetMetadata(
            dataset_id="histopathology_breakhis_connector",
            name="Breast Cancer Histopathology (H&E Biopsy Images)",
            description="Microscopic biopsy image manifest and patch dataset connector covering 40X-400X H&E images.",
            original_source="BreakHis / Clinical Pathology Repository",
            source_url="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
            license="Non-Commercial Research Use",
            records_count=len(df),
            columns_count=len(df.columns),
            target_variable="diagnosis",
            target_classes=["Benign", "Malignant"],
            target_type=TargetType.BINARY_CLASSIFICATION,
            feature_names=list(df.columns),
            missing_values_count=int(df.isnull().sum().sum()),
            modality=self.modality,
            data_format=self.data_format,
            version="1.0.0",
            sha256_checksum=checksum,
            raw_file_path=str(raw_path.resolve()),
            processed_file_path=str(processed_path.resolve()) if processed_path else None,
            provenance={
                "staining": "Hematoxylin & Eosin",
                "biomedical_type": "Microscopic Tissue Histopathology",
                "incompatible_with": "Direct feature-level concatenation with tabular clinical records"
            },
            schema_definition={col: str(df[col].dtype) for col in df.columns},
        )
