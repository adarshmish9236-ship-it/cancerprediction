"""
CANCERWISE AI — Base Connector Interface
Abstract base class for all biomedical dataset connectors.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from backend.schemas.dataset import DatasetMetadata, DataModality, DataFormat


class BaseConnector(ABC):
    """
    Abstract interface for acquiring, ingesting, and parsing biomedical datasets.
    """

    def __init__(self, name: str, modality: DataModality, data_format: DataFormat):
        self.name = name
        self.modality = modality
        self.data_format = data_format

    @abstractmethod
    def fetch_or_download(self, dest_dir: Path, force: bool = False) -> Path:
        """
        Download or acquire the raw dataset files into `dest_dir`.
        Returns the primary raw file path.
        """
        ...

    @abstractmethod
    def load_and_parse(self, raw_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parse the raw dataset into a clean pandas DataFrame and extracted metadata attributes.
        """
        ...

    @abstractmethod
    def create_metadata(
        self,
        df: pd.DataFrame,
        raw_path: Path,
        processed_path: Optional[Path] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> DatasetMetadata:
        """
        Generate standardized DatasetMetadata instance for this dataset.
        """
        ...

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()
