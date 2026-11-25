"""
CANCERWISE AI — Base Model Interface
All ML models implement this abstract interface for consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import logging

import numpy as np

logger = logging.getLogger("cancerwise.models.base")


class BaseModel(ABC):
    """
    Abstract base class for all CANCERWISE AI models.

    Every classifier, regressor, or ensemble must inherit from this class
    and implement the required interface methods.
    """

    def __init__(self, name: str, random_state: int = 42) -> None:
        self.name = name
        self.random_state = random_state
        self._model: Any = None
        self._is_fitted: bool = False
        self._feature_names: Optional[list[str]] = None
        self._metadata: dict = {
            "name": name,
            "version": "0.1.0",
            "phase": 0,
        }

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseModel":
        """Train the model on the provided data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (optional, override if supported)."""
        raise NotImplementedError(f"{self.name} does not support predict_proba.")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk."""
        ...

    def get_feature_names(self) -> Optional[list[str]]:
        return self._feature_names

    def set_feature_names(self, names: list[str]) -> None:
        self._feature_names = names

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def metadata(self) -> dict:
        return self._metadata

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"
