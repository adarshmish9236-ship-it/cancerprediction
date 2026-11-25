"""
CANCERWISE AI — Phase 0 Verification Suite
Verifies that all components of Phase 0 are correctly initialized.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_DIRS = [
    "backend/api", "backend/core", "backend/services", "backend/schemas",
    "backend/middleware",
    "frontend/pages", "frontend/components", "frontend/layouts", "frontend/assets",
    "data/raw", "data/interim", "data/processed", "data/external", "data/metadata",
    "data_mining/classification", "data_mining/regression", "data_mining/semi_supervised",
    "data_mining/multitask", "data_mining/multiclass", "data_mining/one_class",
    "data_mining/structured_output", "data_mining/kernel_learning", "data_mining/large_scale",
    "preprocessing", "feature_engineering", "evaluation", "explainability",
    "models/trained", "models/checkpoints", "models/registry",
    "experiments", "reports", "notebooks", "scripts", "tests",
    "configs", "logs", "docs",
]

REQUIRED_FILES = [
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "README.md",
    "PROJECT_ROADMAP.md",
    "backend/main.py",
    "backend/core/config.py",
    "backend/core/logging_setup.py",
    "backend/core/base_model.py",
    "frontend/app.py",
    "preprocessing/pipeline.py",
    "evaluation/metrics.py",
    "configs/settings.yaml",
    "configs/logging_config.yaml",
    "docs/SYSTEM_ARCHITECTURE.md",
    "docs/TECHNOLOGY_STACK.md",
    "docs/INSTALLATION_GUIDE.md",
    "docs/DEVELOPMENT_GUIDELINES.md",
]


# ============================================================
# 1. Structure Tests
# ============================================================
class TestProjectStructure:

    @pytest.mark.parametrize("rel_path", REQUIRED_DIRS)
    def test_directory_exists(self, rel_path: str) -> None:
        d = PROJECT_ROOT / rel_path
        assert d.is_dir(), f"Missing directory: {rel_path}"

    @pytest.mark.parametrize("rel_path", REQUIRED_FILES)
    def test_file_exists(self, rel_path: str) -> None:
        f = PROJECT_ROOT / rel_path
        assert f.is_file(), f"Missing file: {rel_path}"


# ============================================================
# 2. Configuration Tests
# ============================================================
class TestConfiguration:

    def test_config_loads(self) -> None:
        from backend.core.config import get_config
        config = get_config()
        assert config is not None

    def test_config_project_name(self) -> None:
        from backend.core.config import get_config
        config = get_config()
        assert "CANCERWISE" in str(config.get("project", "name")).upper()

    def test_config_api_port(self) -> None:
        from backend.core.config import get_config
        config = get_config()
        port = config.get("api", "port")
        assert isinstance(port, int) and port > 0

    def test_config_singleton(self) -> None:
        from backend.core.config import get_config
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2


# ============================================================
# 3. Logging Tests
# ============================================================
class TestLogging:

    def test_logging_setup(self) -> None:
        from backend.core.logging_setup import setup_logging
        setup_logging(force=True)
        logger = logging.getLogger("cancerwise.test")
        assert logger is not None

    def test_get_logger(self) -> None:
        from backend.core.logging_setup import get_logger
        logger = get_logger("test.phase0")
        assert logger.name.startswith("cancerwise")

    def test_logging_writes(self, tmp_path: Path) -> None:
        from backend.core.logging_setup import setup_logging
        setup_logging(log_dir=tmp_path, force=True)
        logger = logging.getLogger("cancerwise.test.write")
        logger.info("Phase 0 logging test message")
        log_file = tmp_path / "cancerwise_app.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Phase 0 logging test message" in content


# ============================================================
# 4. Base Model Interface Tests
# ============================================================
class TestBaseModel:

    def test_base_model_is_abstract(self) -> None:
        from backend.core.base_model import BaseModel
        with pytest.raises(TypeError):
            BaseModel(name="test")  # Cannot instantiate abstract class

    def test_base_model_concrete_subclass(self) -> None:
        from backend.core.base_model import BaseModel
        import numpy as np

        class DummyModel(BaseModel):
            def fit(self, X, y, **kw): self._is_fitted = True; return self
            def predict(self, X): return np.zeros(len(X))
            def save(self, path): pass
            @classmethod
            def load(cls, path): return cls("dummy")

        m = DummyModel(name="dummy")
        assert not m.is_fitted
        m.fit(np.zeros((10, 5)), np.zeros(10))
        assert m.is_fitted


# ============================================================
# 5. Preprocessing Tests
# ============================================================
class TestPreprocessing:

    def test_preprocessor_imports(self) -> None:
        from preprocessing.pipeline import CancerwisePreprocessor
        assert CancerwisePreprocessor is not None

    def test_preprocessor_fit_transform(self) -> None:
        import numpy as np
        import pandas as pd
        from preprocessing.pipeline import CancerwisePreprocessor

        df = pd.DataFrame({
            "age": [45, 52, None, 38, 61],
            "radius": [14.2, 20.1, 12.5, None, 17.8],
        })
        pp = CancerwisePreprocessor(missing_strategy="median", scaling_strategy="standard")
        result = pp.fit_transform(df)
        assert result.shape == (5, 2)
        assert not np.any(np.isnan(result))


# ============================================================
# 6. Evaluation Tests
# ============================================================
class TestEvaluation:

    def test_evaluate_classifier(self) -> None:
        import numpy as np
        from evaluation.metrics import evaluate_classifier

        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.85, 0.6, 0.95, 0.8])

        metrics = evaluate_classifier(y_true, y_pred, y_proba)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0


# ============================================================
# 7. Installed Dependencies Tests
# ============================================================
class TestDependencies:

    @pytest.mark.parametrize("pkg_name", [
        "numpy", "scipy", "pandas", "polars", "sklearn", "xgboost",
        "lightgbm", "catboost", "torch", "torchvision", "shap", "lime",
        "imblearn", "mlflow", "fastapi", "uvicorn", "streamlit", "plotly",
        "matplotlib", "seaborn", "pytest", "pydantic", "yaml",
    ])
    def test_core_package_importable(self, pkg_name: str) -> None:
        import importlib
        mod = importlib.import_module(pkg_name)
        assert mod is not None
