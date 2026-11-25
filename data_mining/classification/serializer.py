"""
CANCERWISE AI — Model Serialization & Artifact Management
Persists trained pipelines, preprocessing transformations, selected feature lists,
hyperparameters, and clinical validation manifests.

CRITICAL RULE: Never serialize a trained model without its associated preprocessing pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from backend.core.config import PROJECT_ROOT

logger = logging.getLogger("cancerwise.classification.serializer")


class ModelSerializer:
    """
    Manages persistence and versioning of machine learning pipelines and metadata.
    """

    def __init__(
        self,
        trained_dir: Optional[Path] = None,
        registry_dir: Optional[Path] = None,
    ):
        self.trained_dir = trained_dir or (PROJECT_ROOT / "models" / "trained")
        self.registry_dir = registry_dir or (PROJECT_ROOT / "models" / "registry")
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model_id: str,
        pipeline: Any,
        selected_features: List[str],
        algorithm: str,
        hyperparameters: Dict[str, Any],
        test_metrics: Dict[str, Any],
        training_config: Dict[str, Any],
        version: str = "v1.0.0",
    ) -> Path:
        """
        Saves the fitted pipeline bundle (.joblib) and writes manifest JSON.
        """
        if not hasattr(pipeline, "predict"):
            raise ValueError("Pipeline object must provide a 'predict' method.")

        bundle = {
            "model_id": model_id,
            "version": version,
            "algorithm": algorithm,
            "pipeline": pipeline,
            "selected_features": selected_features,
            "hyperparameters": hyperparameters,
            "test_metrics": test_metrics,
            "training_config": training_config,
        }

        artifact_file = self.trained_dir / f"{model_id}.joblib"
        joblib.dump(bundle, artifact_file, compress=3)

        manifest = {
            "model_id": model_id,
            "version": version,
            "algorithm": algorithm,
            "artifact_file": str(artifact_file.name),
            "selected_features": selected_features,
            "num_features": len(selected_features),
            "hyperparameters": hyperparameters,
            "test_metrics": test_metrics,
            "training_config": training_config,
            "file_size_kb": round(artifact_file.stat().st_size / 1024.0, 2),
        }

        manifest_file = self.registry_dir / f"{model_id}.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info("Saved model artifact '%s' (%s KB)", model_id, manifest["file_size_kb"])
        return artifact_file

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Loads serialized model bundle from disk."""
        artifact_file = self.trained_dir / f"{model_id}.joblib"
        if not artifact_file.exists():
            # Check with .joblib extension if already appended
            if Path(model_id).exists():
                artifact_file = Path(model_id)
            else:
                raise FileNotFoundError(f"Model artifact '{model_id}' does not exist.")

        bundle = joblib.load(artifact_file)
        return bundle

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists all registered models from manifest files."""
        models: List[Dict[str, Any]] = []
        for manifest_file in self.registry_dir.glob("*.json"):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    models.append(manifest)
            except Exception as e:
                logger.warning("Could not read manifest '%s': %s", manifest_file, e)
        return models

    def get_manifest(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Loads manifest metadata for a given model ID."""
        manifest_file = self.registry_dir / f"{model_id}.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def delete_model(self, model_id: str) -> bool:
        """Deletes model binary and manifest."""
        artifact_file = self.trained_dir / f"{model_id}.joblib"
        manifest_file = self.registry_dir / f"{model_id}.json"
        deleted = False
        if artifact_file.exists():
            artifact_file.unlink()
            deleted = True
        if manifest_file.exists():
            manifest_file.unlink()
            deleted = True
        return deleted
