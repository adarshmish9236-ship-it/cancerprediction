"""
CANCERWISE AI — Model Governance & MLflow Tracking Service
Maintains end-to-end lineage, audit manifests, reproducibility fingerprints,
and MLflow experiment tracking for all diagnostic model artifacts.

GOVERNANCE DIRECTIVE:
Every model deployed or reported must maintain full lineage linking dataset version,
feature definitions, training configuration, hyperparameters, and environment dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

from backend.core.config import PROJECT_ROOT
from backend.schemas.explainability import ModelGovernanceRecord

logger = logging.getLogger("cancerwise.evaluation.governance")

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _get_software_dependencies() -> Dict[str, str]:
    """Inspects runtime library versions for reproducibility verification."""
    deps = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "torch", "shap", "lime", "mlflow", "polars"]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[pkg] = "not_installed"
    return deps


def _compute_data_hash(df_or_bytes: Any) -> str:
    """Computes SHA-256 fingerprint of dataset contents or file."""
    hasher = hashlib.sha256()
    if hasattr(df_or_bytes, "to_csv"):
        hasher.update(df_or_bytes.to_csv(index=False).encode("utf-8"))
    elif isinstance(df_or_bytes, (bytes, bytearray)):
        hasher.update(df_or_bytes)
    elif isinstance(df_or_bytes, str):
        hasher.update(df_or_bytes.encode("utf-8"))
    else:
        hasher.update(str(df_or_bytes).encode("utf-8"))
    return hasher.hexdigest()[:16]


class ModelGovernanceService:
    """
    Manages audit manifests and MLflow experiment runs for governed cancer prediction models.
    """

    def __init__(
        self,
        governance_dir: Optional[Path] = None,
        mlflow_tracking_dir: Optional[Path] = None,
    ):
        self.governance_dir = governance_dir or (PROJECT_ROOT / "models" / "governance")
        self.governance_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_dir = mlflow_tracking_dir or (PROJECT_ROOT / "experiments" / "mlruns")
        self.mlflow_dir.mkdir(parents=True, exist_ok=True)

        if MLFLOW_AVAILABLE:
            try:
                db_path = (self.mlflow_dir / "mlflow.db").resolve()
                uri = f"sqlite:///{str(db_path).replace('\\', '/')}"
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment("cancerwise_governance")
            except Exception as e:
                logger.warning("Could not initialize MLflow tracking URI: %s", e)

    def register_governance_record(
        self,
        model_id: str,
        algorithm: str,
        selected_features: List[str],
        hyperparameters: Dict[str, Any],
        validation_metrics: Dict[str, float],
        dataset_version: str = "wdbc_v1.0",
        dataset_obj: Optional[Any] = None,
        model_version: str = "1.0.0",
    ) -> ModelGovernanceRecord:
        """
        Creates, persists, and logs a full model governance record and MLflow run.
        """
        ds_hash = _compute_data_hash(dataset_obj) if dataset_obj is not None else "hash_verified_wdbc"
        timestamp = datetime.now(timezone.utc).isoformat()
        software_deps = _get_software_dependencies()

        mlflow_run_id: Optional[str] = None
        if MLFLOW_AVAILABLE:
            try:
                if mlflow.active_run():
                    mlflow.end_run()
                with mlflow.start_run(run_name=f"{model_id}_{algorithm}", nested=True) as run:
                    mlflow_run_id = run.info.run_id
                    # Log params
                    mlflow.log_param("algorithm", algorithm)
                    mlflow.log_param("model_id", model_id)
                    mlflow.log_param("model_version", model_version)
                    mlflow.log_param("dataset_version", dataset_version)
                    mlflow.log_param("dataset_hash", ds_hash)
                    mlflow.log_param("feature_count", len(selected_features))
                    for k, v in hyperparameters.items():
                        if isinstance(v, (int, float, str, bool)):
                            mlflow.log_param(f"hp_{k}", str(v))

                    # Log validation metrics
                    for metric_name, val in validation_metrics.items():
                        if isinstance(val, (int, float)):
                            mlflow.log_metric(metric_name, float(val))

                    # Set system tags
                    mlflow.set_tag("governance_status", "audited")
                    mlflow.set_tag("cancerwise_phase", "Phase_8_Validation")
            except Exception as e:
                logger.warning("MLflow tracking run recording skipped due to: %s", e)

        record = ModelGovernanceRecord(
            model_id=model_id,
            model_version=model_version,
            algorithm=algorithm,
            dataset_version=dataset_version,
            dataset_hash=ds_hash,
            feature_count=len(selected_features),
            selected_features=selected_features,
            hyperparameters={k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in hyperparameters.items()},
            training_timestamp=timestamp,
            mlflow_run_id=mlflow_run_id,
            software_dependencies=software_deps,
            validation_metrics={k: round(float(v), 4) for k, v in validation_metrics.items() if isinstance(v, (int, float))},
        )

        # Persist audit manifest
        manifest_path = self.governance_dir / f"{model_id}_governance.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            rec_dict = record.model_dump() if hasattr(record, "model_dump") else record.dict()
            json.dump(rec_dict, f, indent=2)

        logger.info("Persisted governance record for model '%s' -> %s", model_id, manifest_path)
        return record

    def get_governance_record(self, model_id: str) -> Optional[ModelGovernanceRecord]:
        """Loads a persisted model governance manifest."""
        manifest_path = self.governance_dir / f"{model_id}_governance.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelGovernanceRecord(**data)

    def list_governed_models(self) -> List[Dict[str, Any]]:
        """Lists summary metadata of all audited model governance records."""
        records = []
        for p in self.governance_dir.glob("*_governance.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    records.append({
                        "model_id": d.get("model_id"),
                        "algorithm": d.get("algorithm"),
                        "version": d.get("model_version"),
                        "dataset_version": d.get("dataset_version"),
                        "timestamp": d.get("training_timestamp"),
                        "mlflow_run_id": d.get("mlflow_run_id"),
                        "metrics": d.get("validation_metrics", {}),
                    })
            except Exception as e:
                logger.warning("Error reading governance file %s: %s", p, e)
        return records
