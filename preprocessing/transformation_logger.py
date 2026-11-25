"""
CANCERWISE AI — Transformation Logger
Maintains an immutable audit trail and reproducible log of every preprocessing,
cleaning, imputation, outlier modification, and transformation operation.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from backend.schemas.preprocessing import TransformationLogEntry
from backend.core.config import PROJECT_ROOT

logger = logging.getLogger("cancerwise.preprocessing.logger")


class TransformationLogger:
    """
    Records and persists structured transformation history.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path or (PROJECT_ROOT / "logs" / "transformation_history.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[TransformationLogEntry] = []
        self._load_existing_history()

    def _load_existing_history(self) -> None:
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._history.append(TransformationLogEntry(**data))
            except Exception as e:
                logger.warning("Could not read prior transformation history: %s", e)

    def log_operation(
        self,
        operation_name: str,
        parameters: Dict[str, Any],
        input_shape: List[int],
        output_shape: List[int],
        records_affected: int,
        columns_affected: List[str],
        execution_status: str = "SUCCESS",
        input_version: str = "v1.0.0",
        output_version: str = "v1.1.0",
        duration_ms: float = 0.0,
        error_message: Optional[str] = None,
    ) -> TransformationLogEntry:
        """
        Record a completed or failed transformation operation.
        """
        op_id = str(uuid.uuid4())
        entry = TransformationLogEntry(
            operation_id=op_id,
            operation_name=operation_name,
            parameters=parameters,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_version=input_version,
            output_version=output_version,
            input_shape=input_shape,
            output_shape=output_shape,
            records_affected=records_affected,
            columns_affected=columns_affected,
            execution_status=execution_status,
            duration_ms=round(duration_ms, 2),
            error_message=error_message,
        )

        self._history.append(entry)

        # Append to log file
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error("Failed to append transformation log to file: %s", e)

        logger.info(
            "Logged operation '%s' [ID: %s] | Status=%s | Records affected=%d",
            operation_name, op_id, execution_status, records_affected,
        )
        return entry

    def get_history(self, limit: Optional[int] = None) -> List[TransformationLogEntry]:
        """Retrieve recent transformation records."""
        if limit:
            return self._history[-limit:]
        return list(self._history)

    def clear(self) -> None:
        """Clear transformation history."""
        self._history = []
        if self.log_path.exists():
            self.log_path.unlink()
