"""
CANCERWISE AI — Centralized Logging Setup
Configures structured logging for the entire platform.
"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOGGING_CONFIG_FILE = PROJECT_ROOT / "configs" / "logging_config.yaml"

_initialized = False


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    config_file: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Initialize logging for CANCERWISE AI.

    Priority:
      1. configs/logging_config.yaml (if present)
      2. Programmatic fallback setup

    Args:
        level:       Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir:     Directory for log files. Defaults to <project_root>/logs.
        config_file: Path to logging YAML config. Defaults to configs/logging_config.yaml.
        force:       Re-initialize even if already set up.
    """
    global _initialized
    if _initialized and not force:
        return

    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = config_file or LOGGING_CONFIG_FILE

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Patch file handler paths to absolute log_dir paths
        for handler in cfg.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = str(log_dir / Path(handler["filename"]).name)

        # Remove json formatter if python-json-logger not installed
        formatters = cfg.get("formatters", {})
        if "json" in formatters:
            try:
                import pythonjsonlogger  # noqa: F401
            except ImportError:
                del formatters["json"]

        logging.config.dictConfig(cfg)
    else:
        # Fallback: simple programmatic config
        _setup_fallback(level, log_dir)

    _initialized = True
    logging.getLogger("cancerwise").info(
        "CANCERWISE AI logging initialized | level=%s | log_dir=%s", level, log_dir
    )


def _setup_fallback(level: str, log_dir: Path) -> None:
    """Minimal fallback logging config."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    cw_logger = logging.getLogger("cancerwise")
    cw_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    cw_logger.propagate = False

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    cw_logger.addHandler(ch)

    # File
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "cancerwise_app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    cw_logger.addHandler(fh)

    # Error file
    eh = logging.handlers.RotatingFileHandler(
        log_dir / "cancerwise_errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    cw_logger.addHandler(eh)


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the 'cancerwise' hierarchy.

    Usage:
        from backend.core.logging_setup import get_logger
        logger = get_logger(__name__)
        logger.info("Hello from %s", __name__)
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(f"cancerwise.{name}" if not name.startswith("cancerwise") else name)
