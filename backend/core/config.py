"""
CANCERWISE AI — Centralized Configuration Manager
Loads settings from YAML config file and environment variables.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env if it exists
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

logger = logging.getLogger("cancerwise.config")


def _resolve_path_vars(value: str, context: dict) -> str:
    """Resolve ${paths.root} style references in YAML string values."""
    if "${paths.root}" in value:
        root = context.get("paths", {}).get("root", str(PROJECT_ROOT))
        value = value.replace("${paths.root}", root)
    return value


def _resolve_nested(obj: Any, context: dict) -> Any:
    """Recursively resolve path variables in a nested dict."""
    if isinstance(obj, dict):
        return {k: _resolve_nested(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_nested(item, context) for item in obj]
    if isinstance(obj, str):
        return _resolve_path_vars(obj, context)
    return obj


class CancerwiseConfig:
    """
    Singleton configuration container for CANCERWISE AI.

    Reads from configs/settings.yaml and merges with environment variables.
    Access settings via attribute-style: config.api.port, config.training.batch_size
    """

    _instance: Optional["CancerwiseConfig"] = None

    def __new__(cls) -> "CancerwiseConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self) -> None:
        if self._loaded:
            return
        self._data: dict = {}
        self._load()
        self._loaded = True

    def _load(self) -> None:
        """Load and parse settings.yaml."""
        if not SETTINGS_FILE.exists():
            logger.warning("settings.yaml not found at %s — using defaults", SETTINGS_FILE)
            self._data = {}
            return

        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Resolve ${paths.root} and similar references
        self._data = _resolve_nested(raw, raw)

        # Override with environment variables where applicable
        self._apply_env_overrides()
        logger.debug("Configuration loaded from %s", SETTINGS_FILE)

    def _apply_env_overrides(self) -> None:
        """Allow .env variables to override YAML config."""
        env_map = {
            "LOG_LEVEL":           ("logging", "level"),
            "API_HOST":            ("api", "host"),
            "API_PORT":            ("api", "port"),
            "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
            "MLFLOW_EXPERIMENT_NAME": ("mlflow", "experiment_name"),
            "MODEL_RANDOM_SEED":   ("data_mining", "random_seed"),
        }
        for env_key, (section, key) in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                if section not in self._data:
                    self._data[section] = {}
                # Coerce types
                current = self._data[section].get(key)
                if isinstance(current, int):
                    val = int(val)
                elif isinstance(current, float):
                    val = float(val)
                elif isinstance(current, bool):
                    val = val.lower() in ("true", "1", "yes")
                self._data[section][key] = val

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a value by dot-path: config.get('api', 'port')."""
        node = self._data
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        section = self._data.get(name)
        if section is None:
            raise AttributeError(f"Configuration section '{name}' not found.")
        # Return a simple namespace-style object
        if isinstance(section, dict):
            return _SectionProxy(section)
        return section


class _SectionProxy:
    """Thin wrapper to allow dot-access on a dict section."""

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_data")
        if name not in d:
            raise AttributeError(f"Config key '{name}' not found.")
        val = d[name]
        if isinstance(val, dict):
            return _SectionProxy(val)
        return val

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_data"))


@lru_cache(maxsize=1)
def get_config() -> CancerwiseConfig:
    """Return the singleton config instance (cached)."""
    return CancerwiseConfig()
