"""Load the YAML configuration files in ``configs/`` into one dictionary."""
from pathlib import Path
from typing import Any, Dict, Union

import yaml

_FILES = {"runtime": "config.yaml", "dataset": "dataset.yaml", "model": "model.yaml"}


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: Union[str, Path] = "configs") -> Dict[str, Any]:
    """Return ``{"runtime": ..., "dataset": ..., "model": ...}``."""
    base = Path(config_dir)
    return {key: load_yaml(base / name) for key, name in _FILES.items()}
