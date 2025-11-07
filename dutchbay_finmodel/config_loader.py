from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml

from .core import ModelConfig


def load_config(path: str) -> ModelConfig:
    cfg_path = pathlib.Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    data: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    valid_keys = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return ModelConfig(**filtered)
