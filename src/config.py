from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    project = config.setdefault("project", {})
    project["root"] = ROOT
    for key in ("data_dir", "results_dir"):
        value = Path(project.get(key, key.removesuffix("_dir")))
        project[key] = value if value.is_absolute() else ROOT / value
    return config


def ensure_directories(config: dict[str, Any]) -> None:
    data_dir = Path(config["project"]["data_dir"])
    results_dir = Path(config["project"]["results_dir"])
    for path in (
        data_dir / "raw",
        data_dir / "processed",
        data_dir / "models",
        data_dir / "cache",
        results_dir / "charts",
    ):
        path.mkdir(parents=True, exist_ok=True)

