"""YAML configuration loading and unresolved-decision detection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tau_sr.constants import UNSET
from tau_sr.exceptions import ConfigurationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file into a dictionary."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{config_path} must contain a mapping")
    return loaded


def load_config_dir(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load all YAML files in a configuration directory."""

    directory = Path(config_dir)
    return {path.stem: load_yaml(path) for path in sorted(directory.glob("*.yaml"))}


def find_unset_values(value: Any, prefix: str = "") -> list[str]:
    """Return dotted paths whose value is explicitly UNSET."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_unset_values(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_unset_values(child, f"{prefix}[{index}]"))
    elif value == UNSET:
        found.append(prefix)
    return found


def require_fields(config: Mapping[str, Any], fields: list[str], *, context: str) -> None:
    """Ensure required top-level fields are present."""

    missing = [field for field in fields if field not in config]
    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(f"{context} missing required fields: {joined}")
