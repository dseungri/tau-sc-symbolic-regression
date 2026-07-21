"""Path utilities for restricted-data aware workflows."""

from __future__ import annotations

import os
from pathlib import Path

from tau_sr.constants import DATA_ROOT_ENV
from tau_sr.exceptions import ConfigurationError, SecurityError


def get_data_root(env_var: str = DATA_ROOT_ENV, *, require_exists: bool = False) -> Path:
    """Return the external data root from an environment variable."""

    value = os.getenv(env_var)
    if not value or value == "UNSET":
        raise ConfigurationError(f"{env_var} is not set to an approved data root")
    root = Path(value).expanduser().resolve()
    if require_exists and not root.exists():
        raise ConfigurationError(f"{env_var} points to a missing path")
    return root


def resolve_under_root(root: Path, relative_or_absolute: str | Path) -> Path:
    """Resolve a path and ensure it remains under root."""

    root = root.expanduser().resolve()
    candidate = Path(relative_or_absolute).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SecurityError(f"path escapes approved root: {candidate}") from exc
    return candidate


def ensure_output_dir(path: str | Path) -> Path:
    """Create and return an explicit output directory."""

    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def reject_in_place_source(source: str | Path, output: str | Path) -> None:
    """Refuse to modify a source file in place."""

    if Path(source).expanduser().resolve() == Path(output).expanduser().resolve():
        raise SecurityError("output path must differ from source path")


def is_incoming_path(path: str | Path) -> bool:
    """Return true when a path appears to point at incoming read-only data."""

    return "00_incoming_readonly" in Path(path).parts
