from __future__ import annotations

import re
from pathlib import Path

from tau_sr.constants import FORBIDDEN_DATA_EXTENSIONS
from tau_sr.intake.file_classification import normalized_extension


def _visible_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.parts)
    ]


def test_no_forbidden_file_extension_is_visible_under_data() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "data").rglob("*"):
        if path.is_file() and path.name not in {"README.md", ".gitkeep"}:
            assert normalized_extension(path) not in FORBIDDEN_DATA_EXTENSIONS


def test_no_real_looking_subject_identifiers_in_fixtures() -> None:
    fixture_root = Path(__file__).resolve().parent / "fixtures"
    pattern = re.compile(r"sub-(?!syn)[A-Za-z0-9]{3,}")
    for path in fixture_root.rglob("*"):
        if path.is_file():
            assert pattern.search(path.read_text(encoding="utf-8")) is None


def test_no_credentials_or_env_files_visible() -> None:
    forbidden = re.compile(r"(credential|secret|token)", re.I)
    for path in _visible_files():
        assert path.name != ".env"
        assert forbidden.search(path.name) is None
