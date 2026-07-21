from __future__ import annotations

import pytest

from tau_sr.exceptions import ConfigurationError, SecurityError
from tau_sr.paths import get_data_root, resolve_under_root


def test_paths_resolve_through_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TAU_SR_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path.resolve()


def test_missing_data_root_raises(monkeypatch) -> None:
    monkeypatch.delenv("TAU_SR_DATA_ROOT", raising=False)
    with pytest.raises(ConfigurationError):
        get_data_root()


def test_resolve_under_root_rejects_escape(tmp_path) -> None:
    with pytest.raises(SecurityError):
        resolve_under_root(tmp_path, "../outside")
