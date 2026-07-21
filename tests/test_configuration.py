from __future__ import annotations

from tau_sr.configuration import find_unset_values, load_config_dir, load_yaml


def test_configuration_files_load() -> None:
    configs = load_config_dir("configs")
    assert "project" in configs
    assert configs["project"]["expected_roi_count"] == 148


def test_required_fields_exist() -> None:
    project = load_yaml("configs/project.yaml")
    assert project["atlas"] == "Destrieux"
    assert project["data_root_env"] == "TAU_SR_DATA_ROOT"


def test_unset_decisions_detected() -> None:
    configs = load_config_dir("configs")
    unset = {name: find_unset_values(config) for name, config in configs.items()}
    assert "preprocessing" in unset
    assert "tau_tracer" in unset["preprocessing"]
    assert "primary_type" in unset["laplacian"]
