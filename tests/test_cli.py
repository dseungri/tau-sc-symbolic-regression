from __future__ import annotations

from tau_sr.cli import main


def test_cli_help_commands_work() -> None:
    assert main(["--help"]) == 0


def test_smoke_test_succeeds(tmp_path) -> None:
    output = tmp_path / "smoke.json"
    assert main(["smoke-test", "--output-report", str(output)]) == 0
    assert output.exists()
