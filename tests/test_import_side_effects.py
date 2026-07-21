from __future__ import annotations


def test_importing_tau_sr_has_no_side_effects() -> None:
    import tau_sr

    assert tau_sr.__version__ == "0.1.0"
