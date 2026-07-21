from __future__ import annotations

from tau_sr.qc.connectome import validate_connectome_matrix
from tau_sr.synthetic.fixtures import malformed_negative_matrix, synthetic_connectome_matrices


def test_connectome_qc_detects_wrong_shapes() -> None:
    count, _, _ = synthetic_connectome_matrices(10)
    report = validate_connectome_matrix(count)
    assert not report.ok
    assert any(message.code == "wrong_shape" for message in report.messages)


def test_connectome_qc_detects_asymmetry() -> None:
    count, _, _ = synthetic_connectome_matrices()
    count[0, 1] += 1.0
    report = validate_connectome_matrix(count)
    assert not report.ok
    assert any(message.code == "asymmetry" for message in report.messages)


def test_connectome_qc_detects_negative_weights() -> None:
    report = validate_connectome_matrix(malformed_negative_matrix())
    assert not report.ok
    assert any(message.code == "negative_weights" for message in report.messages)
