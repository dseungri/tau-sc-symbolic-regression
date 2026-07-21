from __future__ import annotations

from tau_sr.intake.checksums import checksum_records, sha256_file


def test_sha256_results_are_correct(tmp_path) -> None:
    path = tmp_path / "hello.txt"
    path.write_text("hello\n", encoding="utf-8")
    assert sha256_file(path) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def test_recursive_manifest_contains_relative_paths(tmp_path) -> None:
    path = tmp_path / "a" / "b.txt"
    path.parent.mkdir()
    path.write_text("x", encoding="utf-8")
    records = checksum_records(tmp_path)
    assert records[0]["relative_path"] == "a/b.txt"
    assert records[0]["size_bytes"] == 1
