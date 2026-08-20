from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.method_efficacy_experiment import zstd_target


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def test_public_contract_matches_formal_treatment() -> None:
    treatment_path = (
        Path(__file__).resolve().parents[3]
        / "paper"
        / "evidence"
        / "zstd_formal_treatment.json"
    )
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))["target"]

    assert hashlib.sha256(zstd_target.PROGRAM_TEXT.encode()).hexdigest() == (
        treatment["program_text_sha256"]
    )
    assert hashlib.sha256(zstd_target.IGNORE_TEXT.encode()).hexdigest() == (
        treatment["ignore_text_sha256"]
    )


def test_prepare_root_adds_only_frozen_control_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init")
    _git(upstream, "config", "user.name", "test")
    _git(upstream, "config", "user.email", "test@example.invalid")
    (upstream / "lib").mkdir()
    (upstream / "lib" / "zstd.c").write_text(
        "int zstd(void) { return 0; }\n", encoding="utf-8"
    )
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-m", "upstream")
    commit = _git(upstream, "rev-parse", "HEAD")
    monkeypatch.setattr(zstd_target, "UPSTREAM_COMMIT", commit)

    destination = tmp_path / "root"
    result = zstd_target.prepare_root_template(
        upstream=upstream,
        destination=destination,
    )

    assert result["upstream_commit"] == commit
    assert result["changed_paths"] == [".loreleyignore", "loreley.program.md"]
    assert zstd_target.prepare_root_template(
        upstream=upstream,
        destination=destination,
    ) == result
