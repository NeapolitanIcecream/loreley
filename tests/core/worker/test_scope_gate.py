from __future__ import annotations

import subprocess
from pathlib import Path

from loreley.core.campaign_program import parse_campaign_program
from loreley.core.worker.scope_gate import (
    cleanup_scope_gate_untracked_paths,
    validate_campaign_scope,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    (repo / "loreley.program.md").write_text("program\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")


def test_scope_gate_allows_tracked_changes_inside_editable_scope(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(
        b"""## Editable scope
- src/**

## Protected scope
- docs/**
"""
    )
    (tmp_path / "src" / "app.py").write_text("print('changed')\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is True
    assert result.checked_paths == ("src/app.py",)


def test_scope_gate_rejects_protected_scope_even_when_editable_matches(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(
        b"""## Editable scope
- src/**

## Protected scope
- src/app.py
"""
    )
    (tmp_path / "src" / "app.py").write_text("print('blocked')\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.violations[0].code == "protected_scope_modified"
    assert result.violations[0].path == "src/app.py"


def test_scope_gate_rejects_paths_outside_non_empty_editable_scope(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.violations[0].code == "outside_editable_scope"
    assert result.violations[0].path == "README.md"


def test_scope_gate_checks_git_visible_artifact_store_paths(tmp_path: Path) -> None:
    """Regression: Git-visible artifact dirs must not bypass campaign scope."""

    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    (tmp_path / ".loreley").mkdir()
    (tmp_path / ".loreley" / "state.json").write_text("{}\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.violations[0].code == "outside_editable_scope"
    assert result.violations[0].path == ".loreley/state.json"


def test_scope_gate_preserves_significant_filename_whitespace(tmp_path: Path) -> None:
    """Regression: path normalization must not strip valid filename spaces."""

    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/app.py\n")
    (tmp_path / "src" / "app.py ").write_text("print('space')\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.checked_paths == ("src/app.py ",)
    assert result.violations[0].code == "outside_editable_scope"
    assert result.violations[0].path == "src/app.py "


def test_scope_gate_single_star_matches_only_one_path_segment(tmp_path: Path) -> None:
    """Regression: src/* must not allow recursively nested paths."""

    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/*\n")
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "file.py").write_text("print('nested')\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.violations[0].code == "outside_editable_scope"
    assert result.violations[0].path == "src/nested/file.py"


def test_scope_gate_double_star_matches_recursive_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "src" / "nested" / "file.py").write_text("print('nested')\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is True
    assert result.checked_paths == ("src/nested/file.py",)


def test_scope_gate_protects_program_file_by_default(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Goal\nImprove things.\n")
    (tmp_path / "loreley.program.md").write_text("changed\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.violations[0].code == "protected_scope_modified"
    assert result.violations[0].path == "loreley.program.md"


def test_scope_gate_checks_untracked_files_and_rejects_unsafe_symlink(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    (tmp_path / "src" / "new.py").write_text("print('new')\n", encoding="utf-8")
    (tmp_path / "src" / "escape").symlink_to("/tmp")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    codes = {violation.code for violation in result.violations}
    assert "unsafe_symlink_target" in codes
    assert "src/new.py" in result.checked_paths


def test_scope_gate_cleanup_removes_configured_untracked_junk(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    (tmp_path / "src" / "app.py").write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "make").mkdir()
    (tmp_path / "make" / "server.log").write_text("temporary build log\n", encoding="utf-8")

    cleanup = cleanup_scope_gate_untracked_paths(
        worktree=tmp_path,
        path_patterns=("make/server.log",),
    )
    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert cleanup.removed_paths == ("make/server.log",)
    assert cleanup.skipped_paths == ()
    assert not (tmp_path / "make" / "server.log").exists()
    assert result.passed is True
    assert result.checked_paths == ("src/app.py",)


def test_scope_gate_cleanup_does_not_remove_tracked_or_unconfigured_paths(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- src/**\n")
    (tmp_path / "make").mkdir()
    (tmp_path / "make" / "server.log").write_text("tracked\n", encoding="utf-8")
    _git(tmp_path, "add", "make/server.log")
    _git(tmp_path, "commit", "-m", "track server log")
    (tmp_path / "make" / "server.log").write_text("changed tracked\n", encoding="utf-8")
    (tmp_path / "tmp.log").write_text("unconfigured\n", encoding="utf-8")

    cleanup = cleanup_scope_gate_untracked_paths(
        worktree=tmp_path,
        path_patterns=("make/server.log",),
    )
    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert cleanup.removed_paths == ()
    assert cleanup.skipped_paths == ()
    assert (tmp_path / "make" / "server.log").exists()
    assert (tmp_path / "tmp.log").exists()
    assert result.passed is False
    assert {violation.path for violation in result.violations} == {
        "make/server.log",
        "tmp.log",
    }


def test_scope_gate_rejects_unsafe_scope_patterns(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    program = parse_campaign_program(b"## Editable scope\n- ../outside\n")
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    result = validate_campaign_scope(worktree=tmp_path, program=program)

    assert result.passed is False
    assert result.violations[0].code == "invalid_editable_scope_pattern"
    assert result.violations[0].reason == "path_traversal"
