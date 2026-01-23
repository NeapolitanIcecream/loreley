from __future__ import annotations

import subprocess
from pathlib import Path

from git import Repo

from loreley.core.git import has_object, is_shallow_repository, require_commit


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _rev_parse(cwd: Path, ref: str = "HEAD") -> str:
    return _git(cwd, "rev-parse", ref).stdout.strip()


def test_require_commit_unshallows_shallow_clone(tmp_path: Path) -> None:
    """require_commit() should unshallow and resolve commits outside shallow history."""

    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init")
    _git(remote, "config", "user.email", "test@example.com")
    _git(remote, "config", "user.name", "Test User")

    path = remote / "file.txt"
    path.write_text("one\n", encoding="utf-8")
    _git(remote, "add", "file.txt")
    _git(remote, "commit", "-m", "c1")
    commit1 = _rev_parse(remote)

    path.write_text("two\n", encoding="utf-8")
    _git(remote, "commit", "-am", "c2")

    path.write_text("three\n", encoding="utf-8")
    _git(remote, "commit", "-am", "c3")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--depth=1", remote.as_uri(), str(clone))
    repo = Repo(clone)

    assert is_shallow_repository(repo)
    assert not has_object(repo, commit1)

    resolved = require_commit(repo, commit1, fetch_depth=1)
    assert resolved == commit1
    assert has_object(repo, commit1)
    assert not is_shallow_repository(repo)

