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


def test_require_commit_fetches_hash_outside_single_branch_refspec(
    tmp_path: Path,
) -> None:
    """Worker commits remain fetchable from a single-branch scheduler clone."""

    source = tmp_path / "source"
    source.mkdir(parents=True)
    _git(source, "init", "-b", "experiment-root")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test User")
    (source / "file.txt").write_text("root\n", encoding="utf-8")
    _git(source, "add", "file.txt")
    _git(source, "commit", "-m", "root")

    remote = tmp_path / "target.git"
    _git(tmp_path, "clone", "--bare", str(source), str(remote))
    scheduler = tmp_path / "scheduler"
    _git(
        tmp_path,
        "clone",
        "--single-branch",
        "--branch",
        "experiment-root",
        str(remote),
        str(scheduler),
    )
    worker = tmp_path / "worker"
    _git(tmp_path, "clone", str(remote), str(worker))
    _git(worker, "config", "user.email", "test@example.com")
    _git(worker, "config", "user.name", "Test User")
    _git(worker, "switch", "-c", "candidate")
    (worker / "file.txt").write_text("candidate\n", encoding="utf-8")
    _git(worker, "commit", "-am", "candidate")
    candidate_commit = _rev_parse(worker)
    _git(worker, "push", "origin", "candidate")

    repo = Repo(scheduler)
    assert not has_object(repo, candidate_commit)

    resolved = require_commit(repo, candidate_commit)

    assert resolved == candidate_commit
    assert has_object(repo, candidate_commit)
