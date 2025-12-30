from __future__ import annotations

from contextlib import contextmanager
import uuid

import pytest

from loreley.config import Settings
from loreley.core.worker.repository import RepositoryError, WorkerRepository


class _DummyGitError(Exception):
    def __init__(self, command: list[str], status: int = 1, stdout: str = "", stderr: str = "") -> None:
        super().__init__("dummy")
        self.command = command
        self.status = status
        self.stdout = stdout
        self.stderr = stderr


class _FakeGit:
    def __init__(self) -> None:
        self.worktree_calls: list[tuple[str, ...]] = []
        self.checkout_calls: list[tuple[str, ...]] = []

    def worktree(self, *args: str) -> None:
        self.worktree_calls.append(tuple(args))

    def checkout(self, *args: str) -> None:
        self.checkout_calls.append(tuple(args))


class _FakeRepo:
    def __init__(self, git: _FakeGit) -> None:
        self.git = git


def _make_repo(settings: Settings, tmp_path) -> WorkerRepository:
    settings.worker_repo_remote_url = "https://example.invalid/repo.git"
    settings.worker_repo_worktree = str(tmp_path / "repo")
    settings.worker_repo_job_branch_prefix = "jobs"
    return WorkerRepository(settings=settings)


def test_sanitize_value_masks_credentials() -> None:
    masked = WorkerRepository._sanitize_value("https://user:token@example.com/repo.git")
    assert "***@" in masked
    assert "token" not in masked

    unchanged = WorkerRepository._sanitize_value("git@github.com:org/repo.git")
    assert unchanged == "git@github.com:org/repo.git"


def test_format_job_branch_applies_prefix_and_sanitises(tmp_path, settings: Settings) -> None:
    repo = _make_repo(settings, tmp_path)
    branch = repo._format_job_branch("Job ID 123 !!")
    assert branch.startswith("jobs/")
    assert " " not in branch
    assert "!" not in branch


def test_wrap_git_error_sanitises_command(tmp_path, settings: Settings) -> None:
    repo = _make_repo(settings, tmp_path)
    exc = _DummyGitError(
        ["git", "clone", "https://user:pw@example.com/repo.git"],
        status=128,
        stdout="out",
        stderr="err",
    )
    wrapped = repo._wrap_git_error(exc, "Clone failed")

    assert isinstance(wrapped, RepositoryError)
    assert "***@" in str(wrapped)
    assert wrapped.returncode == 128
    assert wrapped.cmd == ("git", "clone", "https://user:pw@example.com/repo.git")


def test_checkout_for_job_creates_branch(monkeypatch: pytest.MonkeyPatch, tmp_path, settings: Settings) -> None:
    repo = _make_repo(settings, tmp_path)
    repo.git_dir.mkdir(parents=True, exist_ok=True)
    base_git = _FakeGit()
    base_repo = _FakeRepo(base_git)
    job_git = _FakeGit()
    job_repo = _FakeRepo(job_git)

    @contextmanager
    def _noop_lock():
        yield

    job_id = uuid.uuid4()
    worktree_path = tmp_path / "job-worktree"

    monkeypatch.setattr(repo, "_repo_lock", _noop_lock)
    monkeypatch.setattr(repo, "prepare", lambda: None)
    monkeypatch.setattr(repo, "_ensure_commit_available", lambda base_commit, repo=None: None)
    monkeypatch.setattr(repo, "_get_repo", lambda: base_repo)
    monkeypatch.setattr(repo, "_open_repo", lambda path: job_repo)
    monkeypatch.setattr(repo, "_allocate_job_worktree_path", lambda job_id, base_commit: worktree_path)
    monkeypatch.setattr(repo, "_ensure_worktree_path_available", lambda path, repo: None)

    with repo.checkout_lease_for_job(job_id=job_id, base_commit="abc123", create_branch=True) as ctx:
        assert ctx.worktree == worktree_path
        expected_branch = repo._format_job_branch(job_id)
        assert ctx.branch_name == expected_branch
        assert ctx.job_id == str(job_id)

    assert ("add", "--detach", str(worktree_path), "abc123") in base_git.worktree_calls
    assert ("remove", "--force", str(worktree_path)) in base_git.worktree_calls
    assert job_git.checkout_calls[0] == ("-B", repo._format_job_branch(job_id), "abc123")


def test_checkout_for_job_detaches_when_branch_not_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    repo = _make_repo(settings, tmp_path)
    repo.git_dir.mkdir(parents=True, exist_ok=True)
    base_git = _FakeGit()
    base_repo = _FakeRepo(base_git)
    job_git = _FakeGit()
    job_repo = _FakeRepo(job_git)

    @contextmanager
    def _noop_lock():
        yield

    worktree_path = tmp_path / "detached-worktree"

    monkeypatch.setattr(repo, "_repo_lock", _noop_lock)
    monkeypatch.setattr(repo, "prepare", lambda: None)
    monkeypatch.setattr(repo, "_ensure_commit_available", lambda base_commit, repo=None: None)
    monkeypatch.setattr(repo, "_get_repo", lambda: base_repo)
    monkeypatch.setattr(repo, "_open_repo", lambda path: job_repo)
    monkeypatch.setattr(repo, "_allocate_job_worktree_path", lambda job_id, base_commit: worktree_path)
    monkeypatch.setattr(repo, "_ensure_worktree_path_available", lambda path, repo: None)

    with repo.checkout_lease_for_job(job_id=None, base_commit="def456", create_branch=False) as ctx:
        assert ctx.branch_name is None
        assert ctx.job_id is None
        assert ctx.worktree == worktree_path

    assert ("add", "--detach", str(worktree_path), "def456") in base_git.worktree_calls
    assert ("remove", "--force", str(worktree_path)) in base_git.worktree_calls
    assert job_git.checkout_calls[0] == ("--detach", "def456")


def test_checkout_lease_preserves_worktree_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    repo = _make_repo(settings, tmp_path)
    repo.git_dir.mkdir(parents=True, exist_ok=True)
    base_git = _FakeGit()
    base_repo = _FakeRepo(base_git)
    job_git = _FakeGit()
    job_repo = _FakeRepo(job_git)

    @contextmanager
    def _noop_lock():
        yield

    worktree_path = tmp_path / "failed-worktree"

    monkeypatch.setattr(repo, "_repo_lock", _noop_lock)
    monkeypatch.setattr(repo, "prepare", lambda: None)
    monkeypatch.setattr(repo, "_ensure_commit_available", lambda base_commit, repo=None: None)
    monkeypatch.setattr(repo, "_get_repo", lambda: base_repo)
    monkeypatch.setattr(repo, "_open_repo", lambda path: job_repo)
    monkeypatch.setattr(repo, "_allocate_job_worktree_path", lambda job_id, base_commit: worktree_path)
    monkeypatch.setattr(repo, "_ensure_worktree_path_available", lambda path, repo: None)

    with pytest.raises(RuntimeError, match="boom"):
        with repo.checkout_lease_for_job(
            job_id=uuid.uuid4(),
            base_commit="deadbeef",
            keep_worktree_on_failure=True,
        ):
            raise RuntimeError("boom")

    assert ("add", "--detach", str(worktree_path), "deadbeef") in base_git.worktree_calls
    assert not any(call[:1] == ("remove",) for call in base_git.worktree_calls)
