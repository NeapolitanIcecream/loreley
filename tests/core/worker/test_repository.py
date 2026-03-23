from __future__ import annotations

from contextlib import contextmanager
import uuid
from typing import Any, cast

import pytest

from loreley.config import Settings
from loreley.core.git import sanitize_value, wrap_git_error
from loreley.core.worker.repository import RepositoryError, WorkerRepository
from loreley.naming import worker_job_branch_prefix


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
    return WorkerRepository(settings=settings)


def test_sanitize_value_masks_credentials() -> None:
    masked = sanitize_value("https://user:token@example.com/repo.git")
    assert "***@" in masked
    assert "token" not in masked

    unchanged = sanitize_value("git@github.com:org/repo.git")
    assert unchanged == "git@github.com:org/repo.git"


def test_format_job_branch_applies_prefix_and_sanitises(tmp_path, settings: Settings) -> None:
    repo = _make_repo(settings, tmp_path)
    branch = repo._format_job_branch("Job ID 123 !!")
    expected_prefix = worker_job_branch_prefix(settings.experiment_id).strip("/") + "/"
    assert branch.startswith(expected_prefix)
    assert " " not in branch
    assert "!" not in branch


def test_format_job_branch_includes_attempt_token(tmp_path, settings: Settings) -> None:
    repo = _make_repo(settings, tmp_path)
    attempt_token = uuid.uuid4()

    branch = repo._format_job_branch("job-123", attempt_token=attempt_token)

    assert str(attempt_token)[:8] in branch


def test_wrap_git_error_sanitises_command() -> None:
    exc = _DummyGitError(
        ["git", "clone", "https://user:pw@example.com/repo.git"],
        status=128,
        stdout="out",
        stderr="err",
    )
    wrapped = wrap_git_error(cast(Any, exc), "Clone failed")

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
    monkeypatch.setattr(
        repo,
        "_prepare_base_repo_for_checkout",
        lambda *, base_commit: base_repo,
    )
    monkeypatch.setattr(repo, "_remove_worktree", lambda path: base_git.worktree("remove", "--force", str(path)))
    monkeypatch.setattr(repo, "_open_repo", lambda path: job_repo)
    monkeypatch.setattr(
        repo,
        "_allocate_job_worktree_path",
        lambda job_id, base_commit, attempt_token=None: worktree_path,
    )
    monkeypatch.setattr(repo, "_ensure_worktree_path_available", lambda path, repo: None)
    attempt_token = uuid.uuid4()

    with repo.checkout_lease_for_job(
        job_id=job_id,
        base_commit="abc123",
        create_branch=True,
        attempt_token=attempt_token,
    ) as ctx:
        assert ctx.worktree == worktree_path
        expected_branch = repo._format_job_branch(job_id, attempt_token=attempt_token)
        assert ctx.branch_name == expected_branch
        assert ctx.job_id == str(job_id)

    assert ("add", "--detach", str(worktree_path), "abc123") in base_git.worktree_calls
    assert ("remove", "--force", str(worktree_path)) in base_git.worktree_calls
    assert job_git.checkout_calls[0] == ("-B", repo._format_job_branch(job_id, attempt_token=attempt_token), "abc123")


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
    monkeypatch.setattr(
        repo,
        "_prepare_base_repo_for_checkout",
        lambda *, base_commit: base_repo,
    )
    monkeypatch.setattr(repo, "_remove_worktree", lambda path: base_git.worktree("remove", "--force", str(path)))
    monkeypatch.setattr(repo, "_open_repo", lambda path: job_repo)
    monkeypatch.setattr(
        repo,
        "_allocate_job_worktree_path",
        lambda job_id, base_commit, attempt_token=None: worktree_path,
    )
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
    monkeypatch.setattr(
        repo,
        "_prepare_base_repo_for_checkout",
        lambda *, base_commit: base_repo,
    )
    monkeypatch.setattr(repo, "_remove_worktree", lambda path: base_git.worktree("remove", "--force", str(path)))
    monkeypatch.setattr(repo, "_open_repo", lambda path: job_repo)
    monkeypatch.setattr(
        repo,
        "_allocate_job_worktree_path",
        lambda job_id, base_commit, attempt_token=None: worktree_path,
    )
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


def test_prepare_base_repo_for_checkout_skips_full_upstream_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    """Checkout preparation should avoid the full `prepare()` sync path for hot worker jobs."""

    repo = _make_repo(settings, tmp_path)
    repo.enable_lfs = False
    base_repo = _FakeRepo(_FakeGit())
    calls: list[str] = []

    monkeypatch.setattr(repo, "_ensure_worktree_ready", lambda: calls.append("ensure_worktree_ready"))
    monkeypatch.setattr(repo, "_get_repo", lambda: base_repo)
    monkeypatch.setattr(repo, "_ensure_remote_origin", lambda *, repo=None: calls.append("ensure_remote_origin"))
    monkeypatch.setattr(
        repo,
        "_ensure_commit_available",
        lambda base_commit, repo=None: calls.append(f"ensure_commit_available:{base_commit}"),
    )
    monkeypatch.setattr(repo, "_prune_worktrees", lambda *, repo=None: calls.append("prune_worktrees"))
    monkeypatch.setattr(
        repo,
        "_sync_upstream",
        lambda: (_ for _ in ()).throw(AssertionError("_sync_upstream should not run")),
    )

    prepared = repo._prepare_base_repo_for_checkout(base_commit="abc123")

    assert prepared is base_repo
    assert calls == [
        "ensure_worktree_ready",
        "ensure_remote_origin",
        "ensure_commit_available:abc123",
        "prune_worktrees",
    ]
