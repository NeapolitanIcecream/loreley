from __future__ import annotations

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
        self.checkout_calls: list[tuple[str, ...]] = []
        self.rev_parse_calls: list[tuple[str, ...]] = []

    def rev_parse(self, *args: str) -> None:
        self.rev_parse_calls.append(tuple(args))

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
    fake_git = _FakeGit()
    fake_repo = _FakeRepo(fake_git)
    calls: dict[str, str] = {}

    monkeypatch.setattr(repo, "prepare", lambda: calls.setdefault("prepare", "done"))
    monkeypatch.setattr(repo, "clean_worktree", lambda: calls.setdefault("clean", "done"))
    monkeypatch.setattr(repo, "_ensure_commit_available", lambda base_commit, repo=None: calls.setdefault("ensure", base_commit))
    monkeypatch.setattr(repo, "_get_repo", lambda: fake_repo)

    ctx = repo.checkout_for_job(job_id=uuid.uuid4(), base_commit="abc123", create_branch=True)

    expected_branch = repo._format_job_branch(ctx.job_id)
    assert ctx.branch_name == expected_branch
    assert fake_git.rev_parse_calls[0] == ("--verify", "abc123")
    assert fake_git.checkout_calls[0] == ("-B", expected_branch, "abc123")
    assert calls["ensure"] == "abc123"


def test_checkout_for_job_detaches_when_branch_not_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    repo = _make_repo(settings, tmp_path)
    fake_git = _FakeGit()
    fake_repo = _FakeRepo(fake_git)

    monkeypatch.setattr(repo, "prepare", lambda: None)
    monkeypatch.setattr(repo, "clean_worktree", lambda: None)
    monkeypatch.setattr(repo, "_ensure_commit_available", lambda base_commit, repo=None: None)
    monkeypatch.setattr(repo, "_get_repo", lambda: fake_repo)

    ctx = repo.checkout_for_job(job_id=None, base_commit="def456", create_branch=False)

    assert ctx.branch_name is None
    assert fake_git.checkout_calls[0] == ("--detach", "def456")
