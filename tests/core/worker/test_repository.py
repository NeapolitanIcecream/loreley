from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import uuid
from typing import Any, cast

import pytest

import loreley.core.worker.repository as repository_module
from loreley.config import Settings
from git import Repo

from loreley.core.git import require_commit, sanitize_value, wrap_git_error
from loreley.core.worker.repository import RepositoryError, WorkerRepository
from loreley.db.models import JobStatus
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


class _FakeQueryResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _PruneGit:
    def __init__(self, refs_output: str, head_by_ref: dict[str, str]) -> None:
        self._refs_output = refs_output
        self._head_by_ref = head_by_ref
        self.rev_parse_calls: list[str] = []

    def for_each_ref(self, *args: str) -> str:
        return self._refs_output

    def rev_parse(self, ref_name: str) -> str:
        self.rev_parse_calls.append(ref_name)
        return self._head_by_ref[ref_name]


def _make_repo(settings: Settings, tmp_path) -> WorkerRepository:
    settings.worker_repo_remote_url = "https://example.invalid/repo.git"
    settings.worker_repo_worktree = str(tmp_path / "repo")
    return WorkerRepository(settings=settings)


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = None
    if env is not None:
        full_env = {**os.environ, **env}
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        env=full_env,
    )


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
    monkeypatch.setattr(
        repo,
        "_prepare_base_repo_for_checkout",
        lambda *, base_commit: base_repo,
    )
    monkeypatch.setattr(repo, "_remove_worktree", lambda path: base_git.worktree("remove", "--force", str(path)))
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
    monkeypatch.setattr(
        repo,
        "_prepare_base_repo_for_checkout",
        lambda *, base_commit: base_repo,
    )
    monkeypatch.setattr(repo, "_remove_worktree", lambda path: base_git.worktree("remove", "--force", str(path)))
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


def test_load_protected_job_branch_state_collects_archive_and_job_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    repo = _make_repo(settings, tmp_path)
    prefix = repo.job_branch_prefix
    archived_branch = f"{prefix}/archived-job"
    pending_ingestion_branch = f"{prefix}/pending-ingestion"

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt: Any) -> _FakeQueryResult:
            self.calls += 1
            if self.calls == 1:
                return _FakeQueryResult([("archive-commit",), ("",)])
            if self.calls == 2:
                return _FakeQueryResult(
                    [
                        (
                            "base-commit",
                            "result-commit",
                            "candidate-commit",
                            archived_branch,
                            ["insp-1", "insp-2", ""],
                            JobStatus.RUNNING,
                            None,
                            None,
                        ),
                        (
                            "pending-base",
                            "pending-result",
                            "pending-candidate",
                            pending_ingestion_branch,
                            ["pending-insp"],
                            JobStatus.SUCCEEDED,
                            None,
                            None,
                        ),
                        (
                            "ignored-base",
                            "ignored-result",
                            "ignored-candidate",
                            "other/prefix",
                            ["ignored-insp"],
                            JobStatus.SUCCEEDED,
                            "succeeded",
                            None,
                        ),
                    ]
                )
            raise AssertionError("unexpected query count")

    @contextmanager
    def _session_scope():
        yield _FakeSession()

    monkeypatch.setattr(repository_module, "session_scope", _session_scope, raising=False)

    protected_commits, protected_branches = repo._load_protected_job_branch_state()

    assert protected_commits >= {
        "archive-commit",
        "base-commit",
        "result-commit",
        "candidate-commit",
        "insp-1",
        "insp-2",
        "pending-base",
        "pending-result",
        "pending-candidate",
        "pending-insp",
    }
    assert "ignored-result" not in protected_commits
    assert protected_branches == {archived_branch, pending_ingestion_branch}


def test_load_protected_job_branch_state_keeps_failed_published_candidate_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    """Regression: post-push failures must keep their published candidate branch discoverable."""

    repo = _make_repo(settings, tmp_path)
    failed_branch = f"{repo.job_branch_prefix}/failed-published"

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt: Any) -> _FakeQueryResult:
            self.calls += 1
            if self.calls == 1:
                return _FakeQueryResult([])
            if self.calls == 2:
                return _FakeQueryResult(
                    [
                        (
                            "failed-base",
                            None,
                            "failed-candidate",
                            failed_branch,
                            [],
                            JobStatus.FAILED,
                            None,
                            object(),
                        ),
                    ]
                )
            raise AssertionError("unexpected query count")

    @contextmanager
    def _session_scope():
        yield _FakeSession()

    monkeypatch.setattr(repository_module, "session_scope", _session_scope, raising=False)

    protected_commits, protected_branches = repo._load_protected_job_branch_state()

    assert protected_commits == {"failed-candidate"}
    assert protected_branches == {failed_branch}


def test_load_protected_job_branch_state_keeps_failed_candidate_refs_when_publish_stamp_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    """Regression: a post-push DB failure must not rely on candidate_published_at to retain the ref."""

    repo = _make_repo(settings, tmp_path)
    failed_branch = f"{repo.job_branch_prefix}/failed-without-stamp"

    class _FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _stmt: Any) -> _FakeQueryResult:
            self.calls += 1
            if self.calls == 1:
                return _FakeQueryResult([])
            if self.calls == 2:
                return _FakeQueryResult(
                    [
                        (
                            "failed-base",
                            None,
                            "failed-candidate",
                            failed_branch,
                            [],
                            JobStatus.FAILED,
                            None,
                            None,
                        ),
                    ]
                )
            raise AssertionError("unexpected query count")

    @contextmanager
    def _session_scope():
        yield _FakeSession()

    monkeypatch.setattr(repository_module, "session_scope", _session_scope, raising=False)

    protected_commits, protected_branches = repo._load_protected_job_branch_state()

    assert protected_commits == {"failed-candidate"}
    assert protected_branches == {failed_branch}


def test_prune_stale_job_branches_skips_protected_commits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    settings: Settings,
) -> None:
    repo = _make_repo(settings, tmp_path)
    repo.job_branch_ttl_hours = 1

    @contextmanager
    def _noop_lock():
        yield

    branch_name = repo._format_job_branch(uuid.uuid4())
    ref_name = f"refs/remotes/origin/{branch_name}"
    head_commit = "a" * 40
    git = _PruneGit(f"{ref_name} 1", {ref_name: head_commit})
    deleted: list[str] = []

    monkeypatch.setattr(repo, "_repo_lock", _noop_lock)
    monkeypatch.setattr(repo, "_get_repo", lambda: _FakeRepo(cast(Any, git)))
    monkeypatch.setattr(repo, "_fetch", lambda *, repo=None, refspecs=None: None)
    monkeypatch.setattr(repo, "delete_remote_branch", lambda branch: deleted.append(branch))
    monkeypatch.setattr(
        repo,
        "_load_protected_job_branch_state",
        lambda: ({head_commit}, set()),
        raising=False,
    )

    assert repo.prune_stale_job_branches() == 0
    assert deleted == []
    assert git.rev_parse_calls == [ref_name]


def test_prune_stale_job_branches_preserves_last_ref_for_protected_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    remote = tmp_path / "remote.git"
    remote_uri = remote.as_uri()
    source = tmp_path / "source"
    fresh = tmp_path / "fresh"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", "--no-local", remote_uri, str(source))
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test User")

    tracked = source / "tracked.txt"
    tracked.write_text("root\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "root")
    _git(source, "branch", "-M", "main")
    _git(source, "push", "origin", "main")

    repo = _make_repo(settings, tmp_path)
    repo.remote_url = remote_uri
    repo.worktree = tmp_path / "worker"
    repo._repo = None
    repo._lock_path = repo._resolve_lock_path()
    repo.enable_lfs = False
    repo.job_branch_ttl_hours = 1
    repo.prepare()

    branch_name = repo._format_job_branch(uuid.uuid4())
    tracked.write_text("candidate\n", encoding="utf-8")
    _git(source, "checkout", "-b", branch_name)
    _git(
        source,
        "commit",
        "-am",
        "candidate",
        env={
            "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z",
        },
    )
    protected_commit = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "push", "origin", branch_name)

    monkeypatch.setattr(
        repo,
        "_load_protected_job_branch_state",
        lambda: ({protected_commit}, set()),
        raising=False,
    )

    assert repo.prune_stale_job_branches() == 0

    _git(remote, "reflog", "expire", "--expire=now", "--all")
    _git(remote, "gc", "--prune=now")
    _git(tmp_path, "clone", "--no-local", remote_uri, str(fresh))

    resolved = require_commit(Repo(fresh), protected_commit)
    assert resolved == protected_commit


def test_prune_stale_job_branches_preserves_protected_ancestor_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    remote = tmp_path / "remote.git"
    remote_uri = remote.as_uri()
    source = tmp_path / "source"
    fresh = tmp_path / "fresh"

    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", "--no-local", remote_uri, str(source))
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test User")

    tracked = source / "tracked.txt"
    tracked.write_text("root\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "root")
    _git(source, "branch", "-M", "main")
    _git(source, "push", "origin", "main")

    repo = _make_repo(settings, tmp_path)
    repo.remote_url = remote_uri
    repo.worktree = tmp_path / "worker"
    repo._repo = None
    repo._lock_path = repo._resolve_lock_path()
    repo.enable_lfs = False
    repo.job_branch_ttl_hours = 1
    repo.prepare()

    protected_branch = repo._format_job_branch(uuid.uuid4())
    tracked.write_text("protected\n", encoding="utf-8")
    _git(source, "checkout", "-b", protected_branch)
    _git(
        source,
        "commit",
        "-am",
        "protected",
        env={
            "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z",
        },
    )
    protected_commit = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "push", "origin", protected_branch)

    descendant_branch = repo._format_job_branch(uuid.uuid4())
    tracked.write_text("descendant\n", encoding="utf-8")
    _git(source, "checkout", "-b", descendant_branch)
    _git(
        source,
        "commit",
        "-am",
        "descendant",
        env={
            "GIT_AUTHOR_DATE": "2020-01-02T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-02T00:00:00Z",
        },
    )
    _git(source, "push", "origin", descendant_branch)

    _git(source, "push", "origin", f":{protected_branch}")

    monkeypatch.setattr(
        repo,
        "_load_protected_job_branch_state",
        lambda: ({protected_commit}, set()),
        raising=False,
    )

    assert repo.prune_stale_job_branches() == 0

    _git(remote, "reflog", "expire", "--expire=now", "--all")
    _git(remote, "gc", "--prune=now")
    _git(tmp_path, "clone", "--no-local", remote_uri, str(fresh))

    resolved = require_commit(Repo(fresh), protected_commit)
    assert resolved == protected_commit
