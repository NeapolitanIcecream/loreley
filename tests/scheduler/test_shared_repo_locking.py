from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from queue import Empty
import time
from multiprocessing import get_context
from typing import Any, cast

import pytest
from rich.console import Console

import loreley.scheduler.ingestion as ingestion_mod
import loreley.scheduler.main as scheduler_main
from loreley.scheduler.ingestion import MapElitesIngestion
from loreley.scheduler.startup_approval import RepoStateRootScan
from loreley.naming import resolve_experiment_namespace
from tests._shared_repo_locking import hold_repo_lock


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Shared repo lock coordination tests require POSIX flock semantics.",
)


def _worker_lock_path(repo_root: Path) -> Path:
    resolved = Path(repo_root).expanduser().resolve()
    return resolved.parent / f".{resolved.name}.lock"


def _assert_waits_for_repo_lock(repo_root: Path, action: Any) -> None:
    hold_seconds = 0.5
    ctx = get_context("spawn")
    ready_queue = ctx.Queue()
    proc = ctx.Process(
        target=hold_repo_lock,
        args=(str(_worker_lock_path(repo_root)), hold_seconds, ready_queue),
    )
    proc.start()
    try:
        deadline = time.monotonic() + 5.0
        while True:
            if proc.exitcode is not None:
                raise AssertionError(
                    f"repo lock holder exited before readiness (exitcode={proc.exitcode})",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError("repo lock holder did not report readiness before timeout")
            try:
                assert ready_queue.get(timeout=min(0.1, remaining)) == "locked"
                break
            except Empty:
                continue
        started = time.perf_counter()
        action()
        elapsed = time.perf_counter() - started
        assert elapsed >= hold_seconds * 0.8
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def _make_ingestion(repo_root: Path, settings: Any) -> MapElitesIngestion:
    return MapElitesIngestion(
        settings=settings,
        console=Console(record=True),
        repo_root=repo_root,
        repo=cast(Any, object()),
        manager=cast(Any, object()),
    )


@dataclass
class _FakeGit:
    branch_calls: list[tuple[str, ...]] = field(default_factory=list)

    def branch(self, *args: str) -> None:
        self.branch_calls.append(tuple(args))


@dataclass
class _FakeRepo:
    git: _FakeGit = field(default_factory=_FakeGit)


def test_ingestion_waits_for_shared_worker_repo_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Any,
) -> None:
    """Regression: scheduler ingestion must not fetch from a worker-held base clone."""

    repo_root = tmp_path / "shared-repo"
    ingestion = _make_ingestion(repo_root, settings)
    monkeypatch.setattr(
        ingestion_mod,
        "require_commit",
        lambda _repo, commit_hash, **_kwargs: commit_hash,
    )

    _assert_waits_for_repo_lock(
        repo_root,
        lambda: ingestion._ensure_commit_available("deadbeef"),
    )


def test_startup_root_scan_waits_for_shared_worker_repo_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Any,
) -> None:
    """Regression: scheduler startup must not resolve the root commit while the worker mutates the base clone."""

    repo_root = tmp_path / "shared-repo"
    scheduler = scheduler_main.EvolutionScheduler.__new__(scheduler_main.EvolutionScheduler)
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler.repo_root = repo_root
    scheduler._repo = cast(Any, object())
    scheduler._root_commit_hash = "deadbeef"

    monkeypatch.setattr(
        scheduler_main,
        "require_commit",
        lambda _repo, commit_hash, **_kwargs: commit_hash,
    )
    monkeypatch.setattr(
        scheduler_main,
        "scan_repo_state_root",
        lambda **_kwargs: RepoStateRootScan(root_commit="deadbeef", eligible_files=1),
    )
    monkeypatch.setattr(
        scheduler_main,
        "require_interactive_repo_state_root_approval",
        lambda **_kwargs: None,
    )

    _assert_waits_for_repo_lock(
        repo_root,
        scheduler._startup_scan_and_validate_repo_state_approval,
    )


def test_scheduler_bootstrap_runs_before_repo_open_for_shared_repo_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Any,
) -> None:
    """Regression: constructor must let bootstrap_instance wait on the shared repo before Repo(...) runs."""

    repo_root = tmp_path / "shared-repo"
    bootstrap_started = False
    fake_repo = _FakeRepo()

    monkeypatch.setattr(
        scheduler_main,
        "ensure_database_schema",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_main.EvolutionScheduler,
        "_resolve_repo_root",
        lambda self: repo_root,
    )

    def _bootstrap_instance(**_kwargs: Any) -> tuple[Any, Any]:
        nonlocal bootstrap_started
        bootstrap_started = True
        return cast(Any, object()), settings

    monkeypatch.setattr(scheduler_main, "bootstrap_instance", _bootstrap_instance)
    monkeypatch.setattr(
        scheduler_main.EvolutionScheduler,
        "_init_repo",
        lambda self: fake_repo if bootstrap_started else (_ for _ in ()).throw(
            AssertionError("_init_repo() ran before bootstrap_instance()"),
        ),
    )
    monkeypatch.setattr(
        scheduler_main,
        "require_repo_writable",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_main.EvolutionScheduler,
        "_acquire_experiment_lock",
        lambda self: cast(Any, object()),
    )
    monkeypatch.setattr(
        scheduler_main.EvolutionScheduler,
        "_require_max_total_jobs",
        lambda self: 1,
    )
    monkeypatch.setattr(
        scheduler_main.EvolutionScheduler,
        "_startup_scan_and_validate_repo_state_approval",
        lambda self: None,
    )
    startup_events: list[str] = []

    class _FakeManager:
        def validate_configured_islands(self) -> None:
            startup_events.append("validate_islands")

    monkeypatch.setattr(
        scheduler_main,
        "MapElitesManager",
        lambda **_kwargs: _FakeManager(),
    )
    monkeypatch.setattr(
        scheduler_main,
        "MapElitesSampler",
        lambda **_kwargs: cast(Any, object()),
    )

    class _FakeJobScheduler:
        def count_total_jobs(self) -> int:
            return 0

    def _build_job_scheduler(**_kwargs: Any) -> _FakeJobScheduler:
        startup_events.append("build_job_scheduler")
        return _FakeJobScheduler()

    monkeypatch.setattr(
        scheduler_main,
        "JobScheduler",
        _build_job_scheduler,
    )
    monkeypatch.setattr(
        scheduler_main,
        "MapElitesIngestion",
        lambda **_kwargs: cast(Any, object()),
    )

    scheduler = scheduler_main.EvolutionScheduler(settings=settings)

    assert bootstrap_started is True
    assert scheduler.repo_root == repo_root
    assert scheduler._repo is fake_repo
    assert startup_events == ["validate_islands", "build_job_scheduler"]


def test_primary_objective_branch_update_waits_for_shared_worker_repo_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Any,
) -> None:
    """Primary-objective branch updates must share the worker base-repo lock."""

    repo_root = tmp_path / "shared-repo"
    fake_repo = _FakeRepo()
    scheduler = scheduler_main.EvolutionScheduler.__new__(scheduler_main.EvolutionScheduler)
    scheduler.settings = settings
    scheduler.console = Console(record=True)
    scheduler.repo_root = repo_root
    scheduler._repo = fake_repo

    monkeypatch.setattr(
        scheduler_main,
        "require_commit",
        lambda _repo, commit_hash, **_kwargs: "c" * 40,
    )

    def _run() -> None:
        branch_name = scheduler._create_primary_objective_branch(
            best_commit_hash="deadbeef",
            root_commit_hash=None,
        )
        expected_branch = (
            f"evolution/primary/{resolve_experiment_namespace(settings.experiment_id)}"
        )
        assert branch_name == expected_branch
        assert fake_repo.git.branch_calls == [("-f", branch_name, "c" * 40)]

    _assert_waits_for_repo_lock(repo_root, _run)
