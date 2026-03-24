from __future__ import annotations

from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import time

import pytest

import loreley.core.experiments as experiments
from loreley.core.experiments import (
    _build_slug_from_source,
    _normalise_remote_url,
    bootstrap_instance,
)


def test_build_slug_from_source_basic() -> None:
    slug = _build_slug_from_source("https://github.com/Owner/Repo.git")
    assert slug == "github.com/owner/repo"


def test_normalise_remote_url_canonicalises_and_strips_credentials() -> None:
    https = "https://user:pass@example.com:8443/Owner/Repo.git"
    ssh = "git@github.com:Owner/Repo.git"

    https_norm = _normalise_remote_url(https)
    ssh_norm = _normalise_remote_url(ssh)

    # Credentials and query/fragment should be stripped.
    assert "user" not in https_norm
    assert "pass" not in https_norm
    assert https_norm.startswith("https://example.com:8443/")

    # SCP-style URLs are normalised into a proper ssh:// form.
    assert ssh_norm.startswith("ssh://git@github.com/")
    assert ssh_norm.endswith("/Owner/Repo.git")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
    ).strip()


def _hold_repo_lock(lock_path: str, hold_seconds: float, ready_queue) -> None:
    import fcntl

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        ready_queue.put("locked")
        time.sleep(hold_seconds)
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


@pytest.mark.skipif(
    os.name != "posix",
    reason="Shared repo lock coordination tests require POSIX flock semantics.",
)
def test_bootstrap_instance_waits_for_shared_worker_repo_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings,
) -> None:
    """Regression: scheduler bootstrap must not resolve the root commit while the worker holds the base repo lock."""

    repo_root = tmp_path / "shared-repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test User")
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo_root, "add", "README.md")
    _git(repo_root, "commit", "-m", "init")

    root_commit = _git(repo_root, "rev-parse", "HEAD")
    test_settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": root_commit,
        }
    )
    monkeypatch.setattr(experiments, "_update_instance_metadata", lambda **_kwargs: None)

    hold_seconds = 0.5
    lock_path = repo_root.parent / f".{repo_root.name}.lock"
    ctx = get_context("spawn")
    ready_queue = ctx.Queue()
    proc = ctx.Process(
        target=_hold_repo_lock,
        args=(str(lock_path), hold_seconds, ready_queue),
    )
    proc.start()
    try:
        assert ready_queue.get(timeout=5) == "locked"
        started = time.perf_counter()
        bootstrap_instance(settings=test_settings, repo_root=repo_root)
        elapsed = time.perf_counter() - started
        assert elapsed >= hold_seconds * 0.8
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

