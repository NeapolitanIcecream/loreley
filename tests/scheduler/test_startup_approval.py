from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import stat

import pytest
from git import Repo

import loreley.scheduler.startup_approval as startup_approval


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def test_repo_state_root_approval_requires_tty() -> None:
    with pytest.raises(ValueError, match="stdin is not a TTY"):
        startup_approval.require_interactive_repo_state_root_approval(
            root_commit="deadbeef",
            eligible_files=3,
            repo_root=Path("."),
            stdin_is_tty=False,
        )


def test_repo_state_root_approval_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup_approval.Confirm, "ask", lambda *args, **kwargs: True)
    startup_approval.require_interactive_repo_state_root_approval(
        root_commit="deadbeef",
        eligible_files=3,
        repo_root=Path("."),
        stdin_is_tty=True,
    )


def test_repo_state_root_approval_rejects_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(startup_approval.Confirm, "ask", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="rejected"):
        startup_approval.require_interactive_repo_state_root_approval(
            root_commit="deadbeef",
            eligible_files=3,
            repo_root=Path("."),
            stdin_is_tty=True,
        )


def test_repo_state_root_approval_auto_approve_does_not_require_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup_approval.Confirm, "ask", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()))
    startup_approval.require_interactive_repo_state_root_approval(
        root_commit="deadbeef",
        eligible_files=3,
        repo_root=Path("."),
        stdin_is_tty=False,
        auto_approve=True,
    )


def test_require_repo_writable_accepts_normal_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init")
    repo = Repo(repo_root)

    startup_approval.require_repo_writable(repo_root=repo_root, repo=repo)


def test_require_repo_writable_rejects_readonly_git_dir(tmp_path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - permission model differs on Windows
        pytest.skip("chmod-based writability checks are not reliable on Windows")

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init")
    repo = Repo(repo_root)
    git_dir = Path(repo.git_dir).resolve()
    original_mode = stat.S_IMODE(git_dir.stat().st_mode)

    try:
        git_dir.chmod(0o555)
        try:
            with tempfile.NamedTemporaryFile(dir=str(git_dir), delete=True):
                pytest.skip("Environment does not enforce directory write permissions")
        except OSError:
            pass

        with pytest.raises(ValueError, match="not writable"):
            startup_approval.require_repo_writable(repo_root=repo_root, repo=repo)
    finally:
        git_dir.chmod(original_mode)

