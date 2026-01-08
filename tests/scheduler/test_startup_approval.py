from __future__ import annotations

from pathlib import Path

import pytest

import loreley.scheduler.startup_approval as startup_approval


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

