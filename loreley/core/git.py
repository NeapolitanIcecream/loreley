from __future__ import annotations

"""Shared Git helpers.

This module provides a single, opinionated implementation for ensuring a git
commit is available locally (with fetch + shallow/unshallow handling) and a
consistent error wrapper that redacts credentials from git commands.
"""

import shlex
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

from git import Repo
from git.exc import BadName, GitCommandError
from loguru import logger
from rich.console import Console

log = logger.bind(module="core.git")

__all__ = [
    "RepositoryError",
    "fetch_origin",
    "has_object",
    "is_shallow_repository",
    "require_commit",
    "sanitize_command",
    "sanitize_value",
    "wrap_git_error",
]


class RepositoryError(RuntimeError):
    """Raised when a git operation fails.

    The error optionally captures the raw git command and process outputs for
    debugging, while the string representation is safe for logs.
    """

    def __init__(
        self,
        message: str,
        *,
        cmd: Sequence[str] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(message)
        self.cmd = tuple(cmd) if cmd else None
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _command_tuple(command: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if not command:
        return None
    if isinstance(command, str):
        return (command,)
    return tuple(str(part) for part in command)


def sanitize_value(value: str) -> str:
    """Mask credential-bearing URLs (best-effort)."""

    parsed = urlsplit(value)
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        netloc = f"***@{host}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return value


def sanitize_command(cmd: Sequence[str]) -> str:
    sanitized = [sanitize_value(str(part)) for part in cmd]
    return shlex.join(sanitized)


def wrap_git_error(exc: GitCommandError, context: str) -> RepositoryError:
    """Convert a GitPython command failure into a sanitized RepositoryError."""

    command = _command_tuple(getattr(exc, "command", None))
    suffix = ""
    if command:
        suffix = f": {sanitize_command(command)}"
    status = getattr(exc, "status", None)
    message = f"{context}{suffix} (exit {status})"
    returncode = status if isinstance(status, int) else None
    return RepositoryError(
        message,
        cmd=command,
        returncode=returncode,
        stdout=getattr(exc, "stdout", None),
        stderr=getattr(exc, "stderr", None),
    )


def has_object(repo: Repo, obj_ref: str) -> bool:
    """Return True when obj_ref can be resolved as a commit locally."""

    try:
        repo.commit(obj_ref)
    except (BadName, GitCommandError, ValueError):
        return False
    return True


def is_shallow_repository(repo: Repo) -> bool:
    """Return True when the repository is shallow (best-effort)."""

    try:
        result = repo.git.rev_parse("--is-shallow-repository")
    except GitCommandError:
        return False
    return result.strip().lower() == "true"


def _ensure_remote(repo: Repo, remote: str) -> None:
    name = (remote or "").strip() or "origin"
    try:
        repo.remote(name)
    except ValueError as exc:
        worktree = getattr(repo, "working_tree_dir", None) or "<unknown>"
        raise RepositoryError(f"Git remote {name!r} is not configured for repo {worktree}.") from exc


def fetch_origin(
    repo: Repo,
    *,
    remote: str = "origin",
    fetch_depth: int | None = None,
    refspecs: Sequence[str] | None = None,
) -> None:
    """Fetch from the given remote using a stable, bounded argument set."""

    remote_name = (remote or "").strip() or "origin"
    _ensure_remote(repo, remote_name)

    fetch_args: list[str] = ["--prune", "--tags"]
    if fetch_depth:
        fetch_args.append(f"--depth={int(fetch_depth)}")
    fetch_args.append(remote_name)
    if refspecs:
        fetch_args.extend(str(r) for r in refspecs if str(r).strip())
    try:
        repo.git.fetch(*fetch_args)
    except GitCommandError as exc:
        raise wrap_git_error(exc, f"Failed to fetch from {remote_name}") from exc


def require_commit(
    repo: Repo,
    commit_ref: str,
    *,
    remote: str = "origin",
    fetch_depth: int | None = None,
    console: Console | None = None,
) -> str:
    """Return the canonical full hash for commit_ref, fetching when needed.

    The algorithm is intentionally strict and fail-fast:
    - resolve locally
    - fetch from origin
    - if still missing and the repo is shallow, unshallow and retry
    - raise RepositoryError when the commit cannot be resolved
    """

    commit = (commit_ref or "").strip()
    if not commit:
        raise RepositoryError("Commit reference must be provided.")

    try:
        resolved = repo.commit(commit)
    except (BadName, GitCommandError, ValueError):
        resolved = None
    if resolved is not None:
        return str(getattr(resolved, "hexsha", "") or "").strip()

    remote_name = (remote or "").strip() or "origin"
    if console is not None:
        console.log(f"[yellow]Fetching missing commit[/] {commit}")
    log.info("Commit {} missing locally; fetching from {}", commit, remote_name)

    fetch_origin(repo, remote=remote_name, fetch_depth=fetch_depth)
    try:
        resolved = repo.commit(commit)
    except (BadName, GitCommandError, ValueError):
        resolved = None
    if resolved is not None:
        return str(getattr(resolved, "hexsha", "") or "").strip()

    if is_shallow_repository(repo):
        if console is not None:
            console.log(f"[yellow]Repository is shallow; unshallowing[/] remote={remote_name}")
        log.info("Repository is shallow; unshallowing to retrieve {}", commit)
        try:
            repo.git.fetch("--unshallow", remote_name)
        except GitCommandError as exc:
            raise wrap_git_error(exc, "Failed to unshallow repository") from exc
        try:
            resolved = repo.commit(commit)
        except (BadName, GitCommandError, ValueError):
            resolved = None
        if resolved is not None:
            return str(getattr(resolved, "hexsha", "") or "").strip()

    raise RepositoryError(
        f"Commit {commit} is not available locally after fetching from {remote_name}.",
    )

