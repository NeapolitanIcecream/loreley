from __future__ import annotations

"""Scheduler startup guards that are safe to unit-test without a live database."""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

from git import Repo
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from loreley.config import Settings
from loreley.core.map_elites.repository_files import list_repository_files


@dataclass(frozen=True, slots=True)
class RepoStateRootScan:
    root_commit: str
    eligible_files: int


def _resolve_git_common_dir(git_dir: Path) -> Path:
    """Resolve the common git directory for worktrees (best-effort).

    For a regular repository, this returns git_dir.
    For a linked worktree, git_dir points at ".git/worktrees/<name>" and the
    common directory is resolved via the "commondir" file.
    """

    commondir_path = git_dir / "commondir"
    if not commondir_path.is_file():
        return git_dir
    try:
        raw = commondir_path.read_text(encoding="utf-8").strip()
    except OSError:
        return git_dir
    if not raw:
        return git_dir
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (git_dir / candidate).resolve()
    return candidate.resolve()


def _directory_is_writable(path: Path) -> bool:
    """Return True when a temporary file can be created in path."""

    try:
        if not path.exists() or not path.is_dir():
            return False
        if not os.access(str(path), os.W_OK | os.X_OK):
            return False
        with tempfile.NamedTemporaryFile(dir=str(path), prefix=".loreley-writecheck-", delete=True):
            return True
    except Exception:
        return False


def require_repo_writable(
    *,
    repo_root: Path,
    repo: Repo,
    console: Console | None = None,
) -> None:
    """Fail fast when the git directory is not writable.

    The scheduler requires write access to the git directory to:
    - fetch missing commits (object database updates), and
    - update the best-fitness branch deliverable at the end of a bounded run.
    """

    c = console or Console()
    git_dir = Path(getattr(repo, "git_dir", "") or "").expanduser().resolve()
    if not git_dir:
        raise ValueError("Cannot determine git_dir for scheduler repository.")
    common_dir = _resolve_git_common_dir(git_dir)

    targets = [git_dir]
    if common_dir != git_dir:
        targets.append(common_dir)

    for target in targets:
        if _directory_is_writable(target):
            continue
        repo_root_resolved = Path(repo_root).expanduser().resolve()
        message = (
            "Scheduler repository is not writable. "
            "Write access is required for git fetch and the best-fitness branch deliverable. "
            f"(repo_root={repo_root_resolved} git_dir={git_dir} common_dir={common_dir} failing_path={target})"
        )
        c.print(f"[bold red]{message}[/]")
        raise ValueError(message)

    c.log(
        "[green]Git directory is writable[/] repo_root={} git_dir={} common_dir={}".format(
            Path(repo_root).expanduser().resolve(),
            git_dir,
            common_dir,
        )
    )


def scan_repo_state_root(
    *,
    settings: Settings,
    repo_root: Path,
    repo: Repo,
    root_commit: str,
) -> RepoStateRootScan:
    """Scan eligible repo-state files at the experiment root commit (count only)."""

    files = list_repository_files(
        repo_root=Path(repo_root).resolve(),
        commit_hash=str(root_commit).strip(),
        settings=settings,
        repo=repo,
    )
    return RepoStateRootScan(root_commit=str(root_commit).strip(), eligible_files=len(files))


def require_interactive_repo_state_root_approval(
    *,
    root_commit: str,
    eligible_files: int,
    repo_root: Path,
    details: Mapping[str, object] | None = None,
    console: Console | None = None,
    stdin_is_tty: bool | None = None,
    auto_approve: bool = False,
) -> None:
    """Require operator confirmation before proceeding.

    The scheduler prints a concise summary of the repo-state scale and filtering
    knobs, then prompts the operator to confirm with a y/n question.

    When `auto_approve=True`, no prompt is shown and stdin does not need to be a TTY.
    """

    c = console or Console()
    table = Table(title="Repo-state startup approval", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("root_commit", str(root_commit))
    table.add_row("repo_root", str(Path(repo_root).resolve()))
    table.add_row("eligible_files", str(int(eligible_files)))

    rendered_details = dict(details or {})
    for key in sorted(rendered_details.keys()):
        value = rendered_details[key]
        if value is None:
            continue
        rendered = str(value)
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(str(v) for v in value) if value else "[]"
        table.add_row(str(key), rendered)

    c.print(table)
    if bool(auto_approve):
        c.print("[green]Startup approval auto-approved[/]")
        return
    if stdin_is_tty is None:
        stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if not stdin_is_tty:
        raise ValueError(
            "Interactive confirmation required, but stdin is not a TTY. "
            f"(root_commit={root_commit} eligible_files={eligible_files})"
        )
    approved = Confirm.ask("Start scheduler main loop now?", default=False, console=c)
    if not approved:
        raise ValueError("Startup approval rejected by operator.")


