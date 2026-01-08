from __future__ import annotations

"""Scheduler startup guards that are safe to unit-test without a live database."""

from dataclasses import dataclass
from pathlib import Path
import sys
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


