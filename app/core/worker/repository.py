from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import UUID

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from app.config import Settings, get_settings

console = Console()
log = logger.bind(module="worker.repository")

__all__ = ["WorkerRepository", "RepositoryError", "CheckoutContext"]


class RepositoryError(RuntimeError):
    """Raised when the worker repository fails to perform a git operation."""

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


@dataclass(slots=True, frozen=True)
class CheckoutContext:
    """Metadata returned after checking out a base commit for a job."""

    job_id: str | None
    branch_name: str | None
    base_commit: str
    worktree: Path


class WorkerRepository:
    """Manage the git worktree used by a worker process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        remote_url = self.settings.worker_repo_remote_url
        if not remote_url:
            raise RepositoryError(
                "Worker repository remote is not configured. "
                "Set WORKER_REPO_REMOTE_URL to the upstream git URL.",
            )
        self.remote_url: str = remote_url

        self.worktree = Path(self.settings.worker_repo_worktree).expanduser().resolve()
        self.branch = self.settings.worker_repo_branch
        self.git_bin = self.settings.worker_repo_git_bin
        self.fetch_depth = self.settings.worker_repo_fetch_depth
        self.clean_excludes = tuple(self.settings.worker_repo_clean_excludes)
        self.job_branch_prefix = self.settings.worker_repo_job_branch_prefix.strip("/")
        self.enable_lfs = self.settings.worker_repo_enable_lfs

        self._env = os.environ.copy()
        self._env.setdefault("GIT_TERMINAL_PROMPT", "0")

    @property
    def git_dir(self) -> Path:
        """Return the .git directory location."""
        return self.worktree / ".git"

    def prepare(self) -> None:
        """Ensure the worktree exists and matches the upstream state."""
        steps = (
            ("Preparing worktree", self._ensure_worktree_ready),
            ("Syncing upstream repository", self._sync_upstream),
        )

        with self._progress() as progress:
            for description, action in steps:
                task_id = progress.add_task(description, total=1)
                action()
                progress.update(task_id, completed=1)

    def checkout_for_job(
        self,
        *,
        job_id: str | UUID | None,
        base_commit: str,
        create_branch: bool = True,
    ) -> CheckoutContext:
        """Checkout the requested base commit and optionally create a job branch."""
        if not base_commit:
            raise RepositoryError("Base commit hash must be provided.")

        self.prepare()
        self.clean_worktree()

        # Ensure the base commit is present locally.
        self._fetch([base_commit])
        self._run_git("rev-parse", "--verify", base_commit)

        branch_name: str | None = None
        if create_branch and job_id:
            branch_name = self._format_job_branch(job_id)
            self._run_git("checkout", "-B", branch_name, base_commit)
        else:
            self._run_git("checkout", "--detach", base_commit)

        job_label = str(job_id) if job_id is not None else "N/A"
        console.log(
            f"[bold green]Checked out base commit[/] job={job_label} "
            f"commit={base_commit}",
        )
        log.info(
            "Checked out base commit {} for job {}",
            base_commit,
            job_id,
        )

        return CheckoutContext(
            job_id=str(job_id) if job_id else None,
            branch_name=branch_name,
            base_commit=base_commit,
            worktree=self.worktree,
        )

    def clean_worktree(self) -> None:
        """Reset tracked files and drop untracked artifacts."""
        if not self.git_dir.exists():
            return
        self._run_git("reset", "--hard")

        clean_cmd = ["clean", "-xdf"]
        for pattern in self.clean_excludes:
            clean_cmd.extend(["-e", pattern])
        self._run_git(*clean_cmd)

    def current_commit(self) -> str:
        """Return the current HEAD commit hash."""
        result = self._run_git("rev-parse", "HEAD")
        return result.stdout.strip()

    # Internal helpers -----------------------------------------------------

    def _ensure_worktree_ready(self) -> None:
        if not self.worktree.exists():
            self.worktree.mkdir(parents=True, exist_ok=True)

        if not self.git_dir.exists():
            if any(self.worktree.iterdir()):
                raise RepositoryError(
                    f"Worktree {self.worktree} exists but is not a git repository.",
                )
            console.log(f"[yellow]Cloning repository into[/] {self.worktree}")
            self._clone()

    def _sync_upstream(self) -> None:
        if not self.git_dir.exists():
            return

        self._ensure_remote_origin()
        self._fetch()
        if self.enable_lfs:
            self._sync_lfs()

        # Keep local tracking branch aligned with origin.
        if self.branch:
            self._run_git("checkout", "-B", self.branch, f"origin/{self.branch}")

    def _clone(self) -> None:
        parent = self.worktree.parent
        parent.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [self.git_bin, "clone", "--origin", "origin"]
        if self.fetch_depth:
            cmd.append(f"--depth={self.fetch_depth}")
        if self.branch:
            cmd.extend(["--branch", self.branch])
        cmd.extend([self.remote_url, str(self.worktree)])
        self._run(cmd, cwd=parent)

    def _ensure_remote_origin(self) -> None:
        try:
            current = self._run_git("remote", "get-url", "origin").stdout.strip()
        except RepositoryError:
            current = None

        if not current:
            self._run_git("remote", "add", "origin", self.remote_url)
            return

        if current == self.remote_url:
            return

        log.warning("Updating origin remote from {} to {}", current, self.remote_url)
        self._run_git("remote", "set-url", "origin", self.remote_url)

    def _fetch(self, refspecs: Sequence[str] | None = None) -> None:
        args = ["fetch", "--prune", "--tags"]
        if self.fetch_depth:
            args.append(f"--depth={self.fetch_depth}")
        args.append("origin")
        if refspecs:
            args.extend(refspecs)
        self._run_git(*args)

    def _sync_lfs(self) -> None:
        try:
            self._run_git("lfs", "install", "--local")
            self._run_git("lfs", "fetch", "origin")
        except RepositoryError as exc:
            log.warning("Git LFS sync skipped: {}", exc)

    def _format_job_branch(self, job_id: str | UUID) -> str:
        raw = str(job_id)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
        safe = safe or "job"
        prefix = self.job_branch_prefix
        if prefix:
            return f"{prefix}/{safe}"
        return safe

    def _progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        )

    def _run_git(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return self._run(
            (self.git_bin, *args),
            cwd=cwd or self.worktree,
        )

    def _run(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = tuple(str(part) for part in cmd)
        working_dir = cwd or self.worktree
        working_dir.mkdir(parents=True, exist_ok=True)

        log.debug("Executing command [{}]: {}", working_dir, self._format_cmd(command))
        result = subprocess.run(
            command,
            cwd=str(working_dir),
            env=self._env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RepositoryError(
                f"Command failed with exit code {result.returncode}: "
                f"{self._format_cmd(command)}",
                cmd=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return result

    @staticmethod
    def _format_cmd(cmd: Sequence[str]) -> str:
        return shlex.join(cmd)

