from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from loreley.config import Settings, get_settings

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
        self.job_branch_ttl_hours = max(0, int(self.settings.worker_repo_job_branch_ttl_hours))

        self._env = os.environ.copy()
        self._env.setdefault("GIT_TERMINAL_PROMPT", "0")
        self._repo: Repo | None = None

        author_name = (self.settings.worker_evolution_commit_author or "").strip()
        author_email = (self.settings.worker_evolution_commit_email or "").strip()
        if author_name:
            self._env.setdefault("GIT_AUTHOR_NAME", author_name)
            self._env.setdefault("GIT_COMMITTER_NAME", author_name)
        if author_email:
            self._env.setdefault("GIT_AUTHOR_EMAIL", author_email)
            self._env.setdefault("GIT_COMMITTER_EMAIL", author_email)

        self._git_env: dict[str, str] = {
            key: value for key, value in self._env.items() if key.upper().startswith("GIT_")
        }
        if self.git_bin:
            self._git_env.setdefault("GIT_PYTHON_GIT_EXECUTABLE", self.git_bin)

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
        repo = self._get_repo()
        self._ensure_commit_available(base_commit, repo=repo)
        try:
            repo.git.rev_parse("--verify", base_commit)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, f"Failed to verify commit {base_commit}") from exc

        branch_name: str | None = None
        try:
            if create_branch and job_id:
                branch_name = self._format_job_branch(job_id)
                repo.git.checkout("-B", branch_name, base_commit)
            else:
                repo.git.checkout("--detach", base_commit)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, f"Failed to checkout commit {base_commit}") from exc

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
        repo = self._get_repo()
        try:
            repo.git.reset("--hard")
            clean_args = ["-xdf"]
            for pattern in self.clean_excludes:
                clean_args.extend(["-e", pattern])
            repo.git.clean(*clean_args)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to clean worker worktree") from exc

    def current_commit(self) -> str:
        """Return the current HEAD commit hash."""
        repo = self._get_repo()
        return repo.head.commit.hexsha

    def has_changes(self) -> bool:
        """Return True if the worktree contains staged or unstaged changes."""
        repo = self._get_repo()
        return repo.is_dirty(untracked_files=True)

    def stage_all(self) -> None:
        """Stage all tracked and untracked changes."""
        repo = self._get_repo()
        try:
            repo.git.add("--all")
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to stage worktree changes") from exc

    def commit(self, message: str) -> str:
        """Create a commit with the staged changes and return the hash."""
        repo = self._get_repo()
        try:
            repo.git.commit("-m", message)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to create worker commit") from exc
        return repo.head.commit.hexsha

    def push_branch(
        self,
        branch_name: str,
        *,
        remote: str = "origin",
        force_with_lease: bool = False,
    ) -> None:
        """Publish the current branch to the configured remote."""
        branch = branch_name.strip()
        if not branch:
            raise RepositoryError("Branch name must be provided when pushing.")
        remote_name = remote.strip() or "origin"
        repo = self._get_repo()
        push_args = []
        if force_with_lease:
            push_args.append("--force-with-lease")
        push_args.extend([remote_name, f"{branch}:{branch}"])
        try:
            repo.git.push(*push_args)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, f"Failed to push branch {branch}") from exc
        console.log(
            f"[green]Pushed worker branch[/] branch={branch} remote={remote_name}",
        )
        log.info("Pushed branch {} to {}", branch, remote_name)

    def delete_remote_branch(
        self,
        branch_name: str,
        *,
        remote: str = "origin",
    ) -> None:
        """Remove a remote branch without touching local history."""
        branch = branch_name.strip()
        if not branch:
            raise RepositoryError("Branch name must be provided when deleting.")
        remote_name = remote.strip() or "origin"
        repo = self._get_repo()
        try:
            repo.git.push(remote_name, f":{branch}")
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, f"Failed to delete remote branch {branch}") from exc
        console.log(
            f"[yellow]Deleted remote branch[/] branch={branch} remote={remote_name}",
        )
        log.info("Deleted remote branch {} from {}", branch, remote_name)

    def prune_stale_job_branches(self) -> int:
        """Delete remote job branches that exceeded their retention window."""
        prefix = self.job_branch_prefix
        ttl_hours = self.job_branch_ttl_hours
        if ttl_hours <= 0 or not prefix:
            return 0
        cutoff_ts = datetime.now(timezone.utc).timestamp() - (ttl_hours * 3600)
        repo = self._get_repo()
        try:
            self._fetch(repo=repo)
        except RepositoryError as exc:
            log.warning("Skipping job branch pruning; fetch failed: {}", exc)
            return 0

        pattern = f"refs/remotes/origin/{prefix}/*"
        try:
            output = repo.git.for_each_ref(
                "--format=%(refname) %(committerdate:unix)",
                pattern,
            )
        except GitCommandError as exc:
            log.warning("Failed to enumerate job branches for pruning: {}", exc)
            return 0

        pruned = 0
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            ref_name, _, ts_part = stripped.partition(" ")
            if not ts_part:
                continue
            try:
                commit_ts = int(ts_part)
            except ValueError:
                continue
            if commit_ts >= cutoff_ts:
                continue
            branch = ref_name.replace("refs/remotes/origin/", "", 1)
            if not branch.startswith(prefix):
                continue
            try:
                self.delete_remote_branch(branch)
                pruned += 1
            except RepositoryError as exc:
                log.warning("Failed to delete stale job branch {}: {}", branch, exc)

        if pruned:
            console.log(
                f"[yellow]Pruned {pruned} stale job branch"
                f"{'es' if pruned != 1 else ''} (>={ttl_hours}h old)[/]",
            )
            log.info(
                "Pruned {} stale job branches older than {}h",
                pruned,
                ttl_hours,
            )
        return pruned

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

        repo = self._get_repo()
        self._ensure_remote_origin(repo=repo)
        self._fetch(repo=repo)
        if self.enable_lfs:
            self._sync_lfs(repo=repo)

        # Keep local tracking branch aligned with origin.
        if self.branch:
            self.clean_worktree()
            try:
                repo.git.checkout("-B", self.branch, f"origin/{self.branch}")
            except GitCommandError as exc:
                raise self._wrap_git_error(
                    exc,
                    f"Failed to sync local branch {self.branch}",
                ) from exc

    def _clone(self) -> None:
        parent = self.worktree.parent
        parent.mkdir(parents=True, exist_ok=True)

        clone_kwargs: dict[str, Any] = {}
        if self.branch:
            clone_kwargs["branch"] = self.branch
        if self._git_env:
            clone_kwargs["env"] = self._git_env
        multi_options: list[str] = []
        if self.fetch_depth:
            multi_options.append(f"--depth={self.fetch_depth}")
        if multi_options:
            clone_kwargs["multi_options"] = multi_options

        try:
            repo = Repo.clone_from(
                self.remote_url,
                str(self.worktree),
                **clone_kwargs,
            )
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to clone worker repository") from exc
        self._configure_repo(repo)

    def _ensure_remote_origin(self, *, repo: Repo | None = None) -> None:
        repo = repo or self._get_repo()
        try:
            origin = repo.remote("origin")
        except ValueError:
            origin = None

        if origin is None:
            try:
                repo.create_remote("origin", self.remote_url)
            except GitCommandError as exc:
                raise self._wrap_git_error(exc, "Failed to add origin remote") from exc
            return

        current = origin.url
        if current == self.remote_url:
            return

        log.warning("Updating origin remote from {} to {}", current, self.remote_url)
        try:
            origin.set_url(self.remote_url)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to update origin remote") from exc

    def _fetch(
        self,
        refspecs: Sequence[str] | None = None,
        *,
        repo: Repo | None = None,
    ) -> None:
        repo = repo or self._get_repo()
        fetch_args = ["--prune", "--tags"]
        if self.fetch_depth:
            fetch_args.append(f"--depth={self.fetch_depth}")
        fetch_args.append("origin")
        if refspecs:
            fetch_args.extend(refspecs)
        try:
            repo.git.fetch(*fetch_args)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to fetch from origin") from exc

    def _sync_lfs(self, *, repo: Repo | None = None) -> None:
        repo = repo or self._get_repo()
        try:
            repo.git.lfs("install", "--local")
            repo.git.lfs("fetch", "origin")
        except GitCommandError as exc:
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

    def _get_repo(self) -> Repo:
        if self._repo and self._repo.working_tree_dir:
            if Path(self._repo.working_tree_dir).resolve() == self.worktree:
                return self._repo
        if not self.git_dir.exists():
            raise RepositoryError(
                f"Worktree {self.worktree} is not a git repository.",
            )
        try:
            repo = Repo(self.worktree)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise RepositoryError(
                f"Worktree {self.worktree} is not a git repository.",
            ) from exc
        return self._configure_repo(repo)

    def _configure_repo(self, repo: Repo) -> Repo:
        if self._git_env:
            repo.git.update_environment(**self._git_env)
        self._repo = repo
        return repo

    def _wrap_git_error(self, exc: GitCommandError, context: str) -> RepositoryError:
        command = self._command_tuple(exc.command)
        sanitized = self._sanitize_command(command) if command else None
        suffix = ""
        if sanitized:
            suffix = f": {sanitized}"
        message = f"{context}{suffix} (exit {exc.status})"
        status = exc.status if isinstance(exc.status, int) else None
        return RepositoryError(
            message,
            cmd=command,
            returncode=status,
            stdout=getattr(exc, "stdout", None),
            stderr=getattr(exc, "stderr", None),
        )

    @staticmethod
    def _command_tuple(command: str | Sequence[str] | None) -> tuple[str, ...] | None:
        if not command:
            return None
        if isinstance(command, str):
            return (command,)
        return tuple(str(part) for part in command)

    @staticmethod
    def _sanitize_command(cmd: Sequence[str]) -> str:
        sanitized = [WorkerRepository._sanitize_value(part) for part in cmd]
        return shlex.join(sanitized)

    @staticmethod
    def _sanitize_value(value: str) -> str:
        parsed = urlsplit(value)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            netloc = f"***@{host}"
            return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        return value

    def _ensure_commit_available(
        self,
        commit_hash: str,
        *,
        repo: Repo | None = None,
    ) -> None:
        repo = repo or self._get_repo()
        if self._has_object(commit_hash, repo=repo):
            return

        log.info("Commit {} missing locally; refreshing from origin", commit_hash)
        self._fetch(repo=repo)
        if self._has_object(commit_hash, repo=repo):
            return

        if self._is_shallow(repo=repo):
            log.info("Repository is shallow; unshallowing to retrieve {}", commit_hash)
            try:
                repo.git.fetch("--unshallow", "origin")
            except GitCommandError as exc:
                raise self._wrap_git_error(exc, "Failed to unshallow repository") from exc
            else:
                if self._has_object(commit_hash, repo=repo):
                    return

        raise RepositoryError(
            f"Commit {commit_hash} is not available locally after fetching from origin.",
        )

    def _has_object(self, obj_ref: str, *, repo: Repo | None = None) -> bool:
        repo = repo or self._get_repo()
        try:
            repo.commit(obj_ref)
        except (BadName, GitCommandError, ValueError):
            return False
        return True

    def _is_shallow(self, *, repo: Repo | None = None) -> bool:
        repo = repo or self._get_repo()
        try:
            result = repo.git.rev_parse("--is-shallow-repository")
        except GitCommandError:
            return False
        return result.strip().lower() == "true"

