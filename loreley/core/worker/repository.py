from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID
import uuid

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

        self._lock_path = self._resolve_lock_path()

    @property
    def git_dir(self) -> Path:
        """Return the .git directory location."""
        return self.worktree / ".git"

    @property
    def job_worktrees_root(self) -> Path:
        """Return the directory root used for per-job worktrees."""
        return self.worktree.parent / f"{self.worktree.name}-worktrees"

    @contextmanager
    def checkout_lease_for_job(
        self,
        *,
        job_id: str | UUID | None,
        base_commit: str,
        create_branch: bool = True,
        keep_worktree_on_failure: bool | None = None,
    ) -> Iterator[CheckoutContext]:
        """Yield an isolated git worktree for a single job.

        The worktree is created via `git worktree add` and removed when the
        context exits (unless keep_worktree_on_failure is enabled and an error
        is raised inside the context).
        """
        if not base_commit:
            raise RepositoryError("Base commit hash must be provided.")

        job_uuid: UUID | None
        if isinstance(job_id, UUID):
            job_uuid = job_id
        elif job_id is None:
            job_uuid = None
        else:
            job_uuid = UUID(str(job_id))

        keep_on_failure = bool(keep_worktree_on_failure) if keep_worktree_on_failure is not None else False

        branch_name: str | None = None
        worktree_path: Path | None = None

        with self._repo_lock():
            self.prepare()
            base_repo = self._get_repo()
            # Ensure the base commit is present locally before creating the worktree.
            self._ensure_commit_available(base_commit, repo=base_repo)
            self._prune_worktrees(repo=base_repo)

            worktree_path = self._allocate_job_worktree_path(
                job_id=job_uuid,
                base_commit=base_commit,
            )
            self._ensure_worktree_path_available(worktree_path, repo=base_repo)

            try:
                base_repo.git.worktree("add", "--detach", str(worktree_path), base_commit)
            except GitCommandError as exc:
                raise self._wrap_git_error(
                    exc,
                    f"Failed to create job worktree at {worktree_path}",
                ) from exc

        # Branch checkout happens in the job worktree (outside the base lock).
        job_repo = self._open_repo(worktree_path)
        job_label = str(job_uuid) if job_uuid is not None else "N/A"
        try:
            if create_branch and job_uuid is not None:
                branch_name = self._format_job_branch(job_uuid)
                job_repo.git.checkout("-B", branch_name, base_commit)
            else:
                job_repo.git.checkout("--detach", base_commit)
        except GitCommandError as exc:
            # Best-effort cleanup on checkout failures.
            with self._repo_lock():
                try:
                    self._remove_worktree(worktree_path)
                except Exception:
                    pass
            raise self._wrap_git_error(exc, f"Failed to checkout commit {base_commit}") from exc

        ctx = CheckoutContext(
            job_id=str(job_uuid) if job_uuid else None,
            branch_name=branch_name,
            base_commit=base_commit,
            worktree=worktree_path,
        )

        console.log(
            f"[bold green]Checked out base commit[/] job={job_label} commit={base_commit} "
            f"worktree={worktree_path}",
        )
        log.info(
            "Checked out base commit {} for job {} in worktree {}",
            base_commit,
            job_uuid,
            worktree_path,
        )

        try:
            yield ctx
        except Exception:
            if keep_on_failure:
                log.warning(
                    "Preserving failed job worktree for inspection job={} worktree={}",
                    job_uuid,
                    worktree_path,
                )
                raise
            with self._repo_lock():
                self._remove_worktree(worktree_path)
            raise
        else:
            with self._repo_lock():
                self._remove_worktree(worktree_path)

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
        """Checkout the requested base commit and optionally create a job branch.

        Note: This method returns a checkout context but does not automatically
        clean up any worktree. Prefer `checkout_lease_for_job()` for workers.
        """
        if not base_commit:
            raise RepositoryError("Base commit hash must be provided.")

        job_uuid: UUID | None
        if isinstance(job_id, UUID):
            job_uuid = job_id
        elif job_id is None:
            job_uuid = None
        else:
            job_uuid = UUID(str(job_id))

        with self._repo_lock():
            self.prepare()
            base_repo = self._get_repo()
            self._ensure_commit_available(base_commit, repo=base_repo)
            self._prune_worktrees(repo=base_repo)
            worktree_path = self._allocate_job_worktree_path(
                job_id=job_uuid,
                base_commit=base_commit,
            )
            self._ensure_worktree_path_available(worktree_path, repo=base_repo)
            try:
                base_repo.git.worktree("add", "--detach", str(worktree_path), base_commit)
            except GitCommandError as exc:
                raise self._wrap_git_error(
                    exc,
                    f"Failed to create job worktree at {worktree_path}",
                ) from exc

        job_repo = self._open_repo(worktree_path)
        branch_name: str | None = None
        try:
            if create_branch and job_uuid is not None:
                branch_name = self._format_job_branch(job_uuid)
                job_repo.git.checkout("-B", branch_name, base_commit)
            else:
                job_repo.git.checkout("--detach", base_commit)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, f"Failed to checkout commit {base_commit}") from exc

        job_label = str(job_uuid) if job_uuid is not None else "N/A"
        console.log(
            f"[bold green]Checked out base commit[/] job={job_label} commit={base_commit} worktree={worktree_path}",
        )
        log.info(
            "Checked out base commit {} for job {} in worktree {}",
            base_commit,
            job_uuid,
            worktree_path,
        )

        return CheckoutContext(
            job_id=str(job_uuid) if job_uuid else None,
            branch_name=branch_name,
            base_commit=base_commit,
            worktree=worktree_path,
        )

    def _resolve_worktree_path(self, worktree: Path | None) -> Path:
        if worktree is None:
            return self.worktree
        return Path(worktree).expanduser().resolve()

    def _repo_for_worktree(self, worktree: Path | None = None) -> Repo:
        resolved = self._resolve_worktree_path(worktree)
        if resolved == self.worktree:
            return self._get_repo()
        return self._open_repo(resolved)

    def clean_worktree(self, *, worktree: Path | None = None) -> None:
        """Reset tracked files and drop untracked artifacts."""
        target = self._resolve_worktree_path(worktree)
        if not (target / ".git").exists():
            return
        repo = self._repo_for_worktree(target)
        try:
            repo.git.reset("--hard")
            clean_args = ["-xdf"]
            for pattern in self.clean_excludes:
                clean_args.extend(["-e", pattern])
            repo.git.clean(*clean_args)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to clean worker worktree") from exc

    def current_commit(self, *, worktree: Path | None = None) -> str:
        """Return the current HEAD commit hash."""
        repo = self._repo_for_worktree(worktree)
        return repo.head.commit.hexsha

    def has_changes(self, *, worktree: Path | None = None) -> bool:
        """Return True if the worktree contains staged or unstaged changes."""
        repo = self._repo_for_worktree(worktree)
        return repo.is_dirty(untracked_files=True)

    def stage_all(self, *, worktree: Path | None = None) -> None:
        """Stage all tracked and untracked changes."""
        repo = self._repo_for_worktree(worktree)
        try:
            repo.git.add("--all")
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to stage worktree changes") from exc

    def commit(self, message: str, *, worktree: Path | None = None) -> str:
        """Create a commit with the staged changes and return the hash."""
        repo = self._repo_for_worktree(worktree)
        try:
            repo.git.commit("-m", message)
        except GitCommandError as exc:
            raise self._wrap_git_error(exc, "Failed to create worker commit") from exc
        return repo.head.commit.hexsha

    def push_branch(
        self,
        branch_name: str,
        *,
        worktree: Path | None = None,
        remote: str = "origin",
        force_with_lease: bool = False,
    ) -> None:
        """Publish the current branch to the configured remote."""
        branch = branch_name.strip()
        if not branch:
            raise RepositoryError("Branch name must be provided when pushing.")
        remote_name = remote.strip() or "origin"
        repo = self._repo_for_worktree(worktree)
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
            # Protect fetch/prune operations because they mutate the shared base clone.
            with self._repo_lock():
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

    # Worktree leasing / locking ------------------------------------------

    def _resolve_lock_path(self) -> Path:
        # Keep lock adjacent to the base worktree so multiple worker processes
        # sharing WORKER_REPO_WORKTREE coordinate without additional services.
        return self.worktree.parent / f".{self.worktree.name}.lock"

    @contextmanager
    def _repo_lock(self) -> Iterator[None]:
        """Cross-process lock protecting base repo mutations.

        This lock should be held only for short-lived operations such as clone,
        fetch/sync, and worktree add/remove/prune. Planning/coding/evaluation
        happens in per-job worktrees and should not be performed under this lock.
        """
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._lock_path, "a+", encoding="utf-8")
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            else:  # pragma: no cover - Windows fallback
                import msvcrt

                fh.seek(0)
                if fh.tell() == 0:
                    fh.write("\n")
                    fh.flush()
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                else:  # pragma: no cover - Windows fallback
                    import msvcrt

                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            fh.close()

    def _open_repo(self, worktree: Path) -> Repo:
        """Open a Repo instance for the given worktree path."""
        try:
            repo = Repo(worktree)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            raise RepositoryError(
                f"Worktree {worktree} is not a git repository.",
            ) from exc
        if self._git_env:
            repo.git.update_environment(**self._git_env)
        return repo

    def _allocate_job_worktree_path(self, *, job_id: UUID | None, base_commit: str) -> Path:
        root = self.job_worktrees_root
        if job_id is not None:
            name = str(job_id)
        else:
            suffix = uuid.uuid4().hex[:8]
            short = base_commit[:12] if base_commit else "commit"
            name = f"detached-{short}-{suffix}"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "job"
        return root / safe

    def _ensure_worktree_path_available(self, path: Path, *, repo: Repo) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return

        # Attempt to deregister the worktree first; if that fails, remove the
        # directory from disk to avoid `worktree add` collisions.
        try:
            repo.git.worktree("remove", "--force", str(path))
        except GitCommandError:
            pass
        shutil.rmtree(path, ignore_errors=True)
        try:
            self._prune_worktrees(repo=repo)
        except Exception:
            pass

    def _remove_worktree(self, path: Path) -> None:
        """Remove a previously created worktree (best-effort)."""
        if not self.git_dir.exists():
            return
        repo = self._get_repo()
        try:
            repo.git.worktree("remove", "--force", str(path))
        except GitCommandError as exc:
            log.warning("Failed to remove worktree {}: {}", path, exc)
        shutil.rmtree(path, ignore_errors=True)
        self._prune_worktrees(repo=repo)

    def _prune_worktrees(self, *, repo: Repo) -> None:
        try:
            repo.git.worktree("prune")
        except GitCommandError as exc:
            log.debug("Worktree prune skipped: {}", exc)

