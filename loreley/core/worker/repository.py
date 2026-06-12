from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import UUID
import uuid

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from sqlalchemy import and_, or_, select

from loreley.config import Settings, get_settings
from loreley.core.git import RepositoryError, fetch_origin, require_commit, wrap_git_error
from loreley.core.repo_lock import file_lock, resolve_repo_lock_path
from loreley.db.base import session_scope
from loreley.db.models import CandidateCommit, EvolutionJob, JobStatus, MapElitesArchiveCell
from loreley.naming import worker_job_branch_prefix

console = Console()
log = logger.bind(module="worker.repository")

__all__ = ["WorkerRepository", "RepositoryError", "CheckoutContext"]

@dataclass(slots=True, frozen=True)
class CheckoutContext:
    """Metadata returned after checking out a base commit for a job."""

    job_id: str | None
    branch_name: str | None
    base_commit: str
    worktree: Path


@dataclass(slots=True)
class _ProtectedBranchState:
    commits: set[str]
    branches: set[str]
    prefix: str

    def remember_commit(self, raw_value: str | None) -> None:
        commit = str(raw_value or "").strip()
        if commit:
            self.commits.add(commit)

    def remember_branch(self, raw_value: str | None) -> None:
        branch = str(raw_value or "").strip()
        if not branch:
            return
        if self.prefix and not branch.startswith(f"{self.prefix}/") and branch != self.prefix:
            return
        self.branches.add(branch)


_UNFINISHED_JOB_STATUSES = (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)


def _remember_archive_commits(session: Any, state: _ProtectedBranchState) -> None:
    rows = session.execute(
        select(MapElitesArchiveCell.commit_hash).where(
            MapElitesArchiveCell.commit_hash.is_not(None),
            MapElitesArchiveCell.commit_hash != "",
        )
    ).all()
    for (commit_hash,) in rows:
        state.remember_commit(commit_hash)


def _load_protected_job_rows(session: Any) -> Sequence[Any]:
    return session.execute(
        select(
            EvolutionJob.base_commit_hash,
            EvolutionJob.result_commit_hash,
            EvolutionJob.candidate_commit_hash,
            EvolutionJob.candidate_branch_name,
            EvolutionJob.inspiration_commit_hashes,
            EvolutionJob.status,
            EvolutionJob.ingestion_status,
            EvolutionJob.candidate_published_at,
        ).where(
            or_(
                EvolutionJob.status.in_(_UNFINISHED_JOB_STATUSES),
                and_(EvolutionJob.status == JobStatus.SUCCEEDED, _ingestion_pending()),
                _failed_candidate_ref_present(),
            )
        )
    ).all()


def _load_protected_candidate_rows(session: Any) -> Sequence[Any]:
    return session.execute(
        select(
            CandidateCommit.commit_hash,
            CandidateCommit.candidate_branch_name,
            CandidateCommit.repair_state,
            CandidateCommit.publication_status,
        ).where(
            CandidateCommit.commit_hash.is_not(None),
            CandidateCommit.commit_hash != "",
            CandidateCommit.candidate_branch_name.is_not(None),
            CandidateCommit.candidate_branch_name != "",
            CandidateCommit.repair_state.in_(("eligible", "scheduled", "repairing")),
            CandidateCommit.publication_status == "published",
        )
    ).all()


def _remember_candidate_branch_row(row: Any, state: _ProtectedBranchState) -> None:
    if len(row) < 2:
        return
    state.remember_commit(row[0])
    state.remember_branch(row[1])


def _remember_job_branch_row(row: Any, state: _ProtectedBranchState) -> None:
    (
        base_commit_hash,
        result_commit_hash,
        candidate_commit_hash,
        candidate_branch_name,
        inspiration_commit_hashes,
        status,
        ingestion_status,
        _candidate_published_at,
    ) = row
    if _job_row_protects_full_lineage(status, ingestion_status):
        state.remember_commit(base_commit_hash)
        state.remember_commit(result_commit_hash)
        state.remember_commit(candidate_commit_hash)
        state.remember_branch(candidate_branch_name)
        for commit_hash in inspiration_commit_hashes or ():
            state.remember_commit(commit_hash)
        return
    if status == JobStatus.FAILED and candidate_branch_name:
        state.remember_commit(candidate_commit_hash)
        state.remember_branch(candidate_branch_name)


def _job_row_protects_full_lineage(status: Any, ingestion_status: Any) -> bool:
    normalized = str(ingestion_status or "").strip().lower()
    return status in _UNFINISHED_JOB_STATUSES or (
        status == JobStatus.SUCCEEDED and normalized not in {"succeeded", "skipped"}
    )


def _ingestion_pending() -> Any:
    return or_(
        EvolutionJob.ingestion_status.is_(None),
        EvolutionJob.ingestion_status == "",
        EvolutionJob.ingestion_status.not_in(("succeeded", "skipped")),
    )


def _failed_candidate_ref_present() -> Any:
    return and_(
        EvolutionJob.status == JobStatus.FAILED,
        EvolutionJob.candidate_commit_hash.is_not(None),
        EvolutionJob.candidate_commit_hash != "",
        EvolutionJob.candidate_branch_name.is_not(None),
        EvolutionJob.candidate_branch_name != "",
    )


def _validated_commit_range(base_commit: str, candidate_commit: str) -> tuple[str, str]:
    base = str(base_commit or "").strip()
    candidate = str(candidate_commit or "").strip()
    if not base or not candidate:
        raise RepositoryError("Both base and candidate commits are required for repair patch.")
    return base, candidate


def _ensure_patch_budget(patch: bytes, *, max_bytes: int) -> None:
    if len(patch) <= max(0, int(max_bytes)):
        return
    raise RepositoryError(
        f"Repair patch exceeds configured byte budget ({len(patch)} > {max_bytes})."
    )


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
        self.job_branch_prefix = worker_job_branch_prefix(self.settings.experiment_id).strip("/")
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
        attempt_token: str | UUID | None = None,
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
            base_repo = self._prepare_base_repo_for_checkout(base_commit=base_commit)

            worktree_path = self._allocate_job_worktree_path(
                job_id=job_uuid,
                base_commit=base_commit,
                attempt_token=attempt_token,
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
                branch_name = self._format_job_branch(job_uuid, attempt_token=attempt_token)
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

        # `prepare()` mutates the shared base clone (clone/fetch/reset/checkout),
        # so it must serialize against other worker/scheduler base-repo access.
        with self._repo_lock():
            with self._progress() as progress:
                for description, action in steps:
                    task_id = progress.add_task(description, total=1)
                    action()
                    progress.update(task_id, completed=1)

    def _prepare_base_repo_for_checkout(self, *, base_commit: str) -> Repo:
        """Prepare the shared base clone for a commit-specific checkout.

        Job-specific worktree creation only needs the base clone to exist, the
        `origin` remote to be configured correctly, and the requested commit to
        be available locally. Skipping a full upstream sync here avoids
        serialising every worker behind `fetch + clean + checkout` on the
        shared base clone.
        """

        self._ensure_worktree_ready()
        base_repo = self._get_repo()
        self._ensure_remote_origin(repo=base_repo)
        self._ensure_commit_available(base_commit, repo=base_repo)
        if self.enable_lfs:
            self._sync_lfs(repo=base_repo)
        self._prune_worktrees(repo=base_repo)
        return base_repo

    def checkout_for_job(
        self,
        *,
        job_id: str | UUID | None,
        base_commit: str,
        create_branch: bool = True,
        attempt_token: str | UUID | None = None,
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
            base_repo = self._prepare_base_repo_for_checkout(base_commit=base_commit)
            worktree_path = self._allocate_job_worktree_path(
                job_id=job_uuid,
                base_commit=base_commit,
                attempt_token=attempt_token,
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
                branch_name = self._format_job_branch(job_uuid, attempt_token=attempt_token)
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

    def reset_mixed_to_commit(
        self,
        commit_hash: str,
        *,
        worktree: Path | None = None,
    ) -> None:
        """Move HEAD to a commit while preserving the resulting diff in the worktree."""

        target_commit = str(commit_hash or "").strip()
        if not target_commit:
            raise RepositoryError("Commit hash must be provided for mixed reset.")
        repo = self._repo_for_worktree(worktree)
        try:
            repo.git.reset("--mixed", target_commit)
        except GitCommandError as exc:
            raise self._wrap_git_error(
                exc,
                f"Failed to reset worker worktree to {target_commit}",
            ) from exc

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

    def diff_summary_between_commits(
        self,
        *,
        base_commit: str,
        candidate_commit: str,
        worktree: Path | None = None,
    ) -> str:
        """Return a bounded, human-readable summary of a candidate diff."""

        base = str(base_commit or "").strip()
        candidate = str(candidate_commit or "").strip()
        if not base or not candidate:
            raise RepositoryError("Both base and candidate commits are required for repair diff.")
        target = self._resolve_worktree_path(worktree)
        try:
            result = subprocess.run(
                [self.git_bin or "git", "-C", str(target), "diff", "--stat", f"{base}..{candidate}"],
                check=True,
                capture_output=True,
                text=True,
                env=self._env,
            )
        except subprocess.CalledProcessError as exc:
            raise RepositoryError(
                f"Failed to summarize repair diff {base}..{candidate}: {exc.stderr or exc}"
            ) from exc
        return result.stdout.strip()

    def apply_diff_between_commits(
        self,
        *,
        base_commit: str,
        candidate_commit: str,
        worktree: Path | None = None,
        max_bytes: int = 65_536,
    ) -> None:
        """Apply the patch represented by base..candidate into a checked-out worktree."""

        base, candidate = _validated_commit_range(base_commit, candidate_commit)
        target = self._resolve_worktree_path(worktree)
        patch = self._diff_patch_bytes(base=base, candidate=candidate, target=target)
        if not patch:
            return
        _ensure_patch_budget(patch, max_bytes=max_bytes)
        self._apply_patch_bytes(patch, base=base, candidate=candidate, target=target)

    def _diff_patch_bytes(self, *, base: str, candidate: str, target: Path) -> bytes:
        try:
            diff_result = subprocess.run(
                [self.git_bin or "git", "-C", str(target), "diff", "--binary", f"{base}..{candidate}"],
                check=True,
                capture_output=True,
                env=self._env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            raise RepositoryError(
                f"Failed to build repair patch {base}..{candidate}: {stderr or exc}"
            ) from exc
        return diff_result.stdout or b""

    def _apply_patch_bytes(
        self,
        patch: bytes,
        *,
        base: str,
        candidate: str,
        target: Path,
    ) -> None:
        try:
            subprocess.run(
                [self.git_bin or "git", "-C", str(target), "apply", "--whitespace=nowarn"],
                input=patch,
                check=True,
                capture_output=True,
                env=self._env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            log.warning(
                "Repair patch application failed base={} candidate={}: {}",
                base,
                candidate,
                stderr or exc,
            )
            raise RepositoryError(
                f"Failed to apply repair patch {base}..{candidate}: {stderr or exc}"
            ) from exc

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

        try:
            protected_commits, protected_branches = self._load_protected_job_branch_state()
        except Exception as exc:
            log.warning("Skipping job branch pruning; protected ref lookup failed: {}", exc)
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
            if branch in protected_branches:
                log.debug("Skipping protected job branch {}", branch)
                continue
            try:
                head_commit = str(repo.git.rev_parse(ref_name)).strip()
            except GitCommandError as exc:
                log.warning("Skipping stale job branch {} because its head commit could not be resolved: {}", branch, exc)
                continue
            if self._branch_contains_protected_commit(
                repo=repo,
                head_commit=head_commit,
                protected_commits=protected_commits,
            ):
                log.debug("Skipping protected job branch {} at {}", branch, head_commit)
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

    def _load_protected_job_branch_state(self) -> tuple[set[str], set[str]]:
        """Return commit hashes and branch names that must not lose their last ref."""

        state = _ProtectedBranchState(
            commits=set(),
            branches=set(),
            prefix=self.job_branch_prefix.strip("/"),
        )
        with session_scope() as session:
            _remember_archive_commits(session, state)
            job_rows = _load_protected_job_rows(session)
            candidate_rows = _load_protected_candidate_rows(session)
            for row in candidate_rows:
                _remember_candidate_branch_row(row, state)
        for row in job_rows:
            _remember_job_branch_row(row, state)
        return state.commits, state.branches

    @staticmethod
    def _commit_matches_protected_commit(head_commit: str, protected_commits: set[str]) -> bool:
        """Return True when `head_commit` matches a protected full or short hash."""

        normalized_head = str(head_commit or "").strip().lower()
        if not normalized_head:
            return False
        for protected in protected_commits:
            candidate = str(protected or "").strip().lower()
            if not candidate:
                continue
            if normalized_head == candidate or normalized_head.startswith(candidate) or candidate.startswith(normalized_head):
                return True
        return False

    @classmethod
    def _branch_contains_protected_commit(
        cls,
        *,
        repo: Repo,
        head_commit: str,
        protected_commits: set[str],
    ) -> bool:
        """Return True when a branch head preserves any protected commit."""

        normalized_head = str(head_commit or "").strip()
        if not normalized_head:
            return False
        if cls._commit_matches_protected_commit(normalized_head, protected_commits):
            return True

        for protected in protected_commits:
            ancestor = str(protected or "").strip()
            if not ancestor:
                continue
            try:
                if cls._is_commit_ancestor(repo=repo, ancestor=ancestor, rev=normalized_head):
                    return True
            except GitCommandError as exc:
                log.warning(
                    "Skipping protected ancestor check ancestor={} head={} because ancestry could not be resolved: {}",
                    ancestor,
                    normalized_head,
                    exc,
                )
        return False

    @staticmethod
    def _is_commit_ancestor(*, repo: Repo, ancestor: str, rev: str) -> bool:
        """Return True when `ancestor` is reachable from `rev`."""

        try:
            repo.git.merge_base(ancestor, rev, is_ancestor=True)
        except GitCommandError as exc:
            if exc.status == 1:
                return False
            raise
        return True

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
        fetch_origin(
            repo,
            remote="origin",
            fetch_depth=self.fetch_depth,
            refspecs=refspecs,
        )

    def _sync_lfs(self, *, repo: Repo | None = None) -> None:
        repo = repo or self._get_repo()
        try:
            repo.git.lfs("install", "--local")
            repo.git.lfs("fetch", "origin")
        except GitCommandError as exc:
            log.warning("Git LFS sync skipped: {}", exc)

    def _format_job_branch(
        self,
        job_id: str | UUID,
        *,
        attempt_token: str | UUID | None = None,
    ) -> str:
        raw = str(job_id)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")
        safe = safe or "job"
        if attempt_token is not None:
            suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", str(attempt_token)).strip("-")
            suffix = suffix[:8] or "attempt"
            safe = f"{safe}-{suffix}"
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
        return wrap_git_error(exc, context)

    def _ensure_commit_available(
        self,
        commit_hash: str,
        *,
        repo: Repo | None = None,
    ) -> None:
        repo = repo or self._get_repo()
        require_commit(
            repo,
            commit_hash,
            remote="origin",
            fetch_depth=self.fetch_depth,
            console=console,
        )

    # Worktree leasing / locking ------------------------------------------

    def _resolve_lock_path(self) -> Path:
        # Keep lock adjacent to the base worktree so multiple worker processes
        # sharing WORKER_REPO_WORKTREE coordinate without additional services.
        return resolve_repo_lock_path(self.worktree)

    @contextmanager
    def _repo_lock(self) -> Iterator[None]:
        """Cross-process lock protecting base repo mutations.

        This lock should be held only for short-lived operations such as clone,
        fetch/sync, and worktree add/remove/prune. Planning/coding/evaluation
        happens in per-job worktrees and should not be performed under this lock.
        """
        with file_lock(self._lock_path):
            yield

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

    def _allocate_job_worktree_path(
        self,
        *,
        job_id: UUID | None,
        base_commit: str,
        attempt_token: str | UUID | None = None,
    ) -> Path:
        root = self.job_worktrees_root
        if job_id is not None:
            name = str(job_id)
        else:
            suffix = uuid.uuid4().hex[:8]
            short = base_commit[:12] if base_commit else "commit"
            name = f"detached-{short}-{suffix}"
        if attempt_token is not None:
            attempt_suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", str(attempt_token)).strip("-")
            attempt_suffix = attempt_suffix[:8] or "attempt"
            name = f"{name}-{attempt_suffix}"
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
