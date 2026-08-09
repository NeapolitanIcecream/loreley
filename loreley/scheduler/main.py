"""Central scheduler that coordinates evolution jobs, workers, and MAP-Elites."""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from loguru import logger
from rich.console import Console
from sqlalchemy import and_, case, func, select

from loreley.config import Settings, get_settings, resolve_objective_contract
from loreley.core.experiments import ExperimentError, bootstrap_instance
from loreley.core.git import RepositoryError, require_commit, wrap_git_error
from loreley.core.job_state import pending_ingestion_job_conditions
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.sampler import MapElitesSampler
from loreley.core.repo_lock import repo_lock
from loreley.core.progress import CampaignProgress, load_campaign_progress
from loreley.db.base import ensure_database_schema, get_engine, session_scope
from loreley.db.locks import (
    AdvisoryLock,
    release_pg_advisory_lock,
    try_acquire_pg_advisory_lock,
    uuid_to_pg_bigint_lock_key,
)
from loreley.db.models import CommitCard, MapElitesArchiveCell, Metric
from loreley.scheduler.baselines import BaselineBootstrapResult, BaselineBootstrapService
from loreley.naming import resolve_experiment_namespace, resolve_experiment_uuid
from loreley.scheduler.ingestion import MapElitesIngestion
from loreley.scheduler.job_scheduler import JobScheduler
from loreley.scheduler.startup_approval import (
    require_repo_writable,
    require_interactive_repo_state_root_approval,
    scan_repo_state_root,
)

console = Console()
log = logger.bind(module="scheduler.main")

__all__ = ["EvolutionScheduler", "main"]


class SchedulerError(RuntimeError):
    """Raised when the scheduler cannot continue."""


class SchedulerLockError(SchedulerError):
    """Raised when a per-experiment scheduler lock cannot be obtained."""


@dataclass(frozen=True, slots=True)
class _RepoStateStartupApproval:
    root_commit: str
    eligible_files: int
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SeedWarmupJobCounts:
    seed_jobs: int
    unfinished_seed_jobs: int
    pending_ingestion_seed_jobs: int


@dataclass(frozen=True, slots=True)
class _SeedSchedulingBudget:
    root_commit_hash: str
    target_samples: int
    available_slots: int
    unfinished_jobs: int
    ordered_islands: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SeedIslandDemand:
    island_id: str
    warmup_samples: int
    remaining_samples: int


@dataclass(frozen=True, slots=True)
class _SeedIslandAllocation:
    demand: _SeedIslandDemand
    count: int


def _remaining_seed_demand(
    *,
    warmup_samples: int,
    target_samples: int,
    counts: _SeedWarmupJobCounts,
) -> int:
    pending = counts.unfinished_seed_jobs + counts.pending_ingestion_seed_jobs
    if warmup_samples < target_samples:
        return max(0, target_samples - warmup_samples - pending)
    return max(0, 1 - pending)


def _allocate_seed_slots(
    demands: tuple[_SeedIslandDemand, ...],
    available_slots: int,
) -> tuple[_SeedIslandAllocation, ...]:
    """Allocate global capacity one slot per island per round."""

    allocations = [0 for _ in demands]
    slots = max(0, int(available_slots))
    while slots > 0:
        made_progress = False
        for index, demand in enumerate(demands):
            if allocations[index] >= demand.remaining_samples:
                continue
            allocations[index] += 1
            slots -= 1
            made_progress = True
            if slots <= 0:
                break
        if not made_progress:
            break
    return tuple(
        _SeedIslandAllocation(demand=demand, count=count)
        for demand, count in zip(demands, allocations)
        if count > 0
    )


def _pending_ingestion_log_suffix(pending_ingestion: int | None) -> str:
    if pending_ingestion is None:
        return ""
    return f" pending_ingestion={pending_ingestion}"


class EvolutionScheduler:
    """Orchestrate job sampling, dispatching, and MAP-Elites maintenance."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        base_settings = settings or get_settings()
        self.settings = base_settings
        self.console = console
        self._advisory_lock: AdvisoryLock | None = None
        # Ensure DB schema exists before repo-specific bootstrap (seed/validate marker).
        ensure_database_schema(settings=base_settings)
        self.repo_root = self._resolve_repo_root()
        try:
            self.repository, effective_settings = bootstrap_instance(
                settings=base_settings,
                repo_root=self.repo_root,
            )
        except ExperimentError as exc:
            raise SchedulerError(str(exc)) from exc
        self.settings = effective_settings
        self._repo = self._init_repo()
        try:
            require_repo_writable(repo_root=self.repo_root, repo=self._repo, console=self.console)
        except ValueError as exc:
            raise SchedulerError(str(exc)) from exc
        self._root_commit_hash = (self.settings.mapelites_experiment_root_commit or "").strip() or None
        self._max_total_jobs = self._require_max_total_jobs()

        # Enforce single-scheduler-per-experiment using a session-level Postgres advisory lock.
        self._advisory_lock = self._acquire_experiment_lock()

        # Fail fast unless the operator explicitly approves the root eligible file count.
        try:
            self._startup_scan_and_validate_repo_state_approval()
        except Exception:
            self.close()
            raise

        try:
            self.manager = MapElitesManager(
                settings=self.settings,
                repo_root=self.repo_root,
            )
            self.manager.validate_configured_islands()
        except Exception as exc:
            self.close()
            raise SchedulerError(
                f"MAP-Elites startup validation failed: {exc}",
            ) from exc
        self.sampler = MapElitesSampler(manager=self.manager, settings=self.settings)
        self.job_scheduler = JobScheduler(
            settings=self.settings,
            console=self.console,
            sampler=self.sampler,
            repo_root=self.repo_root,
        )
        self._total_jobs_count = int(self.job_scheduler.count_total_jobs())
        self.ingestion = MapElitesIngestion(
            settings=self.settings,
            console=self.console,
            repo_root=self.repo_root,
            repo=self._repo,
            manager=self.manager,
        )
        self.baseline_bootstrap = BaselineBootstrapService(
            settings=self.settings,
            repo_root=self.repo_root,
            console=self.console,
        )
        self._stop_requested = False

        # Optionally initialise an explicit experiment root commit so that the
        # archive and database both contain a stable starting point before any
        # evolution jobs run.
        if self._root_commit_hash:
            try:
                self.ingestion.initialise_root_commit(self._root_commit_hash)
            except Exception as exc:
                self.close()
                raise SchedulerError(str(exc)) from exc

    # Public API ------------------------------------------------------------

    def run_forever(self) -> None:
        """Start the scheduler loop until interrupted."""

        interval = max(1.0, float(self.settings.scheduler_poll_interval_seconds))
        try:
            self.console.log(
                "[bold green]Scheduler online[/] repo={} experiment={} interval={}s max_unfinished={}".format(
                    self.repo_root,
                    self.settings.experiment_id,
                    interval,
                    self.settings.scheduler_max_unfinished_jobs,
                ),
            )
            self._install_signal_handlers()
            while not self._stop_requested:
                start = time.perf_counter()
                self.tick()
                elapsed = time.perf_counter() - start
                sleep_for = max(0.0, interval - elapsed)
                # Sleep in small increments so SIGINT/SIGTERM can cut the wait short
                # instead of blocking until the next full tick boundary.
                self._sleep_with_stop(sleep_for)
            self.console.log("[bold yellow]Scheduler stopped[/]")
        finally:
            self.close()

    def tick(self) -> dict[str, int]:
        """Execute a full scheduler cycle."""

        stats: dict[str, int] = {}
        stats["ingested"] = self.ingestion.ingest_completed_jobs()
        reclaim_fn = getattr(self.job_scheduler, "reclaim_stale_running_jobs", None)
        reclaimed = reclaim_fn() if callable(reclaim_fn) else None
        stats["reclaimed_pending"] = int(getattr(reclaimed, "requeued", 0) or 0)
        stats["reclaimed_failed"] = int(getattr(reclaimed, "failed", 0) or 0)
        endpoint = self._identity_endpoint_progress()
        if endpoint is not None and endpoint.identity_target_reached:
            stats["identity_endpoint_reached"] = 1
            stats["identity_count"] = endpoint.distinct_passed_evaluation_identities
            stats["identity_overshoot"] = endpoint.identity_overshoot
            cancel = getattr(
                self.job_scheduler,
                "cancel_pending_for_identity_endpoint",
                None,
            )
            stats["endpoint_cancelled_pending"] = int(cancel() if callable(cancel) else 0)
            stats["dispatched"] = 0
            stats["seed_scheduled"] = 0
            stats["scheduled"] = 0
            stats["unfinished"] = self.job_scheduler.count_unfinished_jobs()
            pending_ingestion = self.ingestion.count_pending_ingestion_jobs()
            stats["pending_ingestion"] = pending_ingestion
            self.console.log(
                "[bold magenta]Scheduler identity endpoint[/] "
                f"identities={endpoint.distinct_passed_evaluation_identities}/"
                f"{endpoint.identity_target} overshoot={endpoint.identity_overshoot} "
                f"unfinished={stats['unfinished']} pending_ingestion={pending_ingestion}"
            )
            if stats["unfinished"] == 0 and pending_ingestion == 0:
                self._create_primary_objective_branch_if_possible()
                self.console.log(
                    "[bold yellow]Identity endpoint drained; shutting down scheduler[/]"
                )
                self.stop()
            return stats
        baseline = self._ensure_campaign_baseline_ready()
        stats["baseline_blocked"] = 0
        if baseline is not None and not baseline.can_dispatch_or_schedule:
            stats["baseline_blocked"] = 1
            stats["dispatched"] = 0
            stats["seed_scheduled"] = 0
            stats["scheduled"] = 0
            stats["unfinished"] = self.job_scheduler.count_unfinished_jobs()
            self.console.log(
                "[bold yellow]Scheduler tick blocked by campaign baseline[/] "
                "status={} key={} reason={}".format(
                    baseline.status,
                    baseline.baseline_key_hash[:12],
                    baseline.failure_kind or baseline.failure_summary or "n/a",
                ),
            )
            return stats
        promote = getattr(self.job_scheduler, "promote_staged_jobs", None)
        stats["staged_promoted"] = int(promote() if callable(promote) else 0)
        stats["dispatched"] = self.job_scheduler.dispatch_pending_jobs()
        unfinished = self.job_scheduler.count_unfinished_jobs()
        stats["seed_scheduled"] = self._maybe_schedule_seed_jobs(unfinished_jobs=unfinished)
        effective_unfinished = unfinished + stats["seed_scheduled"]
        total_jobs = self._get_total_jobs_count()
        if stats["seed_scheduled"] > 0:
            total_jobs = self._adjust_total_jobs_count(stats["seed_scheduled"])
        stats["scheduled"] = self.job_scheduler.schedule_jobs(
            unfinished_jobs=effective_unfinished,
            total_jobs=total_jobs,
            refresh_campaign_program=False,
        )
        stats["unfinished"] = unfinished + stats["seed_scheduled"] + stats["scheduled"]
        if stats["scheduled"] > 0:
            total_jobs_after = self._adjust_total_jobs_count(stats["scheduled"])
        else:
            total_jobs_after = total_jobs

        pending_ingestion = self._terminal_pending_ingestion(
            total_jobs=total_jobs_after,
            unfinished_jobs=stats["unfinished"],
        )
        if pending_ingestion is not None:
            stats["pending_ingestion"] = pending_ingestion

        remaining_total_str = ""
        max_total = self._max_total_jobs
        remaining_total = max(0, max_total - total_jobs_after)
        remaining_total_str = f" remaining_total={remaining_total}/{max_total}"

        self.console.log(
            "[bold magenta]Scheduler tick[/] ingested={ingested} reclaimed_pending={reclaimed_pending} "
            "reclaimed_failed={reclaimed_failed} dispatched={dispatched} seed_scheduled={seed_scheduled} "
            "staged_promoted={staged_promoted} scheduled={scheduled} "
            "unfinished={unfinished}{remaining_total}"
            "{pending_ingestion_suffix}".format(
                **stats,
                remaining_total=remaining_total_str,
                pending_ingestion_suffix=_pending_ingestion_log_suffix(
                    pending_ingestion
                ),
            ),
        )

        if pending_ingestion == 0:
            self._create_primary_objective_branch_if_possible()
            self.console.log(
                "[bold yellow]Scheduler reached max total jobs and all jobs finished; shutting down[/] "
                f"limit={max_total}",
            )
            log.info(
                "Scheduler stopping after reaching max_total_jobs={} (total_jobs={})",
                max_total,
                total_jobs_after,
            )
            self.stop()
        return stats

    def _terminal_pending_ingestion(
        self,
        *,
        total_jobs: int,
        unfinished_jobs: int,
    ) -> int | None:
        if total_jobs < self._max_total_jobs or unfinished_jobs > 0:
            return None
        return self.ingestion.count_pending_ingestion_jobs()

    def _identity_endpoint_progress(self) -> CampaignProgress | None:
        if self.settings.scheduler_max_unique_evaluation_identities is None:
            return None
        with session_scope() as session:
            return load_campaign_progress(session, self.settings)

    def stop(self) -> None:
        """Signal the scheduler loop to exit."""

        if self._stop_requested:
            return
        self._stop_requested = True

    def close(self) -> None:
        """Release any long-lived resources held by the scheduler process."""

        if self._advisory_lock is None:
            return
        try:
            release_pg_advisory_lock(self._advisory_lock)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            self.console.log(f"[yellow]Failed to release scheduler advisory lock[/] reason={exc}")
            log.warning("Failed to release scheduler advisory lock: {}", exc)
        finally:
            self._advisory_lock = None

    # Internal helpers ------------------------------------------------------

    def _require_max_total_jobs(self) -> int:
        raw = getattr(self.settings, "scheduler_max_total_jobs", None)
        if raw is None:
            raise SchedulerError(
                "SCHEDULER_MAX_TOTAL_JOBS is required to enforce a bounded scheduler run.",
            )
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise SchedulerError(
                "SCHEDULER_MAX_TOTAL_JOBS must be a positive integer.",
            ) from exc
        if value <= 0:
            raise SchedulerError(
                "SCHEDULER_MAX_TOTAL_JOBS must be a positive integer.",
            )
        return value

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        terminate = getattr(signal, "SIGTERM", None)
        if terminate is not None:
            signal.signal(terminate, self._handle_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.console.log(f"[yellow]Received signal[/] signum={signum}; shutting down.")
        log.info("Scheduler received signal {}; stopping", signum)
        self.stop()

    def _sleep_with_stop(self, duration: float) -> None:
        """Sleep up to ``duration`` seconds, waking early when a stop is requested."""

        if duration <= 0:
            return

        quantum = 0.5  # seconds; keeps shutdown latency low without busy-waiting
        end = time.perf_counter() + duration
        while not self._stop_requested:
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(quantum, remaining))

    def _get_total_jobs_count(self, *, refresh: bool = False) -> int:
        cached = getattr(self, "_total_jobs_count", None)
        if refresh or cached is None:
            cached = int(self.job_scheduler.count_total_jobs())
            self._total_jobs_count = cached
        return int(cached)

    def _adjust_total_jobs_count(self, delta: int) -> int:
        current = self._get_total_jobs_count()
        updated = max(0, current + int(delta))
        self._total_jobs_count = updated
        return updated

    def _ensure_campaign_baseline_ready(self) -> BaselineBootstrapResult | None:
        root_hash = getattr(self, "_root_commit_hash", None)
        if not root_hash:
            return None
        refresh = getattr(self.job_scheduler, "refresh_campaign_program_for_policy", None)
        if callable(refresh):
            refresh()
        campaign_program = getattr(self.job_scheduler, "campaign_program_snapshot", None)
        return self.baseline_bootstrap.ensure_or_load_baseline(
            root_commit_hash=root_hash,
            campaign_program=campaign_program,
        )

    # DB coordination helpers ----------------------------------------------

    def _acquire_experiment_lock(self) -> AdvisoryLock:
        experiment_id = resolve_experiment_uuid(self.settings.experiment_id)
        key = uuid_to_pg_bigint_lock_key(experiment_id)
        lock = try_acquire_pg_advisory_lock(engine=get_engine(), key=key)
        if lock is None:
            raise SchedulerLockError(
                "Another scheduler instance is already running for this experiment. "
                f"(experiment_id={experiment_id})"
            )
        self.console.log(
            "[green]Acquired scheduler advisory lock[/] experiment={} key={}".format(
                experiment_id,
                key,
            )
        )
        log.info("Acquired scheduler advisory lock for experiment {} (key={})", experiment_id, key)
        return lock

    def _startup_scan_and_validate_repo_state_approval(self) -> None:
        root_commit = self._require_repo_state_startup_root_commit()
        canonical = self._resolve_repo_state_startup_root_commit(root_commit)
        self._root_commit_hash = canonical
        filters = self._repo_state_startup_filters()
        scan = self._scan_repo_state_startup_root(canonical)
        approval = self._build_repo_state_startup_approval(
            root_commit=canonical,
            eligible_files=int(scan.eligible_files),
            filters=filters,
        )
        self._log_repo_state_startup_scan(approval, filters=filters)
        self._request_repo_state_startup_approval(approval)

        self.console.log(
            "[green]Repo-state startup approved[/] root_commit={} eligible_files={}".format(
                approval.root_commit,
                approval.eligible_files,
            )
        )

    def _require_repo_state_startup_root_commit(self) -> str:
        root_commit = (self._root_commit_hash or "").strip()
        if not root_commit:
            raise SchedulerError(
                "MAPELITES_EXPERIMENT_ROOT_COMMIT is required for repo-state startup approval "
                "and incremental-only ingestion."
            )
        return root_commit

    def _resolve_repo_state_startup_root_commit(self, root_commit: str) -> str:
        try:
            with repo_lock(self.repo_root):
                return require_commit(self._repo, root_commit, console=self.console)
        except RepositoryError as exc:
            raise SchedulerError(f"Cannot resolve root commit {root_commit!r} for repo-state scan: {exc}") from exc

    def _scan_repo_state_startup_root(self, canonical_root_commit: str):
        return scan_repo_state_root(
            settings=self.settings,
            repo_root=self.repo_root,
            repo=self._repo,
            root_commit=canonical_root_commit,
        )

    def _repo_state_startup_filters(self) -> dict[str, object]:
        return {
            "allowed_extensions": list(self.settings.mapelites_preprocess_allowed_extensions or []),
            "allowed_filenames": list(self.settings.mapelites_preprocess_allowed_filenames or []),
            "excluded_globs": list(self.settings.mapelites_preprocess_excluded_globs or []),
            "max_file_size_kb": int(self.settings.mapelites_preprocess_max_file_size_kb),
            "root_ignore_files": [".gitignore", ".loreleyignore"],
        }

    def _build_repo_state_startup_approval(
        self,
        *,
        root_commit: str,
        eligible_files: int,
        filters: Mapping[str, object],
    ) -> _RepoStateStartupApproval:
        max_chunks_per_file = self._repo_state_startup_max_chunks_per_file()
        details = {
            "profile": str(getattr(self.settings, "profile", "default")),
            "embedding_model": str(self.settings.mapelites_code_embedding_model),
            "embedding_dimensions": getattr(self.settings, "mapelites_code_embedding_dimensions", None),
            "embedding_batch_size": int(getattr(self.settings, "mapelites_code_embedding_batch_size", 0) or 0),
            "chunk_target_lines": int(getattr(self.settings, "mapelites_chunk_target_lines", 0) or 0),
            "chunk_min_lines": int(getattr(self.settings, "mapelites_chunk_min_lines", 0) or 0),
            "chunk_overlap_lines": int(getattr(self.settings, "mapelites_chunk_overlap_lines", 0) or 0),
            "chunk_max_chunks_per_file": max_chunks_per_file,
            "root_chunk_upper_bound": int(eligible_files) * max_chunks_per_file,
            "pca_target_dims": int(getattr(self.settings, "mapelites_dimensionality_target_dims", 0) or 0),
            "pca_min_fit_samples": int(getattr(self.settings, "mapelites_dimensionality_min_fit_samples", 0) or 0),
            "pca_history_size": int(getattr(self.settings, "mapelites_dimensionality_history_size", 0) or 0),
            "pca_refit_interval": int(getattr(self.settings, "mapelites_dimensionality_refit_interval", 0) or 0),
            "seed_population_size": int(getattr(self.settings, "mapelites_seed_population_size", 0) or 0),
            **dict(filters),
        }
        return _RepoStateStartupApproval(
            root_commit=root_commit,
            eligible_files=int(eligible_files),
            details=details,
        )

    def _repo_state_startup_max_chunks_per_file(self) -> int:
        max_chunks_per_file = int(getattr(self.settings, "mapelites_chunk_max_chunks_per_file", 0) or 0)
        if max_chunks_per_file <= 0:
            return 1
        return max_chunks_per_file

    @staticmethod
    def _log_repo_state_startup_scan(
        approval: _RepoStateStartupApproval,
        *,
        filters: Mapping[str, object],
    ) -> None:
        log.info(
            "Repo-state root scan commit={} eligible_files={} filters={}",
            approval.root_commit,
            approval.eligible_files,
            dict(filters),
        )

    def _request_repo_state_startup_approval(self, approval: _RepoStateStartupApproval) -> None:
        auto_approve = bool(getattr(self.settings, "scheduler_startup_approve", False))
        try:
            require_interactive_repo_state_root_approval(
                root_commit=approval.root_commit,
                eligible_files=approval.eligible_files,
                repo_root=self.repo_root,
                details=approval.details,
                console=self.console,
                auto_approve=auto_approve,
            )
        except ValueError as exc:
            raise SchedulerError(str(exc)) from exc

    # Git helpers -----------------------------------------------------------

    def _resolve_repo_root(self) -> Path:
        candidate = self.settings.scheduler_repo_root or self.settings.worker_repo_worktree
        if candidate:
            return Path(candidate).expanduser().resolve()
        return Path.cwd()

    def _init_repo(self) -> Repo:
        try:
            return Repo(self.repo_root)
        except (NoSuchPathError, InvalidGitRepositoryError) as exc:  # pragma: no cover - filesystem
            raise SchedulerError(f"Scheduler repo {self.repo_root} is not a git repository.") from exc
    # Primary-objective deliverable -----------------------------------------

    def _create_primary_objective_branch_if_possible(self) -> bool:
        """Create the primary-objective branch when a retained candidate exists."""

        best_commit, meta = self._resolve_best_primary_commit()
        if not best_commit:
            self.console.log(
                "[bold yellow]Primary-objective branch not created[/] "
                "reason=no retained candidate with the configured primary objective"
            )
            log.warning(
                "Primary-objective branch not created because no retained "
                "candidate has the configured primary objective"
            )
            return False

        root_commit = meta.get("root_commit_hash")
        primary = resolve_objective_contract(self.settings).primary
        metric_name = meta.get("primary_metric_name") or primary.name
        metric_value = meta.get("primary_metric_value")
        island_id = meta.get("island_id")
        branch_name = self._create_primary_objective_branch(
            best_commit_hash=best_commit,
            root_commit_hash=root_commit,
        )

        metric_value_str: str
        try:
            metric_value_str = (
                f"{float(metric_value):.6f}" if metric_value is not None else "n/a"
            )
        except (TypeError, ValueError):
            metric_value_str = str(metric_value)

        self.console.log(
            "[bold green]Primary-objective branch updated[/] "
            "branch={} commit={} root_commit={} island={} metric={} value={}".format(
                branch_name,
                best_commit,
                root_commit or "n/a",
                island_id or "n/a",
                metric_name or "n/a",
                metric_value_str,
            ),
        )
        log.info(
            "Primary-objective branch updated "
            "(branch={} commit={} root_commit={} island_id={} metric={} value={})",
            branch_name,
            best_commit,
            root_commit,
            island_id,
            metric_name,
            metric_value,
        )
        return True

    def _resolve_best_primary_commit(self) -> tuple[str | None, dict[str, Any]]:
        """Return the retained commit with the best raw primary-objective value."""

        primary = resolve_objective_contract(self.settings).primary
        metric_name = primary.name

        with session_scope() as session:
            order_column = (
                Metric.value.desc()
                if primary.higher_is_better
                else Metric.value.asc()
            )

            conditions: list[Any] = [
                Metric.name == metric_name,
                MapElitesArchiveCell.island_id.in_(
                    tuple(self.settings.mapelites_islands)
                ),
            ]
            if self._root_commit_hash:
                conditions.append(CommitCard.commit_hash != self._root_commit_hash)

            stmt = (
                select(
                    CommitCard.commit_hash,
                    MapElitesArchiveCell.island_id,
                    Metric.value,
                )
                .join(Metric, Metric.commit_card_id == CommitCard.id)
                .join(
                    MapElitesArchiveCell,
                    MapElitesArchiveCell.commit_hash == CommitCard.commit_hash,
                )
                .where(*conditions)
                .order_by(order_column, CommitCard.commit_hash.asc())
                .limit(1)
            )

            row = session.execute(stmt).first()
            if not row:
                return None, {}

            best_commit_hash: str = row[0]
            island_id: str | None = row[1]
            primary_value: float | None = float(row[2]) if row[2] is not None else None

            root_commit_hash = self._find_root_commit_for_experiment_chain(
                session=session,
                start_commit_hash=best_commit_hash,
            )

        meta: dict[str, Any] = {
            "island_id": island_id,
            "primary_metric_name": metric_name,
            "primary_metric_value": primary_value,
            "root_commit_hash": root_commit_hash,
        }
        return best_commit_hash, meta

    def _maybe_schedule_seed_jobs(self, *, unfinished_jobs: int) -> int:
        """Fairly schedule cold-start seed jobs for every empty configured island."""

        budget = self._seed_scheduling_budget(unfinished_jobs=unfinished_jobs)
        if budget is None:
            return 0
        demands = self._seed_island_demands(budget)
        allocations = _allocate_seed_slots(demands, budget.available_slots)
        return self._create_seed_allocations(budget, allocations)

    def _seed_population_target(self) -> int:
        configured_seed_population = max(
            0,
            int(getattr(self.settings, "mapelites_seed_population_size", 0)),
        )
        if configured_seed_population <= 0:
            return 0
        warmup_required = max(
            0,
            int(getattr(self.settings, "mapelites_feature_normalization_warmup_samples", 0) or 0),
        )
        return max(configured_seed_population, warmup_required)

    def _seed_scheduling_budget(
        self,
        *,
        unfinished_jobs: int,
    ) -> _SeedSchedulingBudget | None:
        root_hash = self._root_commit_hash
        target_samples = self._seed_population_target()
        if not root_hash or target_samples <= 0:
            return None
        max_jobs = max(0, int(self.settings.scheduler_max_unfinished_jobs))
        if max_jobs == 0:
            return None
        capacity = max(0, max_jobs - unfinished_jobs)
        if capacity <= 0:
            return None
        total_jobs = self._get_total_jobs_count()
        remaining_total = self._max_total_jobs - total_jobs
        if remaining_total <= 0:
            return None
        islands = tuple(self.settings.mapelites_islands)
        if not islands:
            return None
        start = total_jobs % len(islands)
        ordered_islands = (*islands[start:], *islands[:start])
        return _SeedSchedulingBudget(
            root_commit_hash=root_hash,
            target_samples=target_samples,
            available_slots=min(capacity, remaining_total),
            unfinished_jobs=unfinished_jobs,
            ordered_islands=ordered_islands,
        )

    def _seed_island_demands(
        self,
        budget: _SeedSchedulingBudget,
    ) -> tuple[_SeedIslandDemand, ...]:
        demands: list[_SeedIslandDemand] = []
        for island_id in budget.ordered_islands:
            if self.manager.get_records(island_id):
                continue
            warmup_samples = self.manager.count_pca_history_samples(island_id)
            seed_counts = self._count_seed_warmup_job_counts(island_id=island_id)
            remaining = _remaining_seed_demand(
                warmup_samples=warmup_samples,
                target_samples=budget.target_samples,
                counts=seed_counts,
            )
            if remaining:
                demands.append(
                    _SeedIslandDemand(
                        island_id=island_id,
                        warmup_samples=warmup_samples,
                        remaining_samples=remaining,
                    )
                )
        return tuple(demands)

    def _create_seed_allocations(
        self,
        budget: _SeedSchedulingBudget,
        allocations: tuple[_SeedIslandAllocation, ...],
    ) -> int:
        created_total = 0
        for allocation in allocations:
            demand = allocation.demand
            seed_phase = (
                "warmup"
                if demand.warmup_samples < budget.target_samples
                else "readiness"
            )
            created = self.job_scheduler.create_seed_jobs(
                base_commit_hash=budget.root_commit_hash,
                count=allocation.count,
                island_id=demand.island_id,
                refresh_campaign_program=False,
            )
            created_total += created
            if created:
                self.console.log(
                    "[bold green]Scheduled seed jobs[/] "
                    "count={} root={} island={} phase={} warmup_samples={}/{}".format(
                        created,
                        budget.root_commit_hash,
                        demand.island_id,
                        seed_phase,
                        demand.warmup_samples,
                        budget.target_samples,
                    ),
                )
                log.info(
                    "Scheduled {} seed jobs from root {} on island {} "
                    "(phase={} unfinished_jobs={} warmup_samples={} target_samples={})",
                    created,
                    budget.root_commit_hash,
                    demand.island_id,
                    seed_phase,
                    budget.unfinished_jobs,
                    demand.warmup_samples,
                    budget.target_samples,
                )
        return created_total

    def _count_seed_warmup_job_counts(self, *, island_id: str) -> _SeedWarmupJobCounts:
        from loreley.db.models import EvolutionJob, JobStatus  # Local import to avoid cycles.

        unfinished_seed_statuses = (
            JobStatus.STAGED,
            JobStatus.PENDING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
        )
        succeeded_seed_requiring_ingestion = and_(
            EvolutionJob.is_seed_job.is_(True),
            *pending_ingestion_job_conditions(
                EvolutionJob=EvolutionJob,
                JobStatus=JobStatus,
                func=func,
            ),
        )
        stmt = select(
            func.coalesce(
                func.sum(case((EvolutionJob.is_seed_job.is_(True), 1), else_=0)),
                0,
            ),
            func.coalesce(
                func.sum(
                    case(
                        (
                            and_(
                                EvolutionJob.is_seed_job.is_(True),
                                EvolutionJob.status.in_(unfinished_seed_statuses),
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ),
                0,
            ),
            func.coalesce(
                func.sum(case((succeeded_seed_requiring_ingestion, 1), else_=0)),
                0,
            ),
        ).where(EvolutionJob.island_id == island_id)
        with session_scope() as session:
            row = session.execute(stmt).one()
        return _SeedWarmupJobCounts(
            seed_jobs=int(row[0]),
            unfinished_seed_jobs=int(row[1]),
            pending_ingestion_seed_jobs=int(row[2]),
        )

    def _find_root_commit_for_experiment_chain(
        self,
        *,
        session,
        start_commit_hash: str,
    ) -> str | None:
        """Walk the CommitCard parent chain to find the experiment root commit.

        The root commit is defined as the earliest known parent in the evolution
        chain for this experiment. This may be a commit that does not itself have
        a CommitCard row (for example, the original repository commit used as
        the starting point for the first evolution job).
        """

        current = start_commit_hash
        root: str | None = None
        visited: set[str] = set()

        while current and current not in visited:
            visited.add(current)
            parent_hash = session.execute(
                select(CommitCard.parent_commit_hash).where(
                    CommitCard.commit_hash == current,
                )
            ).scalar_one_or_none()
            if not parent_hash:
                break
            root = parent_hash

            # Continue walking only while the parent itself belongs to this experiment.
            exists = session.execute(
                select(CommitCard.commit_hash).where(
                    CommitCard.commit_hash == parent_hash,
                )
            ).scalar_one_or_none()
            if not exists:
                break

            current = parent_hash

        return root

    def _create_primary_objective_branch(
        self,
        *,
        best_commit_hash: str,
        root_commit_hash: str | None,
    ) -> str:
        """Create or update a stable branch for the declared primary objective."""

        experiment_suffix = resolve_experiment_namespace(self.settings.experiment_id)
        branch_name = f"evolution/primary/{experiment_suffix}"

        try:
            with repo_lock(self.repo_root):
                canonical = require_commit(self._repo, best_commit_hash, console=self.console)
                # Keep the deliverable stable across restarts by force-updating the branch.
                self._repo.git.branch("-f", branch_name, canonical)
        except RepositoryError as exc:
            raise SchedulerError(str(exc)) from exc
        except GitCommandError as exc:
            wrapped = wrap_git_error(
                exc,
                f"Failed to update primary-objective branch {branch_name}",
            )
            raise SchedulerError(str(wrapped)) from exc

        log.info(
            "Updated primary-objective branch {} -> {} (root_commit={})",
            branch_name,
            canonical,
            root_commit_hash,
        )

        return branch_name

def main(
    *,
    settings: Settings | None = None,
    once: bool = False,
    auto_approve: bool = False,
) -> int:
    """Run the Loreley evolution scheduler (once or forever).

    This function is intentionally free of CLI parsing so it can be reused by the
    unified Typer CLI (`loreley scheduler ...`) and by wrapper scripts.
    """

    settings = settings or get_settings()
    if bool(auto_approve):
        settings = settings.model_copy(update={"scheduler_startup_approve": True})

    try:
        scheduler = EvolutionScheduler(settings=settings)
    except SchedulerLockError as exc:
        console.log(f"[bold red]Scheduler refused to start[/] reason={exc}")
        log.error("Scheduler refused to start: {}", exc)
        return 2
    except SchedulerError as exc:
        console.log(f"[bold red]Scheduler startup failed[/] reason={exc}")
        log.error("Scheduler startup failed: {}", exc)
        return 1

    if bool(once):
        try:
            scheduler.tick()
        except Exception as exc:
            console.log(f"[bold red]Scheduler run failed[/] reason={exc}")
            log.exception("Scheduler run failed: {}", exc)
            return 1
        finally:
            scheduler.close()
        return 0

    try:
        scheduler.run_forever()
    except Exception as exc:
        console.log(f"[bold red]Scheduler crashed[/] reason={exc}")
        log.exception("Scheduler crashed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    from loreley.cli import main as loreley_main

    raise SystemExit(loreley_main(["scheduler", *sys.argv[1:]]))
