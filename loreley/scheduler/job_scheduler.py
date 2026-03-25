from __future__ import annotations

"""Job scheduling and dispatch logic used by the evolution scheduler.

This module is intentionally free of the main scheduler loop so that the
core orchestration code in ``loreley.scheduler.main`` can stay focused on
high-level control flow.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from loguru import logger
from rich.console import Console
from sqlalchemy import func, or_, select

from loreley.config import Settings, resolve_default_island_id
from loreley.core.map_elites.sampler import MapElitesSampler, SamplingSnapshot, ScheduledSamplerJob
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobStatus
from loreley.tasks.workers import build_evolution_job_sender_actor

log = logger.bind(module="scheduler.job_scheduler")


def _db_utc_now(session: Any) -> datetime:
    value = session.execute(select(func.now())).scalar_one()
    if not isinstance(value, datetime):
        raise RuntimeError(f"Database current timestamp returned unsupported value: {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class JobLeaseReclaimResult:
    """Summary of stale RUNNING jobs reclaimed during a scheduler tick."""

    requeued: int = 0
    failed: int = 0


@dataclass(slots=True)
class JobScheduler:
    """Encapsulate all logic for producing and dispatching evolution jobs.

    The public methods on this class are deliberately small and side‑effect
    free from the perspective of the caller: they do their database work,
    talk to Dramatiq, and return simple integer counts that the outer
    scheduler loop can use for reporting.
    """

    settings: Settings
    console: Console
    sampler: MapElitesSampler
    _sender_actor: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Build a sender-only actor that targets the experiment-scoped queue.
        self._sender_actor = build_evolution_job_sender_actor(
            settings=self.settings,
        )

    # Measuring -------------------------------------------------------------

    def count_unfinished_jobs(self) -> int:
        """Return the number of jobs that are not yet finished."""

        unfinished_statuses = (
            JobStatus.PENDING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
        )
        with session_scope() as session:
            stmt = (
                select(func.count(EvolutionJob.id))
                .where(EvolutionJob.status.in_(unfinished_statuses))
            )
            return int(session.execute(stmt).scalar_one())

    def count_total_jobs(self) -> int:
        """Return the total number of jobs in the database."""
        with session_scope() as session:
            stmt = select(func.count(EvolutionJob.id))
            return int(session.execute(stmt).scalar_one())

    def reclaim_stale_running_jobs(
        self,
        *,
        now: datetime | None = None,
    ) -> JobLeaseReclaimResult:
        """Recover RUNNING jobs whose lease expired and are no longer heartbeating."""

        batch = max(0, int(self.settings.scheduler_stale_running_reclaim_batch_size))
        if batch == 0:
            return JobLeaseReclaimResult()

        requeued = 0
        failed = 0
        max_recoveries = max(0, int(self.settings.scheduler_stale_running_max_recovery_attempts))

        with session_scope() as session:
            current_time = now or _db_utc_now(session)
            stmt = (
                select(EvolutionJob)
                .where(
                    EvolutionJob.status == JobStatus.RUNNING,
                    or_(
                        EvolutionJob.run_token.is_(None),
                        EvolutionJob.worker_id.is_(None),
                        EvolutionJob.lease_expires_at.is_(None),
                        EvolutionJob.lease_expires_at < current_time,
                    ),
                )
                .order_by(
                    EvolutionJob.lease_expires_at.asc().nullsfirst(),
                    EvolutionJob.started_at.asc(),
                    EvolutionJob.created_at.asc(),
                )
                .limit(batch)
                .with_for_update(skip_locked=True)
            )
            for job in session.execute(stmt).scalars():
                attempts = int(getattr(job, "recovery_count", 0) or 0)
                next_attempt = attempts + 1
                missing_lease = (
                    getattr(job, "run_token", None) is None
                    or getattr(job, "worker_id", None) is None
                    or getattr(job, "lease_expires_at", None) is None
                )
                message = (
                    "Lease metadata missing for RUNNING job; "
                    f"recovered by scheduler (attempt={next_attempt})."
                    if missing_lease
                    else "Lease expired after missing heartbeat; "
                    f"recovered by scheduler (attempt={next_attempt})."
                )
                job.recovery_count = next_attempt
                job.run_token = None
                job.worker_id = None
                job.heartbeat_at = None
                job.lease_expires_at = None
                job.last_error = message
                if attempts >= max_recoveries:
                    job.status = JobStatus.FAILED
                    job.completed_at = current_time
                    failed += 1
                    self.console.log(
                        f"[yellow]Failed stale running job[/] id={job.id} recovery_count={next_attempt}",
                    )
                    log.warning(
                        "Job {} failed after stale lease reclaim (recovery_count={})",
                        job.id,
                        next_attempt,
                    )
                    continue
                job.status = JobStatus.PENDING
                job.started_at = None
                job.completed_at = None
                job.scheduled_at = current_time
                requeued += 1
                self.console.log(
                    f"[yellow]Requeued stale running job[/] id={job.id} recovery_count={job.recovery_count}",
                )
                log.warning(
                    "Job {} requeued after stale lease reclaim (recovery_count={})",
                    job.id,
                    job.recovery_count,
                )
        return JobLeaseReclaimResult(requeued=requeued, failed=failed)

    # Scheduling ------------------------------------------------------------

    def schedule_jobs(self, unfinished_jobs: int, *, total_jobs: int) -> int:
        """Schedule new jobs from MAP-Elites if there is available capacity.

        Parameters
        ----------
        unfinished_jobs:
            Current number of unfinished jobs in the system.
        total_jobs:
            Total number of jobs recorded in the database (used to enforce the global job limit).
        """

        max_jobs = max(0, int(self.settings.scheduler_max_unfinished_jobs))
        if max_jobs == 0:
            return 0

        capacity = max(0, max_jobs - unfinished_jobs)
        if capacity <= 0:
            return 0

        batch = max(1, int(self.settings.scheduler_schedule_batch_size))
        target = min(capacity, batch)

        max_total = getattr(self.settings, "scheduler_max_total_jobs", None)
        if max_total is not None and max_total > 0:
            remaining_total = max_total - int(total_jobs)
            if remaining_total <= 0:
                self.console.log(
                    "[yellow]Scheduler global job limit reached; no new jobs will be scheduled[/] "
                    f"limit={max_total}",
                )
                log.info(
                    "Global scheduler job limit reached: max_total_jobs={} (total_jobs={})",
                    max_total,
                    total_jobs,
                )
                return 0
            target = min(target, remaining_total)

        sampling_snapshot = self.sampler.get_sampling_snapshot()
        if sampling_snapshot is None:
            self.console.log("[yellow]Sampler returned no job[/]")
            return 0
        effective_island = sampling_snapshot.island_id

        scheduled_ids: list[UUID] = []
        selected_base_commits: set[str] = set()
        for _ in range(target):
            job = self._schedule_single_job(
                island_id=effective_island,
                sampling_snapshot=sampling_snapshot,
                excluded_base_commits=selected_base_commits,
            )
            if not job:
                break
            scheduled_ids.append(job.job_id)
            selected_base_commits.add(str(job.base_commit_hash))
        if scheduled_ids:
            self._enqueue_jobs(scheduled_ids)
        return len(scheduled_ids)

    def create_seed_jobs(
        self,
        *,
        base_commit_hash: str,
        count: int,
        island_id: str | None = None,
    ) -> int:
        """Create and enqueue cold-start seed jobs from the root commit.

        Seed jobs use the configured default priority and are immediately promoted
        to QUEUED and sent to Dramatiq.
        """

        if count <= 0:
            return 0

        effective_island = island_id or resolve_default_island_id(self.settings)
        now = datetime.now(timezone.utc)
        jobs: list[EvolutionJob] = []
        goal = (self.settings.worker_evolution_global_goal or "").strip()
        if not goal:
            self.console.log(
                "[bold red]Cannot create seed jobs[/] WORKER_EVOLUTION_GLOBAL_GOAL is empty",
            )
            return 0

        with session_scope() as session:
            for _ in range(count):
                job = EvolutionJob(
                    status=JobStatus.PENDING,
                    base_commit_hash=base_commit_hash,
                    island_id=effective_island,
                    inspiration_commit_hashes=[],
                    goal=goal,
                    constraints=[],
                    acceptance_criteria=[],
                    notes=[],
                    tags=[],
                    iteration_hint=(
                        "Cold-start seed job: design diverse initial directions "
                        "from the root baseline."
                    ),
                    sampling_strategy="seed",
                    sampling_initial_radius=None,
                    sampling_radius_used=None,
                    sampling_fallback_inspirations=None,
                    is_seed_job=True,
                    priority=self.settings.mapelites_sampler_default_priority,
                    scheduled_at=now,
                )
                session.add(job)
                jobs.append(job)
            session.flush()
            job_ids = [job.id for job in jobs]

        if not job_ids:
            return 0

        self.console.log(
            "[bold green]Created seed jobs[/] count={} base={} island={}".format(
                len(job_ids),
                base_commit_hash,
                effective_island,
            ),
        )
        log.info(
            "Created {} seed jobs for base {} on island {}",
            len(job_ids),
            base_commit_hash,
            effective_island,
        )

        self._enqueue_jobs(job_ids)
        return len(job_ids)

    def _schedule_single_job(
        self,
        *,
        island_id: str | None = None,
        sampling_snapshot: SamplingSnapshot | None = None,
        excluded_base_commits: Sequence[str] | None = None,
    ) -> ScheduledSamplerJob | None:
        try:
            scheduled = self.sampler.schedule_job(
                island_id=island_id,
                sampling_snapshot=sampling_snapshot,
                excluded_base_commits=excluded_base_commits,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.console.log(f"[bold red]Sampler failed[/] reason={exc}")
            log.exception("Sampler failed to create a job: {}", exc)
            return None
        if not scheduled:
            self.console.log("[yellow]Sampler returned no job[/]")
            return None
        self.console.log(
            f"[green]Scheduled job[/] id={scheduled.job_id} island={scheduled.island_id} "
            f"base={scheduled.base_commit_hash}",
        )
        return scheduled

    # Dispatching -----------------------------------------------------------

    def dispatch_pending_jobs(self) -> int:
        """Send pending jobs to Dramatiq."""

        batch = max(0, int(self.settings.scheduler_dispatch_batch_size))
        if batch == 0:
            return 0
        pending = self._fetch_pending_job_ids(limit=batch)
        if not pending:
            return 0
        dispatched = self._enqueue_jobs(pending)
        if dispatched:
            self.console.log(f"[cyan]Dispatched {dispatched} job(s) to Dramatiq[/]")
        return dispatched

    def _fetch_pending_job_ids(self, *, limit: int) -> list[UUID]:
        with session_scope() as session:
            stmt = (
                select(EvolutionJob.id)
                .where(
                    EvolutionJob.status == JobStatus.PENDING,
                )
                .order_by(
                    EvolutionJob.priority.desc(),
                    EvolutionJob.scheduled_at.asc(),
                    EvolutionJob.created_at.asc(),
                )
                .limit(limit)
            )
            return list(session.execute(stmt).scalars())

    def _mark_jobs_queued(self, job_ids: Sequence[UUID]) -> list[UUID]:
        ready: list[UUID] = []
        if not job_ids:
            return ready
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            stmt = (
                select(EvolutionJob)
                .where(EvolutionJob.id.in_(job_ids))
                .with_for_update(skip_locked=True)
            )
            for job in session.execute(stmt).scalars():
                if job.status != JobStatus.PENDING:
                    continue
                job.status = JobStatus.QUEUED
                job.scheduled_at = job.scheduled_at or now
                ready.append(job.id)
        return ready

    def _enqueue_jobs(self, job_ids: Sequence[UUID]) -> int:
        if not job_ids:
            return 0

        # IMPORTANT: Enqueue first, then mark QUEUED.
        #
        # If we mark QUEUED before `.send(...)`, a crash or broker failure can leave jobs
        # stuck in QUEUED without any corresponding message in the broker.
        #
        # We allow workers to start jobs that are still PENDING (see EvolutionJobStore),
        # so "send first" is safe and prevents the stuck-QUEUED state.
        sent: list[UUID] = []
        for job_id in job_ids:
            try:
                # Use a sender actor so the message targets the experiment queue.
                self._sender_actor.send(str(job_id))  # type: ignore[attr-defined]
                sent.append(job_id)
                self.console.log(
                    f"[bold green]Enqueued job message[/] id={job_id}",
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.console.log(
                    f"[bold red]Failed to enqueue job[/] id={job_id} reason={exc}",
                )
                log.exception("Failed to enqueue scheduled job {}: {}", job_id, exc)
        if not sent:
            return 0
        marked = self._mark_jobs_queued(sent)
        if len(marked) != len(sent):
            marked_set = set(marked)
            missing = [job_id for job_id in sent if job_id not in marked_set]
            log.debug(
                "Enqueued {} job message(s) but marked {} job(s) as QUEUED (missing={})",
                len(sent),
                len(marked),
                [str(job_id) for job_id in missing[:10]],
            )
        return len(sent)
