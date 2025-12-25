from __future__ import annotations

"""Job scheduling and dispatch logic used by the evolution scheduler.

This module is intentionally free of the main scheduler loop so that the
core orchestration code in ``loreley.scheduler.main`` can stay focused on
high-level control flow.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID

from loguru import logger
from rich.console import Console
from sqlalchemy import func, select

from loreley.config import Settings
from loreley.core.map_elites.sampler import MapElitesSampler, ScheduledSamplerJob
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobStatus
from loreley.tasks.workers import run_evolution_job

log = logger.bind(module="scheduler.job_scheduler")


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
    experiment_id: UUID

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

    # Scheduling ------------------------------------------------------------

    def schedule_jobs(self, unfinished_jobs: int, *, total_scheduled_jobs: int) -> int:
        """Schedule new jobs from MAP-Elites if there is available capacity.

        Parameters
        ----------
        unfinished_jobs:
            Current number of unfinished jobs in the system.
        total_scheduled_jobs:
            Cumulative number of jobs that have been scheduled so far in this
            scheduler process (used to enforce the global job limit).
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
            remaining_total = max_total - total_scheduled_jobs
            if remaining_total <= 0:
                self.console.log(
                    "[yellow]Scheduler global job limit reached; no new jobs will be scheduled[/] "
                    f"limit={max_total}",
                )
                log.info(
                    "Global scheduler job limit reached: max_total_jobs={} (total_scheduled={})",
                    max_total,
                    total_scheduled_jobs,
                )
                return 0
            target = min(target, remaining_total)

        scheduled_ids: list[UUID] = []
        for _ in range(target):
            job = self._schedule_single_job()
            if not job:
                break
            scheduled_ids.append(job.job_id)
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

        effective_island = island_id or self.settings.mapelites_default_island_id or "main"
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
                    experiment_id=self.experiment_id,
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

    def _schedule_single_job(self) -> ScheduledSamplerJob | None:
        try:
            scheduled = self.sampler.schedule_job(experiment_id=self.experiment_id)
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
        ready = self._mark_jobs_queued(pending)
        dispatched = 0
        for job_id in ready:
            try:
                run_evolution_job.send(str(job_id))
                dispatched += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.console.log(
                    f"[bold red]Failed to enqueue job[/] id={job_id} reason={exc}",
                )
                log.exception("Failed to enqueue job {}: {}", job_id, exc)
        if dispatched:
            self.console.log(f"[cyan]Dispatched {dispatched} job(s) to Dramatiq[/]")
        return dispatched

    def _fetch_pending_job_ids(self, *, limit: int) -> list[UUID]:
        with session_scope() as session:
            stmt = (
                select(EvolutionJob.id)
                .where(EvolutionJob.status == JobStatus.PENDING)
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
                .with_for_update()
            )
            for job in session.execute(stmt).scalars():
                if job.status != JobStatus.PENDING:
                    continue
                job.status = JobStatus.QUEUED
                job.scheduled_at = job.scheduled_at or now
                ready.append(job.id)
        return ready

    def _enqueue_jobs(self, job_ids: Sequence[UUID]) -> None:
        if not job_ids:
            return
        queued = self._mark_jobs_queued(job_ids)
        for job_id in queued:
            try:
                run_evolution_job.send(str(job_id))
                self.console.log(
                    f"[bold green]Queued job[/] id={job_id}",
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.console.log(
                    f"[bold red]Failed to enqueue job[/] id={job_id} reason={exc}",
                )
                log.exception("Failed to enqueue scheduled job {}: {}", job_id, exc)


