from __future__ import annotations

"""Job scheduling and dispatch logic used by the evolution scheduler.

This module is intentionally free of the main scheduler loop so that the
core orchestration code in ``loreley.scheduler.main`` can stay focused on
high-level control flow.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
from typing import Any, Sequence
from uuid import UUID

from loguru import logger
from rich.console import Console
from sqlalchemy import and_, func, or_, select

from loreley.config import Settings, resolve_default_island_id
from loreley.core.campaign_program import (
    CampaignProjectionInput,
    CampaignProgramSnapshot,
    apply_campaign_program_projection,
    load_campaign_program_from_repo,
    load_campaign_program_snapshot_by_hash,
    persist_campaign_program,
)
from loreley.core.map_elites.sampler import MapElitesSampler, SamplingSnapshot, ScheduledSamplerJob
from loreley.core.repair_coordination import repair_tokens_available, with_repair_scheduling_lock
from loreley.core.worker.repair import (
    REPAIR_MODE_REBASE_FROM_NEAREST_VIABLE,
    repair_failure_kind_allowlist,
)
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    DiagnosticCapsule,
    EvaluationAttempt,
    EvolutionJob,
    JobStatus,
    MapElitesRepoStateAggregate,
)
from loreley.scheduler.baselines import (
    BASELINE_STATUS_DEGRADED,
    BASELINE_STATUS_VALID,
    load_latest_matching_baseline,
)
from loreley.tasks.workers import build_evolution_job_sender_actor

log = logger.bind(module="scheduler.job_scheduler")


SUPPORTED_REPAIR_MODES = frozenset({REPAIR_MODE_REBASE_FROM_NEAREST_VIABLE})


def _validated_failed_candidate_repair_mode(settings: Settings) -> str:
    mode = str(getattr(settings, "failed_candidate_repair_mode", "") or "")
    if mode not in SUPPORTED_REPAIR_MODES:
        supported = ", ".join(sorted(SUPPORTED_REPAIR_MODES))
        raise ValueError(
            "Unsupported FAILED_CANDIDATE_REPAIR_MODE="
            f"{mode!r}; supported values: {supported}"
        )
    return mode


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


@dataclass(frozen=True, slots=True)
class ScheduledRepairJob:
    """Result descriptor for a failed-candidate repair job."""

    job_id: UUID
    repair_source_candidate_id: UUID
    base_commit_hash: str


class FailedCandidateRepairSampler:
    """Strict repair-pool sampler for ADR 0048 MVP repair jobs."""

    def __init__(self, *, settings: Settings, rng: random.Random | None = None) -> None:
        self.settings = settings
        self._repair_mode = _validated_failed_candidate_repair_mode(settings)
        self._rng = rng or random.Random(int(getattr(settings, "mapelites_sampler_seed", 0) or 0))

    def count_active_repair_jobs(self) -> int:
        active = (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)
        with session_scope() as session:
            return int(
                session.execute(
                    select(func.count(EvolutionJob.id)).where(
                        EvolutionJob.job_kind == "repair",
                        EvolutionJob.status.in_(active),
                    )
                ).scalar_one()
            )

    def schedule_one(self) -> ScheduledRepairJob | None:
        if not bool(self.settings.failed_candidate_repair_enabled):
            return None
        with session_scope() as session:
            candidate = self._select_candidate(session=session)
            if candidate is None:
                return None
            if self._has_active_repair_job(session=session, source_id=candidate.id):
                return None
            goal = self._repair_goal(candidate)
            campaign_program = load_campaign_program_snapshot_by_hash(
                session=session,
                program_hash=getattr(candidate, "campaign_program_hash", None),
            )
            projection = apply_campaign_program_projection(
                CampaignProjectionInput(
                    snapshot=campaign_program,
                    goal=goal,
                    constraints=(),
                    acceptance_criteria=(),
                    notes=(),
                    default_goal=self.settings.worker_evolution_global_goal,
                    preserve_existing_goal=True,
                )
            )
            now = _db_utc_now(session)
            job = EvolutionJob(
                status=JobStatus.PENDING,
                base_commit_hash=candidate.nearest_viable_ancestor_hash,
                island_id=candidate.island_id,
                inspiration_commit_hashes=[],
                goal=projection.goal or goal,
                constraints=projection.constraints,
                acceptance_criteria=projection.acceptance_criteria,
                notes=projection.notes,
                tags=["repair"],
                iteration_hint="Repair failed candidate from nearest viable ancestor.",
                sampling_strategy="failed_candidate_repair",
                sampling_initial_radius=None,
                sampling_radius_used=None,
                sampling_fallback_inspirations=None,
                is_seed_job=False,
                job_kind="repair",
                repair_source_candidate_id=candidate.id,
                repair_mode=self._repair_mode,
                campaign_program_hash=getattr(candidate, "campaign_program_hash", None),
                priority=self.settings.mapelites_sampler_default_priority,
                scheduled_at=now,
            )
            session.add(job)
            session.flush()
            candidate.repair_state = "scheduled"
            candidate.repair_attempts = int(candidate.repair_attempts or 0) + 1
            candidate.last_repair_job_id = job.id
            log.info(
                "Repair job scheduled repair_mode={} failure_kind={}",
                job.repair_mode,
                candidate.failure_kind or "unknown",
            )
            return ScheduledRepairJob(
                job_id=job.id,
                repair_source_candidate_id=candidate.id,
                base_commit_hash=str(candidate.nearest_viable_ancestor_hash),
            )

    def _select_candidate(self, *, session: Any) -> CandidateCommit | None:
        allowlist = repair_failure_kind_allowlist(self.settings.failed_candidate_repair_failure_kinds)
        rows = list(
            session.execute(
                select(CandidateCommit)
                .where(
                    CandidateCommit.publication_status == "published",
                    CandidateCommit.evaluation_status == "candidate_failed",
                    CandidateCommit.failure_stage == "evaluation",
                    CandidateCommit.failure_kind.in_(tuple(sorted(allowlist))),
                    CandidateCommit.repair_state == "eligible",
                    CandidateCommit.lifecycle_status == "active",
                    CandidateCommit.nearest_viable_ancestor_hash.is_not(None),
                    CandidateCommit.nearest_viable_ancestor_hash != "",
                    CandidateCommit.repair_source_candidate_id.is_(None),
                    CandidateCommit.failed_depth
                    <= max(0, int(self.settings.failed_candidate_repair_max_depth)),
                    CandidateCommit.repair_attempts
                    < max(0, int(self.settings.failed_candidate_repair_max_attempts)),
                )
                .order_by(
                    CandidateCommit.failed_depth.asc(),
                    CandidateCommit.updated_at.asc(),
                    CandidateCommit.id.asc(),
                )
                .limit(32)
                .with_for_update(skip_locked=True)
            ).scalars()
        )
        eligible = [row for row in rows if self._strictly_eligible(session=session, candidate=row)]
        if not eligible:
            return None
        top = eligible[: min(8, len(eligible))]
        return self._rng.choice(top)

    def _strictly_eligible(self, *, session: Any, candidate: CandidateCommit) -> bool:
        attempt = None
        if candidate.latest_evaluation_attempt_id is not None:
            attempt = session.get(EvaluationAttempt, candidate.latest_evaluation_attempt_id)
        if attempt is None or attempt.repairability != "repairable":
            return False
        if self._has_active_repair_job(session=session, source_id=candidate.id):
            return False
        if candidate.failure_evidence_id is None:
            return False
        capsule = session.get(DiagnosticCapsule, candidate.failure_evidence_id)
        if capsule is None or not capsule.policy_passed:
            return False
        if not self._ancestor_aggregate_ready(
            session=session,
            commit_hash=str(candidate.nearest_viable_ancestor_hash or ""),
        ):
            return False
        if not self._source_campaign_baseline_allows_repair(
            session=session,
            candidate=candidate,
        ):
            return False
        return True

    def _source_campaign_baseline_allows_repair(
        self,
        *,
        session: Any,
        candidate: CandidateCommit,
    ) -> bool:
        policy = str(getattr(self.settings, "baseline_bootstrap_policy", "required") or "required")
        if policy not in {"required", "warn"}:
            policy = "required"
        campaign_program_hash = getattr(candidate, "campaign_program_hash", None)
        if policy == "warn":
            baseline = load_latest_matching_baseline(
                session=session,
                settings=self.settings,
                campaign_program_hash=campaign_program_hash,
                valid_only=False,
            )
            baseline_status = getattr(baseline, "status", None)
            if baseline is not None and baseline_status == BASELINE_STATUS_DEGRADED:
                log.bind(
                    repair_source_candidate_id=str(getattr(candidate, "id", "")),
                    campaign_program_hash=campaign_program_hash,
                    baseline_id=str(getattr(baseline, "id", "")),
                    baseline_status=baseline_status,
                    baseline_policy=policy,
                ).warning("Repair candidate allowed with degraded campaign baseline under warn policy")
            if baseline is not None and baseline_status in {
                BASELINE_STATUS_VALID,
                BASELINE_STATUS_DEGRADED,
            }:
                return True
            log.bind(
                repair_source_candidate_id=str(getattr(candidate, "id", "")),
                campaign_program_hash=campaign_program_hash,
                baseline_status=baseline_status,
                baseline_policy=policy,
            ).info("Repair candidate blocked by missing campaign baseline under warn policy")
            return False

        baseline = load_latest_matching_baseline(
            session=session,
            settings=self.settings,
            campaign_program_hash=campaign_program_hash,
            valid_only=True,
        )
        if baseline is not None and getattr(baseline, "status", None) == BASELINE_STATUS_VALID:
            return True
        log.bind(
            repair_source_candidate_id=str(getattr(candidate, "id", "")),
            campaign_program_hash=campaign_program_hash,
            baseline_policy=policy,
        ).info("Repair candidate blocked by campaign baseline")
        return False

    @staticmethod
    def _has_active_repair_job(*, session: Any, source_id: UUID) -> bool:
        active = (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)
        return (
            session.execute(
                select(EvolutionJob.id)
                .where(
                    EvolutionJob.job_kind == "repair",
                    EvolutionJob.repair_source_candidate_id == source_id,
                    EvolutionJob.status.in_(active),
                )
                .limit(1)
            ).first()
            is not None
        )

    @staticmethod
    def _ancestor_aggregate_ready(*, session: Any, commit_hash: str) -> bool:
        if not commit_hash:
            return False
        return (
            session.execute(
                select(MapElitesRepoStateAggregate.commit_hash)
                .where(MapElitesRepoStateAggregate.commit_hash == commit_hash)
                .limit(1)
            ).first()
            is not None
        )

    def _repair_goal(self, candidate: CandidateCommit) -> str:
        base_goal = (self.settings.worker_evolution_global_goal or "").strip()
        summary = normalize_repair_text(candidate.failure_summary)
        if summary:
            return f"Repair failed candidate validation while preserving useful work. Failure: {summary}"
        return base_goal or "Repair failed candidate validation while preserving useful work."


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
    repo_root: Path | None = None
    _sender_actor: object = field(init=False, repr=False)
    repair_sampler: FailedCandidateRepairSampler = field(init=False, repr=False)
    _repair_tokens: int = field(default=0, init=False, repr=False)
    _repair_completed_normal_jobs_seen: int = field(default=0, init=False, repr=False)
    _campaign_program_snapshot: CampaignProgramSnapshot | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _reported_campaign_program_hashes: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        # Build a sender-only actor that targets the experiment-scoped queue.
        self._sender_actor = build_evolution_job_sender_actor(
            settings=self.settings,
        )
        self.repair_sampler = FailedCandidateRepairSampler(settings=self.settings)
        self._campaign_program_snapshot = self._load_startup_campaign_program()
        if bool(self.settings.failed_candidate_repair_enabled):
            self._repair_completed_normal_jobs_seen = self._count_completed_normal_jobs_best_effort()

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
                    self._update_repair_source_after_terminal_failure(session=session, job=job)
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

    def _update_repair_source_after_terminal_failure(
        self,
        *,
        session: Any,
        job: EvolutionJob,
    ) -> None:
        if str(getattr(job, "job_kind", "") or "").strip().lower() != "repair":
            return
        source_id = getattr(job, "repair_source_candidate_id", None)
        if source_id is None:
            return
        source = session.get(CandidateCommit, source_id)
        if source is None:
            return
        max_attempts = max(0, int(self.settings.failed_candidate_repair_max_attempts))
        source.repair_state = (
            "exhausted"
            if int(getattr(source, "repair_attempts", 0) or 0) >= max_attempts
            else "eligible"
        )

    # Scheduling ------------------------------------------------------------

    def schedule_jobs(
        self,
        unfinished_jobs: int,
        *,
        total_jobs: int,
        refresh_campaign_program: bool = True,
    ) -> int:
        """Schedule new jobs from MAP-Elites if there is available capacity.

        Parameters
        ----------
        unfinished_jobs:
            Current number of unfinished jobs in the system.
        total_jobs:
            Total number of jobs recorded in the database (used to enforce the global job limit).
        """

        if refresh_campaign_program:
            self._refresh_campaign_program_for_policy()
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

        scheduled_ids: list[UUID] = []
        sampling_snapshot = self.sampler.get_sampling_snapshot()
        repair_reservation = self._available_repair_slots(capacity=target)
        normal_target = max(0, target - repair_reservation)
        selected_base_commits: set[str] = set()
        if sampling_snapshot is not None:
            scheduled_ids.extend(
                self._schedule_normal_jobs(
                    capacity=normal_target,
                    sampling_snapshot=sampling_snapshot,
                    selected_base_commits=selected_base_commits,
                )
            )
        else:
            self.console.log("[yellow]Sampler returned no job[/]")

        remaining_capacity = max(0, target - len(scheduled_ids))
        repair_jobs = self._schedule_repair_jobs(capacity=remaining_capacity, accrue_tokens=False)
        scheduled_ids.extend(repair_jobs)
        if len(repair_jobs) < repair_reservation and sampling_snapshot is not None:
            scheduled_ids.extend(
                self._schedule_normal_jobs(
                    capacity=max(0, target - len(scheduled_ids)),
                    sampling_snapshot=sampling_snapshot,
                    selected_base_commits=selected_base_commits,
                )
            )
        if scheduled_ids:
            self._enqueue_jobs(scheduled_ids)
        return len(scheduled_ids)

    def _schedule_normal_jobs(
        self,
        *,
        capacity: int,
        sampling_snapshot: SamplingSnapshot,
        selected_base_commits: set[str],
    ) -> list[UUID]:
        if capacity <= 0:
            return []
        scheduled_ids: list[UUID] = []
        effective_island = sampling_snapshot.island_id
        for _ in range(capacity):
            job = self._schedule_single_job(
                island_id=effective_island,
                sampling_snapshot=sampling_snapshot,
                excluded_base_commits=selected_base_commits,
            )
            if not job:
                break
            scheduled_ids.append(job.job_id)
            selected_base_commits.add(str(job.base_commit_hash))
        return scheduled_ids

    def _available_repair_slots(self, *, capacity: int) -> int:
        if capacity <= 0 or not bool(self.settings.failed_candidate_repair_enabled):
            return 0
        self._accrue_repair_tokens()
        max_per_tick = max(0, int(self.settings.failed_candidate_repair_max_jobs_per_tick))
        max_active = max(0, int(self.settings.failed_candidate_repair_max_active_jobs))
        if max_per_tick <= 0 or max_active <= 0 or self._repair_tokens <= 0:
            return 0
        active = self.repair_sampler.count_active_repair_jobs()
        available_active = max(0, max_active - active)
        return min(capacity, max_per_tick, available_active, self._repair_tokens)

    def _schedule_repair_jobs(self, *, capacity: int, accrue_tokens: bool = True) -> list[UUID]:
        if capacity <= 0 or not bool(self.settings.failed_candidate_repair_enabled):
            return []
        if accrue_tokens:
            self._accrue_repair_tokens()
        max_per_tick = max(0, int(self.settings.failed_candidate_repair_max_jobs_per_tick))
        max_active = max(0, int(self.settings.failed_candidate_repair_max_active_jobs))
        if max_per_tick <= 0 or max_active <= 0 or self._repair_tokens <= 0:
            return []

        def _schedule_locked() -> list[UUID]:
            persistent_tokens = repair_tokens_available(settings=self.settings)
            self._repair_tokens = min(self._repair_tokens, persistent_tokens)
            if self._repair_tokens <= 0:
                return []
            active = self.repair_sampler.count_active_repair_jobs()
            available_active = max(0, max_active - active)
            count = min(capacity, max_per_tick, available_active, self._repair_tokens)
            if count <= 0:
                return []
            scheduled_ids: list[UUID] = []
            for _ in range(count):
                repair = self.repair_sampler.schedule_one()
                if repair is None:
                    break
                self._repair_tokens = max(0, self._repair_tokens - 1)
                scheduled_ids.append(repair.job_id)
                log.info(
                    "Repair token consumed tokens_remaining={}",
                    self._repair_tokens,
                )
            return scheduled_ids

        return with_repair_scheduling_lock(callback=_schedule_locked)

    def _accrue_repair_tokens(self) -> None:
        normal_jobs_per_token = max(1, int(self.settings.failed_candidate_repair_normal_jobs_per_token))
        max_tokens = max(0, int(self.settings.failed_candidate_repair_max_tokens))
        if max_tokens <= 0:
            self._repair_tokens = 0
            return
        completed = self._count_completed_normal_jobs_best_effort()
        delta = max(0, completed - self._repair_completed_normal_jobs_seen)
        if delta <= 0:
            return
        earned = delta // normal_jobs_per_token
        if earned <= 0:
            return
        self._repair_completed_normal_jobs_seen += earned * normal_jobs_per_token
        self._repair_tokens = min(max_tokens, self._repair_tokens + earned)
        log.info(
            "Repair tokens accrued earned={} tokens={} normal_jobs_seen={}",
            earned,
            self._repair_tokens,
            self._repair_completed_normal_jobs_seen,
        )

    def _count_completed_normal_jobs_best_effort(self) -> int:
        try:
            with session_scope() as session:
                return int(
                    session.execute(
                        select(func.count(EvolutionJob.id)).where(
                            EvolutionJob.status == JobStatus.SUCCEEDED,
                            EvolutionJob.job_kind != "repair",
                        )
                    ).scalar_one()
                )
        except Exception as exc:  # pragma: no cover - defensive scheduler accounting
            log.warning("Failed to count completed normal jobs for repair tokens: {}", exc)
            return self._repair_completed_normal_jobs_seen

    def create_seed_jobs(
        self,
        *,
        base_commit_hash: str,
        count: int,
        island_id: str | None = None,
        refresh_campaign_program: bool = True,
    ) -> int:
        """Create and enqueue cold-start seed jobs from the root commit.

        Seed jobs use the configured default priority and are immediately promoted
        to QUEUED and sent to Dramatiq.
        """

        if count <= 0:
            return 0

        if refresh_campaign_program:
            self._refresh_campaign_program_for_policy()
        effective_island = island_id or resolve_default_island_id(self.settings)
        now = datetime.now(timezone.utc)
        jobs: list[EvolutionJob] = []
        default_goal = (self.settings.worker_evolution_global_goal or "").strip()
        projection = apply_campaign_program_projection(
            CampaignProjectionInput(
                snapshot=self._campaign_program_snapshot,
                goal=default_goal,
                constraints=(),
                acceptance_criteria=(),
                notes=(),
                default_goal=default_goal,
            )
        )
        goal = (projection.goal or "").strip()
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
                    constraints=projection.constraints,
                    acceptance_criteria=projection.acceptance_criteria,
                    notes=projection.notes,
                    tags=[],
                    iteration_hint=(
                        "Cold-start seed job: design diverse initial directions "
                        "from the root baseline."
                    ),
                    job_kind="seed",
                    campaign_program_hash=(
                        self._campaign_program_snapshot.raw_sha256
                        if self._campaign_program_snapshot
                        else None
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
                campaign_program=self._campaign_program_snapshot,
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

    # Campaign program -----------------------------------------------------

    def _effective_repo_root(self) -> Path | None:
        if self.repo_root is not None:
            return Path(self.repo_root).expanduser().resolve()
        configured = str(getattr(self.settings, "scheduler_repo_root", "") or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return None

    def _load_startup_campaign_program(self) -> CampaignProgramSnapshot | None:
        repo_root = self._effective_repo_root()
        if repo_root is None:
            return None
        loaded = load_campaign_program_from_repo(repo_root)
        if loaded.snapshot is None:
            return None
        with session_scope() as session:
            persist_campaign_program(
                session=session,
                snapshot=loaded.snapshot,
                raw_markdown=loaded.raw_markdown or "",
            )
        self.console.log(
            "[green]Campaign program loaded[/] hash={} source={} recognized_sections={}".format(
                loaded.snapshot.raw_sha256[:12],
                loaded.snapshot.source_path,
                ",".join(loaded.snapshot.recognized_sections) or "none",
            ),
        )
        return loaded.snapshot

    def _refresh_campaign_program_for_policy(self) -> None:
        repo_root = self._effective_repo_root()
        if repo_root is None:
            return
        loaded = load_campaign_program_from_repo(repo_root)
        current_hash = loaded.snapshot.raw_sha256 if loaded.snapshot else None
        active_hash = (
            self._campaign_program_snapshot.raw_sha256
            if self._campaign_program_snapshot is not None
            else None
        )
        if current_hash == active_hash:
            return
        policy = str(getattr(self.settings, "campaign_program_change_policy", "locked") or "locked")
        report_key = current_hash or "<missing>"
        if policy == "auto":
            if loaded.snapshot is None:
                if report_key in self._reported_campaign_program_hashes:
                    return
                self._reported_campaign_program_hashes.add(report_key)
                log.warning(
                    "Campaign program missing under auto policy; retaining active hash old_hash={}",
                    active_hash,
                )
                return
            with session_scope() as session:
                persist_campaign_program(
                    session=session,
                    snapshot=loaded.snapshot,
                    raw_markdown=loaded.raw_markdown or "",
                )
            self._campaign_program_snapshot = loaded.snapshot
            self.console.log(
                "[yellow]Campaign program auto-updated[/] old_hash={} new_hash={}".format(
                    (active_hash or "none")[:12],
                    loaded.snapshot.raw_sha256[:12],
                ),
            )
            log.warning(
                "Campaign program auto-updated old_hash={} new_hash={}",
                active_hash,
                loaded.snapshot.raw_sha256,
            )
            return
        if report_key in self._reported_campaign_program_hashes:
            return
        self._reported_campaign_program_hashes.add(report_key)
        if policy == "approve":
            log.warning(
                "Campaign program changed but approve workflow is not implemented; retaining startup hash old_hash={} new_hash={}",
                active_hash,
                current_hash,
            )
            self.console.log(
                "[yellow]Campaign program changed[/] approve policy currently retains startup hash "
                f"old_hash={(active_hash or 'none')[:12]} new_hash={(current_hash or 'none')[:12]}",
            )
            return
        log.warning(
            "Campaign program changed under locked policy; retaining startup hash old_hash={} new_hash={}",
            active_hash,
            current_hash,
        )

    @property
    def campaign_program_snapshot(self) -> CampaignProgramSnapshot | None:
        return self._campaign_program_snapshot

    def refresh_campaign_program_for_policy(self) -> None:
        self._refresh_campaign_program_for_policy()

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
            current_time = _db_utc_now(session)
            queued_stale_after = timedelta(
                seconds=max(1, int(self.settings.worker_job_lease_ttl_seconds))
            )
            queued_stale_before = current_time - queued_stale_after
            stmt = (
                select(EvolutionJob.id)
                .where(
                    or_(
                        EvolutionJob.status == JobStatus.PENDING,
                        and_(
                            EvolutionJob.status == JobStatus.QUEUED,
                            or_(
                                EvolutionJob.scheduled_at.is_(None),
                                EvolutionJob.scheduled_at < queued_stale_before,
                            ),
                        ),
                    ),
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
        with session_scope() as session:
            now = _db_utc_now(session)
            stmt = (
                select(EvolutionJob)
                .where(EvolutionJob.id.in_(job_ids))
                .with_for_update(skip_locked=True)
            )
            for job in session.execute(stmt).scalars():
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.QUEUED
                    job.scheduled_at = job.scheduled_at or now
                elif job.status == JobStatus.QUEUED:
                    job.scheduled_at = now
                else:
                    continue
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


def normalize_repair_text(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    return text[:400] or None
