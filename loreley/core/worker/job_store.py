from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.worker.artifacts import resolve_worker_instance_id, write_job_artifacts
from loreley.core.worker.commit_card import build_commit_card_from_git
from loreley.config import Settings, get_settings
from loreley.core.worker.coding import CodingAgentResponse
from loreley.core.worker.evaluator import EvaluationResult
from loreley.core.worker.planning import PlanningAgentResponse
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, EvolutionJob, JobArtifacts, JobStatus, Metric

if TYPE_CHECKING:
    from loreley.core.worker.evolution import JobContext

log = logger.bind(module="worker.job_store")

__all__ = [
    "EvolutionJobStore",
    "EvolutionWorkerError",
    "JobLeaseLost",
    "JobLockConflict",
    "JobPreconditionError",
    "LockedJob",
]


class EvolutionWorkerError(RuntimeError):
    """Raised when the evolution worker cannot complete a job."""


class JobLockConflict(EvolutionWorkerError):
    """Raised when a concurrent worker already locked the target job row."""


class JobPreconditionError(EvolutionWorkerError):
    """Raised when a job cannot start due to invalid or missing preconditions."""


class JobLeaseLost(EvolutionWorkerError):
    """Raised when a worker no longer owns the active lease for a job."""


@dataclass(slots=True)
class LockedJob:
    """Snapshot of the locked EvolutionJob row used to build worker context."""

    job_id: UUID
    run_token: UUID
    worker_id: str
    base_commit_hash: str
    island_id: str | None
    inspiration_commit_hashes: tuple[str, ...]
    goal: str | None
    constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    iteration_hint: str | None
    notes: tuple[str, ...]
    tags: tuple[str, ...]
    is_seed_job: bool
    sampling_strategy: str | None
    sampling_initial_radius: int | None
    sampling_radius_used: int | None
    sampling_fallback_inspirations: int | None

class EvolutionJobStore:
    """Persistence adapter for the evolution worker."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def start_job(
        self,
        job_id: UUID,
    ) -> LockedJob:
        """Lock the job row, validate status, and mark it as running."""

        try:
            with session_scope() as session:
                job_stmt = (
                    select(EvolutionJob)
                    .where(EvolutionJob.id == job_id)
                    .with_for_update(nowait=True)
                )
                job = session.execute(job_stmt).scalar_one_or_none()
                if not job:
                    raise JobPreconditionError(f"Evolution job {job_id} does not exist.")
                if not job.base_commit_hash:
                    raise EvolutionWorkerError("Evolution job is missing base_commit_hash.")

                allowed_statuses = {JobStatus.PENDING, JobStatus.QUEUED}
                if job.status not in allowed_statuses:
                    raise JobPreconditionError(
                        f"Evolution job {job_id} is {job.status} and cannot run.",
                    )

                now = _utc_now()
                run_token = uuid4()
                worker_id = resolve_worker_instance_id()
                job.status = JobStatus.RUNNING
                job.started_at = now
                job.completed_at = None
                job.heartbeat_at = now
                job.lease_expires_at = now + self._lease_ttl()
                job.run_token = run_token
                job.worker_id = worker_id
                job.last_error = None

                return LockedJob(
                    job_id=job.id,
                    run_token=run_token,
                    worker_id=worker_id,
                    base_commit_hash=job.base_commit_hash,
                    island_id=job.island_id,
                    inspiration_commit_hashes=tuple(job.inspiration_commit_hashes or []),
                    goal=(job.goal or None),
                    constraints=tuple(job.constraints or ()),
                    acceptance_criteria=tuple(job.acceptance_criteria or ()),
                    iteration_hint=job.iteration_hint,
                    notes=tuple(job.notes or ()),
                    tags=tuple(job.tags or ()),
                    is_seed_job=bool(getattr(job, "is_seed_job", False)),
                    sampling_strategy=getattr(job, "sampling_strategy", None),
                    sampling_initial_radius=getattr(job, "sampling_initial_radius", None),
                    sampling_radius_used=getattr(job, "sampling_radius_used", None),
                    sampling_fallback_inspirations=getattr(job, "sampling_fallback_inspirations", None),
                )
        except SQLAlchemyError as exc:
            if self._is_lock_conflict(exc):
                raise JobLockConflict(f"Evolution job {job_id} is locked by another worker.") from exc
            raise EvolutionWorkerError(f"Failed to start job {job_id}: {exc}") from exc

    def renew_job_lease(self, job_id: UUID, run_token: UUID) -> datetime:
        """Extend the lease for an active RUNNING job attempt."""

        now = _utc_now()
        lease_expires_at = now + self._lease_ttl()
        stmt = (
            update(EvolutionJob)
            .where(
                EvolutionJob.id == job_id,
                EvolutionJob.status == JobStatus.RUNNING,
                EvolutionJob.run_token == run_token,
            )
            .values(
                heartbeat_at=now,
                lease_expires_at=lease_expires_at,
            )
        )
        try:
            with session_scope() as session:
                result = session.execute(stmt)
                if int(getattr(result, "rowcount", 0) or 0) != 1:
                    raise JobLeaseLost(
                        f"Evolution job {job_id} lease is no longer active for run_token={run_token}.",
                    )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(f"Failed to renew job lease for {job_id}: {exc}") from exc
        return lease_expires_at

    def persist_success(
        self,
        *,
        job_ctx: JobContext,
        plan: PlanningAgentResponse,
        coding: CodingAgentResponse,
        evaluation: EvaluationResult,
        worktree: Path,
        commit_hash: str,
        commit_message: str,
    ) -> None:
        """Persist successful worker execution artifacts.

        Hot-path data (CommitCard + job indices) is written to the DB.
        Cold-path evidence (prompts/raw/logs) is written to disk and referenced
        via the JobArtifacts table.
        """

        subject = normalize_single_line(commit_message) or f"Evolution job {job_ctx.job_id}"
        if "```" in subject or subject.startswith("{") or subject.startswith("["):
            subject = f"Evolution job {job_ctx.job_id}"
        subject = clamp_text(subject, 72)

        change_summary_source = (
            coding.report.summary
            or plan.plan.summary
            or f"Evolution job {job_ctx.job_id}"
        )
        change_summary = clamp_text(normalize_single_line(change_summary_source), 512) or "N/A"

        eval_summary = clamp_text(normalize_single_line(evaluation.summary), 512) or None

        build = build_commit_card_from_git(
            worktree=Path(worktree),
            base_commit=job_ctx.base_commit_hash,
            candidate_commit=commit_hash,
        )
        key_files = [clamp_text(path, 256) for path in build.key_files[:20] if path.strip()]
        highlights = [clamp_text(line, 200) for line in build.highlights[:8] if line.strip()]
        if not highlights:
            highlights = ["No file-level highlights available."]

        tags = [clamp_text(normalize_single_line(tag), 64) for tag in job_ctx.tags if str(tag).strip()]

        artifact_paths: dict[str, str] = {}
        try:
            artifact_paths = write_job_artifacts(
                job_id=job_ctx.job_id,
                run_token=job_ctx.run_token,
                plan=plan,
                coding=coding,
                evaluation=evaluation,
                base_commit_hash=job_ctx.base_commit_hash,
                candidate_commit_hash=commit_hash,
                commit_message=subject,
                settings=self.settings,
            )
        except Exception as exc:  # pragma: no cover - best-effort artifact store
            log.warning("Failed to write artifacts for job {}: {}", job_ctx.job_id, exc)

        try:
            with session_scope() as session:
                job = session.get(EvolutionJob, job_ctx.job_id)
                if not job:
                    raise EvolutionWorkerError(
                        f"Evolution job {job_ctx.job_id} disappeared during persistence.",
                    )
                self._require_active_lease(job=job, job_ctx=job_ctx)
                job.status = JobStatus.SUCCEEDED
                job.completed_at = _utc_now()
                job.heartbeat_at = None
                job.lease_expires_at = None
                job.run_token = None
                job.worker_id = None
                job.plan_summary = plan.plan.summary
                job.result_commit_hash = commit_hash
                job.last_error = None
                job.ingestion_status = None
                job.ingestion_attempts = 0
                job.ingestion_delta = None
                job.ingestion_status_code = None
                job.ingestion_message = None
                job.ingestion_cell_index = None
                job.ingestion_last_attempt_at = None
                job.ingestion_reason = None

                card = CommitCard(
                    commit_hash=commit_hash,
                    parent_commit_hash=job_ctx.base_commit_hash,
                    island_id=job_ctx.island_id,
                    author=self.settings.worker_evolution_commit_author,
                    subject=subject,
                    change_summary=change_summary,
                    evaluation_summary=eval_summary,
                    tags=tags,
                    key_files=key_files,
                    highlights=highlights,
                    job_id=job_ctx.job_id,
                )
                session.add(card)
                for metric in evaluation.metrics:
                    session.add(
                        Metric(
                            commit=card,
                            name=metric.name,
                            value=metric.value,
                            unit=metric.unit,
                            higher_is_better=metric.higher_is_better,
                            details=dict(metric.details or {}),
                        )
                    )
                if artifact_paths:
                    artifact_row = JobArtifacts(
                        job_id=job_ctx.job_id,
                        planning_prompt_path=artifact_paths.get("planning_prompt_path"),
                        planning_raw_output_path=artifact_paths.get("planning_raw_output_path"),
                        planning_plan_json_path=artifact_paths.get("planning_plan_json_path"),
                        coding_prompt_path=artifact_paths.get("coding_prompt_path"),
                        coding_raw_output_path=artifact_paths.get("coding_raw_output_path"),
                        coding_execution_json_path=artifact_paths.get("coding_execution_json_path"),
                        evaluation_json_path=artifact_paths.get("evaluation_json_path"),
                        evaluation_logs_path=artifact_paths.get("evaluation_logs_path"),
                    )
                    merge = getattr(session, "merge", None)
                    if callable(merge):
                        merge(artifact_row)
                    else:  # pragma: no cover - unit-test fallback for simplified sessions
                        session.add(artifact_row)
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(f"Failed to persist results for job {job_ctx.job_id}: {exc}") from exc

    def mark_job_failed(
        self,
        job_id: UUID,
        message: str,
        *,
        run_token: UUID | None = None,
    ) -> bool:
        """Persist failure status for the job while swallowing DB errors."""

        try:
            with session_scope() as session:
                job = session.get(EvolutionJob, job_id)
                if not job:
                    return False
                if job.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
                    return False
                if run_token is not None:
                    active_run_token = getattr(job, "run_token", None)
                    if job.status != JobStatus.RUNNING or active_run_token != run_token:
                        return False
                job.status = JobStatus.FAILED
                job.completed_at = _utc_now()
                job.heartbeat_at = None
                job.lease_expires_at = None
                job.run_token = None
                job.worker_id = None
                job.last_error = message
                return True
        except SQLAlchemyError as exc:
            log.error("Failed to record failure for job {}: {}", job_id, exc)
        return False

    def _lease_ttl(self) -> timedelta:
        ttl_seconds = max(1, int(self.settings.worker_job_lease_ttl_seconds))
        return timedelta(seconds=ttl_seconds)

    @staticmethod
    def _require_active_lease(*, job: EvolutionJob, job_ctx: JobContext) -> None:
        expected = getattr(job_ctx, "run_token", None)
        actual = getattr(job, "run_token", None)
        if expected is None:
            raise JobLeaseLost(f"Evolution job {job_ctx.job_id} is missing a run_token.")
        if job.status != JobStatus.RUNNING or actual != expected:
            raise JobLeaseLost(
                "Evolution job {} lease was lost before persistence "
                "(expected_run_token={} actual_run_token={} status={}).".format(
                    job_ctx.job_id,
                    expected,
                    actual,
                    getattr(job, "status", None),
                )
            )

    @staticmethod
    def _is_lock_conflict(exc: SQLAlchemyError) -> bool:
        """Return True when the DB error indicates a NOWAIT lock conflict."""

        orig = getattr(exc, "orig", None)
        if not orig:
            return False
        pgcode = getattr(orig, "pgcode", None)
        if pgcode == "55P03":  # PostgreSQL lock_not_available
            return True
        message = str(orig).lower()
        return "could not obtain lock" in message or "database is locked" in message


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
