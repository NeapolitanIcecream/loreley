from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, get_settings
from app.core.worker.coding import CodingAgentResponse
from app.core.worker.evaluator import EvaluationResult
from app.core.worker.planning import PlanningAgentResponse
from app.db.base import session_scope
from app.db.models import CommitMetadata, EvolutionJob, JobStatus, Metric

if TYPE_CHECKING:
    from app.core.worker.evolution import JobContext

log = logger.bind(module="worker.job_store")

__all__ = [
    "EvolutionJobStore",
    "EvolutionWorkerError",
    "JobLockConflict",
    "JobPreconditionError",
    "LockedJob",
    "build_plan_payload",
    "build_coding_payload",
    "build_evaluation_payload",
]


class EvolutionWorkerError(RuntimeError):
    """Raised when the evolution worker cannot complete a job."""


class JobLockConflict(EvolutionWorkerError):
    """Raised when a concurrent worker already locked the target job row."""


class JobPreconditionError(EvolutionWorkerError):
    """Raised when a job cannot start due to invalid or missing preconditions."""


@dataclass(slots=True)
class LockedJob:
    """Snapshot of the locked EvolutionJob row used to build worker context."""

    job_id: UUID
    base_commit_hash: str
    island_id: str | None
    payload: dict[str, Any]
    inspiration_commit_hashes: tuple[str, ...]


def build_plan_payload(response: PlanningAgentResponse) -> dict[str, Any]:
    """Convert a planning response into a serializable payload."""

    plan_dict = response.plan.as_dict()
    plan_dict.update(
        {
            "prompt": response.prompt,
            "raw_output": response.raw_output,
            "command": list(response.command),
            "stderr": response.stderr,
            "attempts": response.attempts,
            "duration_seconds": response.duration_seconds,
        }
    )
    return plan_dict


def build_coding_payload(response: CodingAgentResponse) -> dict[str, Any]:
    """Convert a coding response into a serializable payload."""

    execution = response.execution
    return {
        "implementation_summary": execution.implementation_summary,
        "commit_message": execution.commit_message,
        "step_results": [
            {
                "step_id": step.step_id,
                "status": step.status.value,
                "summary": step.summary,
                "files": list(step.files),
                "commands": list(step.commands),
            }
            for step in execution.step_results
        ],
        "tests_executed": list(execution.tests_executed),
        "tests_recommended": list(execution.tests_recommended),
        "follow_up_items": list(execution.follow_up_items),
        "notes": list(execution.notes),
        "raw_output": response.raw_output,
        "prompt": response.prompt,
        "command": list(response.command),
        "stderr": response.stderr,
        "attempts": response.attempts,
        "duration_seconds": response.duration_seconds,
    }


def build_evaluation_payload(result: EvaluationResult) -> dict[str, Any]:
    """Convert an evaluation result into a serializable payload."""

    return {
        "summary": result.summary,
        "metrics": [metric.as_dict() for metric in result.metrics],
        "tests_executed": list(result.tests_executed),
        "logs": list(result.logs),
        "extra": dict(result.extra or {}),
    }


class EvolutionJobStore:
    """Persistence adapter for the evolution worker."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def start_job(self, job_id: UUID) -> LockedJob:
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

                job.status = JobStatus.RUNNING
                job.started_at = _utc_now()
                job.last_error = None

                return LockedJob(
                    job_id=job.id,
                    base_commit_hash=job.base_commit_hash,
                    island_id=job.island_id,
                    payload=dict(job.payload or {}),
                    inspiration_commit_hashes=tuple(job.inspiration_commit_hashes or []),
                )
        except SQLAlchemyError as exc:
            if self._is_lock_conflict(exc):
                raise JobLockConflict(f"Evolution job {job_id} is locked by another worker.") from exc
            raise EvolutionWorkerError(f"Failed to start job {job_id}: {exc}") from exc

    def persist_success(
        self,
        *,
        job_ctx: JobContext,
        plan: PlanningAgentResponse,
        coding: CodingAgentResponse,
        evaluation: EvaluationResult,
        commit_hash: str,
        commit_message: str,
    ) -> None:
        """Persist successful worker execution artifacts."""

        commit_extra = {
            "job": {
                "id": str(job_ctx.job_id),
                "island_id": job_ctx.island_id,
                "goal": job_ctx.goal,
                "constraints": list(job_ctx.constraints),
                "acceptance_criteria": list(job_ctx.acceptance_criteria),
                "notes": list(job_ctx.notes),
                "tags": list(job_ctx.tags),
                "payload": job_ctx.payload,
            },
            "base_commit": job_ctx.base_snapshot.commit_hash,
            "inspirations": [snapshot.commit_hash for snapshot in job_ctx.inspiration_snapshots],
            "plan": build_plan_payload(plan),
            "coding": build_coding_payload(coding),
            "evaluation": build_evaluation_payload(evaluation),
            "worker": {
                "app_name": self.settings.app_name,
                "environment": self.settings.environment,
                "completed_at": _utc_now().isoformat(),
            },
        }
        job_payload = dict(job_ctx.payload)
        job_payload["result"] = {
            "commit_hash": commit_hash,
            "plan_summary": plan.plan.summary,
            "tests_executed": list(coding.execution.tests_executed),
            "tests_recommended": list(coding.execution.tests_recommended),
            "evaluation_summary": evaluation.summary,
            "metrics": [metric.as_dict() for metric in evaluation.metrics],
        }

        try:
            with session_scope() as session:
                job = session.get(EvolutionJob, job_ctx.job_id)
                if not job:
                    raise EvolutionWorkerError(
                        f"Evolution job {job_ctx.job_id} disappeared during persistence.",
                    )
                job.status = JobStatus.SUCCEEDED
                job.completed_at = _utc_now()
                job.plan_summary = plan.plan.summary
                job.payload = job_payload
                job.last_error = None

                metadata = CommitMetadata(
                    commit_hash=commit_hash,
                    parent_commit_hash=job_ctx.base_commit_hash,
                    island_id=job_ctx.island_id,
                    author=self.settings.worker_evolution_commit_author,
                    message=commit_message,
                    evaluation_summary=evaluation.summary,
                    tags=list(job_ctx.tags),
                    extra_context=commit_extra,
                )
                session.add(metadata)
                for metric in evaluation.metrics:
                    session.add(
                        Metric(
                            commit_hash=commit_hash,
                            name=metric.name,
                            value=metric.value,
                            unit=metric.unit,
                            higher_is_better=metric.higher_is_better,
                            details=dict(metric.details or {}),
                        )
                    )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(f"Failed to persist results for job {job_ctx.job_id}: {exc}") from exc

    def mark_job_failed(self, job_id: UUID, message: str) -> None:
        """Persist failure status for the job while swallowing DB errors."""

        try:
            with session_scope() as session:
                job = session.get(EvolutionJob, job_id)
                if not job:
                    return
                if job.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
                    return
                job.status = JobStatus.FAILED
                job.completed_at = _utc_now()
                job.last_error = message
        except SQLAlchemyError as exc:
            log.error("Failed to record failure for job {}: {}", job_id, exc)

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

