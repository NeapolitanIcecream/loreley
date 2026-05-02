from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError

from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.worker.artifacts import (
    FixedJobArtifactPaths,
    JobArtifactWriteRequest,
    JobArtifactWriteResult,
    resolve_worker_instance_id,
    write_job_artifacts,
)
from loreley.core.worker.commit_card import build_commit_card_from_git
from loreley.config import Settings, get_settings
from loreley.core.worker.coding import CodingAgentResponse
from loreley.core.worker.evaluator import EvaluationResult
from loreley.core.worker.planning import PlanningAgentResponse
from loreley.db.base import session_scope
from loreley.db.models import (
    CommitCard,
    EvaluationArtifactRecord,
    EvolutionJob,
    JobArtifacts,
    JobStatus,
    Metric,
)

if TYPE_CHECKING:
    from loreley.core.worker.evolution import JobContext

log = logger.bind(module="worker.job_store")

_WORKER_ID_MAX_CHARS = int(getattr(EvolutionJob.__table__.c.worker_id.type, "length", 128) or 128)
_WORKER_ID_HASH_CHARS = 12

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


@dataclass(slots=True)
class _PersistSuccessPayload:
    subject: str
    change_summary: str
    eval_summary: str | None
    key_files: list[str]
    highlights: list[str]
    tags: list[str]
    artifact_result: JobArtifactWriteResult


@dataclass(slots=True)
class _PersistSuccessInput:
    job_ctx: "JobContext"
    plan: PlanningAgentResponse
    coding: CodingAgentResponse
    evaluation: EvaluationResult
    worktree: Path
    commit_hash: str
    commit_message: str


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

                now = _db_utc_now(session)
                run_token = uuid4()
                worker_id = _bounded_worker_instance_id(resolve_worker_instance_id())
                job.status = JobStatus.RUNNING
                job.started_at = now
                job.completed_at = None
                job.heartbeat_at = now
                job.lease_expires_at = now + self._lease_ttl()
                job.run_token = run_token
                job.worker_id = worker_id
                job.last_error = None
                job.candidate_commit_hash = None
                job.candidate_branch_name = None
                job.candidate_published_at = None

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

    def record_candidate_commit(
        self,
        job_id: UUID,
        commit_hash: str,
        branch_name: str,
        *,
        run_token: UUID | None = None,
        published: bool = False,
    ) -> None:
        """Persist candidate commit metadata before or after remote publication."""

        candidate_hash = str(commit_hash or "").strip()
        if not candidate_hash:
            raise EvolutionWorkerError("Candidate commit hash must be provided.")
        candidate_branch = str(branch_name or "").strip()
        if not candidate_branch:
            raise EvolutionWorkerError("Candidate branch name must be provided.")

        try:
            with session_scope() as session:
                if run_token is not None:
                    job = self._lock_active_job_for_run(
                        session=session,
                        job_id=job_id,
                        run_token=run_token,
                        action="recording candidate metadata",
                    )
                else:
                    job = session.get(EvolutionJob, job_id)
                    if not job:
                        raise EvolutionWorkerError(
                            f"Evolution job {job_id} disappeared while recording candidate metadata.",
                        )
                if run_token is None and job.status in {JobStatus.SUCCEEDED, JobStatus.CANCELLED}:
                    raise EvolutionWorkerError(
                        f"Evolution job {job_id} cannot record a candidate in status {job.status}.",
                    )
                job.candidate_commit_hash = candidate_hash
                job.candidate_branch_name = candidate_branch
                job.candidate_published_at = _utc_now() if published else None
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(
                f"Failed to record candidate metadata for job {job_id}: {exc}",
            ) from exc

    def renew_job_lease(self, job_id: UUID, run_token: UUID) -> datetime:
        """Extend the lease for an active RUNNING job attempt."""

        try:
            with session_scope() as session:
                now = _db_utc_now(session)
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

        request = _PersistSuccessInput(
            job_ctx=job_ctx,
            plan=plan,
            coding=coding,
            evaluation=evaluation,
            worktree=worktree,
            commit_hash=commit_hash,
            commit_message=commit_message,
        )
        try:
            with session_scope() as session:
                job = self._lock_active_job_for_run(
                    session=session,
                    job_id=job_ctx.job_id,
                    run_token=job_ctx.run_token,
                    action="persisting success",
                )
                payload = self._build_success_payload(request)
                self._mark_job_row_succeeded(job, plan=plan, commit_hash=commit_hash)
                card = self._add_commit_card(
                    session=session,
                    job_ctx=job_ctx,
                    commit_hash=commit_hash,
                    payload=payload,
                )
                self._add_metric_rows(session=session, card=card, evaluation=evaluation)
                self._flush_session(session)
                self._merge_fixed_artifacts(
                    session=session,
                    job_id=job_ctx.job_id,
                    fixed=payload.artifact_result.fixed,
                )
                self._add_evaluation_artifact_records(
                    session=session,
                    job_id=job_ctx.job_id,
                    commit_hash=commit_hash,
                    card=card,
                    artifacts=payload.artifact_result.evaluation_artifacts,
                )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(f"Failed to persist results for job {job_ctx.job_id}: {exc}") from exc

    def _build_success_payload(self, request: _PersistSuccessInput) -> _PersistSuccessPayload:
        subject = _success_subject(request.job_ctx.job_id, request.commit_message)
        build = build_commit_card_from_git(
            worktree=Path(request.worktree),
            base_commit=request.job_ctx.base_commit_hash,
            candidate_commit=request.commit_hash,
        )
        return _PersistSuccessPayload(
            subject=subject,
            change_summary=_change_summary(request),
            eval_summary=clamp_text(normalize_single_line(request.evaluation.summary), 512) or None,
            key_files=_commit_card_key_files(build.key_files),
            highlights=_commit_card_highlights(build.highlights),
            tags=_bounded_tags(request.job_ctx.tags),
            artifact_result=self._write_success_artifacts(request, subject=subject),
        )

    def _write_success_artifacts(
        self,
        request: _PersistSuccessInput,
        *,
        subject: str,
    ) -> JobArtifactWriteResult:
        try:
            return _coerce_artifact_write_result(
                write_job_artifacts(
                    JobArtifactWriteRequest(
                        job_id=request.job_ctx.job_id,
                        run_token=request.job_ctx.run_token,
                        plan=request.plan,
                        coding=request.coding,
                        evaluation=request.evaluation,
                        base_commit_hash=request.job_ctx.base_commit_hash,
                        candidate_commit_hash=request.commit_hash,
                        commit_message=subject,
                        worktree=Path(request.worktree),
                        settings=self.settings,
                    )
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort artifact store
            log.warning("Failed to write artifacts for job {}: {}", request.job_ctx.job_id, exc)
        return JobArtifactWriteResult(fixed=FixedJobArtifactPaths())

    @staticmethod
    def _mark_job_row_succeeded(
        job: EvolutionJob,
        *,
        plan: PlanningAgentResponse,
        commit_hash: str,
    ) -> None:
        job.status = JobStatus.SUCCEEDED
        job.completed_at = _utc_now()
        job.heartbeat_at = None
        job.lease_expires_at = None
        job.run_token = None
        job.worker_id = None
        job.plan_summary = plan.plan.summary
        job.candidate_commit_hash = job.candidate_commit_hash or commit_hash
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

    def _add_commit_card(
        self,
        *,
        session: Any,
        job_ctx: JobContext,
        commit_hash: str,
        payload: _PersistSuccessPayload,
    ) -> CommitCard:
        card = CommitCard(
            commit_hash=commit_hash,
            parent_commit_hash=job_ctx.base_commit_hash,
            island_id=job_ctx.island_id,
            author=self.settings.worker_evolution_commit_author,
            subject=payload.subject,
            change_summary=payload.change_summary,
            evaluation_summary=payload.eval_summary,
            tags=payload.tags,
            key_files=payload.key_files,
            highlights=payload.highlights,
            job_id=job_ctx.job_id,
        )
        session.add(card)
        return card

    @staticmethod
    def _add_metric_rows(
        *,
        session: Any,
        card: CommitCard,
        evaluation: EvaluationResult,
    ) -> None:
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

    @staticmethod
    def _flush_session(session: Any) -> None:
        flush = getattr(session, "flush", None)
        if callable(flush):
            flush()

    @staticmethod
    def _merge_fixed_artifacts(
        *,
        session: Any,
        job_id: UUID,
        fixed: FixedJobArtifactPaths,
    ) -> None:
        fixed_paths = fixed.as_dict()
        if not fixed_paths:
            return
        artifact_row = JobArtifacts(
            job_id=job_id,
            planning_prompt_path=fixed_paths.get("planning_prompt_path"),
            planning_raw_output_path=fixed_paths.get("planning_raw_output_path"),
            planning_plan_json_path=fixed_paths.get("planning_plan_json_path"),
            coding_prompt_path=fixed_paths.get("coding_prompt_path"),
            coding_raw_output_path=fixed_paths.get("coding_raw_output_path"),
            coding_execution_json_path=fixed_paths.get("coding_execution_json_path"),
            evaluation_json_path=fixed_paths.get("evaluation_json_path"),
            evaluation_logs_path=fixed_paths.get("evaluation_logs_path"),
        )
        merge = getattr(session, "merge", None)
        if callable(merge):
            merge(artifact_row)
        else:  # pragma: no cover - unit-test fallback for simplified sessions
            session.add(artifact_row)

    @staticmethod
    def _add_evaluation_artifact_records(
        *,
        session: Any,
        job_id: UUID,
        commit_hash: str,
        card: CommitCard,
        artifacts: tuple[Any, ...],
    ) -> None:
        for artifact in artifacts:
            session.add(
                EvaluationArtifactRecord(
                    job_id=job_id,
                    commit_card_id=card.id,
                    commit_hash=commit_hash,
                    key=artifact.key,
                    kind=artifact.kind,
                    mime_type=artifact.mime_type,
                    label=artifact.label,
                    summary=artifact.summary,
                    visibility=artifact.visibility,
                    agent_projection=artifact.agent_projection,
                    storage_path=artifact.storage_path,
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.sha256,
                    diagnostics=[diagnostic.as_dict() for diagnostic in artifact.diagnostics],
                    artifact_metadata=dict(artifact.metadata or {}),
                )
            )

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
                if run_token is not None:
                    try:
                        job = self._lock_active_job_for_run(
                            session=session,
                            job_id=job_id,
                            run_token=run_token,
                            action="marking failure",
                        )
                    except JobLeaseLost:
                        return False
                else:
                    job = session.get(EvolutionJob, job_id)
                    if not job:
                        return False
                    if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
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
    def _lock_active_job_for_run(
        *,
        session: Any,
        job_id: UUID,
        run_token: UUID,
        action: str,
    ) -> EvolutionJob:
        stmt = (
            select(EvolutionJob)
            .where(
                EvolutionJob.id == job_id,
                EvolutionJob.status == JobStatus.RUNNING,
                EvolutionJob.run_token == run_token,
            )
            .with_for_update()
        )
        job = session.execute(stmt).scalar_one_or_none()
        if not job:
            raise JobLeaseLost(
                f"Evolution job {job_id} lease was lost before {action} "
                f"(expected_run_token={run_token}).",
            )
        return job

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


def _db_utc_now(session: Any) -> datetime:
    value = session.execute(select(func.now())).scalar_one()
    if not isinstance(value, datetime):
        raise RuntimeError(f"Database current timestamp returned unsupported value: {value!r}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _bounded_worker_instance_id(value: str) -> str:
    worker_id = str(value or "").strip()
    if len(worker_id) <= _WORKER_ID_MAX_CHARS:
        return worker_id
    digest = hashlib.sha1(worker_id.encode("utf-8")).hexdigest()[:_WORKER_ID_HASH_CHARS]
    suffix = f"-{digest}"
    prefix_budget = max(1, _WORKER_ID_MAX_CHARS - len(suffix))
    bounded = f"{worker_id[:prefix_budget]}{suffix}"
    log.warning(
        "Worker instance id exceeded {} chars; using bounded lease owner id length={} digest={}",
        _WORKER_ID_MAX_CHARS,
        len(bounded),
        digest,
    )
    return bounded


def _success_subject(job_id: UUID, commit_message: str) -> str:
    subject = normalize_single_line(commit_message) or f"Evolution job {job_id}"
    if "```" in subject or subject.startswith("{") or subject.startswith("["):
        subject = f"Evolution job {job_id}"
    return clamp_text(subject, 72)


def _change_summary(request: _PersistSuccessInput) -> str:
    source = (
        request.coding.report.summary
        or request.plan.plan.summary
        or f"Evolution job {request.job_ctx.job_id}"
    )
    return clamp_text(normalize_single_line(source), 512) or "N/A"


def _commit_card_key_files(paths: Sequence[str]) -> list[str]:
    return [clamp_text(path, 256) for path in paths[:20] if path.strip()]


def _commit_card_highlights(lines: Sequence[str]) -> list[str]:
    highlights = [clamp_text(line, 200) for line in lines[:8] if line.strip()]
    return highlights or ["No file-level highlights available."]


def _bounded_tags(tags: Sequence[str]) -> list[str]:
    return [clamp_text(normalize_single_line(tag), 64) for tag in tags if str(tag).strip()]


def _coerce_artifact_write_result(value: Any) -> JobArtifactWriteResult:
    """Accept the current typed write result and old dict-shaped test doubles."""

    if isinstance(value, JobArtifactWriteResult):
        return value
    if isinstance(value, dict):
        allowed = {
            key: str(path)
            for key, path in value.items()
            if key in FixedJobArtifactPaths.__dataclass_fields__ and path
        }
        return JobArtifactWriteResult(fixed=FixedJobArtifactPaths(**allowed))
    return JobArtifactWriteResult(fixed=FixedJobArtifactPaths())
