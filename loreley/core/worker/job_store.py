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
    FailureJobArtifactWriteRequest,
    FixedJobArtifactPaths,
    JobArtifactWriteRequest,
    JobArtifactWriteResult,
    resolve_worker_instance_id,
    write_failure_job_artifacts,
    write_job_artifacts,
)
from loreley.core.worker.commit_card import build_commit_card_from_git
from loreley.config import Settings, get_settings
from loreley.core.worker.coding import CodingAgentResponse
from loreley.core.worker.evaluator import EvaluationOutcome, EvaluationResult
from loreley.core.worker.planning import PlanningAgentResponse
from loreley.core.worker.repair import build_diagnostic_capsule, repair_failure_kind_allowlist
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    CommitCard,
    DiagnosticCapsule,
    EvaluationAttempt,
    EvaluationArtifactRecord,
    EvolutionJob,
    JobArtifacts,
    JobStatus,
    MapElitesRepoStateAggregate,
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
    job_kind: str
    repair_source_candidate_id: UUID | None
    repair_mode: str | None
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
    evaluation_outcome: EvaluationOutcome | None
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
                job_kind = _job_kind_from_job(job)
                if job_kind == "repair" and job.repair_source_candidate_id is not None:
                    source = session.get(CandidateCommit, job.repair_source_candidate_id)
                    if source is not None:
                        source.repair_state = "repairing"

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
                    job_kind=job_kind,
                    repair_source_candidate_id=getattr(job, "repair_source_candidate_id", None),
                    repair_mode=getattr(job, "repair_mode", None),
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
                if run_token is None and job.status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    raise EvolutionWorkerError(
                        f"Evolution job {job_id} cannot record a candidate in status {job.status}.",
                    )
                job.candidate_commit_hash = candidate_hash
                job.candidate_branch_name = candidate_branch
                job.candidate_published_at = _utc_now() if published else None
                self._upsert_candidate_commit_row(
                    session=session,
                    job=job,
                    commit_hash=candidate_hash,
                    branch_name=candidate_branch,
                    published=published,
                    run_token=run_token,
                )
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

    def _upsert_candidate_commit_row(
        self,
        *,
        session: Any,
        job: EvolutionJob,
        commit_hash: str,
        branch_name: str,
        published: bool,
        run_token: UUID | None,
    ) -> CandidateCommit:
        row = session.execute(
            select(CandidateCommit).where(CandidateCommit.commit_hash == commit_hash)
        ).scalar_one_or_none()
        if row is not None and not isinstance(row, CandidateCommit):
            row = None
        job_kind = _job_kind_from_job(job)
        publication_status = "published" if published else "created"
        published_at = _utc_now() if published else None
        if row is None:
            row = CandidateCommit(
                commit_hash=commit_hash,
                git_parent_commit_hash=str(getattr(job, "base_commit_hash", "") or "").strip(),
                nearest_viable_ancestor_hash=_nearest_viable_ancestor_for_job(job),
                island_id=getattr(job, "island_id", None),
                produced_by_job_id=getattr(job, "id", None),
                run_token=run_token or getattr(job, "run_token", None),
                job_kind=job_kind,
                repair_source_candidate_id=getattr(job, "repair_source_candidate_id", None),
                repair_mode=getattr(job, "repair_mode", None),
                candidate_branch_name=branch_name,
                candidate_published_at=published_at,
                publication_status=publication_status,
                evaluation_status="not_evaluated",
                archive_status="not_considered",
                lifecycle_status="active",
                repair_state="audit_only",
                failed_depth=_candidate_failed_depth_for_job(session=session, job=job),
                repair_attempts=0,
                repo_state_aggregate_status="not_required",
                published_at=published_at,
            )
            session.add(row)
            log.info(
                "CandidateCommit recorded job_kind={} publication_status={}",
                row.job_kind,
                row.publication_status,
            )
            return row

        row.git_parent_commit_hash = row.git_parent_commit_hash or str(getattr(job, "base_commit_hash", "") or "").strip()
        row.nearest_viable_ancestor_hash = row.nearest_viable_ancestor_hash or _nearest_viable_ancestor_for_job(job)
        row.island_id = row.island_id or getattr(job, "island_id", None)
        row.produced_by_job_id = row.produced_by_job_id or getattr(job, "id", None)
        row.run_token = row.run_token or run_token or getattr(job, "run_token", None)
        row.job_kind = row.job_kind or job_kind
        row.repair_source_candidate_id = (
            row.repair_source_candidate_id or getattr(job, "repair_source_candidate_id", None)
        )
        row.repair_mode = row.repair_mode or getattr(job, "repair_mode", None)
        row.candidate_branch_name = branch_name
        row.publication_status = publication_status
        if published:
            row.candidate_published_at = published_at
            row.published_at = published_at
        log.info(
            "CandidateCommit updated job_kind={} publication_status={}",
            row.job_kind,
            row.publication_status,
        )
        return row

    def persist_success(
        self,
        *,
        job_ctx: JobContext,
        plan: PlanningAgentResponse,
        coding: CodingAgentResponse,
        evaluation: EvaluationResult,
        evaluation_outcome: EvaluationOutcome | None = None,
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
            evaluation_outcome=evaluation_outcome,
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
                self._record_success_evaluation(
                    session=session,
                    job=job,
                    job_ctx=job_ctx,
                    card=card,
                    commit_hash=commit_hash,
                    evaluation=evaluation,
                    outcome=request.evaluation_outcome,
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
            evaluation = _merge_success_outcome_artifacts(
                evaluation=request.evaluation,
                outcome=request.evaluation_outcome,
            )
            return _coerce_artifact_write_result(
                write_job_artifacts(
                    JobArtifactWriteRequest(
                        job_id=request.job_ctx.job_id,
                        run_token=request.job_ctx.run_token,
                        plan=request.plan,
                        coding=request.coding,
                        evaluation=evaluation,
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

    def _record_success_evaluation(
        self,
        *,
        session: Any,
        job: EvolutionJob,
        job_ctx: JobContext,
        card: CommitCard,
        commit_hash: str,
        evaluation: EvaluationResult,
        outcome: EvaluationOutcome | None,
    ) -> None:
        candidate = self._candidate_for_commit(session=session, commit_hash=commit_hash)
        if candidate is None:
            candidate = self._upsert_candidate_commit_row(
                session=session,
                job=job,
                commit_hash=commit_hash,
                branch_name=str(job.candidate_branch_name or ""),
                published=bool(job.candidate_published_at),
                run_token=job_ctx.run_token,
            )
        self._flush_session(session)
        effective_outcome = outcome or EvaluationOutcome(
            evaluator_name=None,
            candidate_commit_hash=commit_hash,
            outcome_kind="passed",
            result=evaluation,
            started_at=_utc_now(),
            finished_at=_utc_now(),
        )
        attempt = EvaluationAttempt(
            candidate_commit_id=candidate.id,
            job_id=job_ctx.job_id,
            evaluator_name=effective_outcome.evaluator_name,
            evaluator_version=effective_outcome.evaluator_version,
            outcome_kind="passed",
            repairability=None,
            started_at=effective_outcome.started_at,
            finished_at=effective_outcome.finished_at,
        )
        session.add(attempt)
        self._flush_session(session)
        candidate.latest_evaluation_attempt_id = attempt.id
        candidate.evaluation_status = "passed"
        candidate.archive_status = "not_considered"
        candidate.lifecycle_status = "active"
        candidate.failure_stage = None
        candidate.failure_kind = None
        candidate.failure_summary = None
        candidate.failure_evidence_id = None
        candidate.repair_state = "repaired" if candidate.job_kind == "repair" else "audit_only"
        candidate.commit_card_id = card.id
        candidate.evaluated_at = effective_outcome.finished_at or _utc_now()
        log.info(
            "EvaluationAttempt recorded outcome_kind=passed evaluator={}",
            effective_outcome.evaluator_name or "unknown",
        )
        if job_ctx.repair_source_candidate_id is not None:
            source = session.get(CandidateCommit, job_ctx.repair_source_candidate_id)
            if source is not None:
                source.repair_state = "repaired"

    @staticmethod
    def _candidate_for_commit(*, session: Any, commit_hash: str) -> CandidateCommit | None:
        row = session.execute(
            select(CandidateCommit).where(CandidateCommit.commit_hash == commit_hash)
        ).scalar_one_or_none()
        return row if isinstance(row, CandidateCommit) else None

    def persist_failure(
        self,
        *,
        job_ctx: JobContext,
        message: str,
        outcome: EvaluationOutcome,
        plan: PlanningAgentResponse | None = None,
        coding: CodingAgentResponse | None = None,
        worktree: Path | None = None,
        candidate_commit_hash: str | None = None,
    ) -> bool:
        """Persist a structured failed evaluator outcome for the active job lease."""

        commit_hash = str(candidate_commit_hash or "").strip()
        try:
            with session_scope() as session:
                try:
                    job = self._lock_active_job_for_run(
                        session=session,
                        job_id=job_ctx.job_id,
                        run_token=job_ctx.run_token,
                        action="persisting failure",
                    )
                except JobLeaseLost:
                    return False
                artifact_result = self._write_failure_artifacts(
                    job_ctx=job_ctx,
                    message=message,
                    outcome=outcome,
                    plan=plan,
                    coding=coding,
                    worktree=worktree,
                    candidate_commit_hash=commit_hash or None,
                )
                candidate = None
                if commit_hash:
                    candidate = self._candidate_for_commit(session=session, commit_hash=commit_hash)
                    if candidate is None and job.candidate_branch_name:
                        candidate = self._upsert_candidate_commit_row(
                            session=session,
                            job=job,
                            commit_hash=commit_hash,
                            branch_name=job.candidate_branch_name,
                            published=bool(job.candidate_published_at),
                            run_token=job_ctx.run_token,
                        )
                capsule_row = None
                if candidate is not None:
                    self._flush_session(session)
                    capsule = build_diagnostic_capsule(
                        outcome=outcome,
                        max_bytes=self.settings.failed_candidate_repair_max_diagnostic_bytes,
                    )
                    capsule_row = DiagnosticCapsule(
                        candidate_commit_id=candidate.id,
                        job_id=job_ctx.job_id,
                        policy_version=capsule.policy_version,
                        policy_passed=capsule.policy_passed,
                        payload=capsule.payload,
                        omitted_reasons=list(capsule.omitted_reasons),
                    )
                    session.add(capsule_row)
                    self._flush_session(session)
                attempt = self._add_failure_attempt(
                    session=session,
                    job_ctx=job_ctx,
                    outcome=outcome,
                    candidate=candidate,
                    capsule=capsule_row,
                )
                if capsule_row is not None:
                    capsule_row.evaluation_attempt_id = attempt.id
                self._merge_fixed_artifacts(
                    session=session,
                    job_id=job_ctx.job_id,
                    fixed=artifact_result.fixed,
                )
                if commit_hash:
                    self._add_failure_artifact_record(
                        session=session,
                        job_id=job_ctx.job_id,
                        commit_hash=commit_hash,
                        outcome=outcome,
                    )
                    self._add_evaluation_artifact_records_for_failure(
                        session=session,
                        job_id=job_ctx.job_id,
                        commit_hash=commit_hash,
                        artifacts=artifact_result.evaluation_artifacts,
                    )
                self._mark_job_row_failed(job=job, message=message)
                if candidate is not None:
                    self._update_failed_candidate(
                        session=session,
                        job=job,
                        job_ctx=job_ctx,
                        candidate=candidate,
                        outcome=outcome,
                        attempt=attempt,
                        capsule=capsule_row,
                    )
                return True
        except SQLAlchemyError as exc:
            log.error("Failed to persist structured failure for job {}: {}", job_ctx.job_id, exc)
        return False

    def _write_failure_artifacts(
        self,
        *,
        job_ctx: JobContext,
        message: str,
        outcome: EvaluationOutcome,
        plan: PlanningAgentResponse | None,
        coding: CodingAgentResponse | None,
        worktree: Path | None,
        candidate_commit_hash: str | None,
    ) -> JobArtifactWriteResult:
        try:
            return _coerce_artifact_write_result(
                write_failure_job_artifacts(
                    FailureJobArtifactWriteRequest(
                        job_id=job_ctx.job_id,
                        run_token=job_ctx.run_token,
                        base_commit_hash=job_ctx.base_commit_hash,
                        candidate_commit_hash=candidate_commit_hash,
                        message=message,
                        outcome=outcome,
                        plan=plan,
                        coding=coding,
                        worktree=worktree,
                        settings=self.settings,
                    )
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort artifact store
            log.warning("Failed to write failure artifacts for job {}: {}", job_ctx.job_id, exc)
            return JobArtifactWriteResult(fixed=FixedJobArtifactPaths())

    def _add_failure_attempt(
        self,
        *,
        session: Any,
        job_ctx: JobContext,
        outcome: EvaluationOutcome,
        candidate: CandidateCommit | None,
        capsule: DiagnosticCapsule | None,
    ) -> EvaluationAttempt:
        failure = outcome.failure
        attempt = EvaluationAttempt(
            candidate_commit_id=candidate.id if candidate is not None else None,
            job_id=job_ctx.job_id,
            evaluator_name=outcome.evaluator_name,
            evaluator_version=outcome.evaluator_version,
            outcome_kind=outcome.outcome_kind,
            failure_kind=failure.failure_kind if failure else None,
            failure_stage=failure.failure_stage if failure else None,
            repairability=failure.repairability if failure else None,
            safe_failure_summary=failure.safe_failure_summary if failure else None,
            diagnostic_capsule_id=capsule.id if capsule is not None else None,
            artifact_policy_version=failure.policy_version if failure else None,
            started_at=outcome.started_at,
            finished_at=outcome.finished_at,
        )
        session.add(attempt)
        self._flush_session(session)
        log.info(
            "EvaluationAttempt recorded outcome_kind={} evaluator={}",
            outcome.outcome_kind,
            outcome.evaluator_name or "unknown",
        )
        return attempt

    @staticmethod
    def _mark_job_row_failed(*, job: EvolutionJob, message: str) -> None:
        job.status = JobStatus.FAILED
        job.completed_at = _utc_now()
        job.heartbeat_at = None
        job.lease_expires_at = None
        job.run_token = None
        job.worker_id = None
        job.last_error = message

    @staticmethod
    def _add_failure_artifact_record(
        *,
        session: Any,
        job_id: UUID,
        commit_hash: str,
        outcome: EvaluationOutcome,
    ) -> None:
        failure = outcome.failure
        summary = failure.safe_failure_summary if failure else outcome.outcome_kind
        session.add(
            EvaluationArtifactRecord(
                job_id=job_id,
                commit_card_id=None,
                commit_hash=commit_hash,
                key="evaluation_failure",
                kind="failure",
                mime_type="text/plain",
                label="Evaluation failure",
                summary=clamp_text(normalize_single_line(summary), 1024),
                visibility="human_only",
                agent_projection="summary",
                storage_path=None,
                size_bytes=None,
                sha256=None,
                diagnostics=[],
                artifact_metadata={"outcome_kind": outcome.outcome_kind},
            )
        )

    @staticmethod
    def _add_evaluation_artifact_records_for_failure(
        *,
        session: Any,
        job_id: UUID,
        commit_hash: str,
        artifacts: tuple[Any, ...],
    ) -> None:
        for artifact in artifacts:
            session.add(
                EvaluationArtifactRecord(
                    job_id=job_id,
                    commit_card_id=None,
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

    def _update_failed_candidate(
        self,
        *,
        session: Any,
        job: EvolutionJob,
        job_ctx: JobContext,
        candidate: CandidateCommit,
        outcome: EvaluationOutcome,
        attempt: EvaluationAttempt,
        capsule: DiagnosticCapsule | None,
    ) -> None:
        failure = outcome.failure
        candidate.latest_evaluation_attempt_id = attempt.id
        candidate.evaluation_status = outcome.outcome_kind
        candidate.evaluated_at = outcome.finished_at or _utc_now()
        candidate.commit_card_id = None
        candidate.archive_status = "not_applicable" if outcome.outcome_kind != "passed" else "not_considered"
        if failure is not None:
            candidate.failure_stage = failure.failure_stage
            candidate.failure_kind = failure.failure_kind
            candidate.failure_summary = failure.safe_failure_summary
            candidate.failure_evidence_id = capsule.id if capsule is not None else None
        else:
            candidate.failure_stage = "unknown"
            candidate.failure_kind = "unknown"
            candidate.failure_summary = outcome.outcome_kind
        candidate.repo_state_aggregate_status = "not_required"
        candidate.repair_state = self._decide_repair_state(
            session=session,
            job=job,
            job_ctx=job_ctx,
            candidate=candidate,
            outcome=outcome,
            capsule=capsule,
        )
        log.info(
            "Repair eligibility decided outcome_kind={} repair_state={} failure_kind={}",
            outcome.outcome_kind,
            candidate.repair_state,
            candidate.failure_kind or "none",
        )
        if job_ctx.repair_source_candidate_id is not None:
            self._update_repair_source_after_failed_attempt(
                session=session,
                source_id=job_ctx.repair_source_candidate_id,
            )

    def _decide_repair_state(
        self,
        *,
        session: Any,
        job: EvolutionJob,
        job_ctx: JobContext,
        candidate: CandidateCommit,
        outcome: EvaluationOutcome,
        capsule: DiagnosticCapsule | None,
    ) -> str:
        failure = outcome.failure
        if not candidate.commit_hash:
            return "audit_only"
        if _job_kind_from_job(job) == "repair" or job_ctx.repair_source_candidate_id is not None:
            return "ineligible"
        if outcome.outcome_kind != "candidate_failed" or failure is None:
            return "audit_only"
        if candidate.publication_status != "published":
            return "audit_only"
        if failure.failure_stage != "evaluation":
            return "ineligible"
        allowlist = repair_failure_kind_allowlist(self.settings.failed_candidate_repair_failure_kinds)
        if failure.failure_kind not in allowlist:
            return "ineligible"
        if failure.repairability != "repairable":
            return "ineligible"
        if capsule is None or not capsule.policy_passed:
            return "ineligible"
        if not candidate.nearest_viable_ancestor_hash:
            return "ineligible"
        if not self._ancestor_aggregate_ready(
            session=session,
            commit_hash=candidate.nearest_viable_ancestor_hash,
        ):
            return "ineligible"
        if int(candidate.failed_depth or 0) > max(0, int(self.settings.failed_candidate_repair_max_depth)):
            return "ineligible"
        if int(candidate.repair_attempts or 0) >= max(0, int(self.settings.failed_candidate_repair_max_attempts)):
            return "exhausted"
        if candidate.lifecycle_status != "active":
            return "quarantined"
        return "eligible"

    @staticmethod
    def _ancestor_aggregate_ready(*, session: Any, commit_hash: str) -> bool:
        return (
            session.execute(
                select(MapElitesRepoStateAggregate.commit_hash)
                .where(MapElitesRepoStateAggregate.commit_hash == commit_hash)
                .limit(1)
            ).first()
            is not None
        )

    def _update_repair_source_after_failed_attempt(
        self,
        *,
        session: Any,
        source_id: UUID,
    ) -> None:
        source = session.get(CandidateCommit, source_id)
        if source is None:
            return
        max_attempts = max(0, int(self.settings.failed_candidate_repair_max_attempts))
        source.repair_state = "exhausted" if source.repair_attempts >= max_attempts else "eligible"

    def _update_repair_source_after_terminal_failure(
        self,
        *,
        session: Any,
        job: EvolutionJob,
    ) -> None:
        if _job_kind_from_job(job) != "repair":
            return
        source_id = getattr(job, "repair_source_candidate_id", None)
        if source_id is None:
            return
        self._update_repair_source_after_failed_attempt(session=session, source_id=source_id)

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
                self._mark_job_row_failed(job=job, message=message)
                self._update_repair_source_after_terminal_failure(session=session, job=job)
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


def _job_kind_from_job(job: EvolutionJob) -> str:
    raw = str(getattr(job, "job_kind", "") or "").strip().lower()
    if bool(getattr(job, "is_seed_job", False)) and raw in {"", "evolution"}:
        return "seed"
    if raw:
        return raw
    return "evolution"


def _nearest_viable_ancestor_for_job(job: EvolutionJob) -> str | None:
    value = str(getattr(job, "base_commit_hash", "") or "").strip()
    return value or None


def _candidate_failed_depth_for_job(*, session: Any, job: EvolutionJob) -> int:
    source_id = getattr(job, "repair_source_candidate_id", None)
    if source_id is None:
        return 0
    source = session.get(CandidateCommit, source_id)
    if source is None:
        return 1
    return int(getattr(source, "failed_depth", 0) or 0) + 1


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


def _merge_success_outcome_artifacts(
    *,
    evaluation: EvaluationResult,
    outcome: EvaluationOutcome | None,
) -> EvaluationResult:
    if outcome is None or outcome.outcome_kind != "passed":
        return evaluation
    outcome_artifacts = tuple(outcome.artifacts or ())
    outcome_warnings = tuple(outcome.artifact_validation_warnings or ())
    if not outcome_artifacts and not outcome_warnings:
        return evaluation
    return EvaluationResult(
        summary=evaluation.summary,
        metrics=evaluation.metrics,
        tests_executed=evaluation.tests_executed,
        logs=evaluation.logs,
        extra=dict(evaluation.extra or {}),
        artifacts=(*evaluation.artifacts, *outcome_artifacts),
        artifact_validation_warnings=(
            *evaluation.artifact_validation_warnings,
            *outcome_warnings,
        ),
    )


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
