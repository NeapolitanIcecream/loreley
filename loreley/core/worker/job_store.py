from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError

from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.evolution_events import (
    EVALUATION_INVOCATION_STARTED,
    EvolutionStageHandle,
    JOB_FAILED,
    JOB_RUN_STARTED,
    JOB_SUCCEEDED,
    finish_evolution_stage,
    next_event_ordinal,
    record_evolution_event,
    start_evolution_stage,
)
from loreley.core.usage import persist_usage_events
from loreley.core.worker.artifacts import (
    FailureJobArtifactWriteRequest,
    FixedJobArtifactPaths,
    JobArtifactWriteRequest,
    JobArtifactWriteResult,
    resolve_worker_instance_id,
    write_failure_job_artifacts,
    write_job_artifacts,
)
from loreley.core.worker.candidate_identity import (
    evaluation_identity_key,
    normalize_candidate_identity,
)
from loreley.core.worker.commit_card import build_commit_card_from_git
from loreley.config import Settings, get_settings
from loreley.core.worker.coding import CodingAgentResponse
from loreley.core.campaign_program import campaign_program_artifact_payload
from loreley.core.worker.evaluator import (
    EvaluationMetric,
    EvaluationOutcome,
    EvaluationResult,
)
from loreley.core.worker.evaluation_runtime import measurement_payload_sha256
from loreley.core.worker.planning import PlanningAgentResponse
from loreley.core.worker.repair import (
    build_diagnostic_capsule,
    repair_failure_kind_allowlist,
)
from loreley.db.base import session_scope
from loreley.db.models import (
    CandidateCommit,
    CommitCard,
    DiagnosticCapsule,
    EvaluationAttempt,
    EvaluationArtifactRecord,
    EvaluationMeasurement as EvaluationMeasurementRow,
    EvolutionJob,
    JobArtifacts,
    JobStatus,
    MapElitesRepoStateAggregate,
    Metric,
)

if TYPE_CHECKING:
    from loreley.core.worker.evolution import JobContext

log = logger.bind(module="worker.job_store")

_WORKER_ID_MAX_CHARS = int(
    getattr(EvolutionJob.__table__.c.worker_id.type, "length", 128) or 128
)
_WORKER_ID_HASH_CHARS = 12
_FAILURE_ARTIFACT_KEY = "evaluation_failure"


def _unique_evaluation_artifact_record_key(base_key: str, used_keys: set[str]) -> str:
    base = (base_key or "artifact").strip() or "artifact"
    for index in range(1, 1000):
        suffix = "_artifact" if index == 1 else f"_artifact_{index}"
        stem = base[: max(1, 128 - len(suffix))]
        candidate = f"{stem}{suffix}"
        if candidate not in used_keys:
            return candidate
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{digest}"
    return f"{base[: max(1, 128 - len(suffix))]}{suffix}"


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
    execution_mode: str
    input_candidate_commit_hash: str | None
    input_candidate_summary: str | None
    external_submission_key: str
    input_provenance: dict[str, Any]
    archive_ingestion_enabled: bool
    repair_source_candidate_id: UUID | None
    repair_mode: str | None
    campaign_program_hash: str | None
    sampling_strategy: str | None
    sampling_initial_radius: int | None
    sampling_radius_used: int | None
    sampling_fallback_inspirations: int | None
    sampling_ordinal: int | None
    sampling_recipe_hash: str | None
    sampling_recipe_reused: bool
    seed_portfolio_hash: str | None
    seed_direction_id: str | None
    seed_direction_payload: dict[str, Any]
    seed_admission_lane: str | None
    seed_admission_reason: str | None


@dataclass(slots=True, frozen=True)
class CandidateCommitRecord:
    """Candidate publication metadata persisted before or after a push."""

    job_id: UUID
    commit_hash: str
    branch_name: str
    run_token: UUID | None = None
    published: bool = False
    source_tree_hash: str | None = None


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
    plan: PlanningAgentResponse | None
    coding: CodingAgentResponse | None
    evaluation: EvaluationResult
    evaluation_outcome: EvaluationOutcome | None
    worktree: Path
    commit_hash: str
    commit_message: str


@dataclass(slots=True)
class _PersistFailureInput:
    job_ctx: "JobContext"
    message: str
    outcome: EvaluationOutcome
    plan: PlanningAgentResponse | None = None
    coding: CodingAgentResponse | None = None
    worktree: Path | None = None
    candidate_commit_hash: str | None = None


@dataclass(slots=True, frozen=True)
class _CandidateCommitUpsertInput:
    session: Any
    job: EvolutionJob
    commit_hash: str
    branch_name: str | None
    published: bool
    run_token: UUID | None
    source_tree_hash: str | None = None

    @property
    def job_kind(self) -> str:
        return _job_kind_from_job(self.job)

    @property
    def publication_status(self) -> str:
        if (
            str(getattr(self.job, "execution_mode", "agent") or "agent")
            == "evaluate_existing"
        ):
            return "available"
        return "published" if self.published else "created"

    @property
    def published_at(self) -> datetime | None:
        return _utc_now() if self.published else None


@dataclass(slots=True, frozen=True)
class _SuccessEvaluationInput:
    session: Any
    job: EvolutionJob
    job_ctx: "JobContext"
    card: CommitCard
    commit_hash: str
    evaluation: EvaluationResult
    outcome: EvaluationOutcome | None
    artifacts: tuple[Any, ...]


@dataclass(slots=True, frozen=True)
class _FreshMeasurementInput:
    session: Any
    job_ctx: "JobContext"
    commit_hash: str
    outcome: EvaluationOutcome
    attempt: EvaluationAttempt
    candidate_identity: str | None
    identity_key: str | None
    artifacts: tuple[Any, ...]

    def can_accept(self, payload: dict[str, Any]) -> bool:
        outcome = self.outcome
        return all(
            (
                outcome.measurement_executed,
                payload.get("cacheable") is True,
                bool(outcome.measurement_cache_key),
                bool(outcome.measurement_contract_fingerprint),
                bool(self.candidate_identity),
                bool(self.identity_key),
                bool(outcome.evaluator_name),
                bool(outcome.evaluator_version),
                bool(self.job_ctx.campaign_program_hash),
            )
        )


@dataclass(slots=True, frozen=True)
class _FailedCandidateUpdateInput:
    session: Any
    job: EvolutionJob
    job_ctx: "JobContext"
    candidate: CandidateCommit
    outcome: EvaluationOutcome
    attempt: EvaluationAttempt
    capsule: DiagnosticCapsule | None


@dataclass(slots=True, frozen=True)
class _FailureAttemptInput:
    session: Any
    job_ctx: "JobContext"
    outcome: EvaluationOutcome
    candidate: CandidateCommit | None
    capsule: DiagnosticCapsule | None
    candidate_identity: str | None
    identity_key: str | None

    @property
    def failure(self) -> Any:
        return self.outcome.failure


@dataclass(slots=True, frozen=True)
class _RepairStateDecision:
    session: Any
    job: EvolutionJob
    job_ctx: "JobContext"
    candidate: CandidateCommit
    outcome: EvaluationOutcome
    capsule: DiagnosticCapsule | None


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
                job = self._lock_startable_job(session=session, job_id=job_id)
                now = _db_utc_now(session)
                run_token = uuid4()
                worker_id = _bounded_worker_instance_id(resolve_worker_instance_id())
                job_kind = _job_kind_from_job(job)
                self._mark_job_row_running(
                    job, now=now, run_token=run_token, worker_id=worker_id
                )
                record_evolution_event(
                    session,
                    event_type=JOB_RUN_STARTED,
                    job_id=job.id,
                    run_token=run_token,
                    island_id=getattr(job, "island_id", None),
                    occurred_at=now,
                    payload={
                        "job_kind": job_kind,
                        "recovery_count": int(
                            getattr(job, "recovery_count", 0) or 0
                        ),
                    },
                    key_parts=("worker_run",),
                )
                self._mark_repair_source_running(
                    session=session, job=job, job_kind=job_kind
                )
                return _locked_job_from_row(
                    job=job,
                    run_token=run_token,
                    worker_id=worker_id,
                    job_kind=job_kind,
                )
        except SQLAlchemyError as exc:
            if self._is_lock_conflict(exc):
                raise JobLockConflict(
                    f"Evolution job {job_id} is locked by another worker."
                ) from exc
            raise EvolutionWorkerError(f"Failed to start job {job_id}: {exc}") from exc

    def _lock_startable_job(self, *, session: Any, job_id: UUID) -> EvolutionJob:
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
        if job.status not in {JobStatus.PENDING, JobStatus.QUEUED}:
            raise JobPreconditionError(
                f"Evolution job {job_id} is {job.status} and cannot run.",
            )
        return job

    def _mark_job_row_running(
        self,
        job: EvolutionJob,
        *,
        now: datetime,
        run_token: UUID,
        worker_id: str,
    ) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = now
        job.completed_at = None
        job.heartbeat_at = now
        job.lease_expires_at = now + self._lease_ttl()
        job.run_token = run_token
        job.worker_id = worker_id
        job.last_error = None
        job.failure_stage = None
        job.failure_kind = None
        job.candidate_commit_hash = None
        job.candidate_branch_name = None
        job.candidate_published_at = None

    @staticmethod
    def _mark_repair_source_running(
        *,
        session: Any,
        job: EvolutionJob,
        job_kind: str,
    ) -> None:
        source_id = getattr(job, "repair_source_candidate_id", None)
        if job_kind != "repair" or source_id is None:
            return
        source = session.get(CandidateCommit, source_id)
        if source is not None:
            source.repair_state = "repairing"

    def start_stage(
        self,
        *,
        stage: str,
        job_ctx: "JobContext",
        ordinal: int | None = None,
        commit_hash: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> EvolutionStageHandle:
        """Durably record a stage start before the worker enters it."""

        with session_scope() as session:
            return start_evolution_stage(
                session,
                stage=stage,
                job_id=job_ctx.job_id,
                run_token=job_ctx.run_token,
                island_id=job_ctx.island_id,
                commit_hash=commit_hash,
                ordinal=ordinal,
                payload=payload,
            )

    def finish_stage(
        self,
        *,
        handle: EvolutionStageHandle,
        outcome: str,
        failure_kind: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Durably close a planning, coding, or ingestion stage."""

        with session_scope() as session:
            finish_evolution_stage(
                session,
                handle=handle,
                outcome=outcome,
                failure_kind=failure_kind,
                payload=payload,
            )

    def record_candidate_commit(
        self,
        record: CandidateCommitRecord,
    ) -> None:
        """Persist candidate commit metadata before or after remote publication."""

        candidate_hash = str(record.commit_hash or "").strip()
        if not candidate_hash:
            raise EvolutionWorkerError("Candidate commit hash must be provided.")
        candidate_branch = str(record.branch_name or "").strip()

        try:
            with session_scope() as session:
                job = self._job_for_candidate_record(session=session, record=record)
                _validate_candidate_record(
                    job=job,
                    record=record,
                    candidate_branch=candidate_branch,
                )
                _apply_candidate_record_to_job(
                    job=job,
                    record=record,
                    candidate_hash=candidate_hash,
                    candidate_branch=candidate_branch,
                )
                self._upsert_candidate_commit_row(
                    session=session,
                    job=job,
                    commit_hash=candidate_hash,
                    branch_name=candidate_branch or None,
                    published=record.published,
                    run_token=record.run_token,
                    source_tree_hash=str(record.source_tree_hash or "").strip() or None,
                )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(
                f"Failed to record candidate metadata for job {record.job_id}: {exc}",
            ) from exc

    def _job_for_candidate_record(
        self,
        *,
        session: Any,
        record: CandidateCommitRecord,
    ) -> EvolutionJob:
        if record.run_token is not None:
            return self._lock_active_job_for_run(
                session=session,
                job_id=record.job_id,
                run_token=record.run_token,
                action="recording candidate metadata",
            )
        job = session.get(EvolutionJob, record.job_id)
        if job is None:
            raise EvolutionWorkerError(
                f"Evolution job {record.job_id} disappeared while recording candidate metadata."
            )
        return job

    def find_reusable_evaluation(
        self,
        *,
        source_tree_hash: str,
        evaluator_name: str | None,
        evaluator_version: str | None,
        campaign_program_hash: str | None,
        candidate_commit_hash: str,
    ) -> EvaluationOutcome | None:
        """Reuse a passed evaluation for an identical tree and evaluator contract."""

        tree_hash = str(source_tree_hash or "").strip()
        if not tree_hash:
            return None
        with session_scope() as session:
            row = session.execute(
                select(CandidateCommit, CommitCard, EvaluationAttempt)
                .join(CommitCard, CandidateCommit.commit_card_id == CommitCard.id)
                .join(
                    EvaluationAttempt,
                    CandidateCommit.latest_evaluation_attempt_id
                    == EvaluationAttempt.id,
                )
                .where(
                    CandidateCommit.source_tree_hash == tree_hash,
                    CandidateCommit.evaluation_status == "passed",
                    CandidateCommit.campaign_program_hash == campaign_program_hash,
                    EvaluationAttempt.evaluator_name == evaluator_name,
                    EvaluationAttempt.evaluator_version == evaluator_version,
                    EvaluationAttempt.protocol == "one_shot",
                    EvaluationAttempt.outcome_kind == "passed",
                )
                .order_by(
                    CandidateCommit.evaluated_at.desc(), CandidateCommit.id.desc()
                )
                .limit(1)
            ).first()
            if row is None:
                return None
            candidate, card, attempt = row
            metrics = tuple(
                EvaluationMetric(
                    name=metric.name,
                    value=metric.value,
                    unit=metric.unit,
                    higher_is_better=metric.higher_is_better,
                    details=dict(metric.details or {}),
                )
                for metric in session.execute(
                    select(Metric)
                    .where(Metric.commit_card_id == card.id)
                    .order_by(Metric.name.asc())
                ).scalars()
            )
            reused_commit_hash = candidate.commit_hash
            candidate_identity = candidate.candidate_identity
            original_summary = card.evaluation_summary

        now = _utc_now()
        return EvaluationOutcome(
            evaluator_name=attempt.evaluator_name,
            evaluator_version=attempt.evaluator_version,
            candidate_commit_hash=candidate_commit_hash,
            outcome_kind="passed",
            result=EvaluationResult(
                summary=(
                    "Reused a passed evaluation for an identical Git source tree "
                    f"from commit {reused_commit_hash[:12]}."
                ),
                metrics=metrics,
                tests_executed=("exact Git source-tree evaluation reuse",),
                logs=(f"source_tree_hash={tree_hash}",),
                extra={
                    "evaluation_reused": True,
                    "reused_source_commit_hash": reused_commit_hash,
                    "source_tree_hash": tree_hash,
                    "original_evaluation_summary": original_summary,
                },
                candidate_identity=candidate_identity,
            ),
            started_at=now,
            finished_at=now,
            reuse_kind="exact_tree",
            reused_from_attempt_id=(
                str(attempt.id) if getattr(attempt, "id", None) is not None else None
            ),
        )

    def record_evaluation_observation(
        self,
        *,
        job_ctx: "JobContext",
        candidate_commit_hash: str | None,
        outcome: EvaluationOutcome,
    ) -> UUID:
        """Persist every evaluator observation, including intermediate rework attempts."""

        with session_scope() as session:
            self._lock_active_job_for_run(
                session=session,
                job_id=job_ctx.job_id,
                run_token=job_ctx.run_token,
                action="recording evaluation observation",
            )
            commit_hash = str(candidate_commit_hash or "").strip()
            candidate = (
                self._candidate_for_commit(session=session, commit_hash=commit_hash)
                if commit_hash
                else None
            )
            candidate_identity = normalize_candidate_identity(
                outcome.result.candidate_identity
                if outcome.result is not None
                else outcome.prepared_candidate_identity
            )
            identity_key = evaluation_identity_key(
                candidate_identity=candidate_identity,
                evaluator_name=outcome.evaluator_name,
                evaluator_version=outcome.evaluator_version,
                campaign_program_hash=job_ctx.campaign_program_hash,
                measurement_contract_fingerprint=outcome.measurement_contract_fingerprint,
            )
            failure = outcome.failure
            attempt = EvaluationAttempt(
                candidate_commit_id=candidate.id if candidate is not None else None,
                job_id=job_ctx.job_id,
                run_token=job_ctx.run_token,
                attempt_ordinal=self._next_evaluation_attempt_ordinal(
                    session=session, job_id=job_ctx.job_id, outcome=outcome
                ),
                evaluator_name=outcome.evaluator_name,
                evaluator_version=outcome.evaluator_version,
                campaign_program_hash=job_ctx.campaign_program_hash,
                seed_portfolio_hash=job_ctx.seed_portfolio_hash,
                seed_direction_id=job_ctx.seed_direction_id,
                candidate_identity=candidate_identity,
                evaluation_identity_key=identity_key,
                protocol=outcome.protocol,
                measurement_cache_key=outcome.measurement_cache_key,
                measurement_contract_fingerprint=outcome.measurement_contract_fingerprint,
                measurement_id=_optional_uuid(outcome.measurement_id),
                measurement_reused=outcome.measurement_reused,
                measurement_executed=outcome.measurement_executed,
                reuse_kind=outcome.reuse_kind,
                reused_from_attempt_id=_optional_uuid(outcome.reused_from_attempt_id),
                evaluator_slot=outcome.evaluator_slot,
                evaluator_slot_scope=outcome.evaluator_slot_scope,
                evaluator_slot_wait_seconds=outcome.evaluator_slot_wait_seconds,
                evaluator_slot_acquired_at=outcome.evaluator_slot_acquired_at,
                evaluator_slot_released_at=outcome.evaluator_slot_released_at,
                evaluator_slot_lease_id=_optional_uuid(outcome.evaluator_slot_lease_id),
                evaluator_slot_release_reason=outcome.evaluator_slot_release_reason,
                outcome_kind=outcome.outcome_kind,
                failure_kind=failure.failure_kind if failure else None,
                failure_stage=failure.failure_stage if failure else None,
                repairability=failure.repairability if failure else None,
                safe_failure_summary=failure.safe_failure_summary if failure else None,
                artifact_policy_version=failure.policy_version if failure else None,
                started_at=outcome.started_at,
                finished_at=outcome.finished_at,
            )
            session.add(attempt)
            self._flush_session(session)
            if candidate is not None:
                candidate.latest_evaluation_attempt_id = attempt.id
            return attempt.id

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
            raise EvolutionWorkerError(
                f"Failed to renew job lease for {job_id}: {exc}"
            ) from exc
        return lease_expires_at

    def _upsert_candidate_commit_row(
        self,
        request: _CandidateCommitUpsertInput | None = None,
        **kwargs: Any,
    ) -> CandidateCommit:
        request = _candidate_commit_upsert_input(request, kwargs)
        session = request.session
        commit_hash = request.commit_hash
        row = session.execute(
            select(CandidateCommit).where(CandidateCommit.commit_hash == commit_hash)
        ).scalar_one_or_none()
        if row is not None and not isinstance(row, CandidateCommit):
            row = None
        if row is None:
            return self._insert_candidate_commit_row(request)
        return self._update_candidate_commit_row(row, request)

    def _insert_candidate_commit_row(
        self,
        request: _CandidateCommitUpsertInput,
    ) -> CandidateCommit:
        job = request.job
        published_at = request.published_at
        row = CandidateCommit(
            commit_hash=request.commit_hash,
            git_parent_commit_hash=str(
                getattr(job, "base_commit_hash", "") or ""
            ).strip(),
            nearest_viable_ancestor_hash=_nearest_viable_ancestor_for_job(job),
            island_id=getattr(job, "island_id", None),
            produced_by_job_id=getattr(job, "id", None),
            run_token=request.run_token or getattr(job, "run_token", None),
            job_kind=request.job_kind,
            repair_source_candidate_id=getattr(job, "repair_source_candidate_id", None),
            repair_mode=getattr(job, "repair_mode", None),
            campaign_program_hash=getattr(job, "campaign_program_hash", None),
            seed_portfolio_hash=getattr(job, "seed_portfolio_hash", None),
            seed_direction_id=getattr(job, "seed_direction_id", None),
            seed_admission_lane=getattr(job, "seed_admission_lane", None),
            seed_admission_reason=getattr(job, "seed_admission_reason", None),
            source_tree_hash=request.source_tree_hash,
            candidate_branch_name=request.branch_name,
            candidate_published_at=published_at,
            publication_status=request.publication_status,
            evaluation_status="not_evaluated",
            archive_status="not_considered",
            lifecycle_status="active",
            repair_state="audit_only",
            failed_depth=_candidate_failed_depth_for_job(
                session=request.session, job=job
            ),
            repair_attempts=0,
            repo_state_aggregate_status="not_required",
            published_at=published_at,
        )
        request.session.add(row)
        log.info(
            "CandidateCommit recorded job_kind={} publication_status={}",
            row.job_kind,
            row.publication_status,
        )
        return row

    @staticmethod
    def _update_candidate_commit_row(
        row: CandidateCommit,
        request: _CandidateCommitUpsertInput,
    ) -> CandidateCommit:
        job = request.job
        row.git_parent_commit_hash = row.git_parent_commit_hash or _base_commit_hash(
            job
        )
        row.nearest_viable_ancestor_hash = (
            row.nearest_viable_ancestor_hash or _nearest_viable_ancestor_for_job(job)
        )
        row.island_id = row.island_id or getattr(job, "island_id", None)
        row.produced_by_job_id = row.produced_by_job_id or getattr(job, "id", None)
        row.run_token = (
            row.run_token or request.run_token or getattr(job, "run_token", None)
        )
        row.job_kind = row.job_kind or request.job_kind
        row.repair_source_candidate_id = row.repair_source_candidate_id or getattr(
            job, "repair_source_candidate_id", None
        )
        row.repair_mode = row.repair_mode or getattr(job, "repair_mode", None)
        row.campaign_program_hash = row.campaign_program_hash or getattr(
            job, "campaign_program_hash", None
        )
        row.seed_portfolio_hash = getattr(row, "seed_portfolio_hash", None) or getattr(
            job, "seed_portfolio_hash", None
        )
        row.seed_direction_id = getattr(row, "seed_direction_id", None) or getattr(
            job, "seed_direction_id", None
        )
        row.seed_admission_lane = getattr(row, "seed_admission_lane", None) or getattr(
            job, "seed_admission_lane", None
        )
        row.seed_admission_reason = getattr(row, "seed_admission_reason", None) or getattr(
            job, "seed_admission_reason", None
        )
        row.source_tree_hash = row.source_tree_hash or request.source_tree_hash
        row.candidate_branch_name = request.branch_name
        row.publication_status = request.publication_status
        if request.published:
            published_at = request.published_at
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
        request: _PersistSuccessInput | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist successful worker execution artifacts.

        Hot-path data (CommitCard + job indices) is written to the DB.
        Cold-path evidence (prompts/raw/logs) is written to disk and referenced
        via the JobArtifacts table.
        """

        request = _persist_success_input(request, kwargs)
        try:
            with session_scope() as session:
                job = self._lock_active_job_for_run(
                    session=session,
                    job_id=request.job_ctx.job_id,
                    run_token=request.job_ctx.run_token,
                    action="persisting success",
                )
                payload = self._build_success_payload(request)
                completed_at = _utc_now()
                self._mark_job_row_succeeded(
                    job,
                    plan=request.plan,
                    commit_hash=request.commit_hash,
                    completed_at=completed_at,
                )
                record_evolution_event(
                    session,
                    event_type=JOB_SUCCEEDED,
                    job_id=request.job_ctx.job_id,
                    run_token=request.job_ctx.run_token,
                    island_id=request.job_ctx.island_id,
                    commit_hash=request.commit_hash,
                    occurred_at=completed_at,
                    payload={"outcome": "succeeded"},
                    key_parts=("terminal",),
                )
                card = self._add_commit_card(
                    session=session,
                    job_ctx=request.job_ctx,
                    commit_hash=request.commit_hash,
                    payload=payload,
                )
                self._add_metric_rows(
                    session=session, card=card, evaluation=request.evaluation
                )
                self._flush_session(session)
                self._merge_fixed_artifacts(
                    session=session,
                    job_id=request.job_ctx.job_id,
                    fixed=payload.artifact_result.fixed,
                )
                attempt = self._record_success_evaluation(
                    _SuccessEvaluationInput(
                        session=session,
                        job=job,
                        job_ctx=request.job_ctx,
                        card=card,
                        commit_hash=request.commit_hash,
                        evaluation=request.evaluation,
                        outcome=request.evaluation_outcome,
                        artifacts=payload.artifact_result.evaluation_artifacts,
                    )
                )
                attempt.artifact_paths = dict(payload.artifact_result.fixed.as_dict())
                self._add_evaluation_artifact_records(
                    session=session,
                    job_id=request.job_ctx.job_id,
                    commit_hash=request.commit_hash,
                    card=card,
                    artifacts=payload.artifact_result.evaluation_artifacts,
                    evaluation_attempt_id=attempt.id,
                )
                self._persist_agent_usage(
                    session=session,
                    job_ctx=request.job_ctx,
                    plan=request.plan,
                    coding=request.coding,
                )
        except SQLAlchemyError as exc:
            raise EvolutionWorkerError(
                f"Failed to persist results for job {request.job_ctx.job_id}: {exc}"
            ) from exc

    def _build_success_payload(
        self, request: _PersistSuccessInput
    ) -> _PersistSuccessPayload:
        subject = _success_subject(request.job_ctx.job_id, request.commit_message)
        build = build_commit_card_from_git(
            worktree=Path(request.worktree),
            base_commit=request.job_ctx.base_commit_hash,
            candidate_commit=request.commit_hash,
        )
        return _PersistSuccessPayload(
            subject=subject,
            change_summary=_change_summary(request),
            eval_summary=clamp_text(
                normalize_single_line(request.evaluation.summary), 512
            )
            or None,
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
                        campaign_program=campaign_program_artifact_payload(
                            request.job_ctx.campaign_program
                        ),
                    )
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort artifact store
            log.warning(
                "Failed to write artifacts for job {}: {}", request.job_ctx.job_id, exc
            )
        return JobArtifactWriteResult(fixed=FixedJobArtifactPaths())

    @staticmethod
    def _mark_job_row_succeeded(
        job: EvolutionJob,
        *,
        plan: PlanningAgentResponse | None,
        commit_hash: str,
        completed_at: datetime,
    ) -> None:
        job.status = JobStatus.SUCCEEDED
        job.completed_at = completed_at
        job.heartbeat_at = None
        job.lease_expires_at = None
        job.run_token = None
        job.worker_id = None
        job.plan_summary = (
            plan.plan.summary
            if plan is not None
            else str(getattr(job, "input_candidate_summary", "") or "").strip() or None
        )
        job.candidate_commit_hash = job.candidate_commit_hash or commit_hash
        job.result_commit_hash = commit_hash
        job.last_error = None
        job.failure_stage = None
        job.failure_kind = None
        archive_enabled = bool(getattr(job, "archive_ingestion_enabled", True))
        job.ingestion_status = None if archive_enabled else "skipped"
        job.ingestion_attempts = 0
        job.ingestion_delta = None
        job.ingestion_status_code = None
        job.ingestion_message = None
        job.ingestion_cell_index = None
        job.ingestion_last_attempt_at = None
        job.ingestion_reason = (
            None
            if archive_enabled
            else "Archive ingestion disabled by the persisted job contract."
        )

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
            seed_portfolio_hash=job_ctx.seed_portfolio_hash,
            seed_direction_id=job_ctx.seed_direction_id,
            seed_admission_lane=job_ctx.seed_admission_lane,
            seed_admission_reason=job_ctx.seed_admission_reason,
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

    def _persist_agent_usage(
        self,
        *,
        session: Any,
        job_ctx: "JobContext",
        plan: PlanningAgentResponse | None,
        coding: CodingAgentResponse | None,
    ) -> None:
        events = []
        if plan is not None:
            events.extend(plan.usage_events or ())
        if coding is not None:
            events.extend(coding.usage_events or ())
        if not events:
            return
        inserted = persist_usage_events(
            [
                event.with_context(job_id=job_ctx.job_id, run_token=job_ctx.run_token)
                for event in events
            ],
            session=session,
            settings=self.settings,
        )
        if inserted:
            log.info(
                "Persisted {} agent LLM usage event(s) for job {}",
                inserted,
                job_ctx.job_id,
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
        evaluation_attempt_id: UUID | None = None,
    ) -> None:
        records = [
            EvolutionJobStore._evaluation_artifact_record(
                job_id=job_id,
                commit_card_id=card.id,
                commit_hash=commit_hash,
                artifact=artifact,
                evaluation_attempt_id=evaluation_attempt_id,
            )
            for artifact in artifacts
        ]
        EvolutionJobStore._replace_evaluation_artifact_records(
            session=session, records=records
        )

    @staticmethod
    def _evaluation_artifact_record(
        *,
        job_id: UUID,
        commit_card_id: UUID | None,
        commit_hash: str,
        artifact: Any,
        evaluation_attempt_id: UUID | None = None,
    ) -> EvaluationArtifactRecord:
        return EvaluationArtifactRecord(
            job_id=job_id,
            commit_card_id=commit_card_id,
            evaluation_attempt_id=evaluation_attempt_id,
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

    @staticmethod
    def _replace_evaluation_artifact_records(
        *,
        session: Any,
        records: Sequence[EvaluationArtifactRecord],
    ) -> None:
        keys = tuple(
            dict.fromkeys(
                (record.job_id, record.key)
                for record in records
                if record.evaluation_attempt_id is None
            )
        )
        for job_id, key in keys:
            # Legacy records without an attempt keep their historical latest-per-job
            # projection. Attempt-linked records are append-only and are never deleted.
            session.execute(
                delete(EvaluationArtifactRecord)
                .where(
                    EvaluationArtifactRecord.job_id == job_id,
                    EvaluationArtifactRecord.key == key,
                    EvaluationArtifactRecord.evaluation_attempt_id.is_(None),
                )
                .execution_options(synchronize_session=False)
            )
        for record in records:
            session.add(record)

    @staticmethod
    def _next_evaluation_attempt_ordinal(
        *,
        session: Any,
        job_id: UUID,
        outcome: EvaluationOutcome | None = None,
    ) -> int:
        invocation_ordinal = int(
            getattr(outcome, "invocation_ordinal", 0) or 0
        )
        if invocation_ordinal > 0:
            return invocation_ordinal
        result = session.execute(
            select(
                func.coalesce(func.max(EvaluationAttempt.attempt_ordinal), 0) + 1
            ).where(EvaluationAttempt.job_id == job_id)
        )
        scalar_one = getattr(result, "scalar_one", None)
        if not callable(scalar_one):  # pragma: no cover - simplified unit-test sessions
            return 1
        attempt_ordinal = int(scalar_one())
        invocation_ordinal = next_event_ordinal(
            session,
            event_type=EVALUATION_INVOCATION_STARTED,
            job_id=job_id,
        )
        return max(attempt_ordinal, invocation_ordinal)

    def _record_success_evaluation(
        self,
        request: _SuccessEvaluationInput,
    ) -> EvaluationAttempt:
        candidate = self._candidate_for_commit(
            session=request.session,
            commit_hash=request.commit_hash,
        )
        if candidate is None:
            candidate = self._upsert_candidate_commit_row(
                session=request.session,
                job=request.job,
                commit_hash=request.commit_hash,
                branch_name=str(request.job.candidate_branch_name or ""),
                published=bool(request.job.candidate_published_at),
                run_token=request.job_ctx.run_token,
            )
        self._flush_session(request.session)
        effective_outcome = request.outcome or EvaluationOutcome(
            evaluator_name=None,
            candidate_commit_hash=request.commit_hash,
            outcome_kind="passed",
            result=request.evaluation,
            started_at=_utc_now(),
            finished_at=_utc_now(),
        )
        candidate_identity = normalize_candidate_identity(
            request.evaluation.candidate_identity
        )
        identity_key = evaluation_identity_key(
            candidate_identity=candidate_identity,
            evaluator_name=effective_outcome.evaluator_name,
            evaluator_version=effective_outcome.evaluator_version,
            campaign_program_hash=request.job_ctx.campaign_program_hash,
            measurement_contract_fingerprint=(
                effective_outcome.measurement_contract_fingerprint
            ),
        )
        attempt = self._existing_observation_attempt(
            session=request.session,
            outcome=effective_outcome,
        )
        if attempt is None:
            attempt = EvaluationAttempt(
                candidate_commit_id=candidate.id,
                job_id=request.job_ctx.job_id,
                run_token=request.job_ctx.run_token,
                attempt_ordinal=self._next_evaluation_attempt_ordinal(
                    session=request.session,
                    job_id=request.job_ctx.job_id,
                    outcome=effective_outcome,
                ),
                evaluator_name=effective_outcome.evaluator_name,
                evaluator_version=effective_outcome.evaluator_version,
                campaign_program_hash=request.job_ctx.campaign_program_hash,
                seed_portfolio_hash=request.job_ctx.seed_portfolio_hash,
                seed_direction_id=request.job_ctx.seed_direction_id,
                candidate_identity=candidate_identity,
                evaluation_identity_key=identity_key,
                protocol=effective_outcome.protocol,
                measurement_cache_key=effective_outcome.measurement_cache_key,
                measurement_contract_fingerprint=(
                    effective_outcome.measurement_contract_fingerprint
                ),
                measurement_id=_optional_uuid(effective_outcome.measurement_id),
                measurement_reused=effective_outcome.measurement_reused,
                measurement_executed=effective_outcome.measurement_executed,
                reuse_kind=effective_outcome.reuse_kind,
                reused_from_attempt_id=_optional_uuid(
                    effective_outcome.reused_from_attempt_id
                ),
                evaluator_slot=effective_outcome.evaluator_slot,
                evaluator_slot_scope=effective_outcome.evaluator_slot_scope,
                evaluator_slot_wait_seconds=effective_outcome.evaluator_slot_wait_seconds,
                evaluator_slot_acquired_at=effective_outcome.evaluator_slot_acquired_at,
                evaluator_slot_released_at=effective_outcome.evaluator_slot_released_at,
                evaluator_slot_lease_id=_optional_uuid(
                    effective_outcome.evaluator_slot_lease_id
                ),
                evaluator_slot_release_reason=(
                    effective_outcome.evaluator_slot_release_reason
                ),
                outcome_kind="passed",
                repairability=None,
                started_at=effective_outcome.started_at,
                finished_at=effective_outcome.finished_at,
            )
            request.session.add(attempt)
            self._flush_session(request.session)
        else:
            attempt.candidate_commit_id = candidate.id
            attempt.candidate_identity = candidate_identity
            attempt.evaluation_identity_key = identity_key
            attempt.outcome_kind = "passed"
            attempt.finished_at = effective_outcome.finished_at
        self._accept_fresh_measurement(
            _FreshMeasurementInput(
                session=request.session,
                job_ctx=request.job_ctx,
                commit_hash=request.commit_hash,
                outcome=effective_outcome,
                attempt=attempt,
                candidate_identity=candidate_identity,
                identity_key=identity_key,
                artifacts=request.artifacts,
            )
        )
        candidate.latest_evaluation_attempt_id = attempt.id
        candidate.evaluation_status = "passed"
        candidate.candidate_identity = candidate_identity
        candidate.evaluation_identity_key = identity_key
        candidate.archive_status = "not_considered"
        candidate.lifecycle_status = "active"
        candidate.failure_stage = None
        candidate.failure_kind = None
        candidate.failure_summary = None
        candidate.failure_evidence_id = None
        candidate.repair_state = (
            "repaired" if candidate.job_kind == "repair" else "audit_only"
        )
        candidate.commit_card_id = request.card.id
        candidate.evaluated_at = effective_outcome.finished_at or _utc_now()
        log.info(
            "EvaluationAttempt recorded outcome_kind=passed evaluator={}",
            effective_outcome.evaluator_name or "unknown",
        )
        if request.job_ctx.repair_source_candidate_id is not None:
            source = request.session.get(
                CandidateCommit, request.job_ctx.repair_source_candidate_id
            )
            if source is not None:
                source.repair_state = "repaired"
        return attempt

    @staticmethod
    def _existing_observation_attempt(
        *,
        session: Any,
        outcome: EvaluationOutcome,
    ) -> EvaluationAttempt | None:
        attempt_id = _optional_uuid(outcome.persisted_attempt_id)
        if attempt_id is None:
            return None
        row = session.get(EvaluationAttempt, attempt_id)
        return row if isinstance(row, EvaluationAttempt) else None

    def _accept_fresh_measurement(
        self,
        request: _FreshMeasurementInput,
    ) -> None:
        outcome = request.outcome
        if outcome.protocol != "phased-v1":
            return
        if outcome.measurement_reused:
            request.attempt.measurement_id = _optional_uuid(outcome.measurement_id)
            request.attempt.reused_from_attempt_id = _optional_uuid(
                outcome.reused_from_attempt_id
            )
            return
        payload = dict(outcome.measurement_payload or {})
        if not request.can_accept(payload):
            return
        _require_intact_measurement_evidence(
            evidence=outcome.measurement_evidence,
            artifacts=request.artifacts,
        )
        existing = request.session.execute(
            select(EvaluationMeasurementRow).where(
                EvaluationMeasurementRow.cache_key == outcome.measurement_cache_key
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = EvaluationMeasurementRow(
                cache_key=outcome.measurement_cache_key,
                candidate_identity=request.candidate_identity,
                evaluation_identity_key=request.identity_key,
                evaluator_name=outcome.evaluator_name,
                evaluator_version=outcome.evaluator_version,
                campaign_program_hash=request.job_ctx.campaign_program_hash,
                measurement_contract_fingerprint=(
                    outcome.measurement_contract_fingerprint
                ),
                payload=payload,
                payload_sha256=measurement_payload_sha256(payload),
                evidence_manifest=[
                    item.as_dict() for item in outcome.measurement_evidence
                ],
                source_job_id=request.job_ctx.job_id,
                source_candidate_commit_hash=request.commit_hash,
                source_evaluation_attempt_id=request.attempt.id,
            )
            request.session.add(existing)
            self._flush_session(request.session)
        request.attempt.measurement_id = existing.id

    @staticmethod
    def _candidate_for_commit(
        *, session: Any, commit_hash: str
    ) -> CandidateCommit | None:
        row = session.execute(
            select(CandidateCommit).where(CandidateCommit.commit_hash == commit_hash)
        ).scalar_one_or_none()
        return row if isinstance(row, CandidateCommit) else None

    def persist_failure(
        self,
        request: _PersistFailureInput | None = None,
        **kwargs: Any,
    ) -> bool:
        """Persist a structured failed evaluator outcome for the active job lease."""

        request = _persist_failure_input(request, kwargs)
        commit_hash = str(request.candidate_commit_hash or "").strip()
        try:
            with session_scope() as session:
                job = self._lock_job_for_failure(session=session, request=request)
                if job is None:
                    return False
                artifact_result = self._write_failure_artifacts(request)
                candidate = self._candidate_for_failure(
                    session=session,
                    job=job,
                    request=request,
                    commit_hash=commit_hash,
                )
                capsule_row = self._add_diagnostic_capsule_for_failure(
                    session=session,
                    request=request,
                    candidate=candidate,
                )
                attempt = self._add_failure_attempt(
                    session=session,
                    job_ctx=request.job_ctx,
                    outcome=request.outcome,
                    candidate=candidate,
                    capsule=capsule_row,
                )
                attempt.artifact_paths = dict(artifact_result.fixed.as_dict())
                _link_capsule_attempt(capsule_row, attempt)
                self._record_failure_artifacts(
                    session=session,
                    request=request,
                    commit_hash=commit_hash,
                    artifact_result=artifact_result,
                    evaluation_attempt_id=attempt.id,
                )
                self._persist_agent_usage(
                    session=session,
                    job_ctx=request.job_ctx,
                    plan=request.plan,
                    coding=request.coding,
                )
                failure = request.outcome.failure
                completed_at = _utc_now()
                self._mark_job_row_failed(
                    job=job,
                    message=request.message,
                    failure_stage=failure.failure_stage if failure else "evaluation",
                    failure_kind=failure.failure_kind if failure else "unknown",
                    completed_at=completed_at,
                )
                record_evolution_event(
                    session,
                    event_type=JOB_FAILED,
                    job_id=request.job_ctx.job_id,
                    run_token=request.job_ctx.run_token,
                    island_id=request.job_ctx.island_id,
                    commit_hash=commit_hash or None,
                    occurred_at=completed_at,
                    payload={
                        "failure_stage": (
                            failure.failure_stage if failure else "evaluation"
                        ),
                        "failure_kind": (
                            failure.failure_kind if failure else "unknown"
                        ),
                    },
                    key_parts=("terminal",),
                )
                self._update_candidate_after_failure(
                    _FailedCandidateUpdateInput(
                        session=session,
                        job=job,
                        job_ctx=request.job_ctx,
                        candidate=candidate,
                        outcome=request.outcome,
                        attempt=attempt,
                        capsule=capsule_row,
                    )
                    if candidate is not None
                    else None,
                    session=session,
                    job_ctx=request.job_ctx,
                )
                return True
        except SQLAlchemyError as exc:
            log.error(
                "Failed to persist structured failure for job {}: {}",
                request.job_ctx.job_id,
                exc,
            )
        return False

    def _lock_job_for_failure(
        self,
        *,
        session: Any,
        request: _PersistFailureInput,
    ) -> EvolutionJob | None:
        try:
            return self._lock_active_job_for_run(
                session=session,
                job_id=request.job_ctx.job_id,
                run_token=request.job_ctx.run_token,
                action="persisting failure",
            )
        except JobLeaseLost:
            return None

    def _candidate_for_failure(
        self,
        *,
        session: Any,
        job: EvolutionJob,
        request: _PersistFailureInput,
        commit_hash: str,
    ) -> CandidateCommit | None:
        if not commit_hash:
            return None
        candidate = self._candidate_for_commit(session=session, commit_hash=commit_hash)
        if candidate is not None or not job.candidate_branch_name:
            return candidate
        return self._upsert_candidate_commit_row(
            session=session,
            job=job,
            commit_hash=commit_hash,
            branch_name=job.candidate_branch_name,
            published=bool(job.candidate_published_at),
            run_token=request.job_ctx.run_token,
        )

    def _add_diagnostic_capsule_for_failure(
        self,
        *,
        session: Any,
        request: _PersistFailureInput,
        candidate: CandidateCommit | None,
    ) -> DiagnosticCapsule | None:
        if candidate is None:
            return None
        self._flush_session(session)
        capsule = build_diagnostic_capsule(
            outcome=request.outcome,
            max_bytes=self.settings.failed_candidate_repair_max_diagnostic_bytes,
        )
        capsule_row = DiagnosticCapsule(
            candidate_commit_id=candidate.id,
            job_id=request.job_ctx.job_id,
            policy_version=capsule.policy_version,
            policy_passed=capsule.policy_passed,
            payload=capsule.payload,
            omitted_reasons=list(capsule.omitted_reasons),
        )
        session.add(capsule_row)
        self._flush_session(session)
        return capsule_row

    def _record_failure_artifacts(
        self,
        *,
        session: Any,
        request: _PersistFailureInput,
        commit_hash: str,
        artifact_result: JobArtifactWriteResult,
        evaluation_attempt_id: UUID | None = None,
    ) -> None:
        self._merge_fixed_artifacts(
            session=session,
            job_id=request.job_ctx.job_id,
            fixed=artifact_result.fixed,
        )
        if not commit_hash:
            return
        failure_record = self._failure_artifact_record(
            job_id=request.job_ctx.job_id,
            commit_hash=commit_hash,
            outcome=request.outcome,
            evaluation_attempt_id=evaluation_attempt_id,
        )
        materialized_records = self._disambiguate_evaluation_artifact_record_keys(
            self._evaluation_artifact_records_for_failure(
                job_id=request.job_ctx.job_id,
                commit_hash=commit_hash,
                artifacts=artifact_result.evaluation_artifacts,
                evaluation_attempt_id=evaluation_attempt_id,
            ),
            reserved_keys={failure_record.key},
        )
        records = [
            failure_record,
            *materialized_records,
        ]
        self._replace_evaluation_artifact_records(session=session, records=records)

    def _update_candidate_after_failure(
        self,
        update: _FailedCandidateUpdateInput | None,
        *,
        session: Any,
        job_ctx: "JobContext",
    ) -> None:
        if update is not None:
            self._update_failed_candidate(update)
        elif job_ctx.repair_source_candidate_id is not None:
            self._update_repair_source_after_failed_attempt(
                session=session,
                source_id=job_ctx.repair_source_candidate_id,
            )

    def _write_failure_artifacts(
        self,
        request: _PersistFailureInput,
    ) -> JobArtifactWriteResult:
        try:
            commit_hash = str(request.candidate_commit_hash or "").strip() or None
            return _coerce_artifact_write_result(
                write_failure_job_artifacts(
                    FailureJobArtifactWriteRequest(
                        job_id=request.job_ctx.job_id,
                        run_token=request.job_ctx.run_token,
                        base_commit_hash=request.job_ctx.base_commit_hash,
                        candidate_commit_hash=commit_hash,
                        message=request.message,
                        outcome=request.outcome,
                        plan=request.plan,
                        coding=request.coding,
                        worktree=request.worktree,
                        settings=self.settings,
                        campaign_program=campaign_program_artifact_payload(
                            request.job_ctx.campaign_program
                        ),
                    )
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort artifact store
            log.warning(
                "Failed to write failure artifacts for job {}: {}",
                request.job_ctx.job_id,
                exc,
            )
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
        candidate_identity = normalize_candidate_identity(
            outcome.prepared_candidate_identity
        )
        identity_key = evaluation_identity_key(
            candidate_identity=candidate_identity,
            evaluator_name=outcome.evaluator_name,
            evaluator_version=outcome.evaluator_version,
            campaign_program_hash=job_ctx.campaign_program_hash,
            measurement_contract_fingerprint=outcome.measurement_contract_fingerprint,
        )
        request = _FailureAttemptInput(
            session=session,
            job_ctx=job_ctx,
            outcome=outcome,
            candidate=candidate,
            capsule=capsule,
            candidate_identity=candidate_identity,
            identity_key=identity_key,
        )
        attempt = self._existing_observation_attempt(session=session, outcome=outcome)
        if attempt is None:
            attempt = _new_failure_attempt(
                request=request,
                attempt_ordinal=self._next_evaluation_attempt_ordinal(
                    session=session, job_id=job_ctx.job_id, outcome=outcome
                ),
            )
            session.add(attempt)
            self._flush_session(session)
        else:
            _update_failure_attempt(attempt=attempt, request=request)
        log.info(
            "EvaluationAttempt recorded outcome_kind={} evaluator={}",
            outcome.outcome_kind,
            outcome.evaluator_name or "unknown",
        )
        return attempt

    @staticmethod
    def _mark_job_row_failed(
        *,
        job: EvolutionJob,
        message: str,
        failure_stage: str | None = None,
        failure_kind: str | None = None,
        completed_at: datetime,
    ) -> None:
        job.status = JobStatus.FAILED
        job.completed_at = completed_at
        job.heartbeat_at = None
        job.lease_expires_at = None
        job.run_token = None
        job.worker_id = None
        job.last_error = message
        job.failure_stage = (
            clamp_text(normalize_single_line(failure_stage or ""), 32) or None
        )
        job.failure_kind = (
            clamp_text(normalize_single_line(failure_kind or ""), 64) or None
        )

    @staticmethod
    def _failure_artifact_record(
        *,
        job_id: UUID,
        commit_hash: str,
        outcome: EvaluationOutcome,
        evaluation_attempt_id: UUID | None = None,
    ) -> EvaluationArtifactRecord:
        failure = outcome.failure
        summary = failure.safe_failure_summary if failure else outcome.outcome_kind
        return EvaluationArtifactRecord(
            job_id=job_id,
            commit_card_id=None,
            evaluation_attempt_id=evaluation_attempt_id,
            commit_hash=commit_hash,
            key=_FAILURE_ARTIFACT_KEY,
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

    @staticmethod
    def _disambiguate_evaluation_artifact_record_keys(
        records: Sequence[EvaluationArtifactRecord],
        *,
        reserved_keys: set[str],
    ) -> list[EvaluationArtifactRecord]:
        used = set(reserved_keys)
        disambiguated: list[EvaluationArtifactRecord] = []
        for record in records:
            original_key = str(record.key or "").strip()
            if original_key in used:
                record.key = _unique_evaluation_artifact_record_key(original_key, used)
                metadata = dict(record.artifact_metadata or {})
                metadata.setdefault("original_key", original_key)
                metadata.setdefault("key_collision", "failure_artifact")
                record.artifact_metadata = metadata
            used.add(str(record.key))
            disambiguated.append(record)
        return disambiguated

    @staticmethod
    def _evaluation_artifact_records_for_failure(
        *,
        job_id: UUID,
        commit_hash: str,
        artifacts: tuple[Any, ...],
        evaluation_attempt_id: UUID | None = None,
    ) -> list[EvaluationArtifactRecord]:
        return [
            EvolutionJobStore._evaluation_artifact_record(
                job_id=job_id,
                commit_card_id=None,
                commit_hash=commit_hash,
                artifact=artifact,
                evaluation_attempt_id=evaluation_attempt_id,
            )
            for artifact in artifacts
        ]

    def _update_failed_candidate(
        self,
        request: _FailedCandidateUpdateInput,
    ) -> None:
        candidate = request.candidate
        self._apply_failure_status_to_candidate(request)
        candidate.repair_state = self._decide_repair_state(
            _RepairStateDecision(
                session=request.session,
                job=request.job,
                job_ctx=request.job_ctx,
                candidate=candidate,
                outcome=request.outcome,
                capsule=request.capsule,
            )
        )
        log.info(
            "Repair eligibility decided outcome_kind={} repair_state={} failure_kind={}",
            request.outcome.outcome_kind,
            candidate.repair_state,
            candidate.failure_kind or "none",
        )
        if request.job_ctx.repair_source_candidate_id is not None:
            self._update_repair_source_after_failed_attempt(
                session=request.session,
                source_id=request.job_ctx.repair_source_candidate_id,
            )

    @staticmethod
    def _apply_failure_status_to_candidate(
        request: _FailedCandidateUpdateInput,
    ) -> None:
        candidate = request.candidate
        failure = request.outcome.failure
        candidate.latest_evaluation_attempt_id = request.attempt.id
        candidate.evaluation_status = request.outcome.outcome_kind
        candidate.evaluated_at = request.outcome.finished_at or _utc_now()
        candidate.commit_card_id = None
        candidate.archive_status = _failed_candidate_archive_status(request.outcome)
        candidate.repo_state_aggregate_status = "not_required"
        if failure is None:
            candidate.failure_stage = "unknown"
            candidate.failure_kind = "unknown"
            candidate.failure_summary = request.outcome.outcome_kind
            return
        candidate.failure_stage = failure.failure_stage
        candidate.failure_kind = failure.failure_kind
        candidate.failure_summary = failure.safe_failure_summary
        candidate.failure_evidence_id = (
            request.capsule.id if request.capsule is not None else None
        )

    def _decide_repair_state(
        self,
        request: _RepairStateDecision | None = None,
        **kwargs: Any,
    ) -> str:
        request = _repair_state_decision(request, kwargs)
        failure = request.outcome.failure
        candidate = request.candidate
        if not candidate.commit_hash:
            return "audit_only"
        if (
            _job_kind_from_job(request.job) == "repair"
            or request.job_ctx.repair_source_candidate_id is not None
        ):
            return "ineligible"
        if request.outcome.outcome_kind != "candidate_failed" or failure is None:
            return "audit_only"
        blocker = self._repair_eligibility_blocker(request)
        if blocker is not None:
            return blocker
        return self._repair_budget_state(request)

    def _repair_eligibility_blocker(self, request: _RepairStateDecision) -> str | None:
        candidate = request.candidate
        failure = request.outcome.failure
        if candidate.publication_status != "published":
            return "audit_only"
        if failure is None or not _failure_allows_repair(
            failure,
            allowed=repair_failure_kind_allowlist(
                self.settings.failed_candidate_repair_failure_kinds
            ),
        ):
            return "ineligible"
        if request.capsule is None or not request.capsule.policy_passed:
            return "ineligible"
        if not candidate.nearest_viable_ancestor_hash:
            return "ineligible"
        if not self._ancestor_aggregate_ready(
            session=request.session,
            commit_hash=candidate.nearest_viable_ancestor_hash,
        ):
            return "ineligible"
        return None

    def _repair_budget_state(self, request: _RepairStateDecision) -> str:
        candidate = request.candidate
        if (
            candidate.repair_source_candidate_id is not None
            or int(candidate.failed_depth or 0) != 0
        ):
            return "ineligible"
        if int(candidate.repair_attempts or 0) >= max(
            0, int(self.settings.failed_candidate_repair_max_attempts)
        ):
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
        source.repair_state = (
            "exhausted" if source.repair_attempts >= max_attempts else "eligible"
        )

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
        self._update_repair_source_after_failed_attempt(
            session=session, source_id=source_id
        )

    def mark_job_failed(
        self,
        job_id: UUID,
        message: str,
        *,
        run_token: UUID | None = None,
        failure_stage: str | None = None,
        failure_kind: str | None = None,
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
                    if job.status in {
                        JobStatus.SUCCEEDED,
                        JobStatus.FAILED,
                        JobStatus.CANCELLED,
                    }:
                        return False
                effective_run_token = run_token or getattr(job, "run_token", None)
                completed_at = _utc_now()
                self._mark_job_row_failed(
                    job=job,
                    message=message,
                    failure_stage=failure_stage,
                    failure_kind=failure_kind,
                    completed_at=completed_at,
                )
                record_evolution_event(
                    session,
                    event_type=JOB_FAILED,
                    job_id=job_id,
                    run_token=effective_run_token,
                    island_id=getattr(job, "island_id", None),
                    commit_hash=(
                        str(getattr(job, "candidate_commit_hash", "") or "").strip()
                        or None
                    ),
                    occurred_at=completed_at,
                    payload={
                        "failure_stage": failure_stage or "unknown",
                        "failure_kind": failure_kind or "unknown",
                    },
                    key_parts=("terminal",),
                )
                self._update_repair_source_after_terminal_failure(
                    session=session, job=job
                )
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
        raise RuntimeError(
            f"Database current timestamp returned unsupported value: {value!r}"
        )
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


def _locked_job_from_row(
    *,
    job: EvolutionJob,
    run_token: UUID,
    worker_id: str,
    job_kind: str,
) -> LockedJob:
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
        execution_mode=str(getattr(job, "execution_mode", "agent") or "agent"),
        input_candidate_commit_hash=(
            str(getattr(job, "input_candidate_commit_hash", "") or "").strip() or None
        ),
        input_candidate_summary=(
            str(getattr(job, "input_candidate_summary", "") or "").strip() or None
        ),
        external_submission_key=str(
            getattr(job, "external_submission_key", "") or ""
        ).strip(),
        input_provenance=dict(getattr(job, "input_provenance", {}) or {}),
        archive_ingestion_enabled=bool(getattr(job, "archive_ingestion_enabled", True)),
        repair_source_candidate_id=getattr(job, "repair_source_candidate_id", None),
        repair_mode=getattr(job, "repair_mode", None),
        campaign_program_hash=getattr(job, "campaign_program_hash", None),
        sampling_strategy=getattr(job, "sampling_strategy", None),
        sampling_initial_radius=getattr(job, "sampling_initial_radius", None),
        sampling_radius_used=getattr(job, "sampling_radius_used", None),
        sampling_fallback_inspirations=getattr(
            job, "sampling_fallback_inspirations", None
        ),
        sampling_ordinal=getattr(job, "sampling_ordinal", None),
        sampling_recipe_hash=getattr(job, "sampling_recipe_hash", None),
        sampling_recipe_reused=bool(getattr(job, "sampling_recipe_reused", False)),
        seed_portfolio_hash=(
            str(getattr(job, "seed_portfolio_hash", "") or "").strip() or None
        ),
        seed_direction_id=(
            str(getattr(job, "seed_direction_id", "") or "").strip() or None
        ),
        seed_direction_payload=dict(
            getattr(job, "seed_direction_payload", {}) or {}
        ),
        seed_admission_lane=(
            str(getattr(job, "seed_admission_lane", "") or "").strip() or None
        ),
        seed_admission_reason=(
            str(getattr(job, "seed_admission_reason", "") or "").strip() or None
        ),
    )


def _base_commit_hash(job: EvolutionJob) -> str:
    return str(getattr(job, "base_commit_hash", "") or "").strip()


def _optional_uuid(value: object) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _nearest_viable_ancestor_for_job(job: EvolutionJob) -> str | None:
    return _base_commit_hash(job) or None


def _persist_success_input(
    request: _PersistSuccessInput | None,
    values: dict[str, Any],
) -> _PersistSuccessInput:
    if request is not None:
        return request
    values.setdefault("evaluation_outcome", None)
    return _PersistSuccessInput(**values)


def _persist_failure_input(
    request: _PersistFailureInput | None,
    values: dict[str, Any],
) -> _PersistFailureInput:
    if request is not None:
        return request
    return _PersistFailureInput(**values)


def _candidate_commit_upsert_input(
    request: _CandidateCommitUpsertInput | None,
    values: dict[str, Any],
) -> _CandidateCommitUpsertInput:
    if request is not None:
        return request
    return _CandidateCommitUpsertInput(**values)


def _repair_state_decision(
    request: _RepairStateDecision | None,
    values: dict[str, Any],
) -> _RepairStateDecision:
    if request is not None:
        return request
    return _RepairStateDecision(**values)


def _failed_candidate_archive_status(outcome: EvaluationOutcome) -> str:
    if outcome.outcome_kind == "passed":
        return "not_considered"
    return "not_applicable"


def _is_supplied_candidate_job(job: EvolutionJob) -> bool:
    return (
        str(getattr(job, "execution_mode", "agent") or "agent").strip().lower()
        == "evaluate_existing"
    )


def _validate_candidate_record(
    *,
    job: EvolutionJob,
    record: CandidateCommitRecord,
    candidate_branch: str,
) -> None:
    terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
    if record.run_token is None and job.status in terminal:
        raise EvolutionWorkerError(
            f"Evolution job {record.job_id} cannot record a candidate in status {job.status}."
        )
    supplied = _is_supplied_candidate_job(job)
    if not candidate_branch and not supplied:
        raise EvolutionWorkerError("Candidate branch name must be provided.")
    if supplied and record.published:
        raise EvolutionWorkerError(
            "A supplied candidate is already remote-reachable and must not be "
            "published as a worker branch."
        )


def _apply_candidate_record_to_job(
    *,
    job: EvolutionJob,
    record: CandidateCommitRecord,
    candidate_hash: str,
    candidate_branch: str,
) -> None:
    job.candidate_commit_hash = candidate_hash
    job.candidate_branch_name = candidate_branch or None
    job.candidate_published_at = _utc_now() if record.published else None


def _failure_allows_repair(failure: Any, *, allowed: set[str]) -> bool:
    return (
        failure.failure_stage == "evaluation"
        and failure.failure_kind in allowed
        and failure.repairability == "repairable"
    )


def _optional_failure_value(failure: Any, name: str) -> Any:
    return getattr(failure, name, None) if failure is not None else None


def _new_failure_attempt(
    *,
    request: _FailureAttemptInput,
    attempt_ordinal: int,
) -> EvaluationAttempt:
    outcome = request.outcome
    failure = request.failure
    return EvaluationAttempt(
        candidate_commit_id=getattr(request.candidate, "id", None),
        job_id=request.job_ctx.job_id,
        run_token=request.job_ctx.run_token,
        attempt_ordinal=attempt_ordinal,
        evaluator_name=outcome.evaluator_name,
        evaluator_version=outcome.evaluator_version,
        campaign_program_hash=request.job_ctx.campaign_program_hash,
        seed_portfolio_hash=request.job_ctx.seed_portfolio_hash,
        seed_direction_id=request.job_ctx.seed_direction_id,
        candidate_identity=request.candidate_identity,
        evaluation_identity_key=request.identity_key,
        protocol=outcome.protocol,
        measurement_cache_key=outcome.measurement_cache_key,
        measurement_contract_fingerprint=outcome.measurement_contract_fingerprint,
        measurement_id=_optional_uuid(outcome.measurement_id),
        measurement_reused=outcome.measurement_reused,
        measurement_executed=outcome.measurement_executed,
        reuse_kind=outcome.reuse_kind,
        reused_from_attempt_id=_optional_uuid(outcome.reused_from_attempt_id),
        evaluator_slot=outcome.evaluator_slot,
        evaluator_slot_scope=outcome.evaluator_slot_scope,
        evaluator_slot_wait_seconds=outcome.evaluator_slot_wait_seconds,
        evaluator_slot_acquired_at=outcome.evaluator_slot_acquired_at,
        evaluator_slot_released_at=outcome.evaluator_slot_released_at,
        evaluator_slot_lease_id=_optional_uuid(outcome.evaluator_slot_lease_id),
        evaluator_slot_release_reason=outcome.evaluator_slot_release_reason,
        outcome_kind=outcome.outcome_kind,
        failure_kind=_optional_failure_value(failure, "failure_kind"),
        failure_stage=_optional_failure_value(failure, "failure_stage"),
        repairability=_optional_failure_value(failure, "repairability"),
        safe_failure_summary=_optional_failure_value(failure, "safe_failure_summary"),
        diagnostic_capsule_id=getattr(request.capsule, "id", None),
        artifact_policy_version=_optional_failure_value(failure, "policy_version"),
        started_at=outcome.started_at,
        finished_at=outcome.finished_at,
    )


def _update_failure_attempt(
    *,
    attempt: EvaluationAttempt,
    request: _FailureAttemptInput,
) -> None:
    failure = request.failure
    attempt.candidate_commit_id = getattr(request.candidate, "id", None)
    attempt.outcome_kind = request.outcome.outcome_kind
    attempt.failure_kind = _optional_failure_value(failure, "failure_kind")
    attempt.failure_stage = _optional_failure_value(failure, "failure_stage")
    attempt.repairability = _optional_failure_value(failure, "repairability")
    attempt.safe_failure_summary = _optional_failure_value(
        failure, "safe_failure_summary"
    )
    attempt.diagnostic_capsule_id = getattr(request.capsule, "id", None)
    attempt.artifact_policy_version = _optional_failure_value(failure, "policy_version")
    attempt.finished_at = request.outcome.finished_at


def _require_intact_measurement_evidence(
    *,
    evidence: Sequence[Any],
    artifacts: Sequence[Any],
) -> None:
    """Require every cache evidence item to be backed by intact stored bytes."""

    artifacts_by_key = {
        str(getattr(artifact, "key", "") or ""): artifact
        for artifact in artifacts
        if str(getattr(artifact, "key", "") or "")
    }
    for item in evidence:
        key = str(getattr(item, "key", "") or "")
        artifact = artifacts_by_key.get(key)
        if artifact is None:
            raise EvolutionWorkerError(
                f"Cacheable measurement evidence {key!r} has no stored evaluator artifact."
            )
        _require_measurement_artifact_metadata(item=item, artifact=artifact, key=key)
        _require_measurement_artifact_payload(artifact=artifact, key=key)


def _require_measurement_artifact_metadata(
    *, item: Any, artifact: Any, key: str
) -> None:
    expected_sha = str(getattr(item, "sha256", "") or "").lower()
    expected_size = getattr(item, "size_bytes", None)
    observed_sha = str(getattr(artifact, "sha256", "") or "").lower()
    observed_size = getattr(artifact, "size_bytes", None)
    sha_matches = observed_sha == expected_sha
    size_matches = expected_size is None or int(observed_size or -1) == int(
        expected_size
    )
    if not (sha_matches and size_matches):
        raise EvolutionWorkerError(
            f"Cacheable measurement evidence {key!r} does not match its stored artifact."
        )


def _require_measurement_artifact_payload(*, artifact: Any, key: str) -> None:
    storage_path = str(getattr(artifact, "storage_path", "") or "").strip()
    path = Path(storage_path) if storage_path else None
    if path is None or not path.is_file():
        raise EvolutionWorkerError(
            f"Cacheable measurement evidence {key!r} has no intact stored payload."
        )
    observed_size = int(getattr(artifact, "size_bytes", None) or -1)
    observed_sha = str(getattr(artifact, "sha256", "") or "").lower()
    if path.stat().st_size != observed_size or _sha256_path(path) != observed_sha:
        raise EvolutionWorkerError(
            f"Cacheable measurement evidence {key!r} stored payload changed before acceptance."
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_capsule_attempt(
    capsule_row: DiagnosticCapsule | None,
    attempt: EvaluationAttempt,
) -> None:
    if capsule_row is not None:
        capsule_row.evaluation_attempt_id = attempt.id


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
        (request.coding.report.summary if request.coding is not None else "")
        or (request.plan.plan.summary if request.plan is not None else "")
        or (
            request.job_ctx.input_candidate_summary
            or f"Evolution job {request.job_ctx.job_id}"
        )
    )
    # Coding summaries are already bounded to 800 characters.  Preserve that
    # complete projection so later trajectory compression does not discard a
    # second layer of information.
    return clamp_text(normalize_single_line(source), 800) or "N/A"


def _commit_card_key_files(paths: Sequence[str]) -> list[str]:
    return [clamp_text(path, 256) for path in paths[:20] if path.strip()]


def _commit_card_highlights(lines: Sequence[str]) -> list[str]:
    highlights = [clamp_text(line, 200) for line in lines[:8] if line.strip()]
    return highlights or ["No file-level highlights available."]


def _bounded_tags(tags: Sequence[str]) -> list[str]:
    return [
        clamp_text(normalize_single_line(tag), 64) for tag in tags if str(tag).strip()
    ]


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
