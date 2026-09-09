"""Durable fresh-comparison contexts and fail-closed archive handoff.

Only Loreley writes this ledger. Plugins receive an issued context and return
measurement evidence. A prepared cycle fences the next measurement until its
job is ingested, even though the worker releases its live evaluator slot first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select, text
from sqlalchemy.orm import Session

from loreley.config import Settings
from loreley.core.comparison_contract import validate_comparison_result
from loreley.core.evolution_events import (
    COMPARISON_DECIDED, COMPARISON_MEASURED, COMPARISON_PREPARED,
    record_evolution_event, sanitized_event_payload,
)
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.worker.evaluation_runtime import EvaluationRuntimeError
from loreley.db.base import session_scope
from loreley.db.models import EvaluationAttempt, EvolutionEvent, EvolutionJob, JobStatus


def enabled(settings: Settings, *, is_seed_job: bool = False) -> bool:
    return getattr(settings, "mapelites_admission_policy", "pareto") == "fresh_comparison" and not is_seed_job


def requires_fresh_admission(session: Session | None, job_id: UUID) -> bool:
    """The core-issued context is durable admission intent, even after a restart."""
    if not isinstance(session, Session):
        return False
    return session.execute(select(EvolutionEvent.id).where(
        EvolutionEvent.event_type == COMPARISON_PREPARED,
        EvolutionEvent.job_id == job_id,
    ).limit(1)).first() is not None


def lock_comparison(session: Session, settings: Settings) -> None:
    """Serialize issue/admit transactions, never hold a DB transaction to measure."""
    if session.get_bind().dialect.name != "postgresql":
        raise EvaluationRuntimeError("Fresh comparison requires PostgreSQL.")
    digest = hashlib.sha256(f"loreley:comparison:v1:{settings.experiment_id}".encode()).digest()
    key = int.from_bytes(digest[:8], "big", signed=True)
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def _active_job(session: Session, job_id: UUID, run_token: UUID) -> EvolutionJob:
    job = session.execute(select(EvolutionJob).where(
        EvolutionJob.id == job_id,
    ).with_for_update()).scalar_one()
    if job.status != JobStatus.RUNNING or job.run_token != run_token:
        raise EvaluationRuntimeError("Fresh comparison belongs to an inactive worker run.")
    return job


def _pending_cycle(session: Session, job_id: UUID) -> bool:
    """Run-token mismatches release abandoned runs; succeeded jobs need ingestion."""
    active = and_(EvolutionJob.status == JobStatus.RUNNING,
                  EvolutionJob.run_token == EvolutionEvent.run_token)
    awaiting = and_(EvolutionJob.status == JobStatus.SUCCEEDED,
                    or_(EvolutionJob.ingestion_status.is_(None),
                        EvolutionJob.ingestion_status.not_in(("succeeded", "skipped"))))
    return session.execute(select(EvolutionEvent.id).join(
        EvolutionJob, EvolutionJob.id == EvolutionEvent.job_id,
    ).where(EvolutionEvent.event_type == COMPARISON_PREPARED,
            EvolutionJob.id != job_id, or_(active, awaiting)).limit(1)).first() is not None


def _pending_bootstrap(session: Session) -> bool:
    terminal = ("succeeded", "skipped")
    awaiting = and_(EvolutionJob.status == JobStatus.SUCCEEDED,
                    or_(EvolutionJob.ingestion_status.is_(None),
                        EvolutionJob.ingestion_status.not_in(terminal)))
    return session.execute(select(EvolutionJob.id).where(
        EvolutionJob.is_seed_job.is_(True),
        or_(EvolutionJob.status.in_((JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)),
            awaiting),
    ).limit(1)).first() is not None


def wait_for_bootstrap(*, job_id: UUID, run_token: UUID, deadline: float) -> None:
    """Wait before taking the sole evaluator slot, which seeds also need."""
    while monotonic() < deadline:
        with session_scope() as session:
            _active_job(session, job_id, run_token)
            if not _pending_bootstrap(session):
                return
        sleep(min(.5, max(0., deadline - monotonic())))
    raise EvaluationRuntimeError("Seed bootstrap has not completed ingestion.")


def _write_event(session: Session, *, event_type: str, job_id: UUID,
                 run_token: UUID | None, context: Mapping[str, Any],
                 payload: Mapping[str, Any]) -> None:
    # The event system bounds strings. Identity-bearing fields must never be
    # silently truncated or normalized on their way into the durable context.
    expected = {k: v for k, v in payload.items() if v is not None}
    if sanitized_event_payload(event_type, payload) != expected:
        raise EvaluationRuntimeError("Fresh comparison event exceeds its identity contract.")
    receipt = record_evolution_event(
        session, event_type=event_type, job_id=job_id, run_token=run_token,
        island_id=str(context["island_id"]), commit_hash=str(context["candidate_commit_hash"]),
        payload=payload, key_parts=(str(context["context_id"]),),
    )
    if not receipt.inserted:
        existing = session.get(EvolutionEvent, receipt.event_id)
        if existing is None or existing.payload != expected:
            raise EvaluationRuntimeError("Conflicting replay of a fresh comparison event.")


def prepare_context(*, settings: Settings, job_id: UUID, run_token: UUID,
                    commit_hash: str, island_id: str, repo_root: Path,
                    evaluator_name: str, evaluator_version: str,
                    campaign_program_hash: str, candidate_identity: str,
                    measurement_contract_fingerprint: str, deadline: float) -> dict[str, Any]:
    manager = MapElitesManager(settings=settings, repo_root=repo_root)
    while monotonic() < deadline:
        with session_scope() as session:
            lock_comparison(session, settings)
            _active_job(session, job_id, run_token)
            if _pending_bootstrap(session):
                raise EvaluationRuntimeError("Seed bootstrap changed before fresh measurement.")
            if not _pending_cycle(session, job_id):
                context = manager.prepare_comparison(
                    commit_hash=commit_hash, island_id=island_id,
                    repo_root=repo_root, snapshot_session=session,
                )
                context.update(
                    context_id=str(uuid4()), evaluator_name=evaluator_name,
                    evaluator_version=evaluator_version,
                    campaign_program_hash=campaign_program_hash,
                    candidate_identity_sha256=hashlib.sha256(candidate_identity.encode()).hexdigest(),
                    contract_sha256=hashlib.sha256(measurement_contract_fingerprint.encode()).hexdigest(),
                )
                _write_event(session, event_type=COMPARISON_PREPARED, job_id=job_id,
                             run_token=run_token, context=context, payload=context)
                return context
        sleep(min(.5, max(0., deadline - monotonic())))
    raise EvaluationRuntimeError("Previous fresh comparison has not completed ingestion.")


def _issued_context(session: Session, *, job_id: UUID, run_token: UUID,
                    context_id: str) -> dict[str, Any]:
    row = session.execute(select(EvolutionEvent).where(
        EvolutionEvent.event_type == COMPARISON_PREPARED,
        EvolutionEvent.job_id == job_id, EvolutionEvent.run_token == run_token,
        EvolutionEvent.payload["context_id"].astext == context_id,
    )).scalar_one_or_none()
    if row is None:
        raise EvaluationRuntimeError("No core-issued fresh comparison matches this result.")
    context = dict(row.payload)
    context.setdefault("incumbent_commit_hash", None)
    return context


def record_measurement(*, settings: Settings, job_id: UUID, run_token: UUID,
                       context: Mapping[str, Any], result: Any) -> None:
    if not isinstance(context, Mapping) or not context.get("context_id"):
        raise EvaluationRuntimeError("Fresh measurement is missing its issued context ID.")
    with session_scope() as session:
        _active_job(session, job_id, run_token)
        issued = _issued_context(session, job_id=job_id, run_token=run_token,
                                 context_id=str(context["context_id"]))
        decision = validate_comparison_result(
            result.extra.get("fresh_comparison"), issued, result.metrics,
            minimum_confidence=settings.mapelites_comparison_confidence,
        )
        payload = {k: v for k, v in decision.as_dict().items() if k != "evidence"}
        _write_event(session, event_type=COMPARISON_MEASURED, job_id=job_id,
                     run_token=run_token, context=issued, payload=payload)


def load_handoff(*, session: Session, settings: Settings, job_id: UUID,
                 commit_hash: str, metrics: Sequence[Mapping[str, Any]]) -> tuple[dict, dict]:
    """Bind the gate to the persisted successful attempt, not a mutable old score."""
    lock_comparison(session, settings)
    job = session.execute(select(EvolutionJob).where(
        EvolutionJob.id == job_id,
    ).with_for_update()).scalar_one()
    if job.status != JobStatus.SUCCEEDED or job.result_commit_hash != commit_hash:
        raise EvaluationRuntimeError("Fresh comparison job changed before ingestion.")
    attempt = session.execute(select(EvaluationAttempt).where(
        EvaluationAttempt.job_id == job_id, EvaluationAttempt.outcome_kind == "passed",
    ).order_by(EvaluationAttempt.created_at.desc(), EvaluationAttempt.id.desc()).limit(1)).scalar_one_or_none()
    if (attempt is None or attempt.run_token is None or attempt.measurement_reused
            or not attempt.measurement_executed or attempt.protocol != "phased-v1"):
        raise EvaluationRuntimeError("Fresh comparison needs a newly measured successful attempt.")
    row = session.execute(select(EvolutionEvent).where(
        EvolutionEvent.event_type == COMPARISON_MEASURED,
        EvolutionEvent.job_id == job_id, EvolutionEvent.run_token == attempt.run_token,
        EvolutionEvent.commit_hash == commit_hash,
    ).order_by(EvolutionEvent.occurred_at.desc(), EvolutionEvent.id.desc()).limit(1)).scalar_one_or_none()
    if row is None:
        raise EvaluationRuntimeError("Successful attempt has no fresh comparison evidence.")
    decision = dict(row.payload)
    context = _issued_context(session, job_id=job_id, run_token=attempt.run_token,
                              context_id=decision["context_id"])
    _verify_attempt(context, attempt, job, commit_hash)
    evidence = {k: v for k, v in decision.items() if k not in {"allowed", "point_gain"}}
    evidence["complete"] = True
    checked = validate_comparison_result(
        evidence, context, metrics, minimum_confidence=settings.mapelites_comparison_confidence,
    )
    if checked.allowed != decision["allowed"]:
        raise EvaluationRuntimeError("Fresh comparison decision changed during persistence.")
    return context, decision


def _verify_attempt(context: Mapping[str, Any], attempt: EvaluationAttempt,
                    job: EvolutionJob, commit_hash: str) -> None:
    if attempt.campaign_program_hash != job.campaign_program_hash:
        raise EvaluationRuntimeError("Successful attempt has a different campaign contract.")
    expected = {
        "candidate_commit_hash": commit_hash, "island_id": job.island_id,
        "campaign_program_hash": job.campaign_program_hash,
        "evaluator_name": attempt.evaluator_name, "evaluator_version": attempt.evaluator_version,
        "candidate_identity_sha256": hashlib.sha256(str(attempt.candidate_identity or "").encode()).hexdigest(),
        "contract_sha256": hashlib.sha256(str(attempt.measurement_contract_fingerprint or "").encode()).hexdigest(),
    }
    if any(context.get(k) != v for k, v in expected.items()):
        raise EvaluationRuntimeError("Fresh comparison context does not match the successful attempt.")


def acknowledge(session: Session, *, job_id: UUID, context: Mapping[str, Any],
                allowed: bool) -> None:
    _write_event(session, event_type=COMPARISON_DECIDED, job_id=job_id, run_token=None,
                 context=context, payload={"context_id": context["context_id"],
                                           "allowed": allowed, "cell_index": context["cell_index"]})
