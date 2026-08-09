"""Shared campaign progress queries for status, scheduling, and APIs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import func, select

from loreley.config import Settings
from loreley.db.models import (
    CandidateCommit,
    EvaluationAttempt,
    EvaluationConcurrencyContract,
    EvaluationResourceLease,
    EvolutionJob,
    JobStatus,
    MapElitesArchiveCell,
)


@dataclass(frozen=True, slots=True)
class CampaignProgress:
    terminal_jobs: int
    succeeded_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    running_jobs: int
    queued_jobs: int
    pending_jobs: int
    distinct_passed_source_trees: int
    distinct_passed_evaluation_identities: int
    passed_candidates_without_identity: int
    real_measurements: int
    measurement_reuses: int
    exact_tree_reuses: int
    archive_entries: int
    archive_unique_evaluation_identities: int
    occupied_coordinates: int
    staged_jobs: int = 0
    failures_by_kind: dict[str, int] = field(default_factory=dict)
    evaluator_slot_holders: int = 0
    evaluator_slot_waiters: int = 0
    scheduler_max_unfinished_jobs: int = 0
    configured_worker_processes: int | None = None
    evaluator_max_concurrency: int | None = None
    identity_target: int | None = None
    identity_target_reached: bool = False
    identity_overshoot: int = 0

    @property
    def unfinished_jobs(self) -> int:
        return self.running_jobs + self.queued_jobs + self.pending_jobs

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unfinished_jobs"] = self.unfinished_jobs
        return payload


def load_campaign_progress(session: Any, settings: Settings) -> CampaignProgress:
    """Load all campaign-wide progress counts with explicit identity semantics."""

    statuses = Counter()
    for status, count in session.execute(
        select(EvolutionJob.status, func.count()).group_by(EvolutionJob.status)
    ).all():
        normalized = getattr(status, "value", status)
        statuses[str(normalized).lower()] = int(count or 0)

    succeeded = statuses[JobStatus.SUCCEEDED.value]
    failed = statuses[JobStatus.FAILED.value]
    cancelled = statuses[JobStatus.CANCELLED.value]
    staged = statuses[JobStatus.STAGED.value]
    running = statuses[JobStatus.RUNNING.value]
    queued = statuses[JobStatus.QUEUED.value]
    pending = statuses[JobStatus.PENDING.value]

    distinct_trees = _scalar_count(
        session,
        select(func.count(func.distinct(CandidateCommit.source_tree_hash))).where(
            CandidateCommit.evaluation_status == "passed",
            CandidateCommit.source_tree_hash.is_not(None),
            CandidateCommit.source_tree_hash != "",
        ),
    )
    distinct_identities = _scalar_count(
        session,
        select(func.count(func.distinct(CandidateCommit.evaluation_identity_key))).where(
            CandidateCommit.evaluation_status == "passed",
            CandidateCommit.evaluation_identity_key.is_not(None),
            CandidateCommit.evaluation_identity_key != "",
        ),
    )
    missing_identity = _scalar_count(
        session,
        select(func.count()).select_from(CandidateCommit).where(
            CandidateCommit.evaluation_status == "passed",
            (
                CandidateCommit.evaluation_identity_key.is_(None)
                | (CandidateCommit.evaluation_identity_key == "")
            ),
        ),
    )
    real_measurements = _scalar_count(
        session,
        select(func.count()).select_from(EvaluationAttempt).where(
            EvaluationAttempt.measurement_executed.is_(True)
        ),
    )
    measurement_reuses = _scalar_count(
        session,
        select(func.count()).select_from(EvaluationAttempt).where(
            EvaluationAttempt.reuse_kind == "measurement"
        ),
    )
    exact_tree_reuses = _scalar_count(
        session,
        select(func.count()).select_from(EvaluationAttempt).where(
            EvaluationAttempt.reuse_kind == "exact_tree"
        ),
    )
    archive_entries = _scalar_count(
        session,
        select(func.count()).select_from(MapElitesArchiveCell),
    )
    archive_unique_identities = _scalar_count(
        session,
        select(func.count(func.distinct(CandidateCommit.evaluation_identity_key)))
        .select_from(MapElitesArchiveCell)
        .join(CandidateCommit, CandidateCommit.commit_hash == MapElitesArchiveCell.commit_hash)
        .where(
            CandidateCommit.evaluation_identity_key.is_not(None),
            CandidateCommit.evaluation_identity_key != "",
        ),
    )
    coordinate_rows = (
        select(MapElitesArchiveCell.island_id, MapElitesArchiveCell.cell_index)
        .distinct()
        .subquery()
    )
    occupied_coordinates = _scalar_count(
        session,
        select(func.count()).select_from(coordinate_rows),
    )

    failures_by_kind = {
        str(kind or "unclassified"): int(count or 0)
        for kind, count in session.execute(
            select(EvolutionJob.failure_kind, func.count())
            .where(EvolutionJob.status == JobStatus.FAILED)
            .group_by(EvolutionJob.failure_kind)
        ).all()
    }
    active_lease_base = (
        select(EvaluationResourceLease.status, func.count())
        .join(EvolutionJob, EvolutionJob.id == EvaluationResourceLease.job_id)
        .where(
            EvaluationResourceLease.resource_kind == "evaluator_slot",
            EvolutionJob.status == JobStatus.RUNNING,
            EvolutionJob.run_token == EvaluationResourceLease.run_token,
            EvaluationResourceLease.status.in_(("waiting", "acquired")),
        )
        .group_by(EvaluationResourceLease.status)
    )
    slot_counts = {
        str(status): int(count or 0)
        for status, count in session.execute(active_lease_base).all()
    }
    persisted_limits = {
        row.max_concurrency
        for row in session.execute(select(EvaluationConcurrencyContract)).scalars()
    }
    configured_e = settings.worker_evaluator_max_concurrency
    if len(persisted_limits) == 1:
        configured_e = next(iter(persisted_limits))

    target = settings.scheduler_max_unique_evaluation_identities
    reached = target is not None and distinct_identities >= int(target)
    return CampaignProgress(
        terminal_jobs=succeeded + failed + cancelled,
        succeeded_jobs=succeeded,
        failed_jobs=failed,
        cancelled_jobs=cancelled,
        staged_jobs=staged,
        running_jobs=running,
        queued_jobs=queued,
        pending_jobs=pending,
        distinct_passed_source_trees=distinct_trees,
        distinct_passed_evaluation_identities=distinct_identities,
        passed_candidates_without_identity=missing_identity,
        real_measurements=real_measurements,
        measurement_reuses=measurement_reuses,
        exact_tree_reuses=exact_tree_reuses,
        archive_entries=archive_entries,
        archive_unique_evaluation_identities=archive_unique_identities,
        occupied_coordinates=occupied_coordinates,
        failures_by_kind=failures_by_kind,
        evaluator_slot_holders=slot_counts.get("acquired", 0),
        evaluator_slot_waiters=slot_counts.get("waiting", 0),
        scheduler_max_unfinished_jobs=int(settings.scheduler_max_unfinished_jobs),
        configured_worker_processes=getattr(settings, "worker_processes", None),
        evaluator_max_concurrency=configured_e,
        identity_target=target,
        identity_target_reached=reached,
        identity_overshoot=max(0, distinct_identities - int(target)) if target else 0,
    )


def _scalar_count(session: Any, statement: Any) -> int:
    return int(session.execute(statement).scalar_one() or 0)


__all__ = ["CampaignProgress", "load_campaign_progress"]
