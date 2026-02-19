"""Database query helpers for MAP-Elites manager."""

from __future__ import annotations

from typing import Iterator, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from loreley.config import Settings
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, MapElitesPcaHistory, MapElitesRepoStateAggregate, Metric

from .types import IslandState
from .vector_math import mean_and_maybe_l2_normalize_from_sum

__all__ = [
    "_IN_QUERY_BATCH_SIZE",
    "iter_query_batches",
    "load_commit_vectors",
    "load_commit_fitnesses",
]

_IN_QUERY_BATCH_SIZE = 500


def iter_query_batches(values: Sequence[str], *, batch_size: int) -> Iterator[Sequence[str]]:
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}.")
    for idx in range(0, len(values), batch_size):
        yield values[idx: idx + batch_size]


def load_commit_vectors(
    *,
    island_id: str,
    commit_hashes: Sequence[str],
    state: IslandState,
    snapshot_session: Session | None,
    settings: Settings,
) -> dict[str, tuple[float, ...]]:
    needed = {str(commit).strip() for commit in commit_hashes if str(commit).strip()}
    if not needed:
        return {}

    vectors: dict[str, tuple[float, ...]] = {}
    for entry in state.history:
        commit = str(entry.commit_hash or "").strip()
        if commit and commit in needed:
            vectors[commit] = tuple(float(v) for v in entry.vector)
    missing = sorted(needed.difference(vectors.keys()))
    if not missing:
        return vectors

    def _fill_from_db(session: Session) -> None:
        for commit_batch in iter_query_batches(missing, batch_size=_IN_QUERY_BATCH_SIZE):
            stmt = (
                select(MapElitesPcaHistory)
                .where(MapElitesPcaHistory.island_id == island_id)
                .where(MapElitesPcaHistory.commit_hash.in_(commit_batch))
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                commit = str(row.commit_hash or "").strip()
                if not commit or commit in vectors:
                    continue
                vec = tuple(float(v) for v in (row.vector or []))
                if vec:
                    vectors[commit] = vec

    if snapshot_session is not None:
        _fill_from_db(snapshot_session)
    else:
        with session_scope() as owned:
            _fill_from_db(owned)

    still_missing = sorted(needed.difference(vectors.keys()))
    if not still_missing:
        return vectors

    normalize = bool(settings.mapelites_dimensionality_penultimate_normalize)

    def _fill_from_aggregates(session: Session) -> None:
        for commit_batch in iter_query_batches(still_missing, batch_size=_IN_QUERY_BATCH_SIZE):
            stmt = select(MapElitesRepoStateAggregate).where(
                MapElitesRepoStateAggregate.commit_hash.in_(commit_batch)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                commit = str(row.commit_hash or "").strip()
                if not commit or commit in vectors:
                    continue
                file_count = int(row.file_count or 0)
                if file_count <= 0:
                    continue
                vec = mean_and_maybe_l2_normalize_from_sum(
                    row.sum_vector or (),
                    file_count,
                    normalize=normalize,
                )
                if vec:
                    vectors[commit] = vec

    if snapshot_session is not None:
        _fill_from_aggregates(snapshot_session)
    else:
        with session_scope() as owned:
            _fill_from_aggregates(owned)

    return vectors


def load_commit_fitnesses(
    *,
    commit_hashes: Sequence[str],
    snapshot_session: Session | None,
    settings: Settings,
) -> dict[str, float]:
    metric_name = (settings.mapelites_fitness_metric or "").strip()
    if not metric_name:
        return {}

    needed = sorted({str(commit).strip() for commit in commit_hashes if str(commit).strip()})
    if not needed:
        return {}

    direction = 1.0 if settings.mapelites_fitness_higher_is_better else -1.0
    fitnesses: dict[str, float] = {}

    def _fill(session: Session) -> None:
        stmt = (
            select(CommitCard.commit_hash, Metric.value)
            .join(Metric, Metric.commit_card_id == CommitCard.id)
            .where(CommitCard.commit_hash.in_(needed))
            .where(Metric.name == metric_name)
        )
        for commit_hash, value in session.execute(stmt).all():
            commit = str(commit_hash or "").strip()
            if not commit:
                continue
            try:
                fitnesses[commit] = float(value) * direction
            except (TypeError, ValueError):
                continue

    if snapshot_session is not None:
        _fill(snapshot_session)
    else:
        with session_scope() as session:
            _fill(session)

    return fitnesses
