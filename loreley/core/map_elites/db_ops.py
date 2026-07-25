"""Database query helpers for MAP-Elites manager rebuilds."""

from __future__ import annotations

from typing import Iterator, Sequence

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from loreley.config import Settings, resolve_objective_contract
from loreley.db.base import session_scope
from loreley.db.models import (
    CommitCard,
    MapElitesPcaHistory,
    MapElitesRepoStateAggregate,
    Metric,
)

from .objectives import ObjectiveContractError, ResolvedObjectives
from .types import IslandState
from .vector_math import mean_and_maybe_l2_normalize_from_sum

log = logger.bind(module="map_elites.db_ops")

__all__ = [
    "_IN_QUERY_BATCH_SIZE",
    "iter_query_batches",
    "load_commit_vectors",
    "load_commit_objectives",
]

_IN_QUERY_BATCH_SIZE = 500


def iter_query_batches(
    values: Sequence[str],
    *,
    batch_size: int,
) -> Iterator[Sequence[str]]:
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}.")
    for index in range(0, len(values), batch_size):
        yield values[index : index + batch_size]


def load_commit_vectors(
    *,
    island_id: str,
    commit_hashes: Sequence[str],
    state: IslandState,
    snapshot_session: Session | None,
    settings: Settings,
) -> dict[str, tuple[float, ...]]:
    needed = {
        str(commit).strip()
        for commit in commit_hashes
        if str(commit).strip()
    }
    if not needed:
        return {}
    vectors = {
        str(entry.commit_hash): tuple(float(value) for value in entry.vector)
        for entry in state.history
        if str(entry.commit_hash or "").strip() in needed and entry.vector
    }
    missing = sorted(needed.difference(vectors))

    def _fill_from_history(session: Session) -> None:
        for batch in iter_query_batches(missing, batch_size=_IN_QUERY_BATCH_SIZE):
            rows = session.execute(
                select(MapElitesPcaHistory).where(
                    MapElitesPcaHistory.island_id == island_id,
                    MapElitesPcaHistory.commit_hash.in_(batch),
                )
            ).scalars().all()
            for row in rows:
                vector = tuple(float(value) for value in (row.vector or ()))
                if vector:
                    vectors[str(row.commit_hash)] = vector

    if missing:
        if snapshot_session is not None:
            _fill_from_history(snapshot_session)
        else:
            with session_scope() as session:
                _fill_from_history(session)
    still_missing = sorted(needed.difference(vectors))
    normalize = bool(settings.mapelites_dimensionality_penultimate_normalize)

    def _fill_from_aggregates(session: Session) -> None:
        for batch in iter_query_batches(still_missing, batch_size=_IN_QUERY_BATCH_SIZE):
            rows = session.execute(
                select(MapElitesRepoStateAggregate).where(
                    MapElitesRepoStateAggregate.commit_hash.in_(batch)
                )
            ).scalars().all()
            for row in rows:
                file_count = int(row.file_count or 0)
                if file_count <= 0:
                    continue
                vector = mean_and_maybe_l2_normalize_from_sum(
                    row.sum_vector or (),
                    file_count,
                    normalize=normalize,
                )
                if vector:
                    vectors[str(row.commit_hash)] = vector

    if still_missing:
        if snapshot_session is not None:
            _fill_from_aggregates(snapshot_session)
        else:
            with session_scope() as session:
                _fill_from_aggregates(session)
    return vectors


def load_commit_objectives(
    *,
    commit_hashes: Sequence[str],
    snapshot_session: Session | None,
    settings: Settings,
) -> dict[str, ResolvedObjectives]:
    """Load only commits whose complete metric vectors satisfy the contract."""

    needed = sorted(
        {
            str(commit).strip()
            for commit in commit_hashes
            if str(commit).strip()
        }
    )
    if not needed:
        return {}
    contract = resolve_objective_contract(settings)
    metrics_by_commit: dict[str, list[dict[str, object]]] = {}

    def _fill(session: Session) -> None:
        for batch in iter_query_batches(needed, batch_size=_IN_QUERY_BATCH_SIZE):
            rows = session.execute(
                select(
                    CommitCard.commit_hash,
                    Metric.name,
                    Metric.value,
                    Metric.higher_is_better,
                )
                .join(Metric, Metric.commit_card_id == CommitCard.id)
                .where(
                    CommitCard.commit_hash.in_(batch),
                    Metric.name.in_(contract.names),
                )
            )
            for commit_hash, name, value, higher_is_better in rows:
                metrics_by_commit.setdefault(str(commit_hash), []).append(
                    {
                        "name": str(name),
                        "value": value,
                        "higher_is_better": higher_is_better,
                    }
                )

    if snapshot_session is not None:
        _fill(snapshot_session)
    else:
        with session_scope() as session:
            _fill(session)

    resolved: dict[str, ResolvedObjectives] = {}
    for commit_hash in needed:
        try:
            resolved[commit_hash] = contract.resolve(
                metrics_by_commit.get(commit_hash, ())
            )
        except ObjectiveContractError as exc:
            log.warning(
                "Skipping commit with incomplete objective contract "
                "(commit={} reason={})",
                commit_hash,
                exc,
            )
    return resolved
