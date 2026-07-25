"""Postgres persistence for independent Pareto MAP-Elites islands."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import numpy as np
from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from loreley.core.map_elites.objectives import ObjectiveContract
from loreley.db.base import session_scope
from loreley.db.models import MapElitesArchiveCell, MapElitesPcaHistory, MapElitesState

from .dimension_reduction import PCAProjection, PcaHistoryEntry
from .pareto_archive import ParetoCandidate
from .types import IslandState

log = logger.bind(module="map_elites.snapshot")

Vector = tuple[float, ...]

__all__ = [
    "DatabaseSnapshotStore",
    "SnapshotElite",
    "SnapshotStoreError",
    "SnapshotUpdate",
    "apply_snapshot",
    "deserialize_history",
    "deserialize_projection",
    "ensure_supported_snapshot_meta",
    "serialize_projection",
    "validate_snapshot_contract",
]

UNSUPPORTED_META_KEYS = ("archive", "history")


class SnapshotStoreError(RuntimeError):
    """Raised when MAP-Elites snapshot state cannot be loaded or persisted."""


def ensure_supported_snapshot_meta(
    meta: Mapping[str, Any] | None,
    *,
    island_id: str,
) -> None:
    """Fail fast if old embedded archive/history payloads remain."""

    if not meta:
        return
    for key in UNSUPPORTED_META_KEYS:
        if key in meta:
            raise ValueError(
                "Unsupported MAP-Elites snapshot payload detected; run the "
                "native database migration. "
                f"(island_id={island_id} key={key})"
            )


@dataclass(slots=True, frozen=True)
class SnapshotElite:
    """Persistence payload for one retained Pareto member."""

    cell_index: int
    commit_hash: str
    objective_values: Vector
    measures: Vector
    timestamp: float


@dataclass(slots=True, frozen=True)
class _PersistedCandidate:
    candidate: ParetoCandidate
    cell_index: int


@dataclass(slots=True)
class SnapshotUpdate:
    """One atomic metadata/history/archive update for an island."""

    objective_contract: ObjectiveContract | None = None
    lower_bounds: Sequence[float] | None = None
    upper_bounds: Sequence[float] | None = None
    projection: PCAProjection | None = None
    samples_since_fit: int | None = None
    history_limit: int | None = None
    history_upsert: PcaHistoryEntry | None = None
    history_seen_at: float | None = None
    front_replace: Sequence[SnapshotElite] | None = None
    archive_replace: Sequence[SnapshotElite] | None = None
    clear: bool = False


class DatabaseSnapshotStore:
    """Postgres-backed island snapshots with atomic Pareto-front replacement."""

    @staticmethod
    def _raise_store_error(*, action: str, island_id: str, exc: Exception) -> None:
        if isinstance(exc, SQLAlchemyError):
            message = (
                f"Failed to {action} MAP-Elites snapshot for island {island_id}: {exc}"
            )
        else:
            message = f"Unexpected error while {action} snapshot for island {island_id}: {exc}"
        log.exception(message)
        raise SnapshotStoreError(message) from exc

    def load(
        self,
        island_id: str,
        *,
        history_limit: int | None = None,
        session: Session | None = None,
    ) -> dict[str, Any] | None:
        try:
            if session is not None:
                return self._load_in_session(
                    session,
                    island_id=island_id,
                    history_limit=history_limit,
                )
            with session_scope() as session:
                return self._load_in_session(
                    session,
                    island_id=island_id,
                    history_limit=history_limit,
                )
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            self._raise_store_error(action="load", island_id=island_id, exc=exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._raise_store_error(action="loading", island_id=island_id, exc=exc)

    def _load_in_session(
        self,
        session: Session,
        *,
        island_id: str,
        history_limit: int | None,
    ) -> dict[str, Any] | None:
        state = session.execute(
            select(MapElitesState).where(MapElitesState.island_id == island_id)
        ).scalar_one_or_none()
        if state is None:
            return None
        meta = dict(state.snapshot or {})
        ensure_supported_snapshot_meta(meta, island_id=island_id)
        return {
            **meta,
            "island_id": island_id,
            "history": self._load_history_entries(
                session,
                island_id=island_id,
                limit=history_limit,
            ),
            "archive": self._load_archive_entries(
                session,
                island_id=island_id,
            ),
        }

    def apply_update(
        self,
        island_id: str,
        *,
        update: SnapshotUpdate,
        session: Session | None = None,
    ) -> None:
        now = (
            float(update.history_seen_at)
            if update.history_seen_at is not None
            else time.time()
        )
        if session is None:
            self._apply_update_with_owned_session(
                island_id=island_id,
                update=update,
                now=now,
            )
            return
        self._apply_update_with_caller_session(
            session,
            island_id=island_id,
            update=update,
            now=now,
        )

    def _apply_update_with_owned_session(
        self,
        *,
        island_id: str,
        update: SnapshotUpdate,
        now: float,
    ) -> None:
        try:
            with session_scope() as owned_session:
                self._apply_update_in_session(
                    owned_session,
                    island_id=island_id,
                    update=update,
                    now=now,
                )
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            self._raise_store_error(action="persist", island_id=island_id, exc=exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._raise_store_error(action="persisting", island_id=island_id, exc=exc)

    def _apply_update_with_caller_session(
        self,
        session: Session,
        *,
        island_id: str,
        update: SnapshotUpdate,
        now: float,
    ) -> None:
        try:
            if bool(getattr(session, "in_nested_transaction", lambda: False)()):
                self._apply_update_in_session(
                    session,
                    island_id=island_id,
                    update=update,
                    now=now,
                )
                return
            with session.begin_nested():
                self._apply_update_in_session(
                    session,
                    island_id=island_id,
                    update=update,
                    now=now,
                )
        except ValueError:
            raise
        except SQLAlchemyError as exc:
            self._raise_store_error(action="persist", island_id=island_id, exc=exc)

    def _apply_update_in_session(
        self,
        session: Session,
        *,
        island_id: str,
        update: SnapshotUpdate,
        now: float,
    ) -> None:
        existing = session.execute(
            select(MapElitesState).where(MapElitesState.island_id == island_id)
        ).scalar_one_or_none()
        meta = dict(existing.snapshot or {}) if existing else {}
        ensure_supported_snapshot_meta(meta, island_id=island_id)
        self._apply_contract_meta(
            meta,
            contract=update.objective_contract,
            island_id=island_id,
        )
        meta["last_update_at"] = now
        if update.lower_bounds is not None:
            meta["lower_bounds"] = [float(value) for value in update.lower_bounds]
        if update.upper_bounds is not None:
            meta["upper_bounds"] = [float(value) for value in update.upper_bounds]
        if update.samples_since_fit is not None:
            meta["samples_since_fit"] = max(0, int(update.samples_since_fit))
        meta["projection"] = serialize_projection(update.projection)

        if existing is None:
            session.add(MapElitesState(island_id=island_id, snapshot=meta))
        else:
            existing.snapshot = meta

        if update.clear:
            session.execute(
                delete(MapElitesArchiveCell).where(
                    MapElitesArchiveCell.island_id == island_id
                )
            )
            session.execute(
                delete(MapElitesPcaHistory).where(
                    MapElitesPcaHistory.island_id == island_id
                )
            )
            return

        if update.archive_replace is not None:
            self._replace_archive(
                session,
                island_id=island_id,
                elites=update.archive_replace,
            )
        elif update.front_replace is not None:
            self._replace_front(
                session,
                island_id=island_id,
                elites=update.front_replace,
            )

        if update.history_upsert is not None:
            self._upsert_history(
                session,
                island_id=island_id,
                entry=update.history_upsert,
                now=now,
            )
            self._prune_history_entries(
                session,
                island_id=island_id,
                history_limit=update.history_limit,
            )

    @staticmethod
    def _apply_contract_meta(
        meta: dict[str, Any],
        *,
        contract: ObjectiveContract | None,
        island_id: str,
    ) -> None:
        persisted = str(meta.get("objective_contract_fingerprint") or "").strip()
        if contract is None:
            if persisted:
                return
            raise ValueError(
                "Objective contract is required when persisting an island "
                f"(island={island_id})."
            )
        if persisted and persisted != contract.fingerprint:
            raise ValueError(
                "Objective contract fingerprint mismatch "
                f"(island={island_id} stored={persisted} configured={contract.fingerprint})."
            )
        meta["objective_contract"] = contract.as_payload()
        meta["objective_contract_fingerprint"] = contract.fingerprint

    def _replace_front(
        self,
        session: Session,
        *,
        island_id: str,
        elites: Sequence[SnapshotElite],
    ) -> None:
        payload = tuple(elites)
        if not payload:
            raise ValueError("A front replacement must identify one behavior cell.")
        cell_indices = {int(elite.cell_index) for elite in payload}
        if len(cell_indices) != 1:
            raise ValueError("A front replacement cannot span multiple behavior cells.")
        cell_index = next(iter(cell_indices))
        session.execute(
            delete(MapElitesArchiveCell).where(
                MapElitesArchiveCell.island_id == island_id,
                MapElitesArchiveCell.cell_index == cell_index,
            )
        )
        self._insert_elites(session, island_id=island_id, elites=payload)

    def _replace_archive(
        self,
        session: Session,
        *,
        island_id: str,
        elites: Sequence[SnapshotElite],
    ) -> None:
        session.execute(
            delete(MapElitesArchiveCell).where(
                MapElitesArchiveCell.island_id == island_id
            )
        )
        self._insert_elites(session, island_id=island_id, elites=elites)

    def _insert_elites(
        self,
        session: Session,
        *,
        island_id: str,
        elites: Sequence[SnapshotElite],
    ) -> None:
        values = [
            {
                "island_id": island_id,
                "cell_index": int(elite.cell_index),
                "commit_hash": str(elite.commit_hash),
                "objective_values": [float(value) for value in elite.objective_values],
                "measures": [float(value) for value in elite.measures],
                "timestamp": float(elite.timestamp),
            }
            for elite in elites
        ]
        if values:
            session.execute(pg_insert(MapElitesArchiveCell).values(values))

    @staticmethod
    def _upsert_history(
        session: Session,
        *,
        island_id: str,
        entry: PcaHistoryEntry,
        now: float,
    ) -> None:
        stmt = pg_insert(MapElitesPcaHistory).values(
            island_id=island_id,
            commit_hash=str(entry.commit_hash),
            vector=[float(value) for value in entry.vector],
            embedding_model=str(entry.embedding_model),
            last_seen_at=float(now),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                MapElitesPcaHistory.island_id,
                MapElitesPcaHistory.commit_hash,
            ],
            set_={
                "vector": stmt.excluded.vector,
                "embedding_model": stmt.excluded.embedding_model,
                "last_seen_at": stmt.excluded.last_seen_at,
            },
        )
        session.execute(stmt)

    def _prune_history_entries(
        self,
        session: Session,
        *,
        island_id: str,
        history_limit: int | None,
    ) -> None:
        effective_limit = int(history_limit or 0)
        if effective_limit <= 0:
            return
        stale = list(
            session.execute(
                select(MapElitesPcaHistory.commit_hash)
                .where(MapElitesPcaHistory.island_id == island_id)
                .order_by(
                    MapElitesPcaHistory.last_seen_at.desc(),
                    MapElitesPcaHistory.commit_hash.asc(),
                )
                .offset(effective_limit)
            )
            .scalars()
            .all()
        )
        if stale:
            session.execute(
                delete(MapElitesPcaHistory).where(
                    MapElitesPcaHistory.island_id == island_id,
                    MapElitesPcaHistory.commit_hash.in_(stale),
                )
            )

    @staticmethod
    def _load_archive_entries(
        session: Session,
        *,
        island_id: str,
    ) -> list[dict[str, Any]]:
        rows = list(
            session.execute(
                select(MapElitesArchiveCell)
                .where(MapElitesArchiveCell.island_id == island_id)
                .order_by(
                    MapElitesArchiveCell.cell_index.asc(),
                    MapElitesArchiveCell.commit_hash.asc(),
                )
            )
            .scalars()
            .all()
        )
        return [
            {
                "index": int(row.cell_index),
                "commit_hash": str(row.commit_hash or ""),
                "objective_values": [
                    float(value) for value in (row.objective_values or ())
                ],
                "measures": [float(value) for value in (row.measures or ())],
                "timestamp": float(row.timestamp or 0.0),
            }
            for row in rows
        ]

    @staticmethod
    def _load_history_entries(
        session: Session,
        *,
        island_id: str,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(MapElitesPcaHistory)
            .where(MapElitesPcaHistory.island_id == island_id)
            .order_by(
                MapElitesPcaHistory.last_seen_at.desc(),
                MapElitesPcaHistory.commit_hash.asc(),
            )
        )
        if int(limit or 0) > 0:
            stmt = stmt.limit(int(limit or 0))
        rows = list(session.execute(stmt).scalars().all())
        rows.reverse()
        return [
            {
                "commit_hash": str(row.commit_hash or ""),
                "vector": [float(value) for value in (row.vector or ())],
                "embedding_model": str(row.embedding_model or ""),
            }
            for row in rows
        ]


def apply_snapshot(
    *,
    state: IslandState,
    snapshot: Mapping[str, Any],
    island_id: str,
    objective_contract: ObjectiveContract,
) -> None:
    """Restore a persisted island and validate its objective interpretation."""

    validate_snapshot_contract(
        snapshot,
        island_id=island_id,
        objective_contract=objective_contract,
    )
    _restore_snapshot_metadata(state, snapshot=snapshot)
    state.archive.clear()
    persisted_candidates = _deserialize_archive_candidates(
        snapshot,
        island_id=island_id,
        objective_contract=objective_contract,
    )
    _restore_archive_candidates(
        state,
        persisted_candidates=persisted_candidates,
        island_id=island_id,
    )
    _sync_state_indexes(state)


def validate_snapshot_contract(
    snapshot: Mapping[str, Any],
    *,
    island_id: str,
    objective_contract: ObjectiveContract,
) -> None:
    stored_fingerprint = str(
        snapshot.get("objective_contract_fingerprint") or ""
    ).strip()
    if stored_fingerprint != objective_contract.fingerprint:
        raise ValueError(
            "Objective contract fingerprint mismatch "
            f"(island={island_id} stored={stored_fingerprint or 'missing'} "
            f"configured={objective_contract.fingerprint})."
        )
    stored_contract = snapshot.get("objective_contract")
    if stored_contract != objective_contract.as_payload():
        raise ValueError(f"Objective contract payload mismatch (island={island_id}).")


def _restore_snapshot_metadata(
    state: IslandState,
    *,
    snapshot: Mapping[str, Any],
) -> None:
    lower_bounds = snapshot.get("lower_bounds")
    upper_bounds = snapshot.get("upper_bounds")
    if isinstance(lower_bounds, Sequence):
        state.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    if isinstance(upper_bounds, Sequence):
        state.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
    state.samples_since_fit = _nonnegative_int(snapshot.get("samples_since_fit"))
    history_payload = snapshot.get("history") or ()
    state.history = deserialize_history(history_payload) if history_payload else ()
    state.projection = deserialize_projection(snapshot.get("projection"))


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _deserialize_archive_candidates(
    snapshot: Mapping[str, Any],
    *,
    island_id: str,
    objective_contract: ObjectiveContract,
) -> tuple[_PersistedCandidate, ...]:
    archive_entries = tuple(snapshot.get("archive") or ())
    return tuple(
        _deserialize_archive_candidate(
            entry,
            island_id=island_id,
            objective_contract=objective_contract,
        )
        for entry in archive_entries
    )


def _deserialize_archive_candidate(
    entry: Any,
    *,
    island_id: str,
    objective_contract: ObjectiveContract,
) -> _PersistedCandidate:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Invalid archive entry in island {island_id}.")
    objectives = objective_contract.resolve_values(
        tuple(entry.get("objective_values") or ())
    )
    candidate = ParetoCandidate(
        commit_hash=str(entry.get("commit_hash") or ""),
        objective_values=objectives.values,
        objective_scores=objectives.scores,
        measures=tuple(float(value) for value in (entry.get("measures") or ())),
        timestamp=float(entry.get("timestamp") or 0.0),
    )
    return _PersistedCandidate(
        candidate=candidate,
        cell_index=int(entry.get("index")),
    )


def _restore_archive_candidates(
    state: IslandState,
    *,
    persisted_candidates: Sequence[_PersistedCandidate],
    island_id: str,
) -> None:
    if not persisted_candidates:
        return
    candidates = tuple(item.candidate for item in persisted_candidates)
    outcomes = state.archive.add_many(candidates)
    retained = {
        candidate.commit_hash
        for candidate, outcome in zip(candidates, outcomes)
        if outcome.retained
    }
    expected = {candidate.commit_hash for candidate in candidates}
    if retained != expected:
        raise ValueError(
            "Persisted archive contains dominated, equivalent, or over-capacity "
            f"members (island={island_id})."
        )
    for persisted, outcome in zip(persisted_candidates, outcomes):
        if outcome.cell_index != persisted.cell_index:
            raise ValueError(
                "Persisted behavior cell does not match its measures "
                f"(island={island_id} commit={persisted.candidate.commit_hash})."
            )


def _sync_state_indexes(state: IslandState) -> None:
    state.commit_to_index = {}
    state.index_to_commits = {}
    indices = sorted({int(value) for value in state.archive.data().get("index", ())})
    for cell_index in indices:
        commits = tuple(
            candidate.commit_hash for candidate in state.archive.front(cell_index)
        )
        if commits:
            state.index_to_commits[cell_index] = commits
            for commit_hash in commits:
                state.commit_to_index[commit_hash] = cell_index


def deserialize_history(
    payload: Sequence[Mapping[str, Any]],
) -> tuple[PcaHistoryEntry, ...]:
    return tuple(
        PcaHistoryEntry(
            commit_hash=str(item.get("commit_hash") or ""),
            vector=tuple(float(value) for value in (item.get("vector") or ())),
            embedding_model=str(item.get("embedding_model") or ""),
        )
        for item in payload
    )


def serialize_projection(projection: PCAProjection | None) -> dict[str, Any] | None:
    if projection is None:
        return None
    return {
        "feature_count": projection.feature_count,
        "components": [list(row) for row in projection.components],
        "mean": list(projection.mean),
        "explained_variance": list(projection.explained_variance),
        "explained_variance_ratio": list(projection.explained_variance_ratio),
        "sample_count": projection.sample_count,
        "epoch": int(projection.epoch),
        "fitted_at": projection.fitted_at,
        "whiten": projection.whiten,
        "rotation": [list(row) for row in projection.rotation]
        if projection.rotation
        else None,
    }


def deserialize_projection(
    payload: Mapping[str, Any] | None,
) -> PCAProjection | None:
    if not payload:
        return None
    rotation_payload = payload.get("rotation")
    rotation = (
        tuple(tuple(float(value) for value in row) for row in rotation_payload)
        if isinstance(rotation_payload, (list, tuple)) and rotation_payload
        else None
    )
    return PCAProjection(
        feature_count=int(payload.get("feature_count", 0)),
        components=tuple(
            tuple(float(value) for value in row)
            for row in (payload.get("components") or ())
        ),
        mean=tuple(float(value) for value in (payload.get("mean") or ())),
        explained_variance=tuple(
            float(value) for value in (payload.get("explained_variance") or ())
        ),
        explained_variance_ratio=tuple(
            float(value) for value in (payload.get("explained_variance_ratio") or ())
        ),
        sample_count=int(payload.get("sample_count", 0)),
        epoch=max(0, int(payload.get("epoch", 0))),
        fitted_at=float(payload.get("fitted_at", 0.0)),
        whiten=bool(payload.get("whiten", False)),
        rotation=rotation,
    )
