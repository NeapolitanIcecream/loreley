"""Pareto behavior-archive operations and manager-facing record adapters."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from loreley.config import Settings, resolve_objective_contract

from .pareto_archive import ParetoCandidate, ParetoGridArchive
from .snapshot import SnapshotElite
from .types import IslandState, MapElitesRecord

__all__ = [
    "add_batch",
    "add_single",
    "build_archive",
    "build_archive_replace_payload",
    "build_feature_bounds",
    "clip_vector",
    "record_from_candidate",
    "records_from_archive",
    "sync_archive_indexes",
]


def build_feature_bounds(*, target_dims: int) -> tuple[np.ndarray, np.ndarray]:
    dims = int(target_dims)
    return np.zeros(dims, dtype=np.float64), np.ones(dims, dtype=np.float64)


def build_archive(
    *,
    settings: Settings,
    target_dims: int,
    cells_per_dim: int,
    lower_template: np.ndarray,
    upper_template: np.ndarray,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> ParetoGridArchive:
    dims = int(target_dims)
    lower = np.asarray(
        lower_bounds if lower_bounds is not None else lower_template,
        dtype=np.float64,
    )
    upper = np.asarray(
        upper_bounds if upper_bounds is not None else upper_template,
        dtype=np.float64,
    )
    if lower.shape != (dims,) or upper.shape != (dims,):
        lower, upper = build_feature_bounds(target_dims=dims)
    return ParetoGridArchive(
        dims=tuple(int(cells_per_dim) for _ in range(dims)),
        ranges=tuple(zip(lower.tolist(), upper.tolist())),
        objective_count=len(resolve_objective_contract(settings).specs),
        max_front_size=int(settings.mapelites_pareto_front_max_size),
        epsilon=float(settings.mapelites_pareto_epsilon),
    )


def clip_vector(
    *,
    vector: Sequence[float] | np.ndarray,
    settings: Settings,
    clip_radius: float,
    state: IslandState,
) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    effective_radius = float(clip_radius)
    if effective_radius <= 0.0:
        effective_radius = 1.0
    if settings.mapelites_feature_clip:
        arr = np.clip(arr, -effective_radius, effective_radius)
    normalized = (arr + effective_radius) / (2.0 * effective_radius)
    if settings.mapelites_feature_clip:
        return np.clip(normalized, state.lower_bounds, state.upper_bounds)
    return normalized


def add_single(
    *,
    state: IslandState,
    island_id: str,
    commit_hash: str,
    objective_values: Sequence[float],
    objective_scores: Sequence[float],
    measures: np.ndarray,
    timestamp: float | None = None,
) -> tuple[int, float, MapElitesRecord | None]:
    candidate = ParetoCandidate(
        commit_hash=commit_hash,
        objective_values=tuple(float(value) for value in objective_values),
        objective_scores=tuple(float(value) for value in objective_scores),
        measures=tuple(float(value) for value in np.asarray(measures).reshape(-1)),
        timestamp=time.time() if timestamp is None else float(timestamp),
    )
    outcome = state.archive.add(candidate)
    sync_archive_indexes(state)
    if not outcome.retained:
        return 0, 0.0, None
    record = record_from_candidate(
        candidate=candidate,
        island_id=island_id,
        cell_index=outcome.cell_index,
    )
    return 1, float(len(outcome.removed_commit_hashes)), record


def add_batch(
    *,
    state: IslandState,
    island_id: str,
    commit_hashes: Sequence[str],
    objective_values: Sequence[Sequence[float]] | np.ndarray,
    objective_scores: Sequence[Sequence[float]] | np.ndarray,
    measures: Sequence[np.ndarray] | np.ndarray,
    timestamps: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = len(commit_hashes)
    values_matrix = np.asarray(objective_values, dtype=np.float64)
    scores_matrix = np.asarray(objective_scores, dtype=np.float64)
    measures_matrix = np.asarray(measures, dtype=np.float64)
    timestamp_vector = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    expected_objectives = state.archive.objective_count
    expected_measures = len(state.archive.dims)
    if values_matrix.shape != (batch_size, expected_objectives):
        raise ValueError(
            "Raw objective batch shape mismatch "
            f"(expected={(batch_size, expected_objectives)} got={values_matrix.shape})."
        )
    if scores_matrix.shape != values_matrix.shape:
        raise ValueError(
            "Dominance score batch shape mismatch "
            f"(expected={values_matrix.shape} got={scores_matrix.shape})."
        )
    if measures_matrix.shape != (batch_size, expected_measures):
        raise ValueError(
            "Behavior measures batch shape mismatch "
            f"(expected={(batch_size, expected_measures)} got={measures_matrix.shape})."
        )
    if timestamp_vector.shape != (batch_size,):
        raise ValueError(
            "Timestamp batch shape mismatch "
            f"(expected={(batch_size,)} got={timestamp_vector.shape})."
        )

    candidates = tuple(
        ParetoCandidate(
            commit_hash=str(commit_hash),
            objective_values=tuple(values_matrix[index]),
            objective_scores=tuple(scores_matrix[index]),
            measures=tuple(measures_matrix[index]),
            timestamp=float(timestamp_vector[index]),
        )
        for index, commit_hash in enumerate(commit_hashes)
    )
    outcomes = state.archive.add_many(candidates)
    sync_archive_indexes(state)
    statuses = np.asarray([outcome.status for outcome in outcomes], dtype=np.int64)
    values = np.asarray(
        [1.0 if outcome.retained else 0.0 for outcome in outcomes],
        dtype=np.float64,
    )
    return statuses, values


def sync_archive_indexes(state: IslandState) -> None:
    commit_to_index: dict[str, int] = {}
    index_to_commits: dict[int, tuple[str, ...]] = {}
    for cell_index in sorted(
        {int(value) for value in state.archive.data().get("index", ())}
    ):
        commits = tuple(
            candidate.commit_hash
            for candidate in state.archive.front(cell_index)
        )
        if not commits:
            continue
        index_to_commits[cell_index] = commits
        for commit_hash in commits:
            commit_to_index[commit_hash] = cell_index
    state.commit_to_index = commit_to_index
    state.index_to_commits = index_to_commits


def record_from_candidate(
    *,
    candidate: ParetoCandidate,
    island_id: str,
    cell_index: int,
) -> MapElitesRecord:
    return MapElitesRecord(
        commit_hash=candidate.commit_hash,
        island_id=island_id,
        cell_index=int(cell_index),
        objective_values=candidate.objective_values,
        objective_scores=candidate.objective_scores,
        measures=candidate.measures,
        timestamp=candidate.timestamp,
    )


def records_from_archive(
    archive: ParetoGridArchive,
    island_id: str,
) -> tuple[MapElitesRecord, ...]:
    index_by_commit = {
        candidate.commit_hash: cell_index
        for cell_index in sorted({int(value) for value in archive.data().get("index", ())})
        for candidate in archive.front(cell_index)
    }
    return tuple(
        record_from_candidate(
            candidate=candidate,
            island_id=island_id,
            cell_index=index_by_commit[candidate.commit_hash],
        )
        for candidate in archive.records()
    )


def build_archive_replace_payload(
    *,
    state: IslandState,
    island_id: str,
) -> tuple[SnapshotElite, ...]:
    return tuple(
        SnapshotElite(
            cell_index=record.cell_index,
            commit_hash=record.commit_hash,
            objective_values=record.objective_values,
            measures=record.measures,
            timestamp=record.timestamp,
        )
        for record in records_from_archive(state.archive, island_id)
    )
