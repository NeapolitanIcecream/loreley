"""Projection-driven Pareto archive rebuild helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from .dimension_reduction import (
    FinalEmbedding,
    PCAProjection,
    align_pca_projection,
)
from .objectives import ResolvedObjectives
from .pareto_archive import ParetoGridArchive
from .types import IslandState, MapElitesRecord

log = logger.bind(module="map_elites.rebuild")


@dataclass(slots=True, frozen=True)
class _ArchiveRebuildBatch:
    commit_hashes: Sequence[str]
    objective_values: Sequence[Sequence[float]]
    objective_scores: Sequence[Sequence[float]]
    measures: Sequence[np.ndarray] | np.ndarray
    timestamps: Sequence[float]


__all__ = [
    "pad_or_trim",
    "recompute_final_embedding",
    "seed_after_initial_fit",
    "rebuild_after_refit",
]


def pad_or_trim(
    vector: Sequence[float],
    *,
    target_dims: int,
) -> tuple[float, ...]:
    if len(vector) >= target_dims:
        return tuple(float(value) for value in vector[:target_dims])
    return (
        *tuple(float(value) for value in vector),
        *tuple(0.0 for _ in range(target_dims - len(vector))),
    )


def recompute_final_embedding(
    *,
    current: FinalEmbedding,
    projection: PCAProjection | None,
    target_dims: int,
) -> FinalEmbedding:
    if projection is None:
        return current
    reduced = pad_or_trim(
        projection.transform(current.history_entry.vector),
        target_dims=target_dims,
    )
    return FinalEmbedding(
        commit_hash=current.commit_hash,
        vector=reduced,
        dimensions=len(reduced),
        history_entry=current.history_entry,
        projection=projection,
    )


def _replace_archive_from_batch(
    *,
    state: IslandState,
    island_id: str,
    batch: _ArchiveRebuildBatch,
    build_archive: Callable[[], ParetoGridArchive],
    add_batch: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> int:
    replacement = IslandState(
        archive=build_archive(),
        lower_bounds=state.lower_bounds,
        upper_bounds=state.upper_bounds,
        history=state.history,
        projection=state.projection,
        samples_since_fit=state.samples_since_fit,
    )
    retained = 0
    if batch.commit_hashes:
        statuses, _ = add_batch(
            state=replacement,
            island_id=island_id,
            commit_hashes=batch.commit_hashes,
            objective_values=batch.objective_values,
            objective_scores=batch.objective_scores,
            measures=batch.measures,
            timestamps=batch.timestamps,
        )
        retained = int(np.count_nonzero(statuses > 0))
    state.archive = replacement.archive
    state.commit_to_index = replacement.commit_to_index
    state.index_to_commits = replacement.index_to_commits
    return retained


def seed_after_initial_fit(
    *,
    state: IslandState,
    island_id: str,
    projection: PCAProjection,
    skip_commit_hash: str,
    snapshot_session: Session | None,
    target_dims: int,
    load_commit_objectives: Callable[
        [Sequence[str], Session | None],
        Mapping[str, ResolvedObjectives],
    ],
    clip_vector: Callable[[Sequence[float] | np.ndarray, IslandState], np.ndarray],
    build_archive: Callable[[], ParetoGridArchive],
    add_batch: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    skip = str(skip_commit_hash or "").strip()
    candidates = tuple(
        entry
        for entry in state.history
        if str(entry.commit_hash or "").strip()
        and str(entry.commit_hash or "").strip() != skip
    )
    commit_hashes = tuple(str(entry.commit_hash).strip() for entry in candidates)
    objective_by_commit = dict(load_commit_objectives(commit_hashes, snapshot_session))
    eligible = tuple(
        entry
        for entry in candidates
        if str(entry.commit_hash).strip() in objective_by_commit
        and len(entry.vector) == projection.feature_count
    )
    if not eligible:
        return

    vector_matrix = np.asarray([entry.vector for entry in eligible], dtype=np.float64)
    projected = projection.transform_batch(vector_matrix)
    reduced = _fit_dimensions(projected, target_dims=target_dims)
    measures = clip_vector(reduced, state)
    commits = tuple(str(entry.commit_hash).strip() for entry in eligible)
    resolved = tuple(objective_by_commit[commit] for commit in commits)
    inserted = _replace_archive_from_batch(
        state=state,
        island_id=island_id,
        batch=_ArchiveRebuildBatch(
            commit_hashes=commits,
            objective_values=tuple(item.values for item in resolved),
            objective_scores=tuple(item.scores for item in resolved),
            measures=measures,
            timestamps=(time.time(),) * len(commits),
        ),
        build_archive=build_archive,
        add_batch=add_batch,
    )
    log.info(
        "Seeded Pareto archive after initial PCA fit "
        "(island={} retained={} eligible={} skipped={})",
        island_id,
        inserted,
        len(eligible),
        len(candidates) - len(eligible),
    )


def rebuild_after_refit(
    *,
    state: IslandState,
    island_id: str,
    current: FinalEmbedding,
    old_projection: PCAProjection,
    new_projection: PCAProjection,
    previous_records: Sequence[MapElitesRecord],
    snapshot_session: Session | None,
    target_dims: int,
    load_commit_vectors: Callable[
        [str, Sequence[str], IslandState, Session | None],
        Mapping[str, tuple[float, ...]],
    ],
    clip_vector: Callable[[Sequence[float] | np.ndarray, IslandState], np.ndarray],
    build_archive: Callable[[], ParetoGridArchive],
    add_batch: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> FinalEmbedding:
    if not previous_records:
        return recompute_final_embedding(
            current=current,
            projection=new_projection,
            target_dims=target_dims,
        )
    commit_hashes = tuple(record.commit_hash for record in previous_records)
    vectors = dict(
        load_commit_vectors(
            island_id,
            commit_hashes,
            state,
            snapshot_session,
        )
    )
    missing_commits = tuple(
        commit_hash for commit_hash in commit_hashes if commit_hash not in vectors
    )
    if missing_commits:
        raise ValueError(
            "Cannot rebuild Pareto archive after PCA refit; stored vectors are "
            f"missing (island={island_id} commits={missing_commits})."
        )
    for commit_hash in commit_hashes:
        vector = vectors[commit_hash]
        if len(vector) != new_projection.feature_count:
            raise ValueError(
                "Stored commit vector dimensionality mismatch "
                f"(commit={commit_hash} expected={new_projection.feature_count} "
                f"got={len(vector)})."
            )
    aligned = align_pca_projection(
        projection=new_projection,
        reference=old_projection,
        anchors=[vectors[commit] for commit in commit_hashes],
    )
    state.projection = aligned
    projected = aligned.transform_batch(
        np.asarray(
            [vectors[record.commit_hash] for record in previous_records],
            dtype=np.float64,
        )
    )
    measures = clip_vector(
        _fit_dimensions(projected, target_dims=target_dims),
        state,
    )
    inserted = _replace_archive_from_batch(
        state=state,
        island_id=island_id,
        batch=_ArchiveRebuildBatch(
            commit_hashes=tuple(record.commit_hash for record in previous_records),
            objective_values=tuple(
                record.objective_values for record in previous_records
            ),
            objective_scores=tuple(
                record.objective_scores for record in previous_records
            ),
            measures=measures,
            timestamps=tuple(record.timestamp for record in previous_records),
        ),
        build_archive=build_archive,
        add_batch=add_batch,
    )
    log.info(
        "Rebuilt Pareto archive after PCA refit "
        "(island={} retained={} previous={})",
        island_id,
        inserted,
        len(previous_records),
    )
    return recompute_final_embedding(
        current=current,
        projection=aligned,
        target_dims=target_dims,
    )


def _fit_dimensions(matrix: np.ndarray, *, target_dims: int) -> np.ndarray:
    if matrix.shape[1] >= target_dims:
        return matrix[:, :target_dims]
    padded = np.zeros((matrix.shape[0], target_dims), dtype=np.float64)
    padded[:, : matrix.shape[1]] = matrix
    return padded
