"""Projection-driven archive rebuild helpers."""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np
from loguru import logger
from ribs.archives import GridArchive
from sqlalchemy.orm import Session

from loreley.config import Settings

from .dimension_reduction import FinalEmbedding, PCAProjection, PcaHistoryEntry, align_pca_projection
from .snapshot import purge_island_commit_mappings
from .types import IslandState, MapElitesRecord, Vector

log = logger.bind(module="map_elites.rebuild")

__all__ = [
    "pad_or_trim",
    "recompute_final_embedding",
    "seed_after_initial_fit",
    "rebuild_after_refit",
]


def pad_or_trim(vector: Sequence[float], *, target_dims: int) -> tuple[float, ...]:
    if not vector:
        return tuple(0.0 for _ in range(target_dims))
    if len(vector) >= target_dims:
        return tuple(float(v) for v in vector[:target_dims])
    padded = [float(v) for v in vector]
    padded.extend(0.0 for _ in range(target_dims - len(padded)))
    return tuple(padded)


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


def _rebuild_archive_from_batch(
    *,
    state: IslandState,
    island_id: str,
    candidate_commits: Sequence[str],
    candidate_fitnesses: Sequence[float],
    candidate_measures: Sequence[np.ndarray] | np.ndarray,
    candidate_timestamps: Sequence[float],
    commit_to_island: dict[str, str],
    build_archive: Callable[[], GridArchive],
    add_batch: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> int:
    purge_island_commit_mappings(commit_to_island, island_id)
    state.archive = build_archive()
    state.index_to_commit.clear()
    state.commit_to_index.clear()

    if not candidate_commits:
        return 0

    statuses = add_batch(
        state=state,
        island_id=island_id,
        commit_hashes=candidate_commits,
        objectives=candidate_fitnesses,
        measures=candidate_measures,
        timestamps=candidate_timestamps,
        commit_to_island=commit_to_island,
    )[0]
    return int(np.count_nonzero(statuses > 0))


def seed_after_initial_fit(
    *,
    state: IslandState,
    island_id: str,
    projection: PCAProjection,
    skip_commit_hash: str,
    snapshot_session: Session | None,
    settings: Settings,
    target_dims: int,
    commit_to_island: dict[str, str],
    load_commit_fitnesses: Callable[[Sequence[str], Session | None], Mapping[str, float]],
    clip_vector: Callable[[Sequence[float] | np.ndarray, IslandState], np.ndarray],
    build_archive: Callable[[], GridArchive],
    add_batch: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> None:
    """Populate the archive once the first PCA projection becomes available."""

    skip = str(skip_commit_hash or "").strip()
    candidates = [
        entry
        for entry in state.history
        if str(entry.commit_hash or "").strip()
        and str(entry.commit_hash or "").strip() != skip
    ]
    if not candidates:
        return

    commit_hashes = [str(entry.commit_hash).strip() for entry in candidates]
    fitnesses = dict(load_commit_fitnesses(commit_hashes, snapshot_session))

    inserted = 0
    skipped = 0
    timestamp = time.time()
    candidate_commits: list[str] = []
    candidate_fitnesses: list[float] = []
    candidate_measures: np.ndarray = np.asarray([], dtype=np.float64).reshape(0, int(target_dims))
    candidate_timestamps: list[float] = []

    def _fitness_key(entry: PcaHistoryEntry) -> float:
        commit = str(entry.commit_hash or "").strip()
        value = fitnesses.get(commit)
        if value is None:
            value = float(settings.mapelites_fitness_floor)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(settings.mapelites_fitness_floor)

    candidate_vectors: list[tuple[float, ...]] = []
    for entry in sorted(candidates, key=_fitness_key, reverse=True):
        commit = str(entry.commit_hash or "").strip()
        if not commit:
            continue

        fitness = fitnesses.get(commit)
        if fitness is None:
            fitness = float(settings.mapelites_fitness_floor)
        try:
            fitness_value = float(fitness)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not math.isfinite(fitness_value):
            skipped += 1
            continue

        if len(entry.vector) != projection.feature_count:
            skipped += 1
            continue
        candidate_vectors.append(entry.vector)
        candidate_commits.append(commit)
        candidate_fitnesses.append(fitness_value)

    reduced_matrix = np.asarray([], dtype=np.float64).reshape(0, int(target_dims))
    if candidate_vectors:
        try:
            vector_matrix = np.asarray(candidate_vectors, dtype=np.float64)
            projected = projection.transform_batch(vector_matrix)
        except ValueError:
            skipped += len(candidate_vectors)
            candidate_commits.clear()
            candidate_fitnesses.clear()
            candidate_vectors.clear()
            projected = np.asarray([], dtype=np.float64).reshape(0, int(projection.dimensions))

        if projected.size:
            projected_dims = int(projected.shape[1])
            if projected_dims >= target_dims:
                reduced_matrix = projected[:, :target_dims]
            else:
                reduced_matrix = np.zeros((projected.shape[0], target_dims), dtype=np.float64)
                reduced_matrix[:, :projected_dims] = projected

    if reduced_matrix.size:
        candidate_timestamps = [timestamp] * int(reduced_matrix.shape[0])
        candidate_measures = clip_vector(reduced_matrix, state)

    inserted = _rebuild_archive_from_batch(
        state=state,
        island_id=island_id,
        candidate_commits=candidate_commits,
        candidate_fitnesses=candidate_fitnesses,
        candidate_measures=candidate_measures,
        candidate_timestamps=candidate_timestamps,
        commit_to_island=commit_to_island,
        build_archive=build_archive,
        add_batch=add_batch,
    )

    log.info(
        "Seeded MAP-Elites archive after initial PCA fit (island={} inserted={} candidates={} skipped={})",
        island_id,
        inserted,
        len(candidates),
        skipped,
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
    commit_to_island: dict[str, str],
    load_commit_vectors: Callable[
        [str, Sequence[str], IslandState, Session | None],
        Mapping[str, tuple[float, ...]],
    ],
    clip_vector: Callable[[Sequence[float] | np.ndarray, IslandState], np.ndarray],
    build_archive: Callable[[], GridArchive],
    add_batch: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> FinalEmbedding:
    """Align the new PCA projection and rebuild the archive in the new coordinates."""

    if not previous_records:
        return recompute_final_embedding(
            current=current,
            projection=new_projection,
            target_dims=target_dims,
        )

    commit_hashes = tuple(
        str(record.commit_hash or "").strip()
        for record in previous_records
        if str(record.commit_hash or "").strip()
    )
    vectors = dict(load_commit_vectors(island_id, commit_hashes, state, snapshot_session))
    anchors = [vectors[h] for h in commit_hashes if h in vectors]
    aligned = align_pca_projection(
        projection=new_projection,
        reference=old_projection,
        anchors=anchors,
    )
    state.projection = aligned

    inserted = 0
    missing = 0
    candidate_vectors: list[tuple[float, ...]] = []
    candidate_commits: list[str] = []
    candidate_fitnesses: list[float] = []
    candidate_measures: np.ndarray = np.asarray([], dtype=np.float64).reshape(0, int(target_dims))
    candidate_timestamps: list[float] = []
    for record in sorted(previous_records, key=lambda item: float(item.fitness), reverse=True):
        commit = str(record.commit_hash or "").strip()
        if not commit:
            continue
        vec = vectors.get(commit)
        if not vec:
            missing += 1
            continue
        if len(vec) != aligned.feature_count:
            raise ValueError(
                "Stored commit vector dimensionality mismatch "
                f"(commit={commit} expected={aligned.feature_count} got={len(vec)})."
            )
        candidate_vectors.append(vec)
        candidate_commits.append(commit)
        candidate_fitnesses.append(float(record.fitness))
        candidate_timestamps.append(float(record.timestamp))

    reduced_matrix = np.asarray([], dtype=np.float64).reshape(0, int(target_dims))
    if candidate_vectors:
        vector_matrix = np.asarray(candidate_vectors, dtype=np.float64)
        projected = aligned.transform_batch(vector_matrix)
        projected_dims = int(projected.shape[1])
        if projected_dims >= target_dims:
            reduced_matrix = projected[:, :target_dims]
        else:
            reduced_matrix = np.zeros((projected.shape[0], target_dims), dtype=np.float64)
            reduced_matrix[:, :projected_dims] = projected

    if reduced_matrix.size:
        candidate_measures = clip_vector(reduced_matrix, state)

    inserted = _rebuild_archive_from_batch(
        state=state,
        island_id=island_id,
        candidate_commits=candidate_commits,
        candidate_fitnesses=candidate_fitnesses,
        candidate_measures=candidate_measures,
        candidate_timestamps=candidate_timestamps,
        commit_to_island=commit_to_island,
        build_archive=build_archive,
        add_batch=add_batch,
    )

    log.info(
        "Rebuilt MAP-Elites archive after PCA refit (island={} kept={} missing_vectors={})",
        island_id,
        inserted,
        missing,
    )
    return recompute_final_embedding(
        current=current,
        projection=aligned,
        target_dims=target_dims,
    )
