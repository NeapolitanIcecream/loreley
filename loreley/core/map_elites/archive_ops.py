"""Archive-centric MAP-Elites operations."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence, cast

import numpy as np
from loguru import logger
from ribs.archives import GridArchive

from loreley.config import Settings

from .snapshot import SnapshotCellUpsert, to_list
from .types import IslandState, MapElitesRecord, Vector

log = logger.bind(module="map_elites.archive_ops")

__all__ = [
    "build_feature_bounds",
    "build_archive",
    "clip_vector",
    "to_vector",
    "record_from_scalar_row",
    "records_from_store_data",
    "add_single",
    "add_batch",
    "build_archive_replace_payload",
]


def build_feature_bounds(*, target_dims: int) -> tuple[np.ndarray, np.ndarray]:
    dims = int(target_dims)
    lower = np.zeros(dims, dtype=np.float64)
    upper = np.ones(dims, dtype=np.float64)
    return lower, upper


def build_archive(
    *,
    settings: Settings,
    target_dims: int,
    cells_per_dim: int,
    lower_template: np.ndarray,
    upper_template: np.ndarray,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> GridArchive:
    dims = int(target_dims)
    lower = np.asarray(lower_bounds if lower_bounds is not None else lower_template, dtype=np.float64)
    upper = np.asarray(upper_bounds if upper_bounds is not None else upper_template, dtype=np.float64)
    if lower.shape[0] != dims or upper.shape[0] != dims:
        lower, upper = build_feature_bounds(target_dims=dims)

    ranges = tuple(zip(lower.tolist(), upper.tolist()))
    extra_fields = {
        "commit_hash": ((), object),
        "timestamp": ((), np.float64),
    }
    return GridArchive(
        solution_dim=dims,
        dims=tuple(int(cells_per_dim) for _ in range(dims)),
        ranges=ranges,
        learning_rate=settings.mapelites_archive_learning_rate,
        threshold_min=settings.mapelites_archive_threshold_min,
        epsilon=settings.mapelites_archive_epsilon,
        qd_score_offset=settings.mapelites_archive_qd_score_offset,
        extra_fields=extra_fields,
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

    # When clipping is enabled, keep descriptors within [-k, k] before mapping.
    if settings.mapelites_feature_clip:
        arr = np.clip(arr, -effective_radius, effective_radius)

    normalised = (arr + effective_radius) / (2.0 * effective_radius)

    # Only clamp to archive bounds when defensive clipping is on; otherwise
    # allow values outside [0, 1] to surface as archive insert failures.
    if settings.mapelites_feature_clip:
        return np.clip(normalised, state.lower_bounds, state.upper_bounds)
    return normalised


def add_single(
    *,
    state: IslandState,
    island_id: str,
    commit_hash: str,
    fitness: float,
    measures: np.ndarray,
    commit_to_island: dict[str, str],
    timestamp: float | None = None,
) -> tuple[int, float, MapElitesRecord | None]:
    archive = state.archive
    measures_batch = measures.reshape(1, -1)
    solution = measures_batch  # Store embedding itself as the solution payload.
    objective = np.asarray([fitness], dtype=np.float64)
    ts_value = time.time() if timestamp is None else float(timestamp)
    timestamp_batch = np.asarray([ts_value], dtype=np.float64)
    commit_field = np.asarray([commit_hash], dtype=object)

    cell_index = int(np.asarray(archive.index_of(measures_batch)).item())
    previous_commit = state.index_to_commit.get(cell_index)

    add_info = archive.add(
        solution,
        objective,
        measures_batch,
        commit_hash=commit_field,
        timestamp=timestamp_batch,
    )
    status = int(add_info["status"][0])
    delta = float(add_info["value"][0])

    if status <= 0:
        return status, delta, None

    record: MapElitesRecord | None = None
    retrieve_reason = ""
    retrieve_error: Exception | None = None
    try:
        occupied, data = archive.retrieve_single(measures)
        if occupied:
            record = record_from_scalar_row(
                cast(Mapping[str, Any], data),
                island_id,
            )
        else:
            retrieve_reason = "not_occupied"
    except Exception as exc:  # pragma: no cover - defensive
        retrieve_reason = "exception"
        retrieve_error = exc

    if record is None:
        bound = log.bind(
            event="mapelites.archive.retrieve_single_failed",
            reason=retrieve_reason or "unknown",
            island_id=island_id,
            cell_index=int(cell_index),
            commit_hash=str(commit_hash),
        )
        if retrieve_error is not None:
            bound = bound.opt(exception=retrieve_error)
        bound.error(
            "Archive add succeeded but retrieve_single() failed; returning synthetic record.",
        )

        vector = to_vector(measures)
        record = MapElitesRecord(
            commit_hash=str(commit_hash),
            island_id=island_id,
            cell_index=int(cell_index),
            fitness=float(fitness),
            measures=vector,
            solution=vector,
            timestamp=float(ts_value),
        )

    state.index_to_commit[cell_index] = commit_hash
    state.commit_to_index[commit_hash] = cell_index
    commit_to_island[commit_hash] = island_id
    if previous_commit and previous_commit != commit_hash:
        state.commit_to_index.pop(previous_commit, None)
        commit_to_island.pop(previous_commit, None)

    return status, delta, record


def add_batch(
    *,
    state: IslandState,
    island_id: str,
    commit_hashes: Sequence[str],
    objectives: Sequence[float],
    measures: Sequence[np.ndarray],
    timestamps: Sequence[float],
    commit_to_island: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a batch of elites and update commit mappings from add feedback."""

    batch_size = len(commit_hashes)
    if batch_size == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        )
    if len(objectives) != batch_size or len(measures) != batch_size or len(timestamps) != batch_size:
        raise ValueError(
            "Batch archive insertion payload length mismatch "
            f"(commits={batch_size} objectives={len(objectives)} "
            f"measures={len(measures)} timestamps={len(timestamps)})."
        )

    archive = state.archive
    measures_batch = np.asarray(measures, dtype=np.float64)
    if measures_batch.ndim != 2 or measures_batch.shape[0] != batch_size:
        raise ValueError(
            "Batch archive insertion measures shape mismatch "
            f"(expected=({batch_size}, dims) got={measures_batch.shape})."
        )
    objective_batch = np.asarray(objectives, dtype=np.float64).reshape(-1)
    timestamp_batch = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if objective_batch.shape[0] != batch_size or timestamp_batch.shape[0] != batch_size:
        raise ValueError(
            "Batch archive insertion objective/timestamp shape mismatch "
            f"(expected={batch_size} got_objective={objective_batch.shape[0]} "
            f"got_timestamp={timestamp_batch.shape[0]})."
        )

    commit_batch = np.asarray([str(commit or "").strip() for commit in commit_hashes], dtype=object)
    cell_indices = np.asarray(archive.index_of(measures_batch), dtype=np.int64).reshape(-1)
    if cell_indices.shape[0] != batch_size:
        raise RuntimeError(
            "GridArchive.index_of() returned unexpected index shape "
            f"(expected={batch_size} got={cell_indices.shape[0]})."
        )

    add_info = archive.add(
        measures_batch,
        objective_batch,
        measures_batch,
        commit_hash=commit_batch,
        timestamp=timestamp_batch,
    )
    statuses = np.asarray(add_info.get("status", ()), dtype=np.int64).reshape(-1)
    values = np.asarray(add_info.get("value", ()), dtype=np.float64).reshape(-1)
    if statuses.shape[0] != batch_size or values.shape[0] != batch_size:
        raise RuntimeError(
            "GridArchive.add() returned unexpected add_info shape "
            f"(expected={batch_size} got_status={statuses.shape[0]} got_value={values.shape[0]})."
        )

    winners_by_cell: dict[int, tuple[int, float, int]] = {}
    for idx, status_raw in enumerate(statuses):
        status = int(status_raw)
        if status <= 0:
            continue
        cell_index = int(cell_indices[idx])
        rank = (status, float(values[idx]), idx)
        previous = winners_by_cell.get(cell_index)
        if previous is None or (rank[0], rank[1]) > (previous[0], previous[1]):
            winners_by_cell[cell_index] = rank

    for cell_index, (_status, _value, idx) in winners_by_cell.items():
        commit_hash = str(commit_batch[idx] or "").strip()
        previous_commit = state.index_to_commit.get(cell_index)
        state.index_to_commit[cell_index] = commit_hash
        if commit_hash:
            state.commit_to_index[commit_hash] = cell_index
            commit_to_island[commit_hash] = island_id
        if previous_commit and previous_commit != commit_hash:
            state.commit_to_index.pop(previous_commit, None)
            commit_to_island.pop(previous_commit, None)

    return statuses, values


def build_archive_replace_payload(
    *,
    state: IslandState,
    island_id: str,
) -> tuple[SnapshotCellUpsert, ...]:
    if state.archive.empty:
        return tuple()
    data = state.archive.data()
    records = records_from_store_data(
        cast(Mapping[str, Any], data),
        island_id,
    )
    payload = [
        SnapshotCellUpsert(
            cell_index=int(record.cell_index),
            objective=float(record.fitness),
            measures=tuple(float(v) for v in record.measures),
            solution=tuple(float(v) for v in record.solution),
            commit_hash=str(record.commit_hash),
            timestamp=float(record.timestamp),
        )
        for record in records
        if record.commit_hash
    ]
    payload.sort(key=lambda item: int(item.cell_index))
    return tuple(payload)


def records_from_store_data(
    data: Mapping[str, Any],
    island_id: str,
) -> tuple[MapElitesRecord, ...]:
    if not data:
        return ()
    indices = to_list(data.get("index"))
    if not indices:
        return ()
    objectives = to_list(data.get("objective"))
    measures = to_list(data.get("measures"))
    solutions = to_list(data.get("solution"))
    commit_hashes = to_list(data.get("commit_hash"))
    timestamps = to_list(data.get("timestamp"))
    records: list[MapElitesRecord] = []
    for idx, cell_index in enumerate(indices):
        commit_hash = str(commit_hashes[idx]) if idx < len(commit_hashes) else ""
        fitness = float(objectives[idx]) if idx < len(objectives) else 0.0
        timestamp_value = (
            float(timestamps[idx]) if idx < len(timestamps) else time.time()
        )
        record = MapElitesRecord(
            commit_hash=commit_hash,
            island_id=island_id,
            cell_index=int(cell_index),
            fitness=fitness,
            measures=to_vector(measures[idx]) if idx < len(measures) else (),
            solution=to_vector(solutions[idx]) if idx < len(solutions) else (),
            timestamp=timestamp_value,
        )
        records.append(record)
    return tuple(records)


def record_from_scalar_row(data: Mapping[str, Any], island_id: str) -> MapElitesRecord:
    commit_raw = data.get("commit_hash")
    if isinstance(commit_raw, np.ndarray):
        commit_hash = str(commit_raw.item()) if commit_raw.size else ""
    elif isinstance(commit_raw, (list, tuple)):
        commit_hash = str(commit_raw[0]) if commit_raw else ""
    else:
        commit_hash = str(commit_raw or "")
    return MapElitesRecord(
        commit_hash=commit_hash,
        island_id=island_id,
        cell_index=int(data.get("index", -1)),
        fitness=float(data.get("objective", 0.0)),
        measures=to_vector(data.get("measures", ())),
        solution=to_vector(data.get("solution", ())),
        timestamp=float(data.get("timestamp", time.time())),
    )


def to_vector(values: Any) -> Vector:
    if values is None:
        return ()
    return tuple(float(v) for v in np.asarray(values).ravel())
