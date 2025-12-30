"""Orchestrate MAP-Elites archives for evolutionary commit exploration."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import numpy as np
from loguru import logger
from ribs.archives import GridArchive

from loreley.config import Settings, get_settings
from .code_embedding import CommitCodeEmbedding
from .dimension_reduction import (
    FinalEmbedding,
    PenultimateEmbedding,
    PCAProjection,
    reduce_commit_embeddings,
)
from .preprocess import ChangedFile, PreprocessedFile
from .repository_state_embedding import RepoStateEmbeddingStats, embed_repository_state
from .snapshot import (
    SnapshotBackend,
    SnapshotCellUpsert,
    SnapshotUpdate,
    apply_snapshot,
    build_snapshot_backend,
    to_list,
)
from .summarization_embedding import CommitSummaryEmbedding

if TYPE_CHECKING:  # pragma: no cover
    from .sampler import SupportsMapElitesRecord

log = logger.bind(module="map_elites.manager")

__all__ = [
    "CommitEmbeddingArtifacts",
    "MapElitesInsertionResult",
    "MapElitesManager",
    "MapElitesRecord",
]

Vector = tuple[float, ...]


@dataclass(slots=True, frozen=True)
class CommitEmbeddingArtifacts:
    """Lightweight container for intermediate embedding artifacts."""

    repo_state_stats: RepoStateEmbeddingStats | None
    preprocessed_files: tuple[PreprocessedFile, ...]
    code_embedding: CommitCodeEmbedding | None
    summary_embedding: CommitSummaryEmbedding | None
    final_embedding: FinalEmbedding | None

    @property
    def file_count(self) -> int:
        if self.repo_state_stats is not None:
            return int(self.repo_state_stats.files_aggregated)
        return len(self.preprocessed_files)

    @property
    def chunk_count(self) -> int:
        # Repo-state embeddings do not retain chunk-level artifacts.
        return 0


@dataclass(slots=True, frozen=True)
class MapElitesRecord:
    """Snapshot of a single elite stored inside an archive cell."""

    commit_hash: str
    island_id: str
    cell_index: int
    fitness: float
    measures: Vector
    solution: Vector
    timestamp: float

    @property
    def dimensions(self) -> int:
        return len(self.measures)


@dataclass(slots=True, frozen=True)
class MapElitesInsertionResult:
    """Wraps the outcome of adding a commit to the archive."""

    status: int
    delta: float
    record: MapElitesRecord | None
    artifacts: CommitEmbeddingArtifacts
    message: str | None = None

    @property
    def inserted(self) -> bool:
        return self.status > 0 and self.record is not None


@dataclass(slots=True)
class IslandState:
    """Mutable bookkeeping attached to each island."""

    archive: GridArchive
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    history: tuple[PenultimateEmbedding, ...] = field(default_factory=tuple)
    projection: PCAProjection | None = None
    commit_to_index: dict[str, int] = field(default_factory=dict)
    index_to_commit: dict[int, str] = field(default_factory=dict)


class MapElitesManager:
    """Run the embedding pipeline and maintain per-island MAP-Elites archives."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        repo_root: Path | None = None,
        experiment_id: uuid.UUID | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self._target_dims = max(1, self.settings.mapelites_dimensionality_target_dims)
        self._clip_radius = max(0.0, float(self.settings.mapelites_feature_truncation_k))
        if self._clip_radius == 0.0:
            self._clip_radius = 1.0
        self._cells_per_dim = max(2, self.settings.mapelites_archive_cells_per_dim)
        self._lower_template, self._upper_template = self._build_feature_bounds()
        self._grid_shape = tuple(self._cells_per_dim for _ in range(self._target_dims))
        self._archives: dict[str, IslandState] = {}
        self._commit_to_island: dict[str, str] = {}
        self._default_island = self.settings.mapelites_default_island_id or "default"
        # When provided, this experiment_id is used to scope persisted snapshots
        # in the map_elites_states table. If omitted, state persistence is disabled
        # and archives are kept in-memory only.
        exp_id: uuid.UUID | None = None
        if experiment_id is not None:
            if isinstance(experiment_id, uuid.UUID):
                exp_id = experiment_id
            else:
                exp_id = uuid.UUID(str(experiment_id))
        self._experiment_id: uuid.UUID | None = exp_id
        # Backend that knows how to load and persist archive snapshots.
        self._snapshot_backend: SnapshotBackend = build_snapshot_backend(self._experiment_id)

    @staticmethod
    def _infer_snapshot_target_dims(snapshot: Mapping[str, Any]) -> int | None:
        """Infer the archive dimensionality from a persisted snapshot payload.

        This is used to keep long-lived experiments readable even when the current
        process settings differ from the settings used when the snapshot was written.
        """

        if not snapshot:
            return None

        # Prefer archive entries because they directly encode the stored vector shapes.
        archive_entries = snapshot.get("archive")
        if isinstance(archive_entries, (list, tuple)) and archive_entries:
            for entry in archive_entries:
                if not isinstance(entry, Mapping):
                    continue
                measures = entry.get("measures")
                if isinstance(measures, (list, tuple)) and measures:
                    return len(measures)
                solution = entry.get("solution")
                if isinstance(solution, (list, tuple)) and solution:
                    return len(solution)

        for key in ("lower_bounds", "upper_bounds"):
            bounds = snapshot.get(key)
            if isinstance(bounds, (list, tuple)) and bounds:
                return len(bounds)
        return None

    def _adopt_target_dims(self, target_dims: int) -> None:
        target = max(1, int(target_dims))
        if target == self._target_dims:
            return
        self._target_dims = target
        self._lower_template, self._upper_template = self._build_feature_bounds()
        self._grid_shape = tuple(self._cells_per_dim for _ in range(self._target_dims))

    def ingest(
        self,
        *,
        commit_hash: str,
        changed_files: Sequence[ChangedFile | Mapping[str, object]] | None = None,
        metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        island_id: str | None = None,
        repo_root: Path | None = None,
        treeish: str | None = None,
        fitness_override: float | None = None,
    ) -> MapElitesInsertionResult:
        """Process a commit and attempt to insert it into the archive."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        working_dir = Path(repo_root or self.repo_root).resolve()
        embedding_mode = getattr(self.settings, "mapelites_embedding_mode", "repo_state")
        effective_treeish = (treeish or commit_hash).strip() or None

        log.info(
            "Ingesting commit {} for island {} (embedding_mode={} treeish={} treeish_source={})",
            commit_hash,
            effective_island,
            embedding_mode,
            effective_treeish or "<missing>",
            "explicit" if treeish else "defaulted_to_commit_hash",
        )

        update: SnapshotUpdate | None = None

        try:
            code_embedding, repo_stats = embed_repository_state(
                commit_hash=commit_hash,
                repo_root=working_dir,
                treeish=effective_treeish,
                settings=self.settings,
                # Prefer the configured backend; default is DB.
                cache_backend=(self.settings.mapelites_file_embedding_cache_backend or "db"),
            )
            if not code_embedding or not code_embedding.vector:
                artifacts = self._build_artifacts(repo_stats, (), None, None)
                message = "No eligible repository files produced an embedding."
                log.warning("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )
            summary_embedding = None

            final_embedding, history, projection = reduce_commit_embeddings(
                commit_hash=commit_hash,
                code_embedding=code_embedding,
                summary_embedding=summary_embedding,
                history=state.history,
                projection=state.projection,
                settings=self.settings,
            )
            state.history = history
            state.projection = projection

            # Persist PCA state incrementally even when the archive does not change.
            update = SnapshotUpdate(
                lower_bounds=state.lower_bounds.tolist(),
                upper_bounds=state.upper_bounds.tolist(),
                projection=state.projection,
                history_upsert=final_embedding.penultimate if final_embedding else None,
                history_seen_at=time.time(),
                history_limit=self._resolve_history_limit(),
            )

            artifacts = self._build_artifacts(repo_stats, (), code_embedding, final_embedding)

            if not final_embedding:
                message = "Unable to derive final embedding."
                log.warning("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            metrics_map = self._coerce_metrics(metrics)
            fitness = self._resolve_fitness(metrics_map, fitness_override)
            if fitness is None or not math.isfinite(fitness):
                message = "Fitness value is undefined; skipping archive update."
                log.warning("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            vector = self._clip_vector(final_embedding.vector, state)
            if vector.shape[0] != self._target_dims:
                message = (
                    "Final embedding dimensions mismatch with archive "
                    f"(expected {self._target_dims} got {vector.shape[0]})."
                )
                log.error("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            status, delta, record = self._add_to_archive(
                state=state,
                island_id=effective_island,
                commit_hash=commit_hash,
                fitness=fitness,
                measures=vector,
            )

            if update is not None and record is not None:
                update.cell_upsert = SnapshotCellUpsert(
                    cell_index=int(record.cell_index),
                    objective=float(record.fitness),
                    measures=tuple(float(v) for v in record.measures),
                    solution=tuple(float(v) for v in record.solution),
                    commit_hash=str(record.commit_hash),
                    timestamp=float(record.timestamp),
                )

            if record:
                log.info(
                    "Inserted commit {} into island {} (cell={} status={} Δ={:.4f})",
                    commit_hash,
                    effective_island,
                    record.cell_index,
                    status,
                    delta,
                )
            else:
                log.info(
                    "Commit {} did not improve island {} (status={} Δ={:.4f})",
                    commit_hash,
                    effective_island,
                    status,
                    delta,
                )

            return MapElitesInsertionResult(
                status=status,
                delta=delta,
                record=record,
                artifacts=artifacts,
                message=None if status else "Commit not inserted; objective below cell threshold.",
            )
        finally:
            self._persist_island_state(effective_island, state, update=update)

    def get_records(
        self,
        island_id: str | None = None,
    ) -> tuple["SupportsMapElitesRecord", ...]:
        """Return all elites for a given island."""
        effective_island = island_id or self._default_island
        # Lazily initialise and restore snapshots so that callers (UI, scheduler)
        # can observe persisted archives without requiring a prior ingest call.
        state = self._ensure_island(effective_island)
        if state.archive.empty:
            return ()
        data = state.archive.data()
        return self._records_from_store_data(
            cast(Mapping[str, Any], data),
            effective_island,
        )

    def sample_records(
        self,
        island_id: str | None = None,
        *,
        count: int = 1,
    ) -> tuple[MapElitesRecord, ...]:
        """Randomly sample elites for downstream planning."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        if state.archive.empty:
            return ()
        sampled = state.archive.sample_elites(max(1, count))
        return self._records_from_store_data(
            cast(Mapping[str, Any], sampled),
            effective_island,
        )

    def clear_island(self, island_id: str | None = None) -> None:
        """Reset an island archive and clear associated history."""
        effective_island = island_id or self._default_island
        state = self._archives.get(effective_island)
        if not state:
            return
        state.archive.clear()
        state.history = tuple()
        state.projection = None
        for commit in tuple(state.commit_to_index.keys()):
            self._commit_to_island.pop(commit, None)
        state.commit_to_index.clear()
        state.index_to_commit.clear()
        log.info("Cleared MAP-Elites state for island {}", effective_island)
        update = SnapshotUpdate(
            lower_bounds=state.lower_bounds.tolist(),
            upper_bounds=state.upper_bounds.tolist(),
            projection=None,
            clear=True,
            history_seen_at=time.time(),
            history_limit=self._resolve_history_limit(),
        )
        self._persist_island_state(effective_island, state, update=update)

    def describe_island(self, island_id: str | None = None) -> dict[str, Any]:
        """Return basic stats for observability dashboards."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        archive = state.archive
        stats = archive.stats
        best = getattr(stats, "objective_max", None)
        if best is None:
            best = getattr(stats, "obj_max", None)
        return {
            "island_id": effective_island,
            "occupied": int(getattr(stats, "num_elites", 0)),
            "cells": int(np.prod(getattr(archive, "dims", self._grid_shape))),
            "qd_score": float(getattr(stats, "qd_score", 0.0)),
            "best_fitness": float(best or 0.0),
        }

    def _add_to_archive(
        self,
        *,
        state: IslandState,
        island_id: str,
        commit_hash: str,
        fitness: float,
        measures: np.ndarray,
    ) -> tuple[int, float, MapElitesRecord | None]:
        archive = state.archive
        measures_batch = measures.reshape(1, -1)
        solution = measures_batch  # Store embedding itself as the solution payload.
        objective = np.asarray([fitness], dtype=np.float64)
        timestamp = np.asarray([time.time()], dtype=np.float64)
        commit_field = np.asarray([commit_hash], dtype=object)

        cell_index = int(np.asarray(archive.index_of(measures_batch)).item())
        previous_commit = state.index_to_commit.get(cell_index)

        add_info = archive.add(
            solution,
            objective,
            measures_batch,
            commit_hash=commit_field,
            timestamp=timestamp,
        )
        status = int(add_info["status"][0])
        delta = float(add_info["value"][0])

        if status <= 0:
            return status, delta, None

        occupied, data = archive.retrieve_single(measures)
        if not occupied:
            log.error(
                "Archive reported success but retrieval failed for commit {} on island {}",
                commit_hash,
                island_id,
            )
            return status, delta, None

        record = self._record_from_scalar_row(
            cast(Mapping[str, Any], data),
            island_id,
        )
        state.index_to_commit[cell_index] = commit_hash
        state.commit_to_index[commit_hash] = cell_index
        self._commit_to_island[commit_hash] = island_id
        if previous_commit and previous_commit != commit_hash:
            state.commit_to_index.pop(previous_commit, None)
            self._commit_to_island.pop(previous_commit, None)

        return status, delta, record

    def _records_from_store_data(
        self,
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
                measures=self._to_vector(measures[idx]) if idx < len(measures) else (),
                solution=self._to_vector(solutions[idx]) if idx < len(solutions) else (),
                timestamp=timestamp_value,
            )
            records.append(record)
        return tuple(records)

    @staticmethod
    def _record_from_scalar_row(data: Mapping[str, Any], island_id: str) -> MapElitesRecord:
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
            measures=MapElitesManager._to_vector(data.get("measures", ())),
            solution=MapElitesManager._to_vector(data.get("solution", ())),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def _ensure_island(self, island_id: str) -> IslandState:
        state = self._archives.get(island_id)
        if state:
            return state

        snapshot = self._snapshot_backend.load(island_id)
        snapshot_dims = self._infer_snapshot_target_dims(snapshot) if snapshot else None
        effective_dims = self._target_dims
        if snapshot_dims and snapshot_dims != self._target_dims:
            if not self._archives:
                log.warning(
                    "Adopting snapshot dimensionality for experiment {} island {} (settings_dims={} snapshot_dims={})",
                    self._experiment_id,
                    island_id,
                    self._target_dims,
                    snapshot_dims,
                )
                self._adopt_target_dims(snapshot_dims)
                effective_dims = self._target_dims
            else:
                log.warning(
                    "Snapshot dimensionality differs from current manager configuration; building archive per-island "
                    "(experiment={} island={} settings_dims={} snapshot_dims={})",
                    self._experiment_id,
                    island_id,
                    self._target_dims,
                    snapshot_dims,
                )
                effective_dims = int(snapshot_dims)

        if effective_dims == self._target_dims:
            archive = self._build_archive()
            lower_template = self._lower_template
            upper_template = self._upper_template
        else:
            lower_template, upper_template = self._build_feature_bounds(target_dims=effective_dims)
            archive = self._build_archive(
                target_dims=effective_dims,
                lower_bounds=lower_template,
                upper_bounds=upper_template,
            )

        state = IslandState(
            archive=archive,
            lower_bounds=np.asarray(lower_template, dtype=np.float64).copy(),
            upper_bounds=np.asarray(upper_template, dtype=np.float64).copy(),
        )
        if snapshot:
            apply_snapshot(
                state=state,
                snapshot=snapshot,
                island_id=island_id,
                commit_to_island=self._commit_to_island,
            )
        self._archives[island_id] = state
        log.info(
            "Initialized MAP-Elites archive for island {} (cells={} dims={})",
            island_id,
            int(np.prod(getattr(archive, "dims", self._grid_shape))),
            int(len(getattr(archive, "dims", self._grid_shape))),
        )
        return state

    def _build_feature_bounds(self, *, target_dims: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        dims = int(target_dims) if target_dims is not None else int(self._target_dims)
        lower = np.zeros(dims, dtype=np.float64)
        upper = np.ones(dims, dtype=np.float64)
        return lower, upper

    def _build_archive(
        self,
        *,
        target_dims: int | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> GridArchive:
        dims = int(target_dims) if target_dims is not None else int(self._target_dims)
        lower = np.asarray(lower_bounds if lower_bounds is not None else self._lower_template, dtype=np.float64)
        upper = np.asarray(upper_bounds if upper_bounds is not None else self._upper_template, dtype=np.float64)
        if lower.shape[0] != dims or upper.shape[0] != dims:
            lower, upper = self._build_feature_bounds(target_dims=dims)

        ranges = tuple(zip(lower.tolist(), upper.tolist()))
        extra_fields = {
            "commit_hash": ((), object),
            "timestamp": ((), np.float64),
        }
        return GridArchive(
            solution_dim=dims,
            dims=tuple(self._cells_per_dim for _ in range(dims)),
            ranges=ranges,
            learning_rate=self.settings.mapelites_archive_learning_rate,
            threshold_min=self.settings.mapelites_archive_threshold_min,
            epsilon=self.settings.mapelites_archive_epsilon,
            qd_score_offset=self.settings.mapelites_archive_qd_score_offset,
            extra_fields=extra_fields,
        )

    def _clip_vector(self, vector: Vector, state: IslandState) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        clip_radius = self._clip_radius
        if clip_radius <= 0.0:
            clip_radius = 1.0

        # When clipping is enabled, keep descriptors within [-k, k] before mapping.
        if self.settings.mapelites_feature_clip:
            arr = np.clip(arr, -clip_radius, clip_radius)

        normalised = (arr + clip_radius) / (2.0 * clip_radius)

        # Only clamp to archive bounds when defensive clipping is on; otherwise
        # allow values outside [0, 1] to surface as archive insert failures.
        if self.settings.mapelites_feature_clip:
            return np.clip(normalised, state.lower_bounds, state.upper_bounds)
        return normalised

    def _resolve_fitness(
        self,
        metrics: Mapping[str, float],
        override: float | None,
    ) -> float | None:
        if override is not None:
            return float(override)
        metric_name = self.settings.mapelites_fitness_metric
        if not metric_name:
            return None
        value = metrics.get(metric_name)
        if value is None:
            log.warning(
                "Missing metric {!r}; using configured floor {}",
                metric_name,
                self.settings.mapelites_fitness_floor,
            )
            return self.settings.mapelites_fitness_floor
        direction = 1.0 if self.settings.mapelites_fitness_higher_is_better else -1.0
        return float(value) * direction

    @staticmethod
    def _to_vector(values: Any) -> Vector:
        if values is None:
            return ()
        return tuple(float(v) for v in np.asarray(values).ravel())

    def _coerce_metrics(
        self,
        metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    ) -> dict[str, float]:
        if metrics is None:
            return {}
        if isinstance(metrics, Mapping):
            result: dict[str, float] = {}
            for key, value in metrics.items():
                numeric = self._maybe_float(value)
                if numeric is None:
                    continue
                result[str(key)] = numeric
            return result
        aggregated: dict[str, float] = {}
        for entry in metrics:
            if hasattr(entry, "name") and hasattr(entry, "value"):
                value = getattr(entry, "value")
                numeric = self._maybe_float(value)
                if numeric is not None:
                    aggregated[str(getattr(entry, "name"))] = numeric
                continue
            if isinstance(entry, Mapping):
                name = entry.get("name") or entry.get("metric") or entry.get("key")
                value = entry.get("value")
                if not name:
                    continue
                numeric = self._maybe_float(value)
                if numeric is not None:
                    aggregated[str(name)] = numeric
        return aggregated

    @staticmethod
    def _build_artifacts(
        repo_state_stats: RepoStateEmbeddingStats | None,
        preprocessed: Sequence[PreprocessedFile],
        code_embedding: CommitCodeEmbedding | None,
        final_embedding: FinalEmbedding | None,
    ) -> CommitEmbeddingArtifacts:
        return CommitEmbeddingArtifacts(
            repo_state_stats=repo_state_stats,
            preprocessed_files=tuple(preprocessed),
            code_embedding=code_embedding,
            summary_embedding=None,
            final_embedding=final_embedding,
        )

    def _persist_island_state(
        self,
        island_id: str,
        state: IslandState | None,
        *,
        update: SnapshotUpdate | None,
    ) -> None:
        """Persist incremental snapshot updates for an island when enabled."""

        if not state or self._experiment_id is None:
            return
        if update is None:
            return
        self._snapshot_backend.apply_update(island_id, state=state, update=update)

    def _resolve_history_limit(self) -> int:
        """Return the bounded history window size used by the PCA reducer."""

        min_fit = max(
            2,
            int(self.settings.mapelites_dimensionality_min_fit_samples),
            int(self.settings.mapelites_feature_normalization_warmup_samples),
        )
        return max(
            min_fit,
            int(self.settings.mapelites_dimensionality_history_size),
        )

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
