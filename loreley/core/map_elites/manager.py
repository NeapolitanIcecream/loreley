"""Orchestrate MAP-Elites archives for evolutionary commit exploration."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from loreley.config import Settings, get_settings, resolve_default_island_id

from .archive_ops import (
    add_batch as archive_add_batch,
    add_single as archive_add_single,
    build_archive as archive_build_archive,
    build_archive_replace_payload as archive_build_archive_replace_payload,
    build_feature_bounds as archive_build_feature_bounds,
    clip_vector as archive_clip_vector,
    record_from_scalar_row as archive_record_from_scalar_row,
    records_from_store_data as archive_records_from_store_data,
)
from .code_embedding import CommitCodeEmbedding
from .db_ops import (
    iter_query_batches as db_iter_query_batches,
    load_commit_fitnesses as db_load_commit_fitnesses,
    load_commit_vectors as db_load_commit_vectors,
)
from .dimension_reduction import (
    DimensionReducer,
    FinalEmbedding,
    PCAProjection,
    reduce_commit_embeddings,
    resolve_pca_history_limit,
)
from .preprocess import PreprocessedFile
from .rebuild import (
    pad_or_trim as rebuild_pad_or_trim,
    recompute_final_embedding as rebuild_recompute_final_embedding,
    rebuild_after_refit,
    seed_after_initial_fit,
)
from .repository_state_embedding import RepoStateEmbeddingStats, embed_repository_state_incremental
from .snapshot import DatabaseSnapshotStore, SnapshotCellUpsert, SnapshotUpdate, apply_snapshot
from .types import (
    CommitEmbeddingArtifacts,
    IslandState,
    MapElitesInsertionResult,
    MapElitesRecord,
    Vector,
)

log = logger.bind(module="map_elites.manager")

__all__ = [
    "CommitEmbeddingArtifacts",
    "MapElitesInsertionResult",
    "MapElitesManager",
    "MapElitesRecord",
]


class MapElitesManager:
    """Run the embedding pipeline and maintain per-island MAP-Elites archives."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        repo_root: Path | None = None,
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
        self._reducers: dict[str, DimensionReducer] = {}
        self._commit_to_island: dict[str, str] = {}
        self._default_island = resolve_default_island_id(self.settings)
        self._snapshot_store = DatabaseSnapshotStore()
        self._ingest_info_log_every = int(self.settings.mapelites_ingest_info_log_every)
        self._ingest_invocations = 0

    @staticmethod
    def _infer_snapshot_target_dims(snapshot: Mapping[str, Any]) -> int | None:
        """Infer the archive dimensionality from a persisted snapshot payload."""
        if not snapshot:
            return None

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

    def ingest(
        self,
        *,
        commit_hash: str,
        metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        island_id: str | None = None,
        repo_root: Path | None = None,
        fitness_override: float | None = None,
        snapshot_session: Session | None = None,
    ) -> MapElitesInsertionResult:
        """Process a commit and attempt to insert it into the archive."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        working_dir = Path(repo_root or self.repo_root).resolve()
        ingest_index = self._next_ingest_invocation()
        emit_sampled_info = self._should_emit_sampled_ingest_info(ingest_index)

        if emit_sampled_info:
            log.info(
                "Ingesting commit {} for island {}",
                commit_hash,
                effective_island,
            )

        update: SnapshotUpdate | None = None
        result: MapElitesInsertionResult | None = None
        archive_replace_needed = False
        ingest_started_at = time.perf_counter()
        aggregate_hit_count = 0
        incremental_count = 0
        embedding_cache_miss_count = 0
        pca_fit_ms = 0.0
        pca_refit_ms = 0.0
        archive_add_ms = 0.0
        snapshot_apply_ms = 0.0
        did_initial_fit = False
        did_refit = False

        def _skip(
            *,
            message: str,
            artifacts: CommitEmbeddingArtifacts,
            log_level: str = "warning",
        ) -> MapElitesInsertionResult:
            if log_level == "info":
                if emit_sampled_info:
                    log.info("{} {}", message, commit_hash)
            elif log_level == "error":
                log.error("{} {}", message, commit_hash)
            else:
                log.warning("{} {}", message, commit_hash)
            return MapElitesInsertionResult(
                status=0,
                delta=0.0,
                record=None,
                artifacts=artifacts,
                message=message,
            )

        try:
            code_embedding, repo_stats = embed_repository_state_incremental(
                commit_hash=commit_hash,
                repo_root=working_dir,
                settings=self.settings,
            )
            source = str(getattr(repo_stats, "source", "unknown") or "unknown").strip()
            if source == "aggregate_hit":
                aggregate_hit_count += 1
            elif source == "incremental":
                incremental_count += 1
            cache_misses = int(getattr(repo_stats, "cache_misses", 0) or 0)
            if cache_misses < 0:
                raise ValueError(
                    "Repo-state embedding reported a negative cache_misses value "
                    f"(commit={commit_hash} cache_misses={cache_misses})."
                )
            embedding_cache_miss_count += cache_misses
            if not code_embedding or not code_embedding.vector:
                artifacts = self._build_artifacts(repo_stats, (), None, None)
                result = _skip(
                    message="No eligible repository files produced an embedding.",
                    artifacts=artifacts,
                )
                return result

            old_projection = state.projection
            reducer = self._reducers.get(effective_island)
            if reducer is None:
                reducer = DimensionReducer(
                    settings=self.settings,
                    history=state.history,
                    projection=state.projection,
                    samples_since_fit=state.samples_since_fit,
                )
                self._reducers[effective_island] = reducer
            pca_fit_started_at = time.perf_counter()
            final_embedding, history, projection, samples_since_fit = reduce_commit_embeddings(
                commit_hash=commit_hash,
                code_embedding=code_embedding,
                history=state.history,
                projection=state.projection,
                samples_since_fit=state.samples_since_fit,
                settings=self.settings,
                reducer=reducer,
            )
            pca_fit_ms += (time.perf_counter() - pca_fit_started_at) * 1000.0
            state.history = history
            state.projection = projection
            state.samples_since_fit = samples_since_fit

            did_initial_fit = old_projection is None and state.projection is not None
            did_refit = (
                old_projection is not None
                and state.projection is not None
                and state.projection.epoch != old_projection.epoch
            )
            if (
                did_refit
                and final_embedding is not None
                and old_projection is not None
                and state.projection is not None
            ):
                pca_refit_started_at = time.perf_counter()
                final_embedding = self._rebuild_after_projection_refit(
                    state=state,
                    island_id=effective_island,
                    current=final_embedding,
                    old_projection=old_projection,
                    new_projection=state.projection,
                    snapshot_session=snapshot_session,
                )
                pca_refit_ms += (time.perf_counter() - pca_refit_started_at) * 1000.0
                if effective_island in self._reducers:
                    self._reducers[effective_island].set_projection(state.projection)

            update = SnapshotUpdate(
                lower_bounds=state.lower_bounds.tolist(),
                upper_bounds=state.upper_bounds.tolist(),
                projection=state.projection,
                history_upsert=final_embedding.history_entry if final_embedding else None,
                history_seen_at=time.time(),
                samples_since_fit=state.samples_since_fit,
            )
            artifacts = self._build_artifacts(repo_stats, (), code_embedding, final_embedding)

            if not final_embedding:
                result = _skip(
                    message="Unable to derive final embedding.",
                    artifacts=artifacts,
                )
                return result

            if did_initial_fit and state.projection is not None:
                self._seed_archive_after_initial_fit(
                    state=state,
                    island_id=effective_island,
                    projection=state.projection,
                    skip_commit_hash=commit_hash,
                    snapshot_session=snapshot_session,
                )

            if state.projection is None:
                message = "PCA warmup: projection is not ready; skipping archive update."
                if emit_sampled_info:
                    log.info("{} island={} commit={}", message, effective_island, commit_hash)
                result = MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )
                return result

            archive_replace_needed = bool(did_refit or did_initial_fit)

            metrics_map = self._coerce_metrics(metrics)
            fitness = self._resolve_fitness(metrics_map, fitness_override)
            if fitness is None or not math.isfinite(fitness):
                result = _skip(
                    message="Fitness value is undefined; skipping archive update.",
                    artifacts=artifacts,
                )
                return result

            vector = self._clip_vector(final_embedding.vector, state)
            if vector.shape[0] != self._target_dims:
                message = (
                    "Final embedding dimensions mismatch with archive "
                    f"(expected {self._target_dims} got {vector.shape[0]})."
                )
                result = _skip(
                    message=message,
                    artifacts=artifacts,
                    log_level="error",
                )
                return result

            archive_add_started_at = time.perf_counter()
            status, delta, record = self._add_to_archive(
                state=state,
                island_id=effective_island,
                commit_hash=commit_hash,
                fitness=fitness,
                measures=vector,
            )
            archive_add_ms += (time.perf_counter() - archive_add_started_at) * 1000.0

            if update is not None and record is not None and not archive_replace_needed:
                update.cell_upsert = SnapshotCellUpsert(
                    cell_index=int(record.cell_index),
                    objective=float(record.fitness),
                    measures=tuple(float(v) for v in record.measures),
                    solution=tuple(float(v) for v in record.solution),
                    commit_hash=str(record.commit_hash),
                    timestamp=float(record.timestamp),
                )

            if record:
                if emit_sampled_info:
                    log.info(
                        "Inserted commit {} into island {} (cell={} status={} Δ={:.4f})",
                        commit_hash,
                        effective_island,
                        record.cell_index,
                        status,
                        delta,
                    )
            elif status < 0:
                log.warning(
                    "Archive rejected commit {} for island {} (status={} Δ={:.4f})",
                    commit_hash,
                    effective_island,
                    status,
                    delta,
                )
            else:
                if emit_sampled_info:
                    log.info(
                        "Commit {} did not improve island {} (status={} Δ={:.4f})",
                        commit_hash,
                        effective_island,
                        status,
                        delta,
                    )

            message: str | None = None
            if status <= 0:
                if status == 0:
                    message = "Commit not inserted; objective below cell threshold."
                else:
                    message = f"Commit not inserted; archive rejected insertion (status={status})."
            elif record is None:
                message = "Archive reported insertion success but no record could be retrieved."

            result = MapElitesInsertionResult(
                status=status,
                delta=delta,
                record=record,
                artifacts=artifacts,
                message=message,
            )
            return result
        finally:
            snapshot_apply_started_at = time.perf_counter()
            try:
                if update is not None and archive_replace_needed and update.archive_replace is None:
                    update.archive_replace = self._build_archive_replace_payload(
                        state=state,
                        island_id=effective_island,
                    )
                self._persist_island_state(
                    effective_island,
                    state,
                    update=update,
                    session=snapshot_session,
                )
            finally:
                snapshot_apply_ms += (time.perf_counter() - snapshot_apply_started_at) * 1000.0
                emit_stage_metrics = (
                    emit_sampled_info
                    or result is None
                    or did_initial_fit
                    or did_refit
                    or (result is not None and result.status < 0)
                )
                if emit_stage_metrics:
                    log.bind(
                        commit_hash=commit_hash,
                        island_id=effective_island,
                        aggregate_hit_count=aggregate_hit_count,
                        incremental_count=incremental_count,
                        embedding_cache_miss_count=embedding_cache_miss_count,
                        pca_fit_ms=round(pca_fit_ms, 3),
                        pca_refit_ms=round(pca_refit_ms, 3),
                        archive_add_ms=round(archive_add_ms, 3),
                        snapshot_apply_ms=round(snapshot_apply_ms, 3),
                        ingest_total_ms=round((time.perf_counter() - ingest_started_at) * 1000.0, 3),
                        did_initial_fit=did_initial_fit,
                        did_refit=did_refit,
                        status_code=result.status if result is not None else None,
                    ).info("MAP-Elites ingest stage metrics")

    def get_records(
        self,
        island_id: str | None = None,
    ) -> tuple[MapElitesRecord, ...]:
        """Return all elites for a given island."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        if state.archive.empty:
            return ()
        data = state.archive.data()
        return self._records_from_store_data(
            cast(Mapping[str, Any], data),
            effective_island,
        )

    def get_cell_commits(self, island_id: str | None = None) -> dict[int, str]:
        """Return a lightweight mapping of occupied cell indices to commit hashes."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        stats = state.archive.stats
        occupied = int(getattr(stats, "num_elites", 0))
        if occupied <= 0:
            return {}

        cell_commits: dict[int, str] = {}
        for cell_index, commit_hash in state.index_to_commit.items():
            commit = str(commit_hash or "").strip()
            if not commit:
                continue
            try:
                idx = int(cell_index)
            except (TypeError, ValueError):
                continue
            cell_commits[idx] = commit
        if len(cell_commits) != occupied:
            message = (
                "MAP-Elites archive bookkeeping mismatch between occupied cells "
                "and cell->commit mappings."
            )
            log.error(
                "{} island={} occupied={} mapped={}",
                message,
                effective_island,
                occupied,
                len(cell_commits),
            )
            raise RuntimeError(
                f"{message} island={effective_island} occupied={occupied} mapped={len(cell_commits)}"
            )
        return dict(cell_commits)

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
        self._reducers.pop(effective_island, None)
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
        )
        self._persist_island_state(effective_island, state, update=update)

    def describe_island(self, island_id: str | None = None) -> dict[str, Any]:
        """Return basic stats for observability dashboards."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        archive = state.archive
        stats = archive.stats
        occupied = int(getattr(stats, "num_elites", 0))
        cells = int(np.prod(getattr(archive, "dims", self._grid_shape)))
        qd_score = float(getattr(stats, "qd_score", 0.0) or 0.0)
        coverage_value = getattr(stats, "coverage", None)
        if coverage_value is None:
            coverage_value = (occupied / cells) if cells else 0.0
        norm_qd_value = getattr(stats, "norm_qd_score", None)
        if norm_qd_value is None:
            norm_qd_value = (qd_score / cells) if cells else 0.0
        best = getattr(stats, "objective_max", None)
        if best is None:
            best = getattr(stats, "obj_max", None)
        return {
            "island_id": effective_island,
            "occupied": occupied,
            "cells": cells,
            "coverage": float(coverage_value or 0.0),
            "qd_score": qd_score,
            "norm_qd_score": float(norm_qd_value or 0.0),
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
        timestamp: float | None = None,
    ) -> tuple[int, float, MapElitesRecord | None]:
        return archive_add_single(
            state=state,
            island_id=island_id,
            commit_hash=commit_hash,
            fitness=fitness,
            measures=measures,
            commit_to_island=self._commit_to_island,
            timestamp=timestamp,
        )

    def _add_batch_to_archive(
        self,
        *,
        state: IslandState,
        island_id: str,
        commit_hashes: Sequence[str],
        objectives: Sequence[float],
        measures: Sequence[np.ndarray] | np.ndarray,
        timestamps: Sequence[float],
        commit_to_island: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return archive_add_batch(
            state=state,
            island_id=island_id,
            commit_hashes=commit_hashes,
            objectives=objectives,
            measures=measures,
            timestamps=timestamps,
            commit_to_island=commit_to_island or self._commit_to_island,
        )

    def _seed_archive_after_initial_fit(
        self,
        *,
        state: IslandState,
        island_id: str,
        projection: PCAProjection,
        skip_commit_hash: str,
        snapshot_session: Session | None,
    ) -> None:
        seed_after_initial_fit(
            state=state,
            island_id=island_id,
            projection=projection,
            skip_commit_hash=skip_commit_hash,
            snapshot_session=snapshot_session,
            settings=self.settings,
            target_dims=self._target_dims,
            commit_to_island=self._commit_to_island,
            load_commit_fitnesses=lambda commits, session: self._load_commit_fitnesses(
                commit_hashes=commits,
                snapshot_session=session,
            ),
            clip_vector=lambda vector, current_state: self._clip_vector(vector, current_state),
            build_archive=lambda: self._build_archive(),
            add_batch=self._add_batch_to_archive,
        )

    def _rebuild_after_projection_refit(
        self,
        *,
        state: IslandState,
        island_id: str,
        current: FinalEmbedding,
        old_projection: PCAProjection,
        new_projection: PCAProjection,
        snapshot_session: Session | None,
    ) -> FinalEmbedding:
        previous_records = self.get_records(island_id)
        return rebuild_after_refit(
            state=state,
            island_id=island_id,
            current=current,
            old_projection=old_projection,
            new_projection=new_projection,
            previous_records=previous_records,
            snapshot_session=snapshot_session,
            target_dims=self._target_dims,
            commit_to_island=self._commit_to_island,
            load_commit_vectors=lambda i, commits, s, session: self._load_commit_vectors(
                island_id=i,
                commit_hashes=commits,
                state=s,
                snapshot_session=session,
            ),
            clip_vector=lambda vector, current_state: self._clip_vector(vector, current_state),
            build_archive=lambda: self._build_archive(),
            add_batch=self._add_batch_to_archive,
        )

    def _recompute_final_embedding(
        self,
        current: FinalEmbedding,
        projection: PCAProjection | None,
    ) -> FinalEmbedding:
        return rebuild_recompute_final_embedding(
            current=current,
            projection=projection,
            target_dims=self._target_dims,
        )

    def _pad_or_trim(self, vector: Sequence[float]) -> tuple[float, ...]:
        return rebuild_pad_or_trim(vector, target_dims=self._target_dims)

    @staticmethod
    def _iter_query_batches(values: Sequence[str], *, batch_size: int) -> Iterator[Sequence[str]]:
        return db_iter_query_batches(values, batch_size=batch_size)

    def _load_commit_vectors(
        self,
        *,
        island_id: str,
        commit_hashes: Sequence[str],
        state: IslandState,
        snapshot_session: Session | None,
    ) -> dict[str, tuple[float, ...]]:
        return db_load_commit_vectors(
            island_id=island_id,
            commit_hashes=commit_hashes,
            state=state,
            snapshot_session=snapshot_session,
            settings=self.settings,
        )

    def _load_commit_fitnesses(
        self,
        *,
        commit_hashes: Sequence[str],
        snapshot_session: Session | None,
    ) -> dict[str, float]:
        return db_load_commit_fitnesses(
            commit_hashes=commit_hashes,
            snapshot_session=snapshot_session,
            settings=self.settings,
        )

    def _build_archive_replace_payload(
        self,
        *,
        state: IslandState,
        island_id: str,
    ) -> tuple[SnapshotCellUpsert, ...]:
        return archive_build_archive_replace_payload(
            state=state,
            island_id=island_id,
        )

    def _records_from_store_data(
        self,
        data: Mapping[str, Any],
        island_id: str,
    ) -> tuple[MapElitesRecord, ...]:
        return archive_records_from_store_data(data, island_id)

    @staticmethod
    def _record_from_scalar_row(data: Mapping[str, Any], island_id: str) -> MapElitesRecord:
        return archive_record_from_scalar_row(data, island_id)

    def _ensure_island(self, island_id: str) -> IslandState:
        state = self._archives.get(island_id)
        if state:
            return state

        snapshot = self._snapshot_store.load(
            island_id,
            history_limit=resolve_pca_history_limit(self.settings),
        )
        snapshot_dims = self._infer_snapshot_target_dims(snapshot) if snapshot else None
        if snapshot_dims and snapshot_dims != self._target_dims:
            raise ValueError(
                "Snapshot dimensionality mismatch "
                f"(island={island_id} "
                f"settings_dims={self._target_dims} snapshot_dims={snapshot_dims})."
            )

        archive = self._build_archive()
        lower_template = self._lower_template
        upper_template = self._upper_template

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
        self._reducers[island_id] = DimensionReducer(
            settings=self.settings,
            history=state.history,
            projection=state.projection,
            samples_since_fit=state.samples_since_fit,
        )
        log.info(
            "Initialized MAP-Elites archive for island {} (cells={} dims={})",
            island_id,
            int(np.prod(getattr(archive, "dims", self._grid_shape))),
            int(len(getattr(archive, "dims", self._grid_shape))),
        )
        return state

    def _build_feature_bounds(self, *, target_dims: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        dims = int(target_dims) if target_dims is not None else int(self._target_dims)
        return archive_build_feature_bounds(target_dims=dims)

    def _build_archive(
        self,
        *,
        target_dims: int | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
    ) -> Any:
        dims = int(target_dims) if target_dims is not None else int(self._target_dims)
        return archive_build_archive(
            settings=self.settings,
            target_dims=dims,
            cells_per_dim=self._cells_per_dim,
            lower_template=self._lower_template,
            upper_template=self._upper_template,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def _clip_vector(self, vector: Sequence[float] | np.ndarray, state: IslandState) -> np.ndarray:
        return archive_clip_vector(
            vector=vector,
            settings=self.settings,
            clip_radius=self._clip_radius,
            state=state,
        )

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

    def _next_ingest_invocation(self) -> int:
        self._ingest_invocations += 1
        return self._ingest_invocations

    def _should_emit_sampled_ingest_info(self, ingest_index: int) -> bool:
        every = self._ingest_info_log_every
        if every <= 1:
            return True
        return ingest_index == 1 or (ingest_index - 1) % every == 0

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
            final_embedding=final_embedding,
        )

    def _persist_island_state(
        self,
        island_id: str,
        state: IslandState | None,
        *,
        update: SnapshotUpdate | None,
        session: Session | None = None,
    ) -> None:
        """Persist incremental snapshot updates for an island when enabled."""
        if not state:
            return
        if update is None:
            return
        self._snapshot_store.apply_update(island_id, update=update, session=session)

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
