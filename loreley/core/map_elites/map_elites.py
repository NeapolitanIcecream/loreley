"""Orchestrate MAP-Elites archives for evolutionary commit exploration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
from loguru import logger
from ribs.archives import GridArchive
from sqlalchemy import select
from sqlalchemy.orm import Session

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import CommitCard, MapElitesPcaHistory, MapElitesRepoStateAggregate, Metric
from .code_embedding import CommitCodeEmbedding
from .dimension_reduction import (
    FinalEmbedding,
    PcaHistoryEntry,
    PCAProjection,
    align_pca_projection,
    reduce_commit_embeddings,
    resolve_pca_history_limit,
)
from .preprocess import PreprocessedFile
from .repository_state_embedding import (
    RepoStateEmbeddingStats,
    embed_repository_state_incremental,
)
from .snapshot import (
    DatabaseSnapshotStore,
    SnapshotCellUpsert,
    SnapshotUpdate,
    apply_snapshot,
    purge_island_commit_mappings,
    to_list,
)

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
    history: tuple[PcaHistoryEntry, ...] = field(default_factory=tuple)
    projection: PCAProjection | None = None
    samples_since_fit: int = 0
    commit_to_index: dict[str, int] = field(default_factory=dict)
    index_to_commit: dict[int, str] = field(default_factory=dict)


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
        self._commit_to_island: dict[str, str] = {}
        self._default_island = self.settings.mapelites_default_island_id or "default"
        self._snapshot_store = DatabaseSnapshotStore()

    @staticmethod
    def _infer_snapshot_target_dims(snapshot: Mapping[str, Any]) -> int | None:
        """Infer the archive dimensionality from a persisted snapshot payload.

        Persisted MAP-Elites state is single-tenant. When the current process
        settings disagree with the stored snapshot dimensionality, we fail fast
        instead of silently adopting a different dimensionality.
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

        log.info(
            "Ingesting commit {} for island {}",
            commit_hash,
            effective_island,
        )

        update: SnapshotUpdate | None = None

        try:
            code_embedding, repo_stats = embed_repository_state_incremental(
                commit_hash=commit_hash,
                repo_root=working_dir,
                settings=self.settings,
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
            old_projection = state.projection
            final_embedding, history, projection, samples_since_fit = reduce_commit_embeddings(
                commit_hash=commit_hash,
                code_embedding=code_embedding,
                history=state.history,
                projection=state.projection,
                samples_since_fit=state.samples_since_fit,
                settings=self.settings,
            )
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
                final_embedding = self._rebuild_after_projection_refit(
                    state=state,
                    island_id=effective_island,
                    current=final_embedding,
                    old_projection=old_projection,
                    new_projection=state.projection,
                    snapshot_session=snapshot_session,
                )

            # Persist PCA state incrementally even when the archive does not change.
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
                message = "Unable to derive final embedding."
                log.warning("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            if did_initial_fit and state.projection is not None:
                self._seed_archive_after_initial_fit(
                    state=state,
                    island_id=effective_island,
                    projection=state.projection,
                    skip_commit_hash=commit_hash,
                    snapshot_session=snapshot_session,
                )

            # Plan A: keep the archive empty until PCA is available so we never store elites
            # in a pre-projection coordinate system.
            if state.projection is None:
                message = "PCA warmup: projection is not ready; skipping archive update."
                log.info("{} island={} commit={}", message, effective_island, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            archive_replace_needed = did_refit or did_initial_fit
            if update is not None and archive_replace_needed:
                update.archive_replace = self._build_archive_replace_payload(
                    state=state,
                    island_id=effective_island,
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
            if archive_replace_needed and update is not None:
                update.archive_replace = self._build_archive_replace_payload(
                    state=state,
                    island_id=effective_island,
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
            self._persist_island_state(
                effective_island,
                state,
                update=update,
                session=snapshot_session,
            )

    def get_records(
        self,
        island_id: str | None = None,
    ) -> tuple["MapElitesRecord", ...]:
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

    def get_cell_commits(self, island_id: str | None = None) -> dict[int, str]:
        """Return a lightweight mapping of occupied cell indices to commit hashes.

        This method is intended for hot paths (e.g. scheduling) that only need to
        sample occupied archive cells without materializing full archive records.
        """

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

    def _seed_archive_after_initial_fit(
        self,
        *,
        state: IslandState,
        island_id: str,
        projection: PCAProjection,
        skip_commit_hash: str,
        snapshot_session: Session | None,
    ) -> None:
        """Populate the archive once the first PCA projection becomes available.

        During cold-start warmup, Loreley persists PCA history entries but keeps the
        MAP-Elites archive empty. Once PCA is fitted (epoch 0), we project the
        warmup commits into the new coordinate system and insert them into a fresh
        archive so downstream sampling never observes a mixed coordinate system.
        """

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
        fitnesses = self._load_commit_fitnesses(
            commit_hashes=commit_hashes,
            snapshot_session=snapshot_session,
        )

        purge_island_commit_mappings(self._commit_to_island, island_id)
        state.archive = self._build_archive()
        state.index_to_commit.clear()
        state.commit_to_index.clear()

        inserted = 0
        skipped = 0
        timestamp = time.time()

        def _fitness_key(entry: PcaHistoryEntry) -> float:
            commit = str(entry.commit_hash or "").strip()
            value = fitnesses.get(commit)
            if value is None:
                value = float(self.settings.mapelites_fitness_floor)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(self.settings.mapelites_fitness_floor)

        for entry in sorted(candidates, key=_fitness_key, reverse=True):
            commit = str(entry.commit_hash or "").strip()
            if not commit:
                continue

            fitness = fitnesses.get(commit)
            if fitness is None:
                fitness = float(self.settings.mapelites_fitness_floor)
            try:
                fitness_value = float(fitness)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not math.isfinite(fitness_value):
                skipped += 1
                continue

            try:
                reduced = self._pad_or_trim(projection.transform(entry.vector))
            except ValueError:
                skipped += 1
                continue

            measures = self._clip_vector(reduced, state)
            status, _delta, _record = self._add_to_archive(
                state=state,
                island_id=island_id,
                commit_hash=commit,
                fitness=fitness_value,
                measures=measures,
                timestamp=timestamp,
            )
            if status > 0:
                inserted += 1

        log.info(
            "Seeded MAP-Elites archive after initial PCA fit (island={} inserted={} candidates={} skipped={})",
            island_id,
            inserted,
            len(candidates),
            skipped,
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
        """Align the new PCA projection and rebuild the archive in the new coordinates."""

        previous_records = self.get_records(island_id)
        if not previous_records:
            return self._recompute_final_embedding(current, new_projection)

        commit_hashes = tuple(
            str(record.commit_hash or "").strip()
            for record in previous_records
            if str(record.commit_hash or "").strip()
        )
        vectors = self._load_commit_vectors(
            island_id=island_id,
            commit_hashes=commit_hashes,
            state=state,
            snapshot_session=snapshot_session,
        )
        anchors = [vectors[h] for h in commit_hashes if h in vectors]
        aligned = align_pca_projection(
            projection=new_projection,
            reference=old_projection,
            anchors=anchors,
        )
        state.projection = aligned

        # Clear archive + bookkeeping and reinsert previous elites using the new projection.
        purge_island_commit_mappings(self._commit_to_island, island_id)
        state.archive = self._build_archive()
        state.index_to_commit.clear()
        state.commit_to_index.clear()

        inserted = 0
        missing = 0
        for record in sorted(previous_records, key=lambda item: float(item.fitness), reverse=True):
            commit = str(record.commit_hash or "").strip()
            if not commit:
                continue
            vec = vectors.get(commit)
            if not vec:
                missing += 1
                continue
            reduced = self._pad_or_trim(aligned.transform(vec))
            measures = self._clip_vector(reduced, state)
            status, _delta, _ = self._add_to_archive(
                state=state,
                island_id=island_id,
                commit_hash=commit,
                fitness=float(record.fitness),
                measures=measures,
                timestamp=float(record.timestamp),
            )
            if status > 0:
                inserted += 1

        log.info(
            "Rebuilt MAP-Elites archive after PCA refit (island={} kept={} missing_vectors={})",
            island_id,
            inserted,
            missing,
        )
        return self._recompute_final_embedding(current, aligned)

    def _recompute_final_embedding(
        self,
        current: FinalEmbedding,
        projection: PCAProjection | None,
    ) -> FinalEmbedding:
        if projection is None:
            return current
        reduced = self._pad_or_trim(projection.transform(current.history_entry.vector))
        return FinalEmbedding(
            commit_hash=current.commit_hash,
            vector=reduced,
            dimensions=len(reduced),
            history_entry=current.history_entry,
            projection=projection,
        )

    def _pad_or_trim(self, vector: Sequence[float]) -> tuple[float, ...]:
        if not vector:
            return tuple(0.0 for _ in range(self._target_dims))
        if len(vector) >= self._target_dims:
            return tuple(float(v) for v in vector[: self._target_dims])
        padded = [float(v) for v in vector]
        padded.extend(0.0 for _ in range(self._target_dims - len(padded)))
        return tuple(padded)

    def _load_commit_vectors(
        self,
        *,
        island_id: str,
        commit_hashes: Sequence[str],
        state: IslandState,
        snapshot_session: Session | None,
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
            stmt = (
                select(MapElitesPcaHistory)
                .where(MapElitesPcaHistory.island_id == island_id)
                .where(MapElitesPcaHistory.commit_hash.in_(missing))
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

        normalize = bool(self.settings.mapelites_dimensionality_penultimate_normalize)

        def _fill_from_aggregates(session: Session) -> None:
            stmt = select(MapElitesRepoStateAggregate).where(
                MapElitesRepoStateAggregate.commit_hash.in_(still_missing)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                commit = str(row.commit_hash or "").strip()
                if not commit or commit in vectors:
                    continue
                file_count = int(row.file_count or 0)
                if file_count <= 0:
                    continue
                raw = [float(v) / float(file_count) for v in (row.sum_vector or [])]
                vec = tuple(raw)
                if normalize and vec:
                    magnitude = math.sqrt(sum(value * value for value in vec))
                    if magnitude > 0.0:
                        vec = tuple(value / magnitude for value in vec)
                if vec:
                    vectors[commit] = vec

        if snapshot_session is not None:
            _fill_from_aggregates(snapshot_session)
        else:
            with session_scope() as owned:
                _fill_from_aggregates(owned)

        return vectors

    def _load_commit_fitnesses(
        self,
        *,
        commit_hashes: Sequence[str],
        snapshot_session: Session | None,
    ) -> dict[str, float]:
        metric_name = (self.settings.mapelites_fitness_metric or "").strip()
        if not metric_name:
            return {}

        needed = sorted({str(commit).strip() for commit in commit_hashes if str(commit).strip()})
        if not needed:
            return {}

        direction = 1.0 if self.settings.mapelites_fitness_higher_is_better else -1.0
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

    def _build_archive_replace_payload(
        self,
        *,
        state: IslandState,
        island_id: str,
    ) -> tuple[SnapshotCellUpsert, ...]:
        if state.archive.empty:
            return tuple()
        data = state.archive.data()
        records = self._records_from_store_data(
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
