"""Orchestrate MAP-Elites archives for evolutionary commit exploration."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, cast
from uuid import UUID

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from loreley.config import (
    Settings,
    get_settings,
    resolve_default_island_id,
    resolve_objective_contract,
)

from .archive_ops import (
    add_batch as archive_add_batch,
    add_single as archive_add_single,
    build_archive as archive_build_archive,
    build_archive_replace_payload as archive_build_archive_replace_payload,
    build_feature_bounds as archive_build_feature_bounds,
    clip_vector as archive_clip_vector,
    record_from_candidate as archive_record_from_candidate,
    records_from_archive as archive_records_from_archive,
)
from .code_embedding import CommitCodeEmbedding
from .db_ops import (
    iter_query_batches as db_iter_query_batches,
    load_commit_objectives as db_load_commit_objectives,
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
from .objectives import ObjectiveContractError, ResolvedObjectives
from .rebuild import (
    pad_or_trim as rebuild_pad_or_trim,
    recompute_final_embedding as rebuild_recompute_final_embedding,
    rebuild_after_refit,
    seed_after_initial_fit,
)
from .repository_state_embedding import (
    RepoStateEmbeddingStats,
    RepositoryStateEmbedder,
    embed_repository_state_incremental,
)
from .snapshot import DatabaseSnapshotStore, SnapshotElite, SnapshotUpdate, apply_snapshot
from .types import (
    CommitEmbeddingArtifacts,
    IslandState,
    MapElitesInsertionResult,
    MapElitesRecord,
)

log = logger.bind(module="map_elites.manager")

__all__ = [
    "CommitEmbeddingArtifacts",
    "MapElitesInsertionResult",
    "MapElitesManager",
    "MapElitesRecord",
]


@dataclass(slots=True)
class _IngestStageMetrics:
    started_at: float
    aggregate_hit_count: int = 0
    incremental_count: int = 0
    embedding_cache_miss_count: int = 0
    pca_fit_ms: float = 0.0
    pca_refit_ms: float = 0.0
    archive_add_ms: float = 0.0
    snapshot_apply_ms: float = 0.0
    did_initial_fit: bool = False
    did_refit: bool = False


@dataclass(slots=True, frozen=True)
class _RepoStateIngestResult:
    code_embedding: CommitCodeEmbedding | None
    stats: RepoStateEmbeddingStats


@dataclass(slots=True, frozen=True)
class _ProjectionIngestResult:
    final_embedding: FinalEmbedding | None


@dataclass(slots=True, frozen=True)
class _ArchiveCandidate:
    objectives: ResolvedObjectives
    vector: np.ndarray


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
        self._objective_contract = resolve_objective_contract(self.settings)
        self._archives: dict[str, IslandState] = {}
        self._reducers: dict[str, DimensionReducer] = {}
        self._default_island = resolve_default_island_id(self.settings)
        self._snapshot_store = DatabaseSnapshotStore()
        self._ingest_info_log_every = int(self.settings.mapelites_ingest_info_log_every)
        self._ingest_invocations = 0
        self._repo_state_embedder: RepositoryStateEmbedder | None = None
        self._repo_state_embedder_root: Path | None = None

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
        snapshot_session: Session | None = None,
        event_key_prefix: str | None = None,
        event_job_id: UUID | None = None,
        event_ordinal: int | None = None,
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
        stage_metrics = _IngestStageMetrics(started_at=time.perf_counter())

        try:
            repo_state = self._embed_repo_state_for_ingest(
                commit_hash=commit_hash,
                working_dir=working_dir,
                session=snapshot_session,
                stage_metrics=stage_metrics,
            )
            if not repo_state.code_embedding or not repo_state.code_embedding.vector:
                artifacts = self._build_artifacts(repo_state.stats, (), None, None)
                result = self._skip_ingest_result(
                    message="No eligible repository files produced an embedding.",
                    commit_hash=commit_hash,
                    artifacts=artifacts,
                    emit_sampled_info=emit_sampled_info,
                )
                return result

            projection_update = self._update_projection_for_ingest(
                state=state,
                island_id=effective_island,
                commit_hash=commit_hash,
                code_embedding=repo_state.code_embedding,
                snapshot_session=snapshot_session,
                stage_metrics=stage_metrics,
            )

            final_embedding = projection_update.final_embedding
            update = self._build_ingest_snapshot_update(
                state=state,
                final_embedding=final_embedding,
                event_key_prefix=event_key_prefix,
                event_job_id=event_job_id,
                event_ordinal=event_ordinal,
            )
            artifacts = self._build_artifacts(
                repo_state.stats,
                (),
                repo_state.code_embedding,
                final_embedding,
            )

            if not final_embedding:
                result = self._skip_ingest_result(
                    message="Unable to derive final embedding.",
                    commit_hash=commit_hash,
                    artifacts=artifacts,
                    emit_sampled_info=emit_sampled_info,
                )
                return result

            if stage_metrics.did_initial_fit and state.projection is not None:
                self._seed_archive_after_initial_fit(
                    state=state,
                    island_id=effective_island,
                    projection=state.projection,
                    skip_commit_hash=commit_hash,
                    snapshot_session=snapshot_session,
                )

            result = self._skip_warmup_archive_update(
                state=state,
                island_id=effective_island,
                commit_hash=commit_hash,
                artifacts=artifacts,
                emit_sampled_info=emit_sampled_info,
            )
            if result is not None:
                return result

            archive_replace_needed = bool(stage_metrics.did_refit or stage_metrics.did_initial_fit)

            candidate, skip_result = self._validated_archive_candidate(
                state=state,
                commit_hash=commit_hash,
                metrics=metrics,
                final_embedding=final_embedding,
                artifacts=artifacts,
                emit_sampled_info=emit_sampled_info,
            )
            if skip_result is not None:
                result = skip_result
                return result
            candidate = cast(_ArchiveCandidate, candidate)

            result = self._insert_archive_candidate_for_ingest(
                state=state,
                island_id=effective_island,
                commit_hash=commit_hash,
                candidate=candidate,
                update=update,
                archive_replace_needed=archive_replace_needed,
                artifacts=artifacts,
                emit_sampled_info=emit_sampled_info,
                stage_metrics=stage_metrics,
            )
            return result
        finally:
            self._finalize_ingest_state(
                island_id=effective_island,
                commit_hash=commit_hash,
                state=state,
                update=update,
                archive_replace_needed=archive_replace_needed,
                result=result,
                session=snapshot_session,
                emit_sampled_info=emit_sampled_info,
                stage_metrics=stage_metrics,
            )

    def get_records(
        self,
        island_id: str | None = None,
    ) -> tuple[MapElitesRecord, ...]:
        """Return all elites for a given island."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        return archive_records_from_archive(
            state.archive,
            effective_island,
        )

    def validate_configured_islands(self) -> None:
        """Load every configured island before scheduler side effects begin."""

        for island_id in self.settings.mapelites_islands:
            self._ensure_island(island_id)

    def reload_island(
        self,
        island_id: str,
        *,
        snapshot_session: Session | None = None,
    ) -> None:
        """Discard possibly mutated cache state and restore the durable snapshot."""

        effective_island = island_id or self._default_island
        self._archives.pop(effective_island, None)
        self._reducers.pop(effective_island, None)
        self._ensure_island(
            effective_island,
            snapshot_session=snapshot_session,
        )

    def count_pca_history_samples(self, island_id: str | None = None) -> int:
        """Return the number of non-empty PCA history samples for an island."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        return sum(1 for entry in state.history if entry.dimensions > 0)

    def get_cell_fronts(
        self,
        island_id: str | None = None,
    ) -> dict[int, tuple[str, ...]]:
        """Return every Pareto member grouped by occupied behavior cell."""

        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        mapped_elites = sum(len(commits) for commits in state.index_to_commits.values())
        if mapped_elites != state.archive.stats.num_elites:
            raise RuntimeError(
                "Pareto archive bookkeeping mismatch "
                f"(island={effective_island} archive={state.archive.stats.num_elites} "
                f"mapped={mapped_elites})."
            )
        return dict(state.index_to_commits)

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
        sampled = state.archive.sample(max(1, count))
        index_by_commit = state.commit_to_index
        return tuple(
            archive_record_from_candidate(
                candidate=candidate,
                island_id=effective_island,
                cell_index=index_by_commit[candidate.commit_hash],
            )
            for candidate in sampled
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
        state.commit_to_index.clear()
        state.index_to_commits.clear()
        log.info("Cleared MAP-Elites state for island {}", effective_island)
        update = SnapshotUpdate(
            objective_contract=self._objective_contract,
            lower_bounds=state.lower_bounds.tolist(),
            upper_bounds=state.upper_bounds.tolist(),
            projection=None,
            history_limit=resolve_pca_history_limit(self.settings),
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
        occupied = int(stats.num_occupied)
        elites = int(stats.num_elites)
        cells = int(np.prod(getattr(archive, "dims", self._grid_shape)))
        records = archive_records_from_archive(archive, effective_island)
        best_primary = (
            max(records, key=lambda record: record.primary_objective_score)
            if records
            else None
        )
        primary = self._objective_contract.primary
        return {
            "island_id": effective_island,
            "occupied": occupied,
            "elites": elites,
            "cells": cells,
            "coverage": float(stats.coverage),
            "objective_count": len(self._objective_contract.specs),
            "front_max_size": state.archive.max_front_size,
            "primary_metric_name": primary.name,
            "primary_metric_direction": primary.direction,
            "best_primary_value": (
                best_primary.primary_objective_value
                if best_primary is not None
                else None
            ),
        }

    def _embed_repo_state_for_ingest(
        self,
        *,
        commit_hash: str,
        working_dir: Path,
        session: Session | None,
        stage_metrics: _IngestStageMetrics,
    ) -> _RepoStateIngestResult:
        repo_state_embedder = self._get_repo_state_embedder(working_dir)
        code_embedding, repo_stats = embed_repository_state_incremental(
            commit_hash=commit_hash,
            repo_root=working_dir,
            settings=self.settings,
            embedder=repo_state_embedder,
            session=session,
        )
        self._record_repo_state_stats_for_ingest(
            repo_stats=repo_stats,
            commit_hash=commit_hash,
            stage_metrics=stage_metrics,
        )
        return _RepoStateIngestResult(
            code_embedding=code_embedding,
            stats=repo_stats,
        )

    @staticmethod
    def _record_repo_state_stats_for_ingest(
        *,
        repo_stats: RepoStateEmbeddingStats,
        commit_hash: str,
        stage_metrics: _IngestStageMetrics,
    ) -> None:
        source = str(getattr(repo_stats, "source", "unknown") or "unknown").strip()
        if source == "aggregate_hit":
            stage_metrics.aggregate_hit_count += 1
        elif source == "incremental":
            stage_metrics.incremental_count += 1

        cache_misses = int(getattr(repo_stats, "cache_misses", 0) or 0)
        if cache_misses < 0:
            raise ValueError(
                "Repo-state embedding reported a negative cache_misses value "
                f"(commit={commit_hash} cache_misses={cache_misses})."
            )
        stage_metrics.embedding_cache_miss_count += cache_misses

    def _update_projection_for_ingest(
        self,
        *,
        state: IslandState,
        island_id: str,
        commit_hash: str,
        code_embedding: CommitCodeEmbedding,
        snapshot_session: Session | None,
        stage_metrics: _IngestStageMetrics,
    ) -> _ProjectionIngestResult:
        old_projection = state.projection
        old_history = state.history
        old_samples_since_fit = state.samples_since_fit
        reducer = self._ensure_reducer(island_id, state)
        pca_fit_started_at = time.perf_counter()
        try:
            final_embedding, history, projection, samples_since_fit = (
                reduce_commit_embeddings(
                    commit_hash=commit_hash,
                    code_embedding=code_embedding,
                    history=state.history,
                    projection=state.projection,
                    samples_since_fit=state.samples_since_fit,
                    settings=self.settings,
                    reducer=reducer,
                )
            )
            stage_metrics.pca_fit_ms += (
                time.perf_counter() - pca_fit_started_at
            ) * 1000.0
            state.history = history
            state.projection = projection
            state.samples_since_fit = samples_since_fit

            did_initial_fit = old_projection is None and state.projection is not None
            did_refit = (
                old_projection is not None
                and state.projection is not None
                and state.projection.epoch != old_projection.epoch
            )
            stage_metrics.did_initial_fit = did_initial_fit
            stage_metrics.did_refit = did_refit
            if (
                did_refit
                and final_embedding is not None
                and old_projection is not None
                and state.projection is not None
            ):
                pca_refit_started_at = time.perf_counter()
                final_embedding = self._rebuild_after_projection_refit(
                    state=state,
                    island_id=island_id,
                    current=final_embedding,
                    old_projection=old_projection,
                    new_projection=state.projection,
                    snapshot_session=snapshot_session,
                )
                stage_metrics.pca_refit_ms += (
                    time.perf_counter() - pca_refit_started_at
                ) * 1000.0
                reducer.set_projection(state.projection)
        except Exception:
            state.history = old_history
            state.projection = old_projection
            state.samples_since_fit = old_samples_since_fit
            self._reducers[island_id] = DimensionReducer(
                settings=self.settings,
                history=old_history,
                projection=old_projection,
                samples_since_fit=old_samples_since_fit,
            )
            raise

        return _ProjectionIngestResult(
            final_embedding=final_embedding,
        )

    def _ensure_reducer(
        self,
        island_id: str,
        state: IslandState,
    ) -> DimensionReducer:
        reducer = self._reducers.get(island_id)
        if reducer is None:
            reducer = DimensionReducer(
                settings=self.settings,
                history=state.history,
                projection=state.projection,
                samples_since_fit=state.samples_since_fit,
            )
            self._reducers[island_id] = reducer
        return reducer

    def _build_ingest_snapshot_update(
        self,
        *,
        state: IslandState,
        final_embedding: FinalEmbedding | None,
        event_key_prefix: str | None = None,
        event_job_id: UUID | None = None,
        event_ordinal: int | None = None,
    ) -> SnapshotUpdate:
        return SnapshotUpdate(
            objective_contract=self._objective_contract,
            lower_bounds=state.lower_bounds.tolist(),
            upper_bounds=state.upper_bounds.tolist(),
            projection=state.projection,
            history_limit=resolve_pca_history_limit(self.settings),
            history_upsert=final_embedding.history_entry if final_embedding else None,
            history_seen_at=time.time(),
            samples_since_fit=state.samples_since_fit,
            event_key_prefix=event_key_prefix,
            event_job_id=event_job_id,
            event_ordinal=event_ordinal,
        )

    def _skip_ingest_result(
        self,
        *,
        message: str,
        commit_hash: str,
        artifacts: CommitEmbeddingArtifacts,
        emit_sampled_info: bool,
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

    @staticmethod
    def _skip_warmup_archive_update(
        *,
        state: IslandState,
        island_id: str,
        commit_hash: str,
        artifacts: CommitEmbeddingArtifacts,
        emit_sampled_info: bool,
    ) -> MapElitesInsertionResult | None:
        if state.projection is not None:
            return None

        message = "PCA warmup: projection is not ready; skipping archive update."
        if emit_sampled_info:
            log.info("{} island={} commit={}", message, island_id, commit_hash)
        return MapElitesInsertionResult(
            status=0,
            delta=0.0,
            record=None,
            artifacts=artifacts,
            message=message,
        )

    def _validated_archive_candidate(
        self,
        *,
        state: IslandState,
        commit_hash: str,
        metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
        final_embedding: FinalEmbedding,
        artifacts: CommitEmbeddingArtifacts,
        emit_sampled_info: bool,
    ) -> tuple[_ArchiveCandidate | None, MapElitesInsertionResult | None]:
        try:
            objectives = self._objective_contract.resolve(metrics)
        except ObjectiveContractError as exc:
            return None, self._skip_ingest_result(
                message=f"Objective contract rejected metrics: {exc}",
                commit_hash=commit_hash,
                artifacts=artifacts,
                emit_sampled_info=emit_sampled_info,
            )

        vector = self._clip_vector(final_embedding.vector, state)
        if vector.shape[0] != self._target_dims:
            message = (
                "Final embedding dimensions mismatch with archive "
                f"(expected {self._target_dims} got {vector.shape[0]})."
            )
            return None, self._skip_ingest_result(
                message=message,
                commit_hash=commit_hash,
                artifacts=artifacts,
                emit_sampled_info=emit_sampled_info,
                log_level="error",
            )

        return _ArchiveCandidate(objectives=objectives, vector=vector), None

    def _insert_archive_candidate_for_ingest(
        self,
        *,
        state: IslandState,
        island_id: str,
        commit_hash: str,
        candidate: _ArchiveCandidate,
        update: SnapshotUpdate | None,
        archive_replace_needed: bool,
        artifacts: CommitEmbeddingArtifacts,
        emit_sampled_info: bool,
        stage_metrics: _IngestStageMetrics,
    ) -> MapElitesInsertionResult:
        archive_add_started_at = time.perf_counter()
        replacing_existing = commit_hash in state.commit_to_index
        status, delta, record = self._add_to_archive(
            state=state,
            island_id=island_id,
            commit_hash=commit_hash,
            objective_values=candidate.objectives.values,
            objective_scores=candidate.objectives.scores,
            measures=candidate.vector,
        )
        stage_metrics.archive_add_ms += (time.perf_counter() - archive_add_started_at) * 1000.0

        self._record_snapshot_front_replace(
            update=update,
            state=state,
            island_id=island_id,
            record=record,
            replacing_existing=replacing_existing,
            archive_replace_needed=archive_replace_needed,
        )
        self._log_archive_insertion_outcome(
            commit_hash=commit_hash,
            island_id=island_id,
            status=status,
            delta=delta,
            record=record,
            emit_sampled_info=emit_sampled_info,
        )
        return self._build_archive_insertion_result(
            status=status,
            delta=delta,
            record=record,
            artifacts=artifacts,
        )

    @staticmethod
    def _record_snapshot_front_replace(
        *,
        update: SnapshotUpdate | None,
        state: IslandState,
        island_id: str,
        record: MapElitesRecord | None,
        replacing_existing: bool,
        archive_replace_needed: bool,
    ) -> None:
        if update is None or archive_replace_needed:
            return
        records = archive_records_from_archive(state.archive, island_id)
        if replacing_existing:
            update.archive_replace = tuple(
                SnapshotElite(
                    cell_index=item.cell_index,
                    commit_hash=item.commit_hash,
                    objective_values=item.objective_values,
                    measures=item.measures,
                    timestamp=item.timestamp,
                )
                for item in records
            )
            return
        if record is not None:
            update.front_replace = tuple(
                SnapshotElite(
                    cell_index=item.cell_index,
                    commit_hash=item.commit_hash,
                    objective_values=item.objective_values,
                    measures=item.measures,
                    timestamp=item.timestamp,
                )
                for item in records
                if item.cell_index == record.cell_index
            )

    @staticmethod
    def _log_archive_insertion_outcome(
        *,
        commit_hash: str,
        island_id: str,
        status: int,
        delta: float,
        record: MapElitesRecord | None,
        emit_sampled_info: bool,
    ) -> None:
        if record:
            if emit_sampled_info:
                log.info(
                    "Retained commit {} in island {} Pareto front "
                    "(cell={} removed={})",
                    commit_hash,
                    island_id,
                    record.cell_index,
                    int(delta),
                )
        else:
            if emit_sampled_info:
                log.info(
                    "Commit {} was not retained in island {} Pareto front "
                    "(status={} removed={})",
                    commit_hash,
                    island_id,
                    status,
                    int(delta),
                )

    @staticmethod
    def _build_archive_insertion_result(
        *,
        status: int,
        delta: float,
        record: MapElitesRecord | None,
        artifacts: CommitEmbeddingArtifacts,
    ) -> MapElitesInsertionResult:
        message: str | None = None
        if status <= 0:
            message = (
                "Commit not retained; it was dominated, objective-equivalent, "
                "or removed by bounded-front crowding."
            )
        elif record is None:
            message = "Archive reported insertion success but no record could be retrieved."

        return MapElitesInsertionResult(
            status=status,
            delta=delta,
            record=record,
            artifacts=artifacts,
            message=message,
        )

    def _finalize_ingest_state(
        self,
        *,
        island_id: str,
        commit_hash: str,
        state: IslandState,
        update: SnapshotUpdate | None,
        archive_replace_needed: bool,
        result: MapElitesInsertionResult | None,
        session: Session | None,
        emit_sampled_info: bool,
        stage_metrics: _IngestStageMetrics,
    ) -> None:
        snapshot_apply_started_at = time.perf_counter()
        try:
            if update is not None:
                update.archive_change_reason = (
                    "projection_rebuild"
                    if stage_metrics.did_refit
                    else (
                        "projection_initial_fit"
                        if stage_metrics.did_initial_fit
                        else "local_pareto_update"
                    )
                )
                update.projection_epoch = (
                    int(state.projection.epoch)
                    if state.projection is not None
                    else None
                )
            if update is not None and archive_replace_needed and update.archive_replace is None:
                update.archive_replace = self._build_archive_replace_payload(
                    state=state,
                    island_id=island_id,
                )
            self._persist_island_state(
                island_id,
                state,
                update=update,
                session=session,
            )
        finally:
            stage_metrics.snapshot_apply_ms += (
                time.perf_counter() - snapshot_apply_started_at
            ) * 1000.0
            self._emit_ingest_stage_metrics(
                commit_hash=commit_hash,
                island_id=island_id,
                result=result,
                emit_sampled_info=emit_sampled_info,
                stage_metrics=stage_metrics,
            )

    @staticmethod
    def _emit_ingest_stage_metrics(
        *,
        commit_hash: str,
        island_id: str,
        result: MapElitesInsertionResult | None,
        emit_sampled_info: bool,
        stage_metrics: _IngestStageMetrics,
    ) -> None:
        emit_stage_metrics = (
            emit_sampled_info
            or result is None
            or stage_metrics.did_initial_fit
            or stage_metrics.did_refit
            or (result is not None and result.status < 0)
        )
        if emit_stage_metrics:
            log.bind(
                commit_hash=commit_hash,
                island_id=island_id,
                aggregate_hit_count=stage_metrics.aggregate_hit_count,
                incremental_count=stage_metrics.incremental_count,
                embedding_cache_miss_count=stage_metrics.embedding_cache_miss_count,
                pca_fit_ms=round(stage_metrics.pca_fit_ms, 3),
                pca_refit_ms=round(stage_metrics.pca_refit_ms, 3),
                archive_add_ms=round(stage_metrics.archive_add_ms, 3),
                snapshot_apply_ms=round(stage_metrics.snapshot_apply_ms, 3),
                ingest_total_ms=round((time.perf_counter() - stage_metrics.started_at) * 1000.0, 3),
                did_initial_fit=stage_metrics.did_initial_fit,
                did_refit=stage_metrics.did_refit,
                status_code=result.status if result is not None else None,
            ).info("MAP-Elites ingest stage metrics")

    def _add_to_archive(
        self,
        *,
        state: IslandState,
        island_id: str,
        commit_hash: str,
        objective_values: Sequence[float],
        objective_scores: Sequence[float],
        measures: np.ndarray,
        timestamp: float | None = None,
    ) -> tuple[int, float, MapElitesRecord | None]:
        return archive_add_single(
            state=state,
            island_id=island_id,
            commit_hash=commit_hash,
            objective_values=objective_values,
            objective_scores=objective_scores,
            measures=measures,
            timestamp=timestamp,
        )

    def _add_batch_to_archive(
        self,
        *,
        state: IslandState,
        island_id: str,
        commit_hashes: Sequence[str],
        objective_values: Sequence[Sequence[float]] | np.ndarray,
        objective_scores: Sequence[Sequence[float]] | np.ndarray,
        measures: Sequence[np.ndarray] | np.ndarray,
        timestamps: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return archive_add_batch(
            state=state,
            island_id=island_id,
            commit_hashes=commit_hashes,
            objective_values=objective_values,
            objective_scores=objective_scores,
            measures=measures,
            timestamps=timestamps,
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
            target_dims=self._target_dims,
            load_commit_objectives=lambda commits, session: self._load_commit_objectives(
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

    def _load_commit_objectives(
        self,
        *,
        commit_hashes: Sequence[str],
        snapshot_session: Session | None,
    ) -> dict[str, ResolvedObjectives]:
        return db_load_commit_objectives(
            commit_hashes=commit_hashes,
            snapshot_session=snapshot_session,
            settings=self.settings,
        )

    def _build_archive_replace_payload(
        self,
        *,
        state: IslandState,
        island_id: str,
    ) -> tuple[SnapshotElite, ...]:
        return archive_build_archive_replace_payload(
            state=state,
            island_id=island_id,
        )

    def _ensure_island(
        self,
        island_id: str,
        *,
        snapshot_session: Session | None = None,
    ) -> IslandState:
        state = self._archives.get(island_id)
        if state:
            return state

        load_kwargs: dict[str, Any] = {
            "history_limit": resolve_pca_history_limit(self.settings),
        }
        if snapshot_session is not None:
            load_kwargs["session"] = snapshot_session
        snapshot = self._snapshot_store.load(island_id, **load_kwargs)
        snapshot_dims = self._infer_snapshot_target_dims(snapshot) if snapshot else None
        if snapshot_dims and snapshot_dims != self._target_dims:
            raise ValueError(
                "Snapshot dimensionality mismatch "
                f"(island={island_id} "
                f"settings_dims={self._target_dims} snapshot_dims={snapshot_dims})."
            )

        lower_template = np.asarray(
            snapshot.get("lower_bounds", self._lower_template)
            if snapshot
            else self._lower_template,
            dtype=np.float64,
        )
        upper_template = np.asarray(
            snapshot.get("upper_bounds", self._upper_template)
            if snapshot
            else self._upper_template,
            dtype=np.float64,
        )
        archive = self._build_archive(
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
                objective_contract=self._objective_contract,
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

    def _get_repo_state_embedder(self, repo_root: Path) -> RepositoryStateEmbedder:
        resolved_root = Path(repo_root).resolve()
        if (
            self._repo_state_embedder is None
            or self._repo_state_embedder_root != resolved_root
        ):
            self._repo_state_embedder = RepositoryStateEmbedder(
                settings=self.settings,
            )
            self._repo_state_embedder_root = resolved_root
        return self._repo_state_embedder

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
