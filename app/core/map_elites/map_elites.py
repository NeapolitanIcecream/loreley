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
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.config import Settings, get_settings
from app.db.base import session_scope
from app.db.models import MapElitesState
from .chunk import ChunkedFile, chunk_preprocessed_files
from .code_embedding import CommitCodeEmbedding, embed_chunked_files
from .dimension_reduction import (
    FinalEmbedding,
    PenultimateEmbedding,
    PCAProjection,
    reduce_commit_embeddings,
)
from .preprocess import ChangedFile, PreprocessedFile, preprocess_changed_files
from .summarization_embedding import (
    CommitSummaryEmbedding,
    summarize_preprocessed_files,
)

if TYPE_CHECKING:  # pragma: no cover
    from .chunk import PreprocessedArtifact
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

    preprocessed_files: tuple[PreprocessedFile, ...]
    chunked_files: tuple[ChunkedFile, ...]
    code_embedding: CommitCodeEmbedding | None
    summary_embedding: CommitSummaryEmbedding | None
    final_embedding: FinalEmbedding | None

    @property
    def file_count(self) -> int:
        return len(self.preprocessed_files)

    @property
    def chunk_count(self) -> int:
        return sum(len(file.chunks) for file in self.chunked_files)


@dataclass(slots=True, frozen=True)
class MapElitesRecord:
    """Snapshot of a single elite stored inside an archive cell."""

    commit_hash: str
    island_id: str
    cell_index: int
    fitness: float
    measures: Vector
    solution: Vector
    metadata: Mapping[str, Any]
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

    def ingest(
        self,
        *,
        commit_hash: str,
        changed_files: Sequence[ChangedFile | Mapping[str, object]],
        metrics: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        island_id: str | None = None,
        repo_root: Path | None = None,
        treeish: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        fitness_override: float | None = None,
    ) -> MapElitesInsertionResult:
        """Process a commit and attempt to insert it into the archive."""
        effective_island = island_id or self._default_island
        state = self._ensure_island(effective_island)
        working_dir = Path(repo_root or self.repo_root).resolve()

        log.info(
            "Ingesting commit {} for island {} (treeish={})",
            commit_hash,
            effective_island,
            treeish or "working-tree",
        )

        try:
            preprocessed = preprocess_changed_files(
                changed_files,
                repo_root=working_dir,
                settings=self.settings,
                treeish=treeish,
            )
            if not preprocessed:
                artifacts = self._build_artifacts(preprocessed, [], None, None, None)
                message = "No eligible files after preprocessing."
                log.warning("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            chunked = chunk_preprocessed_files(
                cast("Sequence[PreprocessedArtifact]", preprocessed),
                settings=self.settings,
            )
            if not chunked:
                artifacts = self._build_artifacts(preprocessed, chunked, None, None, None)
                message = "Chunking produced no content."
                log.warning("{} {}", message, commit_hash)
                return MapElitesInsertionResult(
                    status=0,
                    delta=0.0,
                    record=None,
                    artifacts=artifacts,
                    message=message,
                )

            code_embedding = embed_chunked_files(chunked, settings=self.settings)
            summary_embedding = summarize_preprocessed_files(preprocessed, settings=self.settings)

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

            artifacts = self._build_artifacts(
                preprocessed,
                chunked,
                code_embedding,
                summary_embedding,
                final_embedding,
            )

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

            payload_metadata: dict[str, Any] = {"metrics": metrics_map}
            if metadata:
                payload_metadata.update(metadata)
            if treeish:
                payload_metadata.setdefault("treeish", treeish)

            status, delta, record = self._add_to_archive(
                state=state,
                island_id=effective_island,
                commit_hash=commit_hash,
                fitness=fitness,
                measures=vector,
                metadata=payload_metadata,
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
            self._persist_island_state(effective_island, state)

    def get_records(
        self,
        island_id: str | None = None,
    ) -> tuple["SupportsMapElitesRecord", ...]:
        """Return all elites for a given island."""
        effective_island = island_id or self._default_island
        state = self._archives.get(effective_island)
        if not state or state.archive.empty:
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
        state = self._archives.get(effective_island)
        if not state or state.archive.empty:
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
        self._persist_island_state(effective_island, state)

    def describe_island(self, island_id: str | None = None) -> dict[str, Any]:
        """Return basic stats for observability dashboards."""
        effective_island = island_id or self._default_island
        state = self._archives.get(effective_island)
        if not state:
            return {"island_id": effective_island, "occupied": 0, "cells": 0}
        archive = state.archive
        stats = archive.stats
        best = getattr(stats, "objective_max", None)
        if best is None:
            best = getattr(stats, "obj_max", None)
        return {
            "island_id": effective_island,
            "occupied": int(getattr(stats, "num_elites", 0)),
            "cells": int(np.prod(self._grid_shape)),
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
        metadata: Mapping[str, Any],
    ) -> tuple[int, float, MapElitesRecord | None]:
        archive = state.archive
        measures_batch = measures.reshape(1, -1)
        solution = measures_batch  # Store embedding itself as the solution payload.
        objective = np.asarray([fitness], dtype=np.float64)
        timestamp = np.asarray([time.time()], dtype=np.float64)
        commit_field = np.asarray([commit_hash], dtype=object)
        metadata_field = np.asarray([dict(metadata)], dtype=object)

        cell_index = int(np.asarray(archive.index_of(measures_batch)).item())
        previous_commit = state.index_to_commit.get(cell_index)

        add_info = archive.add(
            solution,
            objective,
            measures_batch,
            commit_hash=commit_field,
            metadata=metadata_field,
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
        indices = self._to_list(data.get("index"))
        if not indices:
            return ()
        objectives = self._to_list(data.get("objective"))
        measures = self._to_list(data.get("measures"))
        solutions = self._to_list(data.get("solution"))
        commit_hashes = self._to_list(data.get("commit_hash"))
        metadata_entries = self._to_list(data.get("metadata"))
        timestamps = self._to_list(data.get("timestamp"))
        records: list[MapElitesRecord] = []
        for idx, cell_index in enumerate(indices):
            commit_hash = str(commit_hashes[idx]) if idx < len(commit_hashes) else ""
            fitness = float(objectives[idx]) if idx < len(objectives) else 0.0
            metadata = (
                self._coerce_metadata(metadata_entries[idx])
                if idx < len(metadata_entries)
                else {}
            )
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
                metadata=metadata,
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
            metadata=MapElitesManager._coerce_metadata(data.get("metadata")),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def _ensure_island(self, island_id: str) -> IslandState:
        state = self._archives.get(island_id)
        if state:
            return state
        archive = self._build_archive()
        state = IslandState(
            archive=archive,
            lower_bounds=self._lower_template.copy(),
            upper_bounds=self._upper_template.copy(),
        )
        snapshot = self._load_snapshot(island_id)
        if snapshot:
            self._apply_snapshot(state, snapshot, island_id)
        self._archives[island_id] = state
        log.info(
            "Initialized MAP-Elites archive for island {} (cells={} dims={})",
            island_id,
            np.prod(self._grid_shape),
            self._target_dims,
        )
        return state

    def _build_feature_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self._normalise_bounds(
            self.settings.mapelites_feature_lower_bounds,
            fallback=-6.0,
        )
        upper = self._normalise_bounds(
            self.settings.mapelites_feature_upper_bounds,
            fallback=6.0,
        )
        for idx, (lo, hi) in enumerate(zip(lower, upper)):
            if hi <= lo:
                upper[idx] = lo + 1.0
        return lower, upper

    def _build_archive(self) -> GridArchive:
        ranges = tuple(zip(self._lower_template.tolist(), self._upper_template.tolist()))
        extra_fields = {
            "commit_hash": ((), object),
            "metadata": ((), object),
            "timestamp": ((), np.float64),
        }
        return GridArchive(
            solution_dim=self._target_dims,
            dims=self._grid_shape,
            ranges=ranges,
            learning_rate=self.settings.mapelites_archive_learning_rate,
            threshold_min=self.settings.mapelites_archive_threshold_min,
            epsilon=self.settings.mapelites_archive_epsilon,
            qd_score_offset=self.settings.mapelites_archive_qd_score_offset,
            extra_fields=extra_fields,
        )

    def _clip_vector(self, vector: Vector, state: IslandState) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        if not self.settings.mapelites_feature_clip:
            return arr
        return np.clip(arr, state.lower_bounds, state.upper_bounds)

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
    def _coerce_metadata(payload: Any) -> dict[str, Any]:
        if isinstance(payload, Mapping):
            return dict(payload)
        return {}

    @staticmethod
    def _to_vector(values: Any) -> Vector:
        if values is None:
            return ()
        return tuple(float(v) for v in np.asarray(values).ravel())

    def _normalise_bounds(
        self,
        raw: Sequence[float] | None,
        *,
        fallback: float,
    ) -> np.ndarray:
        values = list(raw or [])
        if not values:
            values = [fallback]
        if len(values) < self._target_dims:
            values.extend([values[-1]] * (self._target_dims - len(values)))
        return np.asarray(values[: self._target_dims], dtype=np.float64)

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
        preprocessed: Sequence[PreprocessedFile],
        chunked: Sequence[ChunkedFile],
        code_embedding: CommitCodeEmbedding | None,
        summary_embedding: CommitSummaryEmbedding | None,
        final_embedding: FinalEmbedding | None,
    ) -> CommitEmbeddingArtifacts:
        return CommitEmbeddingArtifacts(
            preprocessed_files=tuple(preprocessed),
            chunked_files=tuple(chunked),
            code_embedding=code_embedding,
            summary_embedding=summary_embedding,
            final_embedding=final_embedding,
        )

    def _load_snapshot(self, island_id: str) -> dict[str, Any] | None:
        """Load a persisted snapshot for the given island and experiment, if any."""

        if self._experiment_id is None:
            # Persistence is disabled when no experiment is configured.
            return None

        try:
            with session_scope() as session:
                stmt = select(MapElitesState).where(
                    MapElitesState.experiment_id == self._experiment_id,
                    MapElitesState.island_id == island_id,
                )
                state = session.execute(stmt).scalar_one_or_none()
                if not state or not state.snapshot:
                    return None
                return dict(state.snapshot)
        except SQLAlchemyError as exc:
            log.error(
                "Failed to load MAP-Elites snapshot for experiment {} island {}: {}",
                self._experiment_id,
                island_id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while loading snapshot for experiment {} island {}: {}",
                self._experiment_id,
                island_id,
                exc,
            )
        return None

    def _apply_snapshot(
        self,
        state: IslandState,
        snapshot: Mapping[str, Any],
        island_id: str,
    ) -> None:
        lower_bounds = snapshot.get("lower_bounds")
        upper_bounds = snapshot.get("upper_bounds")
        if isinstance(lower_bounds, Sequence):
            state.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        if isinstance(upper_bounds, Sequence):
            state.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)

        history_payload = snapshot.get("history") or []
        if history_payload:
            state.history = self._deserialize_history(history_payload)

        projection_payload = snapshot.get("projection")
        if projection_payload:
            state.projection = self._deserialize_projection(projection_payload)

        state.index_to_commit.clear()
        state.commit_to_index.clear()
        self._purge_island_commit_mappings(island_id)
        archive_entries = snapshot.get("archive") or []
        if archive_entries:
            self._restore_archive_entries(state, archive_entries, island_id)

    def _persist_island_state(self, island_id: str, state: IslandState | None) -> None:
        """Persist the current archive snapshot for an island, if enabled."""

        if not state or self._experiment_id is None:
            return

        snapshot = self._build_snapshot(island_id, state)
        try:
            with session_scope() as session:
                stmt = select(MapElitesState).where(
                    MapElitesState.experiment_id == self._experiment_id,
                    MapElitesState.island_id == island_id,
                )
                existing = session.execute(stmt).scalar_one_or_none()
                if existing:
                    existing.snapshot = snapshot
                else:
                    session.add(
                        MapElitesState(
                            experiment_id=self._experiment_id,
                            island_id=island_id,
                            snapshot=snapshot,
                        )
                    )
        except SQLAlchemyError as exc:
            log.error(
                "Failed to persist MAP-Elites snapshot for experiment {} island {}: {}",
                self._experiment_id,
                island_id,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Unexpected error while persisting snapshot for experiment {} island {}: {}",
                self._experiment_id,
                island_id,
                exc,
            )

    def _build_snapshot(self, island_id: str, state: IslandState) -> dict[str, Any]:
        return {
            "island_id": island_id,
            "lower_bounds": state.lower_bounds.tolist(),
            "upper_bounds": state.upper_bounds.tolist(),
            "history": self._serialize_history(state.history),
            "projection": self._serialize_projection(state.projection),
            "archive": self._serialize_archive(state.archive),
        }

    @staticmethod
    def _serialize_history(
        history: Sequence[PenultimateEmbedding],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for entry in history:
            payload.append(
                {
                    "commit_hash": entry.commit_hash,
                    "vector": [float(value) for value in entry.vector],
                    "code_dimensions": entry.code_dimensions,
                    "summary_dimensions": entry.summary_dimensions,
                    "code_model": entry.code_model,
                    "summary_model": entry.summary_model,
                    "summary_embedding_model": entry.summary_embedding_model,
                }
            )
        return payload

    @staticmethod
    def _deserialize_history(
        payload: Sequence[Mapping[str, Any]],
    ) -> tuple[PenultimateEmbedding, ...]:
        history: list[PenultimateEmbedding] = []
        for item in payload:
            vector_values = cast(Sequence[float], item.get("vector") or [])
            vector = tuple(float(value) for value in vector_values)
            history.append(
                PenultimateEmbedding(
                    commit_hash=str(item.get("commit_hash", "")),
                    vector=vector,
                    code_dimensions=int(item.get("code_dimensions", 0)),
                    summary_dimensions=int(item.get("summary_dimensions", 0)),
                    code_model=item.get("code_model"),
                    summary_model=item.get("summary_model"),
                    summary_embedding_model=item.get("summary_embedding_model"),
                )
            )
        return tuple(history)

    @staticmethod
    def _serialize_projection(projection: PCAProjection | None) -> dict[str, Any] | None:
        if not projection:
            return None
        return {
            "feature_count": projection.feature_count,
            "components": [[float(value) for value in row] for row in projection.components],
            "mean": [float(value) for value in projection.mean],
            "explained_variance_ratio": [
                float(value) for value in projection.explained_variance_ratio
            ],
            "sample_count": projection.sample_count,
            "fitted_at": projection.fitted_at,
        }

    @staticmethod
    def _deserialize_projection(payload: Mapping[str, Any] | None) -> PCAProjection | None:
        if not payload:
            return None
        components_payload = cast(Sequence[Sequence[float]], payload.get("components") or [])
        components = tuple(
            tuple(float(value) for value in row) for row in components_payload
        )
        mean_raw = cast(Sequence[float], payload.get("mean") or [])
        mean = tuple(float(value) for value in mean_raw)
        explained_raw = cast(Sequence[float], payload.get("explained_variance_ratio") or [])
        explained = tuple(float(value) for value in explained_raw)
        return PCAProjection(
            feature_count=int(payload.get("feature_count", len(mean))),
            components=components,
            mean=mean,
            explained_variance_ratio=explained,
            sample_count=int(payload.get("sample_count", 0)),
            fitted_at=float(payload.get("fitted_at", time.time())),
        )

    def _serialize_archive(self, archive: GridArchive) -> list[dict[str, Any]]:
        data = archive.data()
        if archive.empty or not isinstance(data, dict):
            return []
        indices = self._to_list(data.get("index"))
        if not indices:
            return []
        objectives = self._to_list(data.get("objective"))
        measures = self._to_list(data.get("measures"))
        solutions = self._to_list(data.get("solution"))
        commit_hashes = self._to_list(data.get("commit_hash"))
        metadata_entries = self._to_list(data.get("metadata"))
        timestamps = self._to_list(data.get("timestamp"))

        entries: list[dict[str, Any]] = []
        for idx, cell_index in enumerate(indices):
            entry = {
                "index": int(cell_index),
                "objective": float(objectives[idx]) if idx < len(objectives) else 0.0,
                "measures": self._array_to_list(measures[idx]) if idx < len(measures) else [],
                "solution": self._array_to_list(solutions[idx]) if idx < len(solutions) else [],
                "commit_hash": str(commit_hashes[idx]) if idx < len(commit_hashes) else "",
                "metadata": (
                    self._coerce_metadata(metadata_entries[idx])
                    if idx < len(metadata_entries)
                    else {}
                ),
                "timestamp": float(timestamps[idx]) if idx < len(timestamps) else time.time(),
            }
            entries.append(entry)
        return entries

    def _restore_archive_entries(
        self,
        state: IslandState,
        entries: Sequence[Mapping[str, Any]],
        island_id: str,
    ) -> None:
        archive = state.archive
        for entry in entries:
            solution_values = self._array_to_list(entry.get("solution"))
            measures_values = self._array_to_list(entry.get("measures"))
            if not solution_values or not measures_values:
                continue
            solution = np.asarray(solution_values, dtype=np.float64)
            measures = np.asarray(measures_values, dtype=np.float64)
            solution_batch = solution.reshape(1, -1)
            measures_batch = measures.reshape(1, -1)
            objective = np.asarray(
                [float(entry.get("objective", 0.0))],
                dtype=np.float64,
            )
            commit_hash = str(entry.get("commit_hash", ""))
            metadata = self._coerce_metadata(entry.get("metadata"))
            timestamp_value = float(entry.get("timestamp", time.time()))
            archive.add(
                solution_batch,
                objective,
                measures_batch,
                commit_hash=np.asarray([commit_hash], dtype=object),
                metadata=np.asarray([metadata], dtype=object),
                timestamp=np.asarray([timestamp_value], dtype=np.float64),
            )
            stored_index = entry.get("index")
            if stored_index is not None:
                cell_index = int(stored_index)
            else:
                cell_index = int(np.asarray(archive.index_of(measures_batch)).item())
            state.index_to_commit[cell_index] = commit_hash
            if commit_hash:
                state.commit_to_index[commit_hash] = cell_index
                self._commit_to_island[commit_hash] = island_id

    def _purge_island_commit_mappings(self, island_id: str) -> None:
        for commit, mapped_island in tuple(self._commit_to_island.items()):
            if mapped_island == island_id:
                self._commit_to_island.pop(commit, None)

    @staticmethod
    def _array_to_list(values: Any) -> list[float]:
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            return values.astype(float).tolist()
        if isinstance(values, (list, tuple)):
            return [float(value) for value in values]
        return [float(values)]

    @staticmethod
    def _to_list(values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            return values.tolist()
        if isinstance(values, list):
            return values
        if isinstance(values, tuple):
            return list(values)
        return [values]

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
