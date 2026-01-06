"""Run PCA over commit embeddings to derive MAP-Elites features."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Sequence

from loguru import logger
from sklearn.decomposition import PCA

from loreley.config import Settings, get_settings
from .code_embedding import CommitCodeEmbedding

__all__ = [
    "PcaHistoryEntry",
    "PCAProjection",
    "FinalEmbedding",
    "DimensionReducer",
    "reduce_commit_embeddings",
]

Vector = tuple[float, ...]

log = logger.bind(module="map_elites.dimension_reduction")


@dataclass(slots=True, frozen=True)
class PcaHistoryEntry:
    """Commit embedding recorded in PCA history."""

    commit_hash: str
    vector: Vector
    embedding_model: str

    @property
    def dimensions(self) -> int:
        return len(self.vector)


@dataclass(slots=True, frozen=True)
class PCAProjection:
    """Serializable PCA projection metadata."""

    feature_count: int
    components: tuple[Vector, ...]
    mean: Vector
    explained_variance: tuple[float, ...]
    explained_variance_ratio: tuple[float, ...]
    sample_count: int
    fitted_at: float
    whiten: bool

    @property
    def dimensions(self) -> int:
        return len(self.components)

    def transform(self, vector: Sequence[float]) -> Vector:
        """Apply stored PCA projection to the provided vector."""
        if len(vector) != self.feature_count:
            raise ValueError(
                "PCA projection expects vectors with "
                f"{self.feature_count} dimensions, received {len(vector)}",
            )
        centered = [value - mean for value, mean in zip(vector, self.mean)]
        transformed = [
            sum(component[idx] * centered[idx] for idx in range(self.feature_count))
            for component in self.components
        ]
        if self.whiten and self.explained_variance:
            variances = list(self.explained_variance)
            if len(variances) < len(transformed):
                variances.extend([1.0] * (len(transformed) - len(variances)))
            transformed = [
                value / math.sqrt(variance) if variance > 0.0 else value
                for value, variance in zip(transformed, variances)
            ]
        return tuple(transformed)

    @classmethod
    def from_model(
        cls,
        model: PCA,
        sample_count: int,
        fitted_at: float | None = None,
    ) -> "PCAProjection":
        if not hasattr(model, "components_") or not hasattr(model, "mean_"):
            raise ValueError("PCA model must be fitted before export.")

        components = tuple(
            tuple(float(value) for value in row) for row in model.components_
        )
        mean = tuple(float(value) for value in model.mean_)
        explained_variance = tuple(
            float(value) for value in getattr(model, "explained_variance_", [])
        )
        explained = tuple(
            float(value) for value in getattr(model, "explained_variance_ratio_", [])
        )
        return cls(
            feature_count=len(mean),
            components=components,
            mean=mean,
            explained_variance=explained_variance,
            explained_variance_ratio=explained,
            sample_count=sample_count,
            fitted_at=fitted_at or time.time(),
            whiten=bool(getattr(model, "whiten", False)),
        )


@dataclass(slots=True, frozen=True)
class FinalEmbedding:
    """Low-dimensional embedding fed into the MAP-Elites grid."""

    commit_hash: str
    vector: Vector
    dimensions: int
    history_entry: PcaHistoryEntry
    projection: PCAProjection | None


class DimensionReducer:
    """Maintain PCA state and produce compact embeddings."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        history: Sequence[PcaHistoryEntry] | None = None,
        projection: PCAProjection | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._target_dims = max(1, self.settings.mapelites_dimensionality_target_dims)
        self._min_fit_samples = max(
            2,
            self.settings.mapelites_dimensionality_min_fit_samples,
            self.settings.mapelites_feature_normalization_warmup_samples,
        )
        self._history_limit = max(
            self._min_fit_samples,
            self.settings.mapelites_dimensionality_history_size,
        )
        self._refit_interval = max(
            0,
            self.settings.mapelites_dimensionality_refit_interval,
        )
        self._normalize_input = self.settings.mapelites_dimensionality_penultimate_normalize

        self._history: OrderedDict[str, PcaHistoryEntry] = OrderedDict()
        self._projection: PCAProjection | None = projection
        self._feature_count: int | None = (
            projection.feature_count if projection else None
        )
        self._samples_since_fit = 0

        if history:
            for entry in history:
                self._record_history(entry, count_for_refit=False)

    @property
    def history(self) -> tuple[PcaHistoryEntry, ...]:
        """Return stored PCA history entries."""
        return tuple(self._history.values())

    @property
    def projection(self) -> PCAProjection | None:
        """Return the currently active PCA projection."""
        return self._projection

    def build_history_entry(
        self,
        *,
        commit_hash: str,
        code_embedding: CommitCodeEmbedding | None = None,
    ) -> PcaHistoryEntry | None:
        """Prepare a PCA history entry from the commit embedding."""
        if not code_embedding or not code_embedding.vector:
            log.warning(
                "Commit {} produced no embeddings; skipping PCA preparation.",
                commit_hash,
            )
            return None

        vector = tuple(code_embedding.vector)
        if self._normalize_input:
            vector = self._l2_normalize(vector)

        return PcaHistoryEntry(
            commit_hash=commit_hash,
            vector=vector,
            embedding_model=str(code_embedding.model),
        )

    def reduce(
        self,
        entry: PcaHistoryEntry,
        *,
        refit: bool | None = None,
    ) -> FinalEmbedding | None:
        """Track a PCA history entry and project it to the target space."""
        if not entry.vector:
            log.warning(
                "PCA history entry for commit {} is empty.",
                entry.commit_hash,
            )
            return None

        self._record_history(entry)
        if refit is True or (refit is None and self._should_refit()):
            self._fit_projection()

        reduced = self._project(entry)
        if not reduced:
            return None

        return FinalEmbedding(
            commit_hash=entry.commit_hash,
            vector=reduced,
            dimensions=len(reduced),
            history_entry=entry,
            projection=self._projection,
        )

    def _record_history(
        self,
        entry: PcaHistoryEntry,
        *,
        count_for_refit: bool = True,
    ) -> None:
        """Store embeddings while respecting history bounds."""
        dimensions = entry.dimensions
        if dimensions == 0:
            return

        if self._feature_count is None:
            self._feature_count = dimensions
        elif dimensions != self._feature_count:
            log.warning(
                "PCA input dimensions changed from {} to {}; resetting PCA state.",
                self._feature_count,
                dimensions,
            )
            self._history.clear()
            self._projection = None
            self._feature_count = dimensions
            self._samples_since_fit = 0

        commit_hash = entry.commit_hash
        is_new_entry = commit_hash not in self._history
        if not is_new_entry:
            self._history.pop(commit_hash)
        self._history[commit_hash] = entry

        if len(self._history) > self._history_limit:
            dropped_hash, _ = self._history.popitem(last=False)
            log.debug("Evicted oldest embedding {}", dropped_hash)

        if count_for_refit and is_new_entry:
            self._samples_since_fit += 1

    def _should_refit(self) -> bool:
        """Return True when PCA should be recomputed."""
        sample_count = len(self._history)
        if sample_count < self._min_fit_samples:
            return False
        if self._projection is None:
            return True
        if self._refit_interval <= 0:
            return False
        return self._samples_since_fit >= self._refit_interval

    def _fit_projection(self) -> PCAProjection | None:
        """Fit PCA using the stored history."""
        samples = list(self._history.values())
        if len(samples) < self._min_fit_samples:
            log.debug(
                "Not enough samples for PCA: have {} require {}",
                len(samples),
                self._min_fit_samples,
            )
            return None

        feature_count = samples[0].dimensions
        if feature_count == 0:
            log.warning("Cannot fit PCA without features.")
            return None

        n_components = min(self._target_dims, len(samples), feature_count)
        if n_components == 0:
            log.warning(
                "PCA target components resolved to 0 (target={}, samples={}, features={})",
                self._target_dims,
                len(samples),
                feature_count,
            )
            return None

        model = PCA(
            n_components=n_components,
            svd_solver="auto",
            whiten=True,
        )
        try:
            model.fit([entry.vector for entry in samples])
        except ValueError as exc:
            log.error("Unable to fit PCA: {}", exc)
            return None

        projection = PCAProjection.from_model(model, len(samples))
        self._projection = projection
        self._samples_since_fit = 0
        log.info(
            "Fitted PCA projection: samples={} components={} variance_retained={:.3f}",
            len(samples),
            projection.dimensions,
            sum(projection.explained_variance_ratio),
        )
        return projection

    def _project(self, entry: PcaHistoryEntry) -> Vector:
        """Apply PCA projection (or fallback) and enforce target dims."""
        vector = entry.vector
        projection = self._projection
        if projection:
            try:
                vector = projection.transform(vector)
            except ValueError as exc:
                log.error(
                    "Stored PCA projection incompatible with commit {}: {}",
                    entry.commit_hash,
                    exc,
                )
                self._projection = None
                vector = entry.vector

        return self._pad_or_trim(vector)

    def _pad_or_trim(self, vector: Sequence[float]) -> Vector:
        if not vector:
            return ()
        if len(vector) >= self._target_dims:
            return tuple(vector[: self._target_dims])
        padded = list(vector)
        padded.extend(0.0 for _ in range(self._target_dims - len(vector)))
        return tuple(padded)

    @staticmethod
    def _l2_normalize(vector: Vector) -> Vector:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector
        return tuple(value / magnitude for value in vector)


def reduce_commit_embeddings(
    *,
    commit_hash: str,
    code_embedding: CommitCodeEmbedding | None,
    history: Sequence[PcaHistoryEntry] | None = None,
    projection: PCAProjection | None = None,
    settings: Settings | None = None,
) -> tuple[
    FinalEmbedding | None,
    tuple[PcaHistoryEntry, ...],
    PCAProjection | None,
]:
    """Convenience helper that runs the full reduction pipeline once.

    Returns a tuple of (final_embedding, updated_history, updated_projection).
    """

    reducer = DimensionReducer(
        settings=settings,
        history=history,
        projection=projection,
    )
    entry = reducer.build_history_entry(
        commit_hash=commit_hash,
        code_embedding=code_embedding,
    )
    if not entry:
        return None, reducer.history, reducer.projection

    reduced = reducer.reduce(entry)
    return reduced, reducer.history, reducer.projection

