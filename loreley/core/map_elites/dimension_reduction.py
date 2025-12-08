"""Fuse embeddings and run PCA to derive MAP-Elites features."""

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
from .summarization_embedding import CommitSummaryEmbedding

__all__ = [
    "PenultimateEmbedding",
    "PCAProjection",
    "FinalEmbedding",
    "DimensionReducer",
    "reduce_commit_embeddings",
]

Vector = tuple[float, ...]

log = logger.bind(module="map_elites.dimension_reduction")


@dataclass(slots=True, frozen=True)
class PenultimateEmbedding:
    """Concatenated code + summary embedding for a single commit."""

    commit_hash: str
    vector: Vector
    code_dimensions: int
    summary_dimensions: int
    code_model: str | None
    summary_model: str | None
    summary_embedding_model: str | None

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
    penultimate: PenultimateEmbedding
    projection: PCAProjection | None


class DimensionReducer:
    """Maintain PCA state and produce compact embeddings."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        history: Sequence[PenultimateEmbedding] | None = None,
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
        self._normalise_penultimate = (
            self.settings.mapelites_dimensionality_penultimate_normalize
        )

        self._history: OrderedDict[str, PenultimateEmbedding] = OrderedDict()
        self._projection: PCAProjection | None = projection
        self._feature_count: int | None = (
            projection.feature_count if projection else None
        )
        self._samples_since_fit = 0

        if history:
            for entry in history:
                self._record_history(entry, count_for_refit=False)

    @property
    def history(self) -> tuple[PenultimateEmbedding, ...]:
        """Return stored penultimate embeddings."""
        return tuple(self._history.values())

    @property
    def projection(self) -> PCAProjection | None:
        """Return the currently active PCA projection."""
        return self._projection

    def build_penultimate(
        self,
        *,
        commit_hash: str,
        code_embedding: CommitCodeEmbedding | None = None,
        summary_embedding: CommitSummaryEmbedding | None = None,
    ) -> PenultimateEmbedding | None:
        """Concatenate embeddings into a single vector."""
        vector_parts: list[float] = []
        code_dims = 0
        summary_dims = 0
        code_model = None
        summary_model = None
        summary_embedding_model = None

        if code_embedding and code_embedding.vector:
            vector_parts.extend(code_embedding.vector)
            code_dims = len(code_embedding.vector)
            code_model = code_embedding.model

        if summary_embedding and summary_embedding.vector:
            vector_parts.extend(summary_embedding.vector)
            summary_dims = len(summary_embedding.vector)
            summary_model = summary_embedding.summary_model
            summary_embedding_model = summary_embedding.embedding_model

        if not vector_parts:
            log.warning(
                "Commit {} produced no embeddings; skipping PCA preparation.",
                commit_hash,
            )
            return None

        vector = tuple(vector_parts)
        if self._normalise_penultimate:
            vector = self._l2_normalise(vector)

        return PenultimateEmbedding(
            commit_hash=commit_hash,
            vector=vector,
            code_dimensions=code_dims,
            summary_dimensions=summary_dims,
            code_model=code_model,
            summary_model=summary_model,
            summary_embedding_model=summary_embedding_model,
        )

    def reduce(
        self,
        penultimate: PenultimateEmbedding,
        *,
        refit: bool | None = None,
    ) -> FinalEmbedding | None:
        """Track penultimate embedding and project to the target space."""
        if not penultimate.vector:
            log.warning(
                "Penultimate embedding for commit {} is empty.",
                penultimate.commit_hash,
            )
            return None

        self._record_history(penultimate)
        if refit is True or (refit is None and self._should_refit()):
            self._fit_projection()

        reduced = self._project(penultimate)
        if not reduced:
            return None

        return FinalEmbedding(
            commit_hash=penultimate.commit_hash,
            vector=reduced,
            dimensions=len(reduced),
            penultimate=penultimate,
            projection=self._projection,
        )

    def _record_history(
        self,
        embedding: PenultimateEmbedding,
        *,
        count_for_refit: bool = True,
    ) -> None:
        """Store embeddings while respecting history bounds."""
        dimensions = embedding.dimensions
        if dimensions == 0:
            return

        if self._feature_count is None:
            self._feature_count = dimensions
        elif dimensions != self._feature_count:
            log.warning(
                "Penultimate dimensions changed from {} to {}; resetting PCA state.",
                self._feature_count,
                dimensions,
            )
            self._history.clear()
            self._projection = None
            self._feature_count = dimensions
            self._samples_since_fit = 0

        commit_hash = embedding.commit_hash
        is_new_entry = commit_hash not in self._history
        if not is_new_entry:
            self._history.pop(commit_hash)
        self._history[commit_hash] = embedding

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

    def _project(self, penultimate: PenultimateEmbedding) -> Vector:
        """Apply PCA projection (or fallback) and enforce target dims."""
        vector = penultimate.vector
        projection = self._projection
        if projection:
            try:
                vector = projection.transform(vector)
            except ValueError as exc:
                log.error(
                    "Stored PCA projection incompatible with commit {}: {}",
                    penultimate.commit_hash,
                    exc,
                )
                self._projection = None
                vector = penultimate.vector

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
    def _l2_normalise(vector: Vector) -> Vector:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector
        return tuple(value / magnitude for value in vector)


def reduce_commit_embeddings(
    *,
    commit_hash: str,
    code_embedding: CommitCodeEmbedding | None,
    summary_embedding: CommitSummaryEmbedding | None,
    history: Sequence[PenultimateEmbedding] | None = None,
    projection: PCAProjection | None = None,
    settings: Settings | None = None,
) -> tuple[
    FinalEmbedding | None,
    tuple[PenultimateEmbedding, ...],
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
    penultimate = reducer.build_penultimate(
        commit_hash=commit_hash,
        code_embedding=code_embedding,
        summary_embedding=summary_embedding,
    )
    if not penultimate:
        return None, reducer.history, reducer.projection

    reduced = reducer.reduce(penultimate)
    return reduced, reducer.history, reducer.projection

