"""Run PCA over commit embeddings to derive MAP-Elites features."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import math
import time
from typing import Sequence

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA

from loreley.config import Settings, get_settings
from .code_embedding import CommitCodeEmbedding

__all__ = [
    "PcaHistoryEntry",
    "PCAProjection",
    "align_pca_projection",
    "FinalEmbedding",
    "DimensionReducer",
    "resolve_pca_history_limit",
    "reduce_commit_embeddings",
]

Vector = tuple[float, ...]

log = logger.bind(module="map_elites.dimension_reduction")


def resolve_pca_history_limit(settings: Settings) -> int:
    """Return the bounded history window size used by the PCA reducer."""
    min_fit = max(
        2,
        int(settings.mapelites_dimensionality_min_fit_samples),
        int(settings.mapelites_feature_normalization_warmup_samples),
    )
    return max(
        min_fit,
        int(settings.mapelites_dimensionality_history_size),
    )


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
    epoch: int
    fitted_at: float
    whiten: bool
    # Optional orthogonal alignment matrix applied after PCA (and whitening, when
    # enabled). Computed with orthogonal Procrustes to align successive PCA refits
    # and reduce coordinate jitter. This matrix may include a reflection.
    rotation: tuple[Vector, ...] | None = None

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
        vector_array = np.asarray(vector, dtype=np.float64)
        mean_array = np.asarray(self.mean, dtype=np.float64)
        components_matrix = np.asarray(self.components, dtype=np.float64)

        centered = vector_array - mean_array
        transformed = components_matrix @ centered
        if self.whiten and self.explained_variance:
            variances = np.asarray(self.explained_variance, dtype=np.float64)
            if variances.size < transformed.size:
                variances = np.pad(
                    variances,
                    (0, transformed.size - variances.size),
                    mode="constant",
                    constant_values=1.0,
                )
            scales = np.sqrt(variances[: transformed.size])
            positive_scales = scales > 0.0
            transformed = transformed.copy()
            transformed[positive_scales] = (
                transformed[positive_scales] / scales[positive_scales]
            )
        if self.rotation:
            rotation_matrix = np.asarray(self.rotation, dtype=np.float64)
            dims = int(transformed.size)
            if rotation_matrix.shape != (dims, dims):
                raise ValueError(
                    "PCA rotation matrix must be square with the same dimensionality "
                    f"as the projection (dims={dims})."
                )
            transformed = transformed @ rotation_matrix
        return tuple(float(value) for value in transformed.tolist())

    @classmethod
    def from_model(
        cls,
        model: PCA,
        sample_count: int,
        *,
        epoch: int = 0,
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
            epoch=max(0, int(epoch)),
            fitted_at=fitted_at or time.time(),
            whiten=bool(getattr(model, "whiten", False)),
            rotation=None,
        )


def align_pca_projection(
    *,
    projection: PCAProjection,
    reference: PCAProjection,
    anchors: Sequence[Sequence[float]],
) -> PCAProjection:
    """Align `projection` outputs to `reference` using orthogonal Procrustes.

    This helper computes an orthogonal matrix R (may include a reflection) that minimises:

        || Z_new R - Z_ref ||_F

    on the provided anchor vectors, where Z_new is produced by `projection` and
    Z_ref is produced by `reference`. The returned projection applies R after
    PCA (and whitening, when enabled).
    """

    if not anchors:
        return projection
    dims = projection.dimensions
    if dims <= 0:
        return projection
    if reference.dimensions != dims:
        log.debug(
            "Skipping PCA alignment due to dimensionality mismatch (ref={} new={})",
            reference.dimensions,
            dims,
        )
        return projection

    ref_rows: list[list[float]] = []
    new_rows: list[list[float]] = []
    for anchor in anchors:
        try:
            ref_rows.append(list(reference.transform(anchor)))
            new_rows.append(list(projection.transform(anchor)))
        except ValueError:
            continue

    if not ref_rows or len(ref_rows) != len(new_rows):
        return projection
    if len(ref_rows) < 2:
        return projection

    z_ref = np.asarray(ref_rows, dtype=np.float64)
    z_new = np.asarray(new_rows, dtype=np.float64)
    if z_ref.shape[1] != dims or z_new.shape[1] != dims:
        return projection

    try:
        m = z_new.T @ z_ref
        u, _s, vt = np.linalg.svd(m)
        r = u @ vt
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.debug("Unable to align PCA projections: {}", exc)
        return projection

    rotation = tuple(tuple(float(v) for v in row) for row in r.tolist())
    return replace(projection, rotation=rotation)


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
        samples_since_fit: int = 0,
    ) -> None:
        self.settings = settings or get_settings()
        self._target_dims = max(1, self.settings.mapelites_dimensionality_target_dims)
        self._min_fit_samples = max(
            2,
            int(self.settings.mapelites_dimensionality_min_fit_samples),
            int(self.settings.mapelites_feature_normalization_warmup_samples),
        )
        self._history_limit = resolve_pca_history_limit(self.settings)
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
        self._samples_since_fit = max(0, int(samples_since_fit or 0))

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

    @property
    def samples_since_fit(self) -> int:
        """Return the number of new samples recorded since the last PCA fit."""
        return int(self._samples_since_fit)

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
            raise ValueError(
                "PCA input dimensions changed from {} to {}. "
                "Loreley expects MAP-Elites embedding dimensions to remain stable for "
                "the lifetime of an instance. Check MAPELITES_CODE_EMBEDDING_DIMENSIONS "
                "and the configured embedding model, then reset the database with "
                "`uv run loreley reset-db --yes` if needed.".format(
                    self._feature_count,
                    dimensions,
                )
            )

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

        seed = int(getattr(self.settings, "mapelites_dimensionality_seed", 0) or 0)
        if seed < 0:
            seed = 0
        model = PCA(
            n_components=n_components,
            svd_solver="auto",
            whiten=True,
            random_state=seed,
        )
        try:
            sample_matrix = np.asarray(
                [entry.vector for entry in samples],
                dtype=np.float64,
            )
            model.fit(sample_matrix)
        except ValueError as exc:
            log.error("Unable to fit PCA: {}", exc)
            return None

        previous_epoch = self._projection.epoch if self._projection else -1
        projection = PCAProjection.from_model(
            model,
            len(samples),
            epoch=previous_epoch + 1,
        )
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
    samples_since_fit: int = 0,
) -> tuple[
    FinalEmbedding | None,
    tuple[PcaHistoryEntry, ...],
    PCAProjection | None,
    int,
]:
    """Convenience helper that runs the full reduction pipeline once.

    Returns a tuple of (final_embedding, updated_history, updated_projection, samples_since_fit).
    """

    reducer = DimensionReducer(
        settings=settings,
        history=history,
        projection=projection,
        samples_since_fit=samples_since_fit,
    )
    entry = reducer.build_history_entry(
        commit_hash=commit_hash,
        code_embedding=code_embedding,
    )
    if not entry:
        return None, reducer.history, reducer.projection, reducer.samples_since_fit

    reduced = reducer.reduce(entry)
    return reduced, reducer.history, reducer.projection, reducer.samples_since_fit

