from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from loreley.config import Settings
from loreley.core.map_elites.dimension_reduction import (
    CommitCodeEmbedding,
    DimensionReducer,
    FinalEmbedding,
    PCAProjection,
    PcaHistoryEntry,
    reduce_commit_embeddings,
)


def _make_entry(vector: Sequence[float], commit_hash: str = "c") -> PcaHistoryEntry:
    return PcaHistoryEntry(
        commit_hash=commit_hash,
        vector=tuple(float(v) for v in vector),
        embedding_model="code",
    )


def test_pca_projection_transform_basic() -> None:
    projection = PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(1.0, 2.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(1.0, 1.0),
        sample_count=10,
        epoch=0,
        fitted_at=123.0,
        whiten=False,
        rotation=None,
    )

    result = projection.transform((2.0, 4.0))
    assert result == (1.0, 2.0)

    with pytest.raises(ValueError):
        projection.transform((1.0, 2.0, 3.0))


def test_pca_projection_transform_with_whiten_and_rotation() -> None:
    projection = PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(0.0, 0.0),
        explained_variance=(4.0, 1.0),
        explained_variance_ratio=(1.0, 1.0),
        sample_count=10,
        epoch=0,
        fitted_at=123.0,
        whiten=True,
        rotation=((0.0, 1.0), (-1.0, 0.0)),
    )

    result = projection.transform((4.0, 2.0))
    assert result == pytest.approx((-2.0, 2.0))


def test_pca_projection_transform_rejects_invalid_rotation_shape() -> None:
    projection = PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(1.0, 1.0),
        sample_count=10,
        epoch=0,
        fitted_at=123.0,
        whiten=False,
        rotation=((1.0, 0.0, 0.0),),
    )

    with pytest.raises(ValueError, match="must be square"):
        projection.transform((1.0, 2.0))


def test_pca_projection_transform_batch_matches_transform() -> None:
    projection = PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(0.0, 0.0),
        explained_variance=(4.0, 1.0),
        explained_variance_ratio=(1.0, 1.0),
        sample_count=10,
        epoch=0,
        fitted_at=123.0,
        whiten=True,
        rotation=((0.0, 1.0), (-1.0, 0.0)),
    )

    vectors = np.asarray(
        [
            (4.0, 2.0),
            (0.0, 0.0),
            (1.0, -1.0),
        ],
        dtype=np.float64,
    )
    projected = projection.transform_batch(vectors)
    expected = np.asarray([projection.transform(row) for row in vectors], dtype=np.float64)
    assert projected == pytest.approx(expected)

    with pytest.raises(ValueError):
        projection.transform_batch(np.asarray([1.0, 2.0], dtype=np.float64))
    with pytest.raises(ValueError):
        projection.transform_batch(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64))


def test_build_history_entry_returns_code_vector(settings: Settings) -> None:
    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(1.0, 2.0),
        model="code-model",
        dimensions=2,
    )

    settings.mapelites_dimensionality_penultimate_normalize = False
    reducer = DimensionReducer(settings=settings)

    entry = reducer.build_history_entry(
        commit_hash="abc",
        code_embedding=code_embedding,
    )
    assert entry is not None
    assert entry.vector == (1.0, 2.0)
    assert entry.embedding_model == "code-model"

    # When no embeddings are provided, return None
    empty = reducer.build_history_entry(commit_hash="empty")
    assert empty is None


def test_build_history_entry_normalizes_vector_when_enabled(settings: Settings) -> None:
    settings.mapelites_dimensionality_penultimate_normalize = True
    reducer = DimensionReducer(settings=settings)

    embedding = CommitCodeEmbedding(
        files=(),
        vector=(3.0, 4.0),
        model="code-model",
        dimensions=2,
    )
    entry = reducer.build_history_entry(commit_hash="norm", code_embedding=embedding)
    assert entry is not None
    assert entry.vector == pytest.approx((0.6, 0.8))

    zero_embedding = CommitCodeEmbedding(
        files=(),
        vector=(0.0, 0.0),
        model="code-model",
        dimensions=2,
    )
    zero_entry = reducer.build_history_entry(
        commit_hash="zero",
        code_embedding=zero_embedding,
    )
    assert zero_entry is not None
    assert zero_entry.vector == (0.0, 0.0)


def test_l2_normalize_rejects_non_1d_vector() -> None:
    with pytest.raises(ValueError, match="1-D"):
        DimensionReducer._l2_normalize(
            ((1.0, 2.0), (3.0, 4.0)),  # type: ignore[arg-type]
        )


def test_history_raises_on_dimension_change(settings: Settings) -> None:
    settings.mapelites_dimensionality_penultimate_normalize = False
    reducer = DimensionReducer(settings=settings)

    first = _make_entry((1.0, 0.0), commit_hash="a")
    second = _make_entry((1.0, 0.0, 0.0), commit_hash="b")

    reducer._record_history(first)  # type: ignore[attr-defined]
    assert len(reducer.history) == 1

    with pytest.raises(ValueError):
        reducer._record_history(second)  # type: ignore[attr-defined]
    assert len(reducer.history) == 1
    assert reducer.history[0].commit_hash == "a"


def test_fit_projection_respects_min_samples_and_target_dims(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 3
    settings.mapelites_dimensionality_min_fit_samples = 2
    settings.mapelites_feature_normalization_warmup_samples = 2
    settings.mapelites_dimensionality_penultimate_normalize = False

    reducer = DimensionReducer(settings=settings)

    first = _make_entry((1.0, 0.0), commit_hash="a")
    second = _make_entry((0.0, 1.0), commit_hash="b")

    reducer._record_history(first)  # type: ignore[attr-defined]
    assert reducer._fit_projection() is None  # type: ignore[attr-defined]

    reducer._record_history(second)  # type: ignore[attr-defined]
    projection = reducer._fit_projection()  # type: ignore[attr-defined]
    assert projection is not None
    assert projection.feature_count == 2
    assert 1 <= projection.dimensions <= settings.mapelites_dimensionality_target_dims


def test_reduce_commit_embeddings_end_to_end(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_dimensionality_min_fit_samples = 1

    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(1.0, 0.0),
        model="code-model",
        dimensions=2,
    )

    final, history, projection, samples_since_fit = reduce_commit_embeddings(
        commit_hash="abc",
        code_embedding=code_embedding,
        history=None,
        projection=None,
        settings=settings,
    )

    assert isinstance(final, FinalEmbedding)
    assert final.commit_hash == "abc"
    assert len(final.vector) == settings.mapelites_dimensionality_target_dims
    assert final.history_entry.embedding_model == "code-model"
    assert len(history) >= 1
    assert samples_since_fit >= 0


def test_reduce_commit_embeddings_refits_after_interval_across_calls(
    settings: Settings,
) -> None:
    """Regression: refit_interval must count across repeated one-shot calls."""

    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_dimensionality_min_fit_samples = 2
    settings.mapelites_feature_normalization_warmup_samples = 2
    settings.mapelites_dimensionality_history_size = 16
    settings.mapelites_dimensionality_refit_interval = 2
    settings.mapelites_dimensionality_penultimate_normalize = False

    history = None
    projection = None
    samples_since_fit = 0

    def _embed(commit: str, vector: Sequence[float]) -> CommitCodeEmbedding:
        return CommitCodeEmbedding(
            files=(),
            vector=tuple(float(v) for v in vector),
            model="code-model",
            dimensions=len(vector),
        )

    # Seed fit after two samples.
    _, history, projection, samples_since_fit = reduce_commit_embeddings(
        commit_hash="a",
        code_embedding=_embed("a", (1.0, 0.0, 0.0)),
        history=history,
        projection=projection,
        samples_since_fit=samples_since_fit,
        settings=settings,
    )
    assert projection is None

    _, history, projection, samples_since_fit = reduce_commit_embeddings(
        commit_hash="b",
        code_embedding=_embed("b", (0.0, 1.0, 0.0)),
        history=history,
        projection=projection,
        samples_since_fit=samples_since_fit,
        settings=settings,
    )
    assert projection is not None
    first_epoch = projection.epoch

    # Accumulate two new samples since the last fit; the second should refit.
    _, history, projection, samples_since_fit = reduce_commit_embeddings(
        commit_hash="c",
        code_embedding=_embed("c", (0.0, 0.0, 1.0)),
        history=history,
        projection=projection,
        samples_since_fit=samples_since_fit,
        settings=settings,
    )
    assert projection is not None
    assert projection.epoch == first_epoch
    assert samples_since_fit == 1

    _, history, projection, samples_since_fit = reduce_commit_embeddings(
        commit_hash="d",
        code_embedding=_embed("d", (1.0, 1.0, 0.0)),
        history=history,
        projection=projection,
        samples_since_fit=samples_since_fit,
        settings=settings,
    )
    assert projection is not None
    assert projection.epoch == first_epoch + 1
    assert samples_since_fit == 0


