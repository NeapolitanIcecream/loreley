from __future__ import annotations

from typing import Sequence

import pytest

from loreley.config import Settings
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import (
    DimensionReducer,
    FinalEmbedding,
    PCAProjection,
    PenultimateEmbedding,
    reduce_commit_embeddings,
)
from loreley.core.map_elites.summarization_embedding import CommitSummaryEmbedding


def _make_penultimate(vector: Sequence[float], commit_hash: str = "c") -> PenultimateEmbedding:
    return PenultimateEmbedding(
        commit_hash=commit_hash,
        vector=tuple(float(v) for v in vector),
        code_dimensions=len(vector),
        summary_dimensions=0,
        code_model="code",
        summary_model=None,
        summary_embedding_model=None,
    )


def test_pca_projection_transform_basic() -> None:
    projection = PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(1.0, 2.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(1.0, 1.0),
        sample_count=10,
        fitted_at=123.0,
        whiten=False,
    )

    result = projection.transform((2.0, 4.0))
    assert result == (1.0, 2.0)

    with pytest.raises(ValueError):
        projection.transform((1.0, 2.0, 3.0))


def test_build_penultimate_concatenates_and_normalises(settings: Settings) -> None:
    code_embedding = CommitCodeEmbedding(
        files=(),
        vector=(1.0, 2.0),
        model="code-model",
        dimensions=2,
    )
    summary_embedding = CommitSummaryEmbedding(
        summaries=(),
        vector=(3.0, 4.0),
        summary_model="summary-model",
        embedding_model="emb-model",
        dimensions=2,
    )

    settings.mapelites_dimensionality_penultimate_normalize = False
    reducer = DimensionReducer(settings=settings)

    penultimate = reducer.build_penultimate(
        commit_hash="abc",
        code_embedding=code_embedding,
        summary_embedding=summary_embedding,
    )
    assert penultimate is not None
    assert penultimate.vector == (1.0, 2.0, 3.0, 4.0)
    assert penultimate.code_dimensions == 2
    assert penultimate.summary_dimensions == 2
    assert penultimate.code_model == "code-model"
    assert penultimate.summary_model == "summary-model"
    assert penultimate.summary_embedding_model == "emb-model"

    # When no embeddings are provided, return None
    empty = reducer.build_penultimate(commit_hash="empty")
    assert empty is None


def test_history_resets_on_dimension_change(settings: Settings) -> None:
    settings.mapelites_dimensionality_penultimate_normalize = False
    reducer = DimensionReducer(settings=settings)

    first = _make_penultimate((1.0, 0.0), commit_hash="a")
    second = _make_penultimate((1.0, 0.0, 0.0), commit_hash="b")

    reducer._record_history(first)  # type: ignore[attr-defined]
    assert len(reducer.history) == 1

    reducer._record_history(second)  # type: ignore[attr-defined]
    assert len(reducer.history) == 1
    assert reducer.history[0].commit_hash == "b"


def test_fit_projection_respects_min_samples_and_target_dims(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 3
    settings.mapelites_dimensionality_min_fit_samples = 2
    settings.mapelites_feature_normalization_warmup_samples = 2
    settings.mapelites_dimensionality_penultimate_normalize = False

    reducer = DimensionReducer(settings=settings)

    first = _make_penultimate((1.0, 0.0), commit_hash="a")
    second = _make_penultimate((0.0, 1.0), commit_hash="b")

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
    summary_embedding = CommitSummaryEmbedding(
        summaries=(),
        vector=(0.0, 1.0),
        summary_model="summary-model",
        embedding_model="emb-model",
        dimensions=2,
    )

    final, history, projection = reduce_commit_embeddings(
        commit_hash="abc",
        code_embedding=code_embedding,
        summary_embedding=summary_embedding,
        history=None,
        projection=None,
        settings=settings,
    )

    assert isinstance(final, FinalEmbedding)
    assert final.commit_hash == "abc"
    assert len(final.vector) == settings.mapelites_dimensionality_target_dims
    assert len(history) >= 1


