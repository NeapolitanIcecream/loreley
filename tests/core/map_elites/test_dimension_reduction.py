from __future__ import annotations

from typing import Sequence

import pytest

from loreley.config import Settings
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.dimension_reduction import (
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
        fitted_at=123.0,
        whiten=False,
    )

    result = projection.transform((2.0, 4.0))
    assert result == (1.0, 2.0)

    with pytest.raises(ValueError):
        projection.transform((1.0, 2.0, 3.0))


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


def test_history_resets_on_dimension_change(settings: Settings) -> None:
    settings.mapelites_dimensionality_penultimate_normalize = False
    reducer = DimensionReducer(settings=settings)

    first = _make_entry((1.0, 0.0), commit_hash="a")
    second = _make_entry((1.0, 0.0, 0.0), commit_hash="b")

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

    final, history, projection = reduce_commit_embeddings(
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


